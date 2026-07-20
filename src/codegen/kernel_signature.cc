/*! \file src/codegen/kernel_signature.cc
 * \brief 实现内核参数契约、签名顺序和启动元数据校验。
 */

#include "codegen/kernel_signature.h"

#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace kxc::codegen {
namespace {

// 判断 dtype code 是否属于当前 NDArray 和内核 ABI 都能解释的 DLPack 集合。
bool IsSupportedDType(DLDataType dtype) {
    switch (dtype.code) {
        case kDLInt:
        case kDLUInt:
        case kDLFloat:
        case kDLBfloat:
        case kDLComplex:
        case kDLBool:
            return true;
        default:
            return false;
    }
}

// 统一判断二的幂对齐；零不表示“默认”，因为签名必须完整自描述。
bool IsPowerOfTwo(uint64_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

// 将角色转换为稳定文本，供诊断和未来 manifest 复用。
const char* RoleName(KernelArgRole role) {
    switch (role) {
        case KernelArgRole::kInput:
            return "input";
        case KernelArgRole::kConstant:
            return "constant";
        case KernelArgRole::kOutput:
            return "output";
    }
    return "unknown";
}

// 返回角色的分段序号，签名只能从 input 单向推进到 constant 和 output。
int RoleOrder(KernelArgRole role) {
    switch (role) {
        case KernelArgRole::kInput:
            return 0;
        case KernelArgRole::kConstant:
            return 1;
        case KernelArgRole::kOutput:
            return 2;
    }
    throw std::invalid_argument("KernelArgSpec contains an unknown role");
}

// 检查 Dim3 的每一维，零维 launch 在 LLVM 和 CUDA 中都没有合法语义。
void ValidateDim3(const Dim3& value, const char* field) {
    if (value.x == 0 || value.y == 0 || value.z == 0) {
        throw std::invalid_argument(std::string(field) + " dimensions must be non-zero");
    }
}

}  // namespace

// 构造参数对象时立即完成验证，避免非法节点进入签名 Array。
KernelArgSpec::KernelArgSpec(String name, KernelArgRole role, DLDataType dtype,
                             Array<int64_t> shape, Device device, uint64_t alignment,
                             bool mutable_data, String constant_key) {
    auto* node = new KernelArgSpecNode();
    node->name = std::move(name);
    node->role = role;
    node->dtype = dtype;
    // 深拷贝调用方 Array，避免其后续 push/erase 改写已经验证的参数契约。
    for (int64_t dimension : shape) node->shape_.push_back(dimension);
    node->device = std::move(device);
    node->alignment = alignment;
    node->mutable_data = mutable_data;
    node->constant_key = std::move(constant_key);
    SetData(node);
    Validate();
}

// 从 ObjectRef 恢复时拒绝错误节点，避免静态转换造成未定义行为。
KernelArgSpec::KernelArgSpec(const ObjectRef& ref) : ObjectRef(ref) {
    if (defined() && !As<KernelArgSpecNode>()) {
        SetData(nullptr);
        throw std::invalid_argument("ObjectRef does not contain KernelArgSpecNode");
    }
    if (defined()) Validate();
}

// rank 是 shape 的计算属性，防止两个字段在迁移或序列化后失配。
int KernelArgSpec::rank() const {
    if (operator->()->shape_.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("KernelArgSpec rank exceeds int range");
    }
    return static_cast<int>(operator->()->shape_.size());
}

// 返回独立容器，防止 Array 的共享可变实现泄漏节点内部 shape。
Array<int64_t> KernelArgSpec::shape() const {
    Array<int64_t> result;
    for (int64_t dimension : operator->()->shape_) result.push_back(dimension);
    return result;
}

// 校验单参数自身不变量；跨参数的名称、key 和角色顺序由 Signature 校验。
void KernelArgSpec::Validate() const {
    const auto* node = operator->();
    if (std::string(node->name).empty()) {
        throw std::invalid_argument("KernelArgSpec name must not be empty");
    }
    if (!node->device.defined()) {
        throw std::invalid_argument("KernelArgSpec device must be defined");
    }
    if (!IsSupportedDType(node->dtype)) {
        throw std::invalid_argument("KernelArgSpec uses an unsupported dtype code");
    }
    if (node->dtype.bits == 0 || node->dtype.bits % 8 != 0 || node->dtype.lanes == 0) {
        throw std::invalid_argument(
            "KernelArgSpec dtype must be byte-aligned with non-zero bits and lanes");
    }
    if (!IsPowerOfTwo(node->alignment)) {
        throw std::invalid_argument("KernelArgSpec alignment must be a power of two");
    }

    for (int64_t dimension : node->shape_) {
        if (dimension < kDynamicDimension) {
            throw std::invalid_argument("KernelArgSpec shape contains an invalid dimension");
        }
        // 常量和输出需要在启动前分配或绑定，当前没有 shape function 时不能动态。
        if (dimension == kDynamicDimension && node->role != KernelArgRole::kInput) {
            throw std::invalid_argument(
                "Only input arguments may contain dynamic dimensions");
        }
    }

    const bool has_constant_key = !std::string(node->constant_key).empty();
    if (node->role == KernelArgRole::kConstant) {
        if (!has_constant_key) {
            throw std::invalid_argument("Constant KernelArgSpec requires constant_key");
        }
        if (node->mutable_data) {
            throw std::invalid_argument("Constant KernelArgSpec must be immutable");
        }
    } else if (has_constant_key) {
        throw std::invalid_argument("Only constant KernelArgSpec may define constant_key");
    }

    // 第一阶段 ABI 不支持 in-place input；输出是唯一允许内核写入的角色。
    if (node->role == KernelArgRole::kOutput && !node->mutable_data) {
        throw std::invalid_argument("Output KernelArgSpec must be mutable");
    }
    if (node->role == KernelArgRole::kInput && node->mutable_data) {
        throw std::invalid_argument("Input KernelArgSpec must be immutable");
    }
    (void)RoleOrder(node->role);
}

// 输出字段顺序固定，便于测试、日志比较和后续稳定序列化。
std::string KernelArgSpec::ToString() const {
    const auto* node = operator->();
    std::ostringstream os;
    os << "KernelArgSpec(name=" << std::string(node->name)
       << ", role=" << RoleName(node->role)
       << ", dtype=" << static_cast<int>(node->dtype.code) << ':'
       << static_cast<int>(node->dtype.bits) << 'x' << node->dtype.lanes
       << ", shape=[";
    for (size_t i = 0; i < node->shape_.size(); ++i) {
        if (i != 0) os << ',';
        os << node->shape_[i];
    }
    os << "], device=" << node->device.ToString()
       << ", alignment=" << node->alignment
       << ", mutable=" << (node->mutable_data ? "true" : "false");
    if (!std::string(node->constant_key).empty()) {
        os << ", constant_key=" << std::string(node->constant_key);
    }
    os << ')';
    return os.str();
}

// 返回类型安全节点；undefined 句柄不能被解释为空契约。
const KernelArgSpecNode* KernelArgSpec::operator->() const {
    const auto* node = As<KernelArgSpecNode>();
    if (!node) throw std::runtime_error("undefined or invalid KernelArgSpec");
    return node;
}

// 构造签名时立即验证全部参数间关系。
KernelSignature::KernelSignature(String symbol, Array<KernelArgSpec> arguments) {
    auto* node = new KernelSignatureNode();
    node->symbol = std::move(symbol);
    // 参数 Array 同样采用逐项复制，使调用方重排原数组不会改变签名 ABI。
    for (const auto& argument : arguments) node->arguments_.push_back(argument);
    SetData(node);
    Validate();
}

// 从 ObjectRef 恢复签名时执行运行时类型检查。
KernelSignature::KernelSignature(const ObjectRef& ref) : ObjectRef(ref) {
    if (defined() && !As<KernelSignatureNode>()) {
        SetData(nullptr);
        throw std::invalid_argument("ObjectRef does not contain KernelSignatureNode");
    }
    if (defined()) Validate();
}

// 参数状态机保证固定的 input -> constant -> output 分段，避免调用方猜测顺序。
void KernelSignature::Validate() const {
    const auto* node = operator->();
    if (std::string(node->symbol).empty()) {
        throw std::invalid_argument("KernelSignature symbol must not be empty");
    }
    if (node->arguments_.empty()) {
        throw std::invalid_argument("KernelSignature must contain arguments");
    }

    std::unordered_set<std::string> names;
    std::unordered_set<std::string> constant_keys;
    int current_role = 0;
    bool saw_output = false;
    Device signature_device;
    for (const auto& argument : node->arguments_) {
        argument.Validate();
        const auto* spec = argument.operator->();
        const int role = RoleOrder(spec->role);
        if (role < current_role) {
            throw std::invalid_argument(
                "KernelSignature arguments must be ordered input, constant, output");
        }
        current_role = role;
        saw_output = saw_output || spec->role == KernelArgRole::kOutput;
        if (!signature_device.defined()) {
            signature_device = spec->device;
        } else if (signature_device != spec->device) {
            throw std::invalid_argument(
                "KernelSignature arguments must reside on one physical device");
        }

        const std::string name = spec->name;
        if (!names.insert(name).second) {
            throw std::invalid_argument("KernelSignature argument names must be unique");
        }
        if (spec->role == KernelArgRole::kConstant) {
            const std::string key = spec->constant_key;
            if (!constant_keys.insert(key).second) {
                throw std::invalid_argument("KernelSignature constant keys must be unique");
            }
        }
    }
    if (!saw_output) {
        throw std::invalid_argument("KernelSignature requires at least one output");
    }
}

// 返回独立参数容器；其中的 KernelArgSpec 本身也不暴露可变 shape。
Array<KernelArgSpec> KernelSignature::arguments() const {
    Array<KernelArgSpec> result;
    for (const auto& argument : operator->()->arguments_) result.push_back(argument);
    return result;
}

// 动态输入从参数 shape 推导，不在节点中保存可能失真的冗余标志。
bool KernelSignature::has_dynamic_input_shape() const {
    for (const auto& argument : operator->()->arguments_) {
        if (argument->role != KernelArgRole::kInput) continue;
        for (int64_t dimension : argument.shape()) {
            if (dimension == kDynamicDimension) return true;
        }
    }
    return false;
}

// 按参数原顺序生成诊断，不能对 Array 排序而改变 ABI 位置。
std::string KernelSignature::ToString() const {
    const auto* node = operator->();
    std::ostringstream os;
    os << "KernelSignature(symbol=" << std::string(node->symbol) << ", arguments=[";
    for (size_t i = 0; i < node->arguments_.size(); ++i) {
        if (i != 0) os << ", ";
        os << node->arguments_[i].ToString();
    }
    os << "])";
    return os.str();
}

// 返回类型安全节点；undefined 句柄不能被解释为空签名。
const KernelSignatureNode* KernelSignature::operator->() const {
    const auto* node = As<KernelSignatureNode>();
    if (!node) throw std::runtime_error("undefined or invalid KernelSignature");
    return node;
}

// 构造启动元数据时立即执行 backend/device 一致性检查。
KernelLaunchMetadata::KernelLaunchMetadata(
    Device device, CodeGenBackend backend, Dim3 grid, Dim3 block,
    uint64_t dynamic_shared_memory_bytes) {
    auto* node = new KernelLaunchMetadataNode();
    node->device = std::move(device);
    node->backend = backend;
    node->grid = grid;
    node->block = block;
    node->dynamic_shared_memory_bytes = dynamic_shared_memory_bytes;
    SetData(node);
    Validate();
}

// 从 ObjectRef 恢复启动元数据时执行运行时类型检查。
KernelLaunchMetadata::KernelLaunchMetadata(const ObjectRef& ref) : ObjectRef(ref) {
    if (defined() && !As<KernelLaunchMetadataNode>()) {
        SetData(nullptr);
        throw std::invalid_argument("ObjectRef does not contain KernelLaunchMetadataNode");
    }
    if (defined()) Validate();
}

// CPU/LLVM 与 CUDA 的启动字段含义不同，因此在对象边界严格隔离两套规则。
void KernelLaunchMetadata::Validate() const {
    const auto* node = operator->();
    if (!node->device.defined()) {
        throw std::invalid_argument("KernelLaunchMetadata device must be defined");
    }
    ValidateDim3(node->grid, "grid");
    ValidateDim3(node->block, "block");

    if (node->backend == CodeGenBackend::kLLVM) {
        if (node->device.device_type() != kCPU) {
            throw std::invalid_argument("LLVM launch metadata requires a CPU device");
        }
        if (node->grid.x != 1 || node->grid.y != 1 || node->grid.z != 1 ||
            node->block.x != 1 || node->block.y != 1 || node->block.z != 1 ||
            node->dynamic_shared_memory_bytes != 0) {
            throw std::invalid_argument(
                "LLVM launch metadata cannot define CUDA launch dimensions");
        }
        return;
    }
    if (node->backend == CodeGenBackend::kCUDA) {
        if (node->device.device_type() != kCUDA) {
            throw std::invalid_argument("CUDA launch metadata requires a CUDA device");
        }
        return;
    }
    throw std::invalid_argument("KernelLaunchMetadata uses a non-executable backend");
}

// 输出不依赖枚举数值，避免未来内部重排影响诊断文本。
std::string KernelLaunchMetadata::ToString() const {
    const auto* node = operator->();
    const char* backend = node->backend == CodeGenBackend::kLLVM ? "llvm" : "cuda";
    std::ostringstream os;
    os << "KernelLaunchMetadata(backend=" << backend
       << ", device=" << node->device.ToString()
       << ", grid=" << node->grid.x << 'x' << node->grid.y << 'x' << node->grid.z
       << ", block=" << node->block.x << 'x' << node->block.y << 'x' << node->block.z
       << ", dynamic_shared_memory_bytes=" << node->dynamic_shared_memory_bytes << ')';
    return os.str();
}

// 返回类型安全节点；undefined 句柄不能被解释为空元数据。
const KernelLaunchMetadataNode* KernelLaunchMetadata::operator->() const {
    const auto* node = As<KernelLaunchMetadataNode>();
    if (!node) throw std::runtime_error("undefined or invalid KernelLaunchMetadata");
    return node;
}

}  // namespace kxc::codegen
