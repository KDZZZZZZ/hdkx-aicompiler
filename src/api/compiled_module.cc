/*! \file src/api/compiled_module.cc
 * \brief 实现编译模块组装、NDArray 契约校验和后端启动分发。
 */

#include "api/compiled_module.h"

#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace kxc::api {
namespace {

// DLPack dtype 必须逐字段相等，尤其不能忽略向量 lanes。
bool SameDType(DLDataType lhs, DLDataType rhs) {
    return lhs.code == rhs.code && lhs.bits == rhs.bits && lhs.lanes == rhs.lanes;
}

// 把 dtype 格式化为稳定诊断文本，避免错误信息只显示枚举 code。
std::string DTypeText(DLDataType dtype) {
    std::ostringstream os;
    os << static_cast<int>(dtype.code) << ':' << static_cast<int>(dtype.bits)
       << 'x' << dtype.lanes;
    return os.str();
}

// 参数错误统一附带 symbol、序号和名称，便于定位大型模型中的 ABI 槽位。
[[noreturn]] void ThrowArgumentError(const codegen::KernelSignature& signature,
                                     size_t index,
                                     const codegen::KernelArgSpec& spec,
                                     const std::string& detail) {
    throw std::invalid_argument(
        "kernel '" + std::string(signature->symbol) + "' argument[" +
        std::to_string(index) + "] '" + std::string(spec->name) + "': " + detail);
}

// 返回 Target 所描述的物理设备，构造函数已经保证 target defined。
Device TargetDevice(const Target& target) {
    return Device(target->device_type, target->device_id);
}

// 容器当前没有 COW，所有对外返回值都必须显式复制容器节点。
Map<String, runtime::NDArray> CopyConstants(
    const Map<String, runtime::NDArray>& source) {
    Map<String, runtime::NDArray> result;
    for (const auto& item : source) result.Set(item.first, item.second);
    return result;
}

// 检查模块常量表与签名 constant 段严格一一对应，拒绝缺项和多余项。
void ValidateConstants(const codegen::KernelSignature& signature,
                       const Map<String, runtime::NDArray>& constants) {
    size_t expected_count = 0;
    const Array<codegen::KernelArgSpec> specs = signature.arguments();
    for (const auto& spec : specs) {
        if (spec->role != codegen::KernelArgRole::kConstant) continue;
        ++expected_count;
        if (!constants.count(spec->constant_key)) {
            throw std::invalid_argument(
                "CompiledModule is missing constant '" +
                std::string(spec->constant_key) + "'");
        }
        const runtime::NDArray& value = constants.at(spec->constant_key);
        if (!value.defined()) {
            throw std::invalid_argument(
                "CompiledModule constant '" + std::string(spec->constant_key) +
                "' is undefined");
        }
        if (!SameDType(value.dtype(), spec->dtype) || value.device() != spec->device) {
            throw std::invalid_argument(
                "CompiledModule constant '" + std::string(spec->constant_key) +
                "' dtype or device does not match its signature");
        }
        const Array<int64_t> expected_shape = spec.shape();
        const Array<int64_t> actual_shape = value.shape();
        if (expected_shape.size() != actual_shape.size()) {
            throw std::invalid_argument(
                "CompiledModule constant '" + std::string(spec->constant_key) +
                "' rank does not match its signature");
        }
        for (size_t i = 0; i < expected_shape.size(); ++i) {
            if (expected_shape[i] != actual_shape[i]) {
                throw std::invalid_argument(
                    "CompiledModule constant '" + std::string(spec->constant_key) +
                    "' shape does not match its signature");
            }
        }
    }
    if (constants.size() != expected_count) {
        throw std::invalid_argument("CompiledModule constant table contains unexpected keys");
    }
}

// 对单个 NDArray 执行所有不依赖其他参数的安全检查。
void ValidateArgument(const codegen::KernelSignature& signature,
                      size_t index,
                      const codegen::KernelArgSpec& spec,
                      const runtime::NDArray& argument,
                      const Map<String, runtime::NDArray>& constants) {
    if (!argument.defined()) {
        ThrowArgumentError(signature, index, spec, "NDArray is undefined");
    }
    if (!argument.storage().defined()) {
        ThrowArgumentError(signature, index, spec, "Storage is undefined");
    }
    if (!SameDType(argument.dtype(), spec->dtype)) {
        ThrowArgumentError(signature, index, spec,
                           "dtype expected " + DTypeText(spec->dtype) +
                               ", actual " + DTypeText(argument.dtype()));
    }
    if (argument.device() != spec->device) {
        ThrowArgumentError(signature, index, spec,
                           "device expected " + spec->device.ToString() +
                               ", actual " + argument.device().ToString());
    }
    if (!argument.IsContiguous()) {
        ThrowArgumentError(signature, index, spec,
                           "layout must be contiguous");
    }

    const Array<int64_t> expected_shape = spec.shape();
    const Array<int64_t> actual_shape = argument.shape();
    if (actual_shape.size() != expected_shape.size()) {
        ThrowArgumentError(signature, index, spec,
                           "rank expected " + std::to_string(expected_shape.size()) +
                               ", actual " + std::to_string(actual_shape.size()));
    }
    for (size_t dimension = 0; dimension < expected_shape.size(); ++dimension) {
        if (actual_shape[dimension] < 0) {
            ThrowArgumentError(signature, index, spec,
                               "actual shape contains a negative dimension");
        }
        if (expected_shape[dimension] != codegen::kDynamicDimension &&
            expected_shape[dimension] != actual_shape[dimension]) {
            ThrowArgumentError(
                signature, index, spec,
                "shape dimension " + std::to_string(dimension) + " expected " +
                    std::to_string(expected_shape[dimension]) + ", actual " +
                    std::to_string(actual_shape[dimension]));
        }
    }

    size_t nbytes = 0;
    try {
        nbytes = argument.NBytes();
        argument.storage().ValidateRange(argument->byte_offset, nbytes);
    } catch (const std::exception& error) {
        ThrowArgumentError(signature, index, spec,
                           std::string("storage range is invalid: ") + error.what());
    }

    // 零元素张量不会被内核解引用，允许其 Storage data 为 nullptr。
    if (nbytes != 0) {
        void* base = argument.storage().data();
        if (!base) {
            ThrowArgumentError(signature, index, spec,
                               "non-empty tensor has a null data pointer");
        }
        const uintptr_t base_address = reinterpret_cast<uintptr_t>(base);
        if (argument->byte_offset >
            std::numeric_limits<uintptr_t>::max() - base_address) {
            ThrowArgumentError(signature, index, spec,
                               "effective data address overflows uintptr_t");
        }
        const uintptr_t effective_address = base_address + argument->byte_offset;
        if (effective_address % spec->alignment != 0) {
            ThrowArgumentError(signature, index, spec,
                               "effective data address does not satisfy alignment " +
                                   std::to_string(spec->alignment));
        }
    }

    // 编译常量不可被调用方替换，否则 constant_key 将失去稳定绑定语义。
    if (spec->role == codegen::KernelArgRole::kConstant) {
        const runtime::NDArray& bound = constants.at(spec->constant_key);
        if (argument.get() != bound.get()) {
            ThrowArgumentError(signature, index, spec,
                               "constant argument does not match the bound payload");
        }
    }
}

}  // namespace

// 组装时校验 Target、签名、元数据、常量表和 executable 的交叉一致性。
CompiledModule::CompiledModule(
    Target target,
    tir::PrimFunc prim_func,
    codegen::KernelSignature signature,
    codegen::KernelLaunchMetadata launch_metadata,
    Map<String, runtime::NDArray> constants,
    codegen::CompiledKernel executable,
    std::shared_ptr<profiling::ProfileContext> profile_context) {
    if (!target.defined()) {
        throw std::invalid_argument("CompiledModule target must be defined");
    }
    if (!signature.defined() || !launch_metadata.defined()) {
        throw std::invalid_argument(
            "CompiledModule signature and launch metadata must be defined");
    }
    signature.Validate();
    launch_metadata.Validate();
    const Device target_device = TargetDevice(target);
    if (launch_metadata->device != target_device) {
        throw std::invalid_argument(
            "CompiledModule target and launch metadata devices do not match");
    }
    if (!executable.defined() || !executable.IsReady()) {
        throw std::invalid_argument("CompiledModule executable must be ready");
    }
    // 要求共享同一签名和元数据节点，彻底消除装配时的 ABI 漂移。
    if (executable.signature().get() != signature.get() ||
        executable.launch_metadata().get() != launch_metadata.get()) {
        throw std::invalid_argument(
            "CompiledModule executable contract does not match module metadata");
    }
    const Array<codegen::KernelArgSpec> specs = signature.arguments();
    for (const auto& spec : specs) {
        if (spec->device != target_device) {
            throw std::invalid_argument(
                "CompiledModule signature device does not match target");
        }
    }
    ValidateConstants(signature, constants);

    SetData(new CompiledModuleNode(
        std::move(target), std::move(prim_func), std::move(signature),
        std::move(launch_metadata), CopyConstants(constants),
        std::move(executable), std::move(profile_context)));
}

// ObjectRef 恢复只允许 CompiledModuleNode，错误类型立即在 API 边界失败。
CompiledModule::CompiledModule(const ObjectRef& ref) : ObjectRef(ref) {
    if (defined() && !As<CompiledModuleNode>()) {
        SetData(nullptr);
        throw std::invalid_argument("ObjectRef does not contain CompiledModuleNode");
    }
}

// 所有检查完成后才把 NDArray 交给 launcher，校验失败不会产生任何后端副作用。
AsyncOperation CompiledModule::Launch(
    const Array<runtime::NDArray>& ordered_arguments,
    const DeviceStream& stream) const {
    const auto* node = operator->();
    if (!node->executable_.IsReady()) {
        throw std::runtime_error(
            "kernel '" + std::string(node->signature_->symbol) +
            "' executable is not ready");
    }
    if (!stream.defined()) {
        throw std::invalid_argument(
            "kernel '" + std::string(node->signature_->symbol) +
            "' requires a defined DeviceStream");
    }
    if (stream.device() != node->launch_metadata_->device) {
        throw std::invalid_argument(
            "kernel '" + std::string(node->signature_->symbol) +
            "' stream device expected " + node->launch_metadata_->device.ToString() +
            ", actual " + stream.device().ToString());
    }

    const Array<codegen::KernelArgSpec> specs = node->signature_.arguments();
    if (ordered_arguments.size() != specs.size()) {
        throw std::invalid_argument(
            "kernel '" + std::string(node->signature_->symbol) +
            "' argument count expected " + std::to_string(specs.size()) +
            ", actual " + std::to_string(ordered_arguments.size()));
    }
    for (size_t i = 0; i < specs.size(); ++i) {
        ValidateArgument(node->signature_, i, specs[i], ordered_arguments[i],
                         node->constants_);
    }
    return node->executable_.Launch(ordered_arguments, stream);
}

// 签名对象内部已经防止 Array 别名修改，可安全返回共享句柄。
codegen::KernelSignature CompiledModule::signature() const {
    return operator->()->signature_;
}

// 启动元数据是经过构造校验的只读对象句柄。
codegen::KernelLaunchMetadata CompiledModule::launch_metadata() const {
    return operator->()->launch_metadata_;
}

// Map 没有 COW，因此这里必须逐项复制，调用方修改副本不会污染模块状态。
Map<String, runtime::NDArray> CompiledModule::constants() const {
    return CopyConstants(operator->()->constants_);
}

// Target 节点由设备模型提供不可变能力快照，返回共享句柄即可。
Target CompiledModule::target() const { return operator->()->target_; }

// PrimFunc 只作为诊断产物保存，执行契约只读取 signature 和 metadata。
tir::PrimFunc CompiledModule::prim_func() const { return operator->()->prim_func_; }

// undefined 模块不 ready；defined 模块继续询问内部 executable 的实时状态。
bool CompiledModule::IsReady() const noexcept {
    const auto* node = As<CompiledModuleNode>();
    return node != nullptr && node->executable_.IsReady();
}

// 当前模块只在 executable ready 后才能组装，因此状态集合保持最小且确定。
String CompiledModule::GetStatus() const {
    return String(IsReady() ? "ready" : "not_ready");
}

// profiling 未启用时使用空字符串，不让调用方依赖空指针检查。
String CompiledModule::GetProfileBundlePath() const {
    const auto* node = operator->();
    return String(node->profile_context_ ? node->profile_context_->bundle_dir() : "");
}

// 所有访问统一进行动态类型检查，避免错误 ObjectRef 被静态解释为模块。
const CompiledModuleNode* CompiledModule::operator->() const {
    const auto* node = As<CompiledModuleNode>();
    if (!node) throw std::runtime_error("undefined or invalid CompiledModule");
    return node;
}

}  // namespace kxc::api
