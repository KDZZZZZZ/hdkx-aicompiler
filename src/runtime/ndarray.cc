/*! \file src/runtime/ndarray.cc
 * \brief 实现 Storage-backed 连续张量、视图和设备间复制。
 */

#include "kxc/runtime/ndarray.h"
#include "kxc/support/object_registration.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "kxc/runtime/device_api.h"

namespace kxc::runtime {

KXC_OBJECT_DEFINE(NDArrayNode)

namespace {

// 计算单元素字节数，并拒绝不可按字节寻址或 lanes 乘法溢出的 dtype。
size_t ElementBytes(DLDataType dtype) {
    if (dtype.lanes == 0 || dtype.bits == 0 || dtype.bits % 8 != 0) {
        throw std::invalid_argument("dtype bits must be byte-aligned and lanes non-zero");
    }
    const size_t bytes = dtype.bits / 8;
    if (bytes > std::numeric_limits<size_t>::max() / dtype.lanes) {
        throw std::overflow_error("dtype byte width overflow");
    }
    return bytes * dtype.lanes;
}

// 计算连续张量字节数，同时校验负维和所有乘法边界。
size_t TensorBytes(const std::vector<int64_t>& shape, DLDataType dtype) {
    const size_t element_bytes = ElementBytes(dtype);
    bool has_zero_dimension = false;
    // 必须扫描全部维度后再处理零尺寸，避免 {0, -1} 绕过负维校验。
    for (int64_t dim : shape) {
        if (dim < 0) throw std::invalid_argument("NDArray shape cannot be negative");
        has_zero_dimension = has_zero_dimension || dim == 0;
    }
    if (has_zero_dimension) return 0;
    // 空 shape 表示标量，因此元素计数从 1 开始；每一步都检查乘法溢出。
    size_t elements = 1;
    for (int64_t dim : shape) {
        const size_t value = static_cast<size_t>(dim);
        if (elements > std::numeric_limits<size_t>::max() / value) {
            throw std::overflow_error("NDArray element count overflow");
        }
        elements *= value;
    }
    if (elements > std::numeric_limits<size_t>::max() / element_bytes) {
        throw std::overflow_error("NDArray byte size overflow");
    }
    return elements * element_bytes;
}

// 将对象系统容器复制为节点内部稳定持有的 shape/stride 存储。
std::vector<int64_t> ToVector(const Array<int64_t>& values) {
    std::vector<int64_t> result;
    result.reserve(values.size());
    for (int64_t value : values) result.push_back(value);
    return result;
}

// 将内部 shape/stride 副本转换为公开对象系统容器。
Array<int64_t> ToArray(const std::vector<int64_t>& values) {
    Array<int64_t> result;
    for (int64_t value : values) result.push_back(value);
    return result;
}

// 按 DLPack 的 code、bits 和 lanes 完整比较 dtype。
bool SameDType(DLDataType a, DLDataType b) {
    return a.code == b.code && a.bits == b.bits && a.lanes == b.lanes;
}

// 判断 dtype code 是否属于当前 NDArray 可安全解释的集合。
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

// 用节点当前成员刷新对外 DLTensor 视图及其借用指针。
void RefreshDLTensor(NDArrayNode* node) {
    // DLTensor 的 shape/strides 借用节点内 vector，成员移动完成后必须刷新指针。
    node->dl_tensor.data = node->storage.data();
    node->dl_tensor.device = ToDLDevice(node->storage.device());
    node->dl_tensor.ndim = static_cast<int32_t>(node->shape_storage.size());
    node->dl_tensor.dtype = node->dtype;
    node->dl_tensor.shape = node->shape_storage.data();
    node->dl_tensor.strides = node->strides_storage.empty()
                                  ? nullptr
                                  : node->strides_storage.data();
    node->dl_tensor.byte_offset = node->byte_offset;
}

// Empty 与 View 共用的构造入口，统一建立经过容量校验的连续张量对象。
NDArray MakeArray(std::vector<int64_t> shape, DLDataType dtype, Storage storage,
                  std::vector<int64_t> strides, size_t byte_offset) {
    // 在发布 DLTensor 借用视图前统一验证 dtype、字节数及 Storage 容量。
    if (!IsSupportedDType(dtype)) throw std::invalid_argument("unsupported dtype code");
    const size_t nbytes = TensorBytes(shape, dtype);
    storage.ValidateRange(byte_offset, nbytes);
    auto* node = new NDArrayNode();
    NDArray array(node);
    node->storage = std::move(storage);
    node->dtype = dtype;
    node->shape_storage = std::move(shape);
    node->strides_storage = std::move(strides);
    node->byte_offset = byte_offset;
    RefreshDLTensor(node);
    return array;
}

}  // namespace

// 将受支持的文本 dtype 解析为 DLPack 描述。
DLDataType DataTypeFromString(const std::string& dtype) {
    if (dtype == "float16") return {kDLFloat, 16, 1};
    if (dtype == "float32") return {kDLFloat, 32, 1};
    if (dtype == "float64") return {kDLFloat, 64, 1};
    if (dtype == "int8") return {kDLInt, 8, 1};
    if (dtype == "int16") return {kDLInt, 16, 1};
    if (dtype == "int32") return {kDLInt, 32, 1};
    if (dtype == "int64") return {kDLInt, 64, 1};
    if (dtype == "uint8") return {kDLUInt, 8, 1};
    if (dtype == "bool") return {kDLBool, 8, 1};
    throw std::invalid_argument("unsupported dtype string: " + dtype);
}

// 从通用对象引用恢复 NDArray，并执行运行时类型检查。
NDArray::NDArray(const ObjectRef& ref) : ObjectRef(ref) {
    if (defined() && !As<NDArrayNode>()) {
        throw std::runtime_error("ObjectRef does not contain NDArrayNode");
    }
}

// 在指定设备分配未初始化的连续张量。
NDArray NDArray::Empty(Array<int64_t> shape, DLDataType dtype, Device device,
                       size_t alignment) {
    std::vector<int64_t> shape_storage = ToVector(shape);
    const size_t nbytes = TensorBytes(shape_storage, dtype);
    return MakeArray(std::move(shape_storage), dtype,
                     Storage::Alloc(device, nbytes, alignment), {}, 0);
}

// 分配并按后端语义清零连续张量。
NDArray NDArray::Zeros(Array<int64_t> shape, DLDataType dtype, Device device,
                       size_t alignment) {
    NDArray result =
        Empty(std::move(shape), dtype, std::move(device), alignment);
    const size_t nbytes = result.NBytes();
    DeviceZero(result.device(), result.storage().data(), result->byte_offset, nbytes);
    return result;
}

// 返回承载当前张量数据的物理设备。
Device NDArray::device() const { return operator->()->storage.device(); }
// 返回 shape 副本，避免调用方修改 DLTensor 借用的内部存储。
Array<int64_t> NDArray::shape() const { return ToArray(operator->()->shape_storage); }
// 返回张量的 DLPack dtype 描述。
DLDataType NDArray::dtype() const { return operator->()->dtype; }
// 返回共享所有权的底层 Storage。
Storage NDArray::storage() const { return operator->()->storage; }
// 计算当前逻辑视图覆盖的连续字节数。
size_t NDArray::NBytes() const {
    return TensorBytes(operator->()->shape_storage, operator->()->dtype);
}

// 验证显式 strides 是否描述连续布局，并拒绝 stride 计算溢出。
bool NDArray::IsContiguous() const {
    const auto* node = operator->();
    if (node->strides_storage.empty()) return true;
    if (node->strides_storage.size() != node->shape_storage.size()) return false;
    int64_t expected = 1;
    for (size_t i = node->shape_storage.size(); i-- > 0;) {
        // 长度为 1 的维度不影响地址连续性，其 stride 可以忽略。
        if (node->shape_storage[i] != 1 && node->strides_storage[i] != expected) {
            return false;
        }
        if (node->shape_storage[i] != 0) {
            if (expected > std::numeric_limits<int64_t>::max() /
                               std::max<int64_t>(node->shape_storage[i], 1)) {
                return false;
            }
            expected *= std::max<int64_t>(node->shape_storage[i], 1);
        }
    }
    return true;
}

// 从 CPU 字节缓冲区同步复制完整张量内容。
void NDArray::CopyFromBytes(const void* source, size_t nbytes) const {
    if (nbytes != NBytes()) throw std::invalid_argument("NDArray byte size mismatch");
    if (nbytes != 0 && source == nullptr) {
        throw std::invalid_argument("non-empty copy requires source");
    }
    DeviceCopySync(Device::CPU(), source, 0, device(), storage().data(),
                   operator->()->byte_offset, nbytes);
}

// 将完整张量内容同步复制到 CPU 字节缓冲区。
void NDArray::CopyToBytes(void* destination, size_t nbytes) const {
    if (nbytes != NBytes()) throw std::invalid_argument("NDArray byte size mismatch");
    if (nbytes != 0 && destination == nullptr) {
        throw std::invalid_argument("non-empty copy requires destination");
    }
    DeviceCopySync(device(), storage().data(), operator->()->byte_offset,
                   Device::CPU(), destination, 0, nbytes);
}

// 在 shape、dtype 和连续布局一致时执行 Storage 级同步复制。
void NDArray::CopyFrom(const NDArray& source) const {
    if (!source.defined()) throw std::invalid_argument("source NDArray is undefined");
    if (operator->()->shape_storage != source->shape_storage ||
        !SameDType(dtype(), source.dtype()) || !IsContiguous() ||
        !source.IsContiguous()) {
        throw std::invalid_argument("NDArray copy requires matching contiguous layout");
    }
    StorageCopySync(source.storage(), source->byte_offset, storage(),
                    operator->()->byte_offset, NBytes());
}

// 在目标设备分配同构张量并完成同步复制。
NDArray NDArray::CopyTo(const Device& destination) const {
    NDArray result = Empty(shape(), dtype(), destination);
    result.CopyFrom(*this);
    return result;
}

// 发起 Storage 级异步复制，返回保活两端内存的完成句柄。
AsyncOperation NDArray::CopyFromAsync(const NDArray& source,
                                      const DeviceStream& stream) const {
    if (operator->()->shape_storage != source->shape_storage ||
        !SameDType(dtype(), source.dtype()) || !IsContiguous() ||
        !source.IsContiguous()) {
        throw std::invalid_argument("NDArray async copy requires matching contiguous layout");
    }
    return StorageCopyAsync(source.storage(), source->byte_offset, storage(),
                            operator->()->byte_offset, NBytes(), stream);
}

// 在同一 Storage 上创建经过容量和连续性校验的视图。
NDArray NDArray::CreateView(Array<int64_t> shape, Array<int64_t> strides,
                            size_t byte_offset) const {
    std::vector<int64_t> shape_storage = ToVector(shape);
    std::vector<int64_t> strides_storage = ToVector(strides);
    // View 共享原 Storage 的引用计数生命周期，当前公共 API 仅接受连续布局。
    NDArray view = MakeArray(std::move(shape_storage), dtype(), storage(),
                             std::move(strides_storage), byte_offset);
    if (!view.IsContiguous()) {
        throw std::invalid_argument("non-contiguous NDArray views are not supported");
    }
    return view;
}

// 返回经过 NDArray 类型约束的底层节点。
const NDArrayNode* NDArray::operator->() const {
    const auto* node = As<NDArrayNode>();
    if (!node) throw std::runtime_error("undefined or invalid NDArray");
    return node;
}

// 暴露只读 DLTensor 视图，其内部借用指针由 NDArrayNode 保活。
const DLTensor* NDArray::operator*() const { return &operator->()->dl_tensor; }

}  // namespace kxc::runtime
