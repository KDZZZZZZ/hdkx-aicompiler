/*! \file src/base/ndarray.cc
 * \brief 实现基础对象、设备、NDArray、Target、执行计划、PassContext 和 profiling 支撑逻辑。
 */

#include "base/ndarray.h"

#include <cstring>
#include <stdexcept>

namespace kxc {
namespace runtime {

namespace {

DLDataType ParseDType(const std::string& dtype_str) {
    if (dtype_str == "float32") return {kDLFloat, 32, 1};
    if (dtype_str == "int32") return {kDLInt, 32, 1};
    if (dtype_str == "float64") return {kDLFloat, 64, 1};
    if (dtype_str == "int64") return {kDLInt, 64, 1};
    if (dtype_str == "int8") return {kDLInt, 8, 1};
    if (dtype_str == "uint8") return {kDLUint, 8, 1};
    if (dtype_str == "bool") return {kDLUint, 1, 1};
    return {kDLFloat, 32, 1};
}

}  // namespace

NDArrayNode::~NDArrayNode() {
    if (dl_tensor.data) {
        delete[] static_cast<char*>(dl_tensor.data);
        dl_tensor.data = nullptr;
    }
}

NDArray::NDArray(Array<int64_t> shape, std::string dtype_str) {
    auto* node = new NDArrayNode();

    for (auto dim : shape) {
        node->shape.push_back(dim);
    }

    node->dl_tensor.shape = node->shape.data();
    node->dl_tensor.ndim = static_cast<int>(node->shape.size());
    node->dl_tensor.strides = nullptr;
    node->dl_tensor.byte_offset = 0;
    node->dl_tensor.ctx = {kDLCPU, 0};
    node->dl_tensor.dtype = ParseDType(dtype_str);

    int64_t elements = 1;
    for (auto dim : node->shape) {
        elements *= dim;
    }
    size_t bytes = static_cast<size_t>(elements) *
                   ((node->dl_tensor.dtype.bits + 7) / 8);
    if (bytes == 0) {
        bytes = 1;
    }

    node->dl_tensor.data = new char[bytes];
    std::memset(node->dl_tensor.data, 0, bytes);
    SetData(node);
}

const NDArrayNode* NDArray::operator->() const {
    return static_cast<const NDArrayNode*>(object_);
}

const DLTensor* NDArray::operator*() const {
    return &operator->()->dl_tensor;
}

size_t NDArray::NBytes() const {
    const NDArrayNode* node = operator->();
    int64_t elements = 1;
    for (int64_t dim : node->shape) {
        elements *= dim;
    }
    size_t bytes = static_cast<size_t>(elements) * ((node->dl_tensor.dtype.bits + 7) / 8);
    return bytes == 0 ? 1 : bytes;
}

void NDArray::CopyFromBytes(const void* data, size_t nbytes) const {
    if (!defined()) {
        throw std::runtime_error("CopyFromBytes requires a defined NDArray");
    }
    if (!data && nbytes != 0) {
        throw std::runtime_error("CopyFromBytes received null data");
    }
    size_t expected = NBytes();
    if (nbytes != expected) {
        throw std::runtime_error("CopyFromBytes byte size mismatch: expected " +
                                 std::to_string(expected) + ", got " +
                                 std::to_string(nbytes));
    }
    std::memcpy(const_cast<void*>(operator->()->dl_tensor.data), data, nbytes);
}

}  // namespace runtime
}  // namespace kxc
