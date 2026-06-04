/*! \file src/base/tensor.cc
 * \brief 实现基础对象、设备、NDArray、Target、执行计划、PassContext 和 profiling 支撑逻辑。
 */

#include "base/tensor.h"

namespace kxc {

Tensor::Tensor(runtime::NDArray data) : data_(std::move(data)) {}

Tensor::Tensor(Array<int64_t> shape, std::string dtype) : data_(std::move(shape), std::move(dtype)) {}

const DLTensor* Tensor::operator->() const {
    return *data_;
}

bool Tensor::defined() const {
    return data_.defined();
}

}  // namespace kxc

