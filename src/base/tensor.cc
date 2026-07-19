/*! \file src/base/tensor.cc
 * \brief 实现基础对象、设备、NDArray、Target、执行计划、PassContext 和 profiling 支撑逻辑。
 */

#include "base/tensor.h"

namespace kxc {

// 包装 NDArray 为前端 Tensor，并共享底层 Storage 生命周期。
Tensor::Tensor(runtime::NDArray data) : data_(std::move(data)) {}

// 将成员访问转发到 NDArray 暴露的只读 DLTensor 视图。
const DLTensor* Tensor::operator->() const {
    return *data_;
}

// 判断代理对象是否持有有效 NDArray。
bool Tensor::defined() const {
    return data_.defined();
}

}  // namespace kxc

