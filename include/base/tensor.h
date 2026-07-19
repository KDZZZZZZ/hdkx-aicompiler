/*! \file include/base/tensor.h
 * \brief 定义基础对象系统、容器、设备、NDArray、Target、PassContext 和 profiling 公共类型。
 */

#pragma once

#include "ndarray.h"

namespace kxc {

class Tensor {
public:
    runtime::NDArray data_;

    Tensor() = default;
    explicit Tensor(runtime::NDArray data);

    const DLTensor* operator->() const;
    bool defined() const;
};

}  // namespace kxc
