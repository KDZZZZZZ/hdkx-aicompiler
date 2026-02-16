#pragma once

#include <string>

#include "base/container.h"
#include "ndarray.h"

namespace kxc {

class Tensor {
public:
    runtime::NDArray data_;

    Tensor() = default;
    explicit Tensor(runtime::NDArray data);
    Tensor(Array<int64_t> shape, std::string dtype = "float32");

    const DLTensor* operator->() const;
    bool defined() const;
};

}  // namespace kxc