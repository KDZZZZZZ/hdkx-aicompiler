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

