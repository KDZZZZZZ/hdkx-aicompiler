#pragma once
#include "ndarray.h"
#include "base/container.h"
#include <vector>
#include <string>

namespace kxc {

// Simple Tensor class wrapping NDArray
class Tensor {
public:
    runtime::NDArray data_;

    Tensor() = default;
    
    // Construct from NDArray
    Tensor(runtime::NDArray data) : data_(data) {}
    
    // Construct new tensor with shape and dtype
    Tensor(Array<int64_t> shape, std::string dtype = "float32") 
        : data_(shape, dtype) {}

    // Allow implicit conversion to NDArray for convenience?
    // operator runtime::NDArray() const { return data_; }
    
    // Access underlying DLTensor
    const DLTensor* operator->() const {
        return *data_;
    }
    
    // Check if defined
    bool defined() const { return data_.defined(); }
};

} // namespace kxc
