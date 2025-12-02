#pragma once
#include "object.h"
#include <vector>
#include <string>

namespace kxc {

class TensorNode : public Object {
public:
    // Minimal tensor representation
    std::vector<int64_t> shape;
    std::string dtype;
    void* data = nullptr;

    const TypeIndex GetTypeId() const override {
        return kKXC_OBJECT_TYPE + 10; // Arbitrary offset
    }
};

class Tensor : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    
    Tensor(std::vector<int64_t> shape, std::string dtype) {
        TensorNode* node = new TensorNode();
        node->shape = std::move(shape);
        node->dtype = std::move(dtype);
        object_ = node;
        if (object_) object_->IncRef();
    }

    const TensorNode* operator->() const {
        return static_cast<const TensorNode*>(object_);
    }
};

}
