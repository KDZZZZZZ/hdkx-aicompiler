#pragma once
#include "relay.h"
#include <string>
#include <unordered_map>
#include <memory>
#include <any>

namespace kxc {

// --- Op (Operator) ---
class OpNode : public RelayNode {
public:
    std::string name;
    std::string description;
    
    // Attribute Map: stores metadata like "TAttrs" -> "Conv2DAttrs"
    // In a real compiler (TVM), this uses tvm::AttrRegistry
    std::unordered_map<std::string, std::any> attrs;

    const TypeIndex GetTypeId() const override {
        return kKXC_OBJECT_TYPE + 11; // Unique TypeIndex
    }
};

class Op : public Relay {
public:
    using Relay::Relay;
    
    explicit Op(std::string name, std::string description = "") {
        OpNode* node = new OpNode();
        node->name = std::move(name);
        node->description = std::move(description);
        object_ = node;
        if (object_) object_->IncRef();
    }

    const OpNode* operator->() const {
        return static_cast<const OpNode*>(object_);
    }

    // Global registry for Ops (simplified)
    static const Op& Get(const std::string& name);
};

// --- Attribute System ---
// Base class for all attributes
class BaseAttrsNode : public Object {
public:
    virtual void VisitAttrs(AttrVisitor& visitor) {}
    
    const TypeIndex GetTypeId() const override {
        return kKXC_OBJECT_TYPE + 12; 
    }
};

class Attrs : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    
    const BaseAttrsNode* operator->() const {
        return static_cast<const BaseAttrsNode*>(object_);
    }
};

// --- Concrete Attributes Definitions ---

// 1. Conv2DAttrs
class Conv2DAttrsNode : public BaseAttrsNode {
public:
    std::vector<int64_t> strides;
    std::vector<int64_t> padding;
    std::vector<int64_t> dilation;
    int groups;
    std::string data_layout;

    void VisitAttrs(AttrVisitor& visitor) override {
        // visitor("strides", &strides);
        // visitor("padding", &padding);
        // ...
    }

    const TypeIndex GetTypeId() const override {
        return kKXC_OBJECT_TYPE + 13;
    }
};

class Conv2DAttrs : public Attrs {
public:
    using Attrs::Attrs;
    
    static Conv2DAttrs Create(std::vector<int64_t> strides, std::vector<int64_t> padding, std::string layout = "NCHW") {
        Conv2DAttrsNode* node = new Conv2DAttrsNode();
        node->strides = std::move(strides);
        node->padding = std::move(padding);
        node->groups = 1;
        node->data_layout = std::move(layout);
        
        Conv2DAttrs attrs;
        attrs.object_ = node;
        if (attrs.object_) attrs.object_->IncRef();
        return attrs;
    }
    
    const Conv2DAttrsNode* operator->() const {
        return static_cast<const Conv2DAttrsNode*>(object_);
    }
};

// 2. DenseAttrs
class DenseAttrsNode : public BaseAttrsNode {
public:
    int64_t units; // Output dimension
    // std::string out_dtype;

    const TypeIndex GetTypeId() const override {
        return kKXC_OBJECT_TYPE + 18;
    }
};

class DenseAttrs : public Attrs {
public:
    using Attrs::Attrs;
    static DenseAttrs Create(int64_t units) {
        DenseAttrsNode* node = new DenseAttrsNode();
        node->units = units;
        DenseAttrs attrs;
        attrs.object_ = node;
        if (attrs.object_) attrs.object_->IncRef();
        return attrs;
    }
    const DenseAttrsNode* operator->() const { return static_cast<const DenseAttrsNode*>(object_); }
};

// 3. MaxPool2DAttrs
class MaxPool2DAttrsNode : public BaseAttrsNode {
public:
    std::vector<int64_t> pool_size;
    std::vector<int64_t> strides;
    std::vector<int64_t> padding;
    
    const TypeIndex GetTypeId() const override { return kKXC_OBJECT_TYPE + 19; }
};

class MaxPool2DAttrs : public Attrs {
public:
    using Attrs::Attrs;
    static MaxPool2DAttrs Create(std::vector<int64_t> pool_size, std::vector<int64_t> strides, std::vector<int64_t> padding) {
        MaxPool2DAttrsNode* node = new MaxPool2DAttrsNode();
        node->pool_size = std::move(pool_size);
        node->strides = std::move(strides);
        node->padding = std::move(padding);
        MaxPool2DAttrs attrs;
        attrs.object_ = node;
        if(attrs.object_) attrs.object_->IncRef();
        return attrs;
    }
    const MaxPool2DAttrsNode* operator->() const { return static_cast<const MaxPool2DAttrsNode*>(object_); }
};

// 4. SoftmaxAttrs
class SoftmaxAttrsNode : public BaseAttrsNode {
public:
    int axis;
    const TypeIndex GetTypeId() const override { return kKXC_OBJECT_TYPE + 20; }
};

class SoftmaxAttrs : public Attrs {
public:
    using Attrs::Attrs;
    static SoftmaxAttrs Create(int axis) {
        SoftmaxAttrsNode* node = new SoftmaxAttrsNode();
        node->axis = axis;
        SoftmaxAttrs attrs;
        attrs.object_ = node;
        if(attrs.object_) attrs.object_->IncRef();
        return attrs;
    }
    const SoftmaxAttrsNode* operator->() const { return static_cast<const SoftmaxAttrsNode*>(object_); }
};

} // namespace kxc
