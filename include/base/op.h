#pragma once
#include "relay.h"
#include <string>
#include <unordered_map>
#include <memory>
#include <any>

namespace kxc {

// Argument Information structure
struct ArgumentInfo {
    std::string name;
    std::string type;
    std::string description;
    bool is_optional;
    std::string default_value_desc; // Description of default value if any
};

// --- Op (Operator) ---
class OpNode : public RelayNode {
public:
    std::string name;
    std::string description;
    
    // Inputs/Arguments metadata
    std::vector<ArgumentInfo> arguments;
    int num_inputs = -1; // -1 means variable or undefined
    
    // Attribute Map: stores metadata like "TAttrs" -> "Conv2DAttrs"
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

// 5. BatchNormAttrs (New Example)
class BatchNormAttrsNode : public BaseAttrsNode {
public:
    double epsilon = 1e-5;
    bool center = true;
    bool scale = true;
    
    const TypeIndex GetTypeId() const override { return kKXC_OBJECT_TYPE + 21; }
};

class BatchNormAttrs : public Attrs {
public:
    using Attrs::Attrs;
    static BatchNormAttrs Create(double epsilon = 1e-5, bool center = true, bool scale = true) {
        BatchNormAttrsNode* node = new BatchNormAttrsNode();
        node->epsilon = epsilon;
        node->center = center;
        node->scale = scale;
        BatchNormAttrs attrs;
        attrs.object_ = node;
        if(attrs.object_) attrs.object_->IncRef();
        return attrs;
    }
    const BatchNormAttrsNode* operator->() const { return static_cast<const BatchNormAttrsNode*>(object_); }
};
class AddAttrsNode : public BaseAttrsNode {
public:
    const TypeIndex GetTypeId() const override { return kKXC_OBJECT_TYPE + 22; }
};


class AddAttrs : public Attrs {
public:
    using Attrs::Attrs;
    static AddAttrs Create() {
        AddAttrsNode* node = new AddAttrsNode();
        AddAttrs attrs;
        attrs.object_ = node;
    }
    const AddAttrsNode* operator->() const { return static_cast<const AddAttrsNode*>(object_); }
};
class CastAttrsNode : public BaseAttrsNode {
public:
    int to;
    const TypeIndex GetTypeId() const override { return kKXC_OBJECT_TYPE + 23; }
};
class CastAttrs : public Attrs {
public:
    using Attrs::Attrs;
    static CastAttrs Create(int to = 0) {
        CastAttrsNode* node = new CastAttrsNode();
        node->to = to;
        CastAttrs attrs;
        attrs.object_ = node;
        if(attrs.object_) attrs.object_->IncRef();
        return attrs;
    }
    const CastAttrsNode* operator->() const { return static_cast<const CastAttrsNode*>(object_); }
};
class ConcatAttrsNode : public BaseAttrsNode {
public:
    int axis;
    const TypeIndex GetTypeId() const override { return kKXC_OBJECT_TYPE + 24; }
};
class ConcatAttrs : public Attrs {
public:
    using Attrs::Attrs;
    static ConcatAttrs Create(int axis = 0) {
        ConcatAttrsNode* node = new ConcatAttrsNode();
        node->axis = axis;
        ConcatAttrs attrs;
        attrs.object_ = node;
        if(attrs.object_) attrs.object_->IncRef();
        return attrs;
    }
    const ConcatAttrsNode* operator->() const { return static_cast<const ConcatAttrsNode*>(object_); }
};
class ConstantAttrsNode : public BaseAttrsNode {
public:
    Tensor value;
    const TypeIndex GetTypeId() const override { return kKXC_OBJECT_TYPE + 25; }
};
class ConstantAttrs : public Attrs {
public:
    using Attrs::Attrs;
    static ConstantAttrs Create(Tensor value) {
        ConstantAttrsNode* node = new ConstantAttrsNode();
        node->value = value;
        ConstantAttrs attrs;
        attrs.object_ = node;
        if(attrs.object_) attrs.object_->IncRef();
        return attrs;
    }
    const ConstantAttrsNode* operator->() const { return static_cast<const ConstantAttrsNode*>(object_); }
};
class ConstantOfShapeAttrsNode : public BaseAttrsNode {
public:
    Tensor value;
    const TypeIndex GetTypeId() const override { return kKXC_OBJECT_TYPE + 26; }
};
class ConstantOfShapeAttrs : public Attrs {
public:
    using Attrs::Attrs;
    static ConstantOfShapeAttrs Create(Tensor value) {
        ConstantOfShapeAttrsNode* node = new ConstantOfShapeAttrsNode();
        node->value = value;
        ConstantOfShapeAttrs attrs;
        attrs.object_ = node;
        if(attrs.object_) attrs.object_->IncRef();
        return attrs;
    }
    const ConstantOfShapeAttrsNode* operator->() const { return static_cast<const ConstantOfShapeAttrsNode*>(object_); }
};
class DivAttrsNode : public BaseAttrsNode {
public:
    const TypeIndex GetTypeId() const override { return kKXC_OBJECT_TYPE + 27; }
};
class DivAttrs : public Attrs {
public:
    using Attrs::Attrs;
    static DivAttrs Create() {
        DivAttrsNode* node = new DivAttrsNode();
        DivAttrs attrs;
        attrs.object_ = node;
        if(attrs.object_) attrs.object_->IncRef();
        return attrs;
    }
    const DivAttrsNode* operator->() const { return static_cast<const DivAttrsNode*>(object_); }
};
class EqualAttrsNode : public BaseAttrsNode {
public:
    const TypeIndex GetTypeId() const override { return kKXC_OBJECT_TYPE + 28; }
};
class EqualAttrs : public Attrs {
public:
    using Attrs::Attrs;
    static EqualAttrs Create() {
        EqualAttrsNode* node = new EqualAttrsNode();
        EqualAttrs attrs;
        attrs.object_ = node;
        if(attrs.object_) attrs.object_->IncRef();
        return attrs;
    }
    const EqualAttrsNode* operator->() const { return static_cast<const EqualAttrsNode*>(object_); }
};
class ErfAttrsNode : public BaseAttrsNode {
public:
    const TypeIndex GetTypeId() const override { return kKXC_OBJECT_TYPE + 29; }
};
class ErfAttrs : public Attrs {
public:
    using Attrs::Attrs;
    static ErfAttrs Create() {
        ErfAttrsNode* node = new ErfAttrsNode();
        ErfAttrs attrs;
        attrs.object_ = node;
        if(attrs.object_) attrs.object_->IncRef();
        return attrs;
    }
    const ErfAttrsNode* operator->() const { return static_cast<const ErfAttrsNode*>(object_); }
};

class ExpandAttrsNode : public BaseAttrsNode {
public:
    const TypeIndex GetTypeId() const override { return kKXC_OBJECT_TYPE + 30; }
};
class ExpandAttrs : public Attrs {
public:
    using Attrs::Attrs;
    static ExpandAttrs Create() {
        ExpandAttrsNode* node = new ExpandAttrsNode();
        ExpandAttrs attrs;
        attrs.object_ = node;
        if(attrs.object_) attrs.object_->IncRef();
        return attrs;
    }
    const ExpandAttrsNode* operator->() const { return static_cast<const ExpandAttrsNode*>(object_); }
};
class GatherAttrsNode : public BaseAttrsNode {
public:
    int axis;
    const TypeIndex GetTypeId() const override { return kKXC_OBJECT_TYPE + 31; }
};
class GatherAttrs : public Attrs {
public:
    using Attrs::Attrs;
    static GatherAttrs Create(int axis = 0) {
        GatherAttrsNode* node = new GatherAttrsNode();
        node->axis = axis;
        GatherAttrs attrs;
        attrs.object_ = node;
        if(attrs.object_) attrs.object_->IncRef();
        return attrs;
    }
    const GatherAttrsNode* operator->() const { return static_cast<const GatherAttrsNode*>(object_); }
};

class ReduceMeanAttrsNode : public BaseAttrsNode {
public:
    std::vector<int64_t> axes;
    int64_t keepdims = 1; 
    const TypeIndex GetTypeId() const override { return kKXC_OBJECT_TYPE + 32; }
};
class ReduceMeanAttrs : public Attrs {
public:
    using Attrs::Attrs;
    static ReduceMeanAttrs Create(std::vector<int64_t> axes, int64_t keepdims = 1) {
        ReduceMeanAttrsNode* node = new ReduceMeanAttrsNode();
        node->axes = std::move(axes);
        node->keepdims = keepdims;
        ReduceMeanAttrs attrs;
        attrs.object_ = node;
        if(attrs.object_) attrs.object_->IncRef();
        return attrs;
    }
    const ReduceMeanAttrsNode* operator->() const { return static_cast<const ReduceMeanAttrsNode*>(object_); }
};

class ReshapeAttrsNode : public BaseAttrsNode {
public:
    int allowzero = 0;
    const TypeIndex GetTypeId() const override { return kKXC_OBJECT_TYPE + 33; }
};
class ReshapeAttrs : public Attrs {
public:
    using Attrs::Attrs;
    static ReshapeAttrs Create(int allowzero = 0) {
        ReshapeAttrsNode* node = new ReshapeAttrsNode();
        node->allowzero = allowzero;
        ReshapeAttrs attrs;
        attrs.object_ = node;
        if(attrs.object_) attrs.object_->IncRef();
        return attrs;
    }
    const ReshapeAttrsNode* operator->() const { return static_cast<const ReshapeAttrsNode*>(object_); }
};

class SplitAttrsNode : public BaseAttrsNode {
public:
    int axis = 0;
    const TypeIndex GetTypeId() const override { return kKXC_OBJECT_TYPE + 34; }
};
class SplitAttrs : public Attrs {
public:
    using Attrs::Attrs;
    static SplitAttrs Create(int axis = 0) {
        SplitAttrsNode* node = new SplitAttrsNode();
        node->axis = axis;
        SplitAttrs attrs;
        attrs.object_ = node;
        if(attrs.object_) attrs.object_->IncRef();
        return attrs;
    }
    const SplitAttrsNode* operator->() const { return static_cast<const SplitAttrsNode*>(object_); }
};

class TransposeAttrsNode : public BaseAttrsNode {
public:
    std::vector<int64_t> perm;
    const TypeIndex GetTypeId() const override { return kKXC_OBJECT_TYPE + 35; }
};
class TransposeAttrs : public Attrs {
public:
    using Attrs::Attrs;
    static TransposeAttrs Create(std::vector<int64_t> perm) {
        TransposeAttrsNode* node = new TransposeAttrsNode();
        node->perm = std::move(perm);
        TransposeAttrs attrs;
        attrs.object_ = node;
        if(attrs.object_) attrs.object_->IncRef();
        return attrs;
    }
    const TransposeAttrsNode* operator->() const { return static_cast<const TransposeAttrsNode*>(object_); }
};

} // namespace kxc
