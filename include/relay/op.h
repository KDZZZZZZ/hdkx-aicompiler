#pragma once
#include "relay.h"
#include "base/container.h"
#include <string>
#include <unordered_map>
#include <memory>
#include <any>

namespace kxc {
namespace relay {

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
    Array<ArgumentInfo> arguments;
    int num_inputs = -1; // -1 means variable or undefined
    
    // Attribute Map: stores metadata like "TAttrs" -> "Conv2DAttrs"
    std::unordered_map<std::string, std::any> attrs;

    KXC_OBJECT_DECLARE
};

KXC_OBJECT_DEFINE(OpNode)

class Op : public Relay {
public:
    using Relay::Relay;
    
    explicit Op(std::string name, std::string description = "") {
        OpNode* node = new OpNode();
        node->name = std::move(name);
        node->description = std::move(description);
        SetData(node);
    }

    const OpNode* operator->() const {
        return static_cast<const OpNode*>(object_);
    }

    // Global registry for Ops (simplified)
    static const Op& Get(const std::string& name);
};

// --- Attribute System ---
#define KXC_DECLARE_ATTRS_NODE \
    KXC_OBJECT_DECLARE

#define KXC_DECLARE_ATTRS_REF(TypeName, NodeName) \
public: \
    using Attrs::Attrs; \
    const NodeName* operator->() const { return static_cast<const NodeName*>(object_); } \
private: \
    friend class TypeName; \
    static TypeName InternalCreate(NodeName* node) { \
        TypeName attrs; \
        attrs.SetData(node); \
        return attrs; \
    }

#define KXC_DEFINE_SIMPLE_ATTRS(TypeName) \
    class TypeName##Node : public BaseAttrsNode { \
    public: \
        KXC_DECLARE_ATTRS_NODE \
    }; \
    KXC_OBJECT_DEFINE(TypeName##Node) \
    class TypeName : public Attrs { \
        KXC_DECLARE_ATTRS_REF(TypeName, TypeName##Node) \
    public: \
        static TypeName Create() { \
            return InternalCreate(new TypeName##Node()); \
        } \
    };

// Base class for all attributes
class BaseAttrsNode : public Object {
public:
    virtual void VisitAttrs(AttrVisitor& visitor) {}
    
    KXC_DECLARE_ATTRS_NODE
};

KXC_OBJECT_DEFINE(BaseAttrsNode)

class Attrs : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    Attrs(ObjectRef n) : ObjectRef(n) {}
    
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
    int channels;
    std::vector<int64_t> kernel_size;
    std::string data_layout;
    std::string kernel_layout;
    std::string out_layout;
    std::string out_dtype;

    void VisitAttrs(AttrVisitor& visitor) override {
        // visitor("strides", &strides);
        // visitor("padding", &padding);
        // ...
    }

    KXC_DECLARE_ATTRS_NODE
};

KXC_OBJECT_DEFINE(Conv2DAttrsNode)

class Conv2DAttrs : public Attrs {
    KXC_DECLARE_ATTRS_REF(Conv2DAttrs, Conv2DAttrsNode)
public:
    static Conv2DAttrs Create(std::vector<int64_t> strides, std::vector<int64_t> padding, 
                             std::vector<int64_t> dilation, int groups, int channels, 
                             std::vector<int64_t> kernel_size, std::string data_layout, 
                             std::string kernel_layout, std::string out_layout, std::string out_dtype) {
        Conv2DAttrsNode* node = new Conv2DAttrsNode();
        node->strides = std::move(strides);
        node->padding = std::move(padding);
        node->dilation = std::move(dilation);
        node->groups = groups;
        node->channels = channels;
        node->kernel_size = std::move(kernel_size);
        node->data_layout = std::move(data_layout);
        node->kernel_layout = std::move(kernel_layout);
        node->out_layout = std::move(out_layout);
        node->out_dtype = std::move(out_dtype);
        return InternalCreate(node);
    }
};

// 2. DenseAttrs
class DenseAttrsNode : public BaseAttrsNode {
public:
    int64_t units; // Output dimension
    std::string out_dtype;
    KXC_DECLARE_ATTRS_NODE
};
KXC_OBJECT_DEFINE(DenseAttrsNode)

class DenseAttrs : public Attrs {
    KXC_DECLARE_ATTRS_REF(DenseAttrs, DenseAttrsNode)
public:
    static DenseAttrs Create(int64_t units, std::string out_dtype) {
        DenseAttrsNode* node = new DenseAttrsNode();
        node->units = std::move(units);
        node->out_dtype = std::move(out_dtype);
        return InternalCreate(node);
    }
};

// 3. MaxPool2DAttrs
class MaxPool2DAttrsNode : public BaseAttrsNode {
public:
    std::vector<int64_t> pool_size;
    std::vector<int64_t> strides;
    std::vector<int64_t> padding;
    std::vector<int64_t> dilation;
    std::string layout;
    bool ceil_mode;
    
    KXC_DECLARE_ATTRS_NODE
};
KXC_OBJECT_DEFINE(MaxPool2DAttrsNode)

class MaxPool2DAttrs : public Attrs {
    KXC_DECLARE_ATTRS_REF(MaxPool2DAttrs, MaxPool2DAttrsNode)
public:
    static MaxPool2DAttrs Create(std::vector<int64_t> strides, std::vector<int64_t> padding, 
                                std::vector<int64_t> dilation, std::vector<int64_t> pool_size,
                                std::string layout, bool ceil_mode) {
        MaxPool2DAttrsNode* node = new MaxPool2DAttrsNode();
        node->strides = std::move(strides);
        node->padding = std::move(padding);
        node->dilation = std::move(dilation);
        node->pool_size = std::move(pool_size);
        node->layout = std::move(layout);
        node->ceil_mode = ceil_mode;
        return InternalCreate(node);
    }
};

// 4. SoftmaxAttrs
class SoftmaxAttrsNode : public BaseAttrsNode {
public:
    int axis;
    KXC_DECLARE_ATTRS_NODE
};
KXC_OBJECT_DEFINE(SoftmaxAttrsNode)

class SoftmaxAttrs : public Attrs {
    KXC_DECLARE_ATTRS_REF(SoftmaxAttrs, SoftmaxAttrsNode)
public:
    static SoftmaxAttrs Create(int axis) {
        SoftmaxAttrsNode* node = new SoftmaxAttrsNode();
        node->axis = std::move(axis);
        return InternalCreate(node);
    }
};

// 5. BatchNormAttrs
class BatchNormAttrsNode : public BaseAttrsNode {
public:
    double epsilon = 1e-5;
    bool center = true;
    bool scale = true;
    
    KXC_DECLARE_ATTRS_NODE
};
KXC_OBJECT_DEFINE(BatchNormAttrsNode)

class BatchNormAttrs : public Attrs {
    KXC_DECLARE_ATTRS_REF(BatchNormAttrs, BatchNormAttrsNode)
public:
    static BatchNormAttrs Create(double epsilon = 1e-5, bool center = true, bool scale = true) {
        BatchNormAttrsNode* node = new BatchNormAttrsNode();
        node->epsilon = std::move(epsilon);
        node->center = std::move(center);
        node->scale = std::move(scale);
        return InternalCreate(node);
    }
};

// Simple Attributes (no fields)
KXC_DEFINE_SIMPLE_ATTRS(AddAttrs)

// Attributes with fields
class CastAttrsNode : public BaseAttrsNode {
public:
    int to;
    KXC_DECLARE_ATTRS_NODE
};
KXC_OBJECT_DEFINE(CastAttrsNode)
class CastAttrs : public Attrs {
    KXC_DECLARE_ATTRS_REF(CastAttrs, CastAttrsNode)
public:
    static CastAttrs Create(int to = 0) {
        CastAttrsNode* node = new CastAttrsNode();
        node->to = std::move(to);
        return InternalCreate(node);
    }
};

class ConcatAttrsNode : public BaseAttrsNode {
public:
    int axis;
    KXC_DECLARE_ATTRS_NODE
};
KXC_OBJECT_DEFINE(ConcatAttrsNode)
class ConcatAttrs : public Attrs {
    KXC_DECLARE_ATTRS_REF(ConcatAttrs, ConcatAttrsNode)
public:
    static ConcatAttrs Create(int axis = 0) {
        ConcatAttrsNode* node = new ConcatAttrsNode();
        node->axis = std::move(axis);
        return InternalCreate(node);
    }
};

class ConstantAttrsNode : public BaseAttrsNode {
public:
    runtime::NDArray value;
    KXC_DECLARE_ATTRS_NODE
};
KXC_OBJECT_DEFINE(ConstantAttrsNode)
class ConstantAttrs : public Attrs {
    KXC_DECLARE_ATTRS_REF(ConstantAttrs, ConstantAttrsNode)
public:
    static ConstantAttrs Create(runtime::NDArray value) {
        ConstantAttrsNode* node = new ConstantAttrsNode();
        node->value = std::move(value);
        return InternalCreate(node);
    }
};

class ConstantOfShapeAttrsNode : public BaseAttrsNode {
public:
    runtime::NDArray value;
    KXC_DECLARE_ATTRS_NODE
};
KXC_OBJECT_DEFINE(ConstantOfShapeAttrsNode)
class ConstantOfShapeAttrs : public Attrs {
    KXC_DECLARE_ATTRS_REF(ConstantOfShapeAttrs, ConstantOfShapeAttrsNode)
public:
    static ConstantOfShapeAttrs Create(runtime::NDArray value) {
        ConstantOfShapeAttrsNode* node = new ConstantOfShapeAttrsNode();
        node->value = std::move(value);
        return InternalCreate(node);
    }
    static ConstantOfShapeAttrs Create() {
        ConstantOfShapeAttrsNode* node = new ConstantOfShapeAttrsNode();
        return InternalCreate(node);
    }
};

KXC_DEFINE_SIMPLE_ATTRS(DivAttrs)
KXC_DEFINE_SIMPLE_ATTRS(EqualAttrs)
KXC_DEFINE_SIMPLE_ATTRS(ErfAttrs)
KXC_DEFINE_SIMPLE_ATTRS(ExpandAttrs)

class GatherAttrsNode : public BaseAttrsNode {
public:
    int axis = 0;
    KXC_DECLARE_ATTRS_NODE
};
KXC_OBJECT_DEFINE(GatherAttrsNode)
class GatherAttrs : public Attrs {
    KXC_DECLARE_ATTRS_REF(GatherAttrs, GatherAttrsNode)
public:
    static GatherAttrs Create(int axis = 0) {
        GatherAttrsNode* node = new GatherAttrsNode();
        node->axis = std::move(axis);
        return InternalCreate(node);
    }
};

class ReduceMeanAttrsNode : public BaseAttrsNode {
public:
    std::vector<int64_t> axes;
    int64_t keepdims = 1; 
    KXC_DECLARE_ATTRS_NODE
};
KXC_OBJECT_DEFINE(ReduceMeanAttrsNode)
class ReduceMeanAttrs : public Attrs {
    KXC_DECLARE_ATTRS_REF(ReduceMeanAttrs, ReduceMeanAttrsNode)
public:
    static ReduceMeanAttrs Create(std::vector<int64_t> axes, int64_t keepdims = 1) {
        ReduceMeanAttrsNode* node = new ReduceMeanAttrsNode();
        node->axes = std::move(axes);
        node->keepdims = std::move(keepdims);
        return InternalCreate(node);
    }
};

class ReshapeAttrsNode : public BaseAttrsNode {
public:
    std::vector<int64_t> newshape;
    int allowzero = 0;
    KXC_DECLARE_ATTRS_NODE
};
KXC_OBJECT_DEFINE(ReshapeAttrsNode)
class ReshapeAttrs : public Attrs {
    KXC_DECLARE_ATTRS_REF(ReshapeAttrs, ReshapeAttrsNode)
public:
    static ReshapeAttrs Create(std::vector<int64_t> newshape, int allowzero = 0) {
        ReshapeAttrsNode* node = new ReshapeAttrsNode();
        node->newshape = std::move(newshape);
        node->allowzero = std::move(allowzero);
        return InternalCreate(node);
    }
};

class SplitAttrsNode : public BaseAttrsNode {
public:
    std::vector<int64_t> split; // indices_or_sections
    int axis = 0;
    KXC_DECLARE_ATTRS_NODE
};
KXC_OBJECT_DEFINE(SplitAttrsNode)
class SplitAttrs : public Attrs {
    KXC_DECLARE_ATTRS_REF(SplitAttrs, SplitAttrsNode)
public:
    static SplitAttrs Create(std::vector<int64_t> split, int axis = 0) {
        SplitAttrsNode* node = new SplitAttrsNode();
        node->split = std::move(split);
        node->axis = std::move(axis);
        return InternalCreate(node);
    }
};

class TransposeAttrsNode : public BaseAttrsNode {
public:
    std::vector<int64_t> perm;
    KXC_DECLARE_ATTRS_NODE
};
KXC_OBJECT_DEFINE(TransposeAttrsNode)
class TransposeAttrs : public Attrs {
    KXC_DECLARE_ATTRS_REF(TransposeAttrs, TransposeAttrsNode)
public:
    static TransposeAttrs Create(std::vector<int64_t> perm) {
        TransposeAttrsNode* node = new TransposeAttrsNode();
        node->perm = std::move(perm);
        return InternalCreate(node);
    }
};

KXC_DEFINE_SIMPLE_ATTRS(ReluAttrs)
KXC_DEFINE_SIMPLE_ATTRS(GlobalAvgPool2DAttrs)

class FlattenAttrsNode : public BaseAttrsNode {
public:
    int axis = 1;
    KXC_DECLARE_ATTRS_NODE
};
KXC_OBJECT_DEFINE(FlattenAttrsNode)
class FlattenAttrs : public Attrs {
    KXC_DECLARE_ATTRS_REF(FlattenAttrs, FlattenAttrsNode)
public:
    static FlattenAttrs Create(int axis = 1) {
        FlattenAttrsNode* node = new FlattenAttrsNode();
        node->axis = std::move(axis);
        return InternalCreate(node);
    }
};

class GemmAttrsNode : public BaseAttrsNode {
public:
    float alpha = 1.0f;
    float beta = 1.0f;
    int transA = 0;
    int transB = 0;
    KXC_DECLARE_ATTRS_NODE
};
KXC_OBJECT_DEFINE(GemmAttrsNode)
class GemmAttrs : public Attrs {
    KXC_DECLARE_ATTRS_REF(GemmAttrs, GemmAttrsNode)
public:
    static GemmAttrs Create(float alpha = 1.0f, float beta = 1.0f, int transA = 0, int transB = 0) {
        GemmAttrsNode* node = new GemmAttrsNode();
        node->alpha = std::move(alpha);
        node->beta = std::move(beta);
        node->transA = std::move(transA);
        node->transB = std::move(transB);
        return InternalCreate(node);
    }
};
} // namespace relay
} // namespace kxc
