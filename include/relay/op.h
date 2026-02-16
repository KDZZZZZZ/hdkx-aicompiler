#pragma once

#include <any>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "base/container.h"
#include "relay.h"

namespace kxc {
namespace relay {

struct ArgumentInfo {
    std::string name;
    std::string type;
    std::string description;
    bool is_optional;
    std::string default_value_desc;
};

class OpNode : public RelayNode {
public:
    std::string name;
    std::string description;
    Array<ArgumentInfo> arguments;
    int num_inputs = -1;
    std::unordered_map<std::string, std::any> attrs;

    KXC_OBJECT_DECLARE
};

KXC_OBJECT_DEFINE(OpNode)

class Op : public Relay {
public:
    using Relay::Relay;

    explicit Op(std::string name, std::string description = "");

    const OpNode* operator->() const { return static_cast<const OpNode*>(object_); }

    static const Op& Get(const std::string& name);
};

#define KXC_DECLARE_ATTRS_NODE KXC_OBJECT_DECLARE

#define KXC_DECLARE_ATTRS_REF(TypeName, NodeName)                                              \
public:                                                                                         \
    using Attrs::Attrs;                                                                         \
    const NodeName* operator->() const { return static_cast<const NodeName*>(object_); }       \
                                                                                                \
private:                                                                                        \
    friend class TypeName;                                                                      \
    static TypeName InternalCreate(NodeName* node) {                                            \
        TypeName attrs;                                                                         \
        attrs.SetData(node);                                                                    \
        return attrs;                                                                           \
    }

#define KXC_DEFINE_SIMPLE_ATTRS(TypeName)             \
    class TypeName##Node : public BaseAttrsNode {     \
    public:                                            \
        KXC_DECLARE_ATTRS_NODE                         \
    };                                                 \
    KXC_OBJECT_DEFINE(TypeName##Node)                 \
    class TypeName : public Attrs {                   \
        KXC_DECLARE_ATTRS_REF(TypeName, TypeName##Node) \
                                                       \
    public:                                            \
        static TypeName Create();                      \
    };

class BaseAttrsNode : public Object {
public:
    virtual void VisitAttrs(AttrVisitor& visitor) { (void)visitor; }

    KXC_DECLARE_ATTRS_NODE
};

KXC_OBJECT_DEFINE(BaseAttrsNode)

class Attrs : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    Attrs(ObjectRef n) : ObjectRef(n) {}

    const BaseAttrsNode* operator->() const { return static_cast<const BaseAttrsNode*>(object_); }
};

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

    void VisitAttrs(AttrVisitor& visitor) override { (void)visitor; }

    KXC_DECLARE_ATTRS_NODE
};

KXC_OBJECT_DEFINE(Conv2DAttrsNode)

class Conv2DAttrs : public Attrs {
    KXC_DECLARE_ATTRS_REF(Conv2DAttrs, Conv2DAttrsNode)

public:
    static Conv2DAttrs Create(std::vector<int64_t> strides, std::vector<int64_t> padding,
                              std::vector<int64_t> dilation, int groups, int channels,
                              std::vector<int64_t> kernel_size, std::string data_layout,
                              std::string kernel_layout, std::string out_layout,
                              std::string out_dtype);
};

class DenseAttrsNode : public BaseAttrsNode {
public:
    int64_t units;
    std::string out_dtype;
    KXC_DECLARE_ATTRS_NODE
};
KXC_OBJECT_DEFINE(DenseAttrsNode)

class DenseAttrs : public Attrs {
    KXC_DECLARE_ATTRS_REF(DenseAttrs, DenseAttrsNode)

public:
    static DenseAttrs Create(int64_t units, std::string out_dtype);
};

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
                                 std::string layout, bool ceil_mode);
};

class SoftmaxAttrsNode : public BaseAttrsNode {
public:
    int axis;
    KXC_DECLARE_ATTRS_NODE
};
KXC_OBJECT_DEFINE(SoftmaxAttrsNode)

class SoftmaxAttrs : public Attrs {
    KXC_DECLARE_ATTRS_REF(SoftmaxAttrs, SoftmaxAttrsNode)

public:
    static SoftmaxAttrs Create(int axis);
};

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
    static BatchNormAttrs Create(double epsilon = 1e-5, bool center = true, bool scale = true);
};

KXC_DEFINE_SIMPLE_ATTRS(AddAttrs)

class CastAttrsNode : public BaseAttrsNode {
public:
    int to;
    KXC_DECLARE_ATTRS_NODE
};
KXC_OBJECT_DEFINE(CastAttrsNode)
class CastAttrs : public Attrs {
    KXC_DECLARE_ATTRS_REF(CastAttrs, CastAttrsNode)

public:
    static CastAttrs Create(int to = 0);
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
    static ConcatAttrs Create(int axis = 0);
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
    static ConstantAttrs Create(runtime::NDArray value);
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
    static ConstantOfShapeAttrs Create(runtime::NDArray value);
    static ConstantOfShapeAttrs Create();
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
    static GatherAttrs Create(int axis = 0);
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
    static ReduceMeanAttrs Create(std::vector<int64_t> axes, int64_t keepdims = 1);
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
    static ReshapeAttrs Create(std::vector<int64_t> newshape, int allowzero = 0);
};

class SplitAttrsNode : public BaseAttrsNode {
public:
    std::vector<int64_t> split;
    int axis = 0;
    KXC_DECLARE_ATTRS_NODE
};
KXC_OBJECT_DEFINE(SplitAttrsNode)
class SplitAttrs : public Attrs {
    KXC_DECLARE_ATTRS_REF(SplitAttrs, SplitAttrsNode)

public:
    static SplitAttrs Create(std::vector<int64_t> split, int axis = 0);
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
    static TransposeAttrs Create(std::vector<int64_t> perm);
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
    static FlattenAttrs Create(int axis = 1);
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
    static GemmAttrs Create(float alpha = 1.0f, float beta = 1.0f, int transA = 0,
                            int transB = 0);
};

class DeviceCopyAttrsNode : public BaseAttrsNode {
public:
    VirtualDevice src_virtual_device;
    VirtualDevice dst_virtual_device;
    bool async = false;
    bool in_group = true;
    KXC_DECLARE_ATTRS_NODE
};
KXC_OBJECT_DEFINE(DeviceCopyAttrsNode)
class DeviceCopyAttrs : public Attrs {
    KXC_DECLARE_ATTRS_REF(DeviceCopyAttrs, DeviceCopyAttrsNode)

public:
    static DeviceCopyAttrs Create(VirtualDevice src_virtual_device,
                                  VirtualDevice dst_virtual_device, bool async = false,
                                  bool in_group = true);
};

class CollectiveAttrsNode : public BaseAttrsNode {
public:
    std::string kind;
    std::string reduce_kind = "sum";
    bool in_group = true;
    int group_id = 0;
    int root_worker = 0;
    KXC_DECLARE_ATTRS_NODE
};
KXC_OBJECT_DEFINE(CollectiveAttrsNode)
class CollectiveAttrs : public Attrs {
    KXC_DECLARE_ATTRS_REF(CollectiveAttrs, CollectiveAttrsNode)

public:
    static CollectiveAttrs Create(std::string kind, std::string reduce_kind = "sum",
                                  bool in_group = true, int group_id = 0, int root_worker = 0);
};

}  // namespace relay
}  // namespace kxc
