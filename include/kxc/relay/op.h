/*! \file include/kxc/relay/op.h
 * \brief 定义 Relay IR 节点、算子注册、attrs 和 Relay 到 TE lowering 属性。
 */

#pragma once

#include <any>
#include <string>
#include <unordered_map>

#include "kxc/support/container.h"
#include "relay.h"

namespace kxc {
namespace relay {

/*! \brief 算子 schema 中单个参数的说明。 */
struct ArgumentInfo {
    std::string name;
    std::string type;
    std::string description;
    bool is_optional;
    std::string default_value_desc;
};

/*! \brief Relay 算子对象，保存算子名称、描述、输入数和任意属性。 */
class OpNode : public RelayNode {
public:
    std::string name;
    std::string description;
    Array<ArgumentInfo> arguments;
    int num_inputs = -1;
    std::unordered_map<std::string, std::any> attrs;

    KXC_OBJECT_DECLARE
};


/*! \brief Relay 算子的引用类型，用于在表达式树中引用全局注册算子。 */
class Op : public Relay {
public:
    using Relay::Relay;

    /*! \brief 构造一个算子引用，并注册/查找对应名称的算子节点。 */
    explicit Op(std::string name, std::string description = "");

    const OpNode* operator->() const { return static_cast<const OpNode*>(object_); }

    /*! \brief 从全局算子表中获取指定名称的算子。 */
    static const Op& Get(const std::string& name);
};

#define KXC_DECLARE_ATTRS_NODE KXC_OBJECT_DECLARE

#define KXC_DECLARE_ATTRS_REF(TypeName, NodeName)                                              \
public:                                                                                         \
    using Attrs::Attrs;                                                                         \
    const NodeName* operator->() const { return static_cast<const NodeName*>(object_); }       \
                                                                                                \
private:                                                                                        \
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
    class TypeName : public Attrs {                   \
        KXC_DECLARE_ATTRS_REF(TypeName, TypeName##Node) \
                                                       \
    public:                                            \
        static TypeName Create();                      \
    };

/*! \brief 所有算子属性节点的基类，提供 TVM 风格的 VisitAttrs 入口。 */
class BaseAttrsNode : public Object {
public:
    virtual void VisitAttrs(AttrVisitor& visitor) { (void)visitor; }

    KXC_DECLARE_ATTRS_NODE
};


/*! \brief 算子属性的引用类型，作为 Call.attrs 的统一承载对象。 */
class Attrs : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    Attrs(ObjectRef n) : ObjectRef(n) {}

    const BaseAttrsNode* operator->() const { return static_cast<const BaseAttrsNode*>(object_); }
};

/*! \brief nn.conv2d 的卷积窗口、布局和输出通道属性。 */
class Conv2DAttrsNode : public BaseAttrsNode {
public:
    Array<int64_t> strides;
    Array<int64_t> padding;
    Array<int64_t> dilation;
    int groups;
    int channels;
    Array<int64_t> kernel_size;
    std::string data_layout;
    std::string kernel_layout;
    std::string out_layout;
    std::string out_dtype;

    void VisitAttrs(AttrVisitor& visitor) override { (void)visitor; }

    KXC_DECLARE_ATTRS_NODE
};


/*! \brief nn.conv2d 属性引用类型。 */
class Conv2DAttrs : public Attrs {
    KXC_DECLARE_ATTRS_REF(Conv2DAttrs, Conv2DAttrsNode)

public:
    static Conv2DAttrs Create(Array<int64_t> strides, Array<int64_t> padding,
                              Array<int64_t> dilation, int groups, int channels,
                              Array<int64_t> kernel_size, std::string data_layout,
                              std::string kernel_layout, std::string out_layout,
                              std::string out_dtype);
};

/*! \brief nn.dense 的输出单元数和输出 dtype 属性。 */
class DenseAttrsNode : public BaseAttrsNode {
public:
    int64_t units;
    std::string out_dtype;
    KXC_DECLARE_ATTRS_NODE
};

/*! \brief nn.dense 属性引用类型。 */
class DenseAttrs : public Attrs {
    KXC_DECLARE_ATTRS_REF(DenseAttrs, DenseAttrsNode)

public:
    static DenseAttrs Create(int64_t units, std::string out_dtype);
};

/*! \brief nn.max_pool2d 的窗口、步幅、padding 和布局属性。 */
class MaxPool2DAttrsNode : public BaseAttrsNode {
public:
    Array<int64_t> pool_size;
    Array<int64_t> strides;
    Array<int64_t> padding;
    Array<int64_t> dilation;
    std::string layout;
    bool ceil_mode;

    KXC_DECLARE_ATTRS_NODE
};

/*! \brief nn.max_pool2d 属性引用类型。 */
class MaxPool2DAttrs : public Attrs {
    KXC_DECLARE_ATTRS_REF(MaxPool2DAttrs, MaxPool2DAttrsNode)

public:
    static MaxPool2DAttrs Create(Array<int64_t> strides, Array<int64_t> padding,
                                 Array<int64_t> dilation, Array<int64_t> pool_size,
                                 std::string layout, bool ceil_mode);
};

/*! \brief nn.softmax 的归一化轴属性。 */
class SoftmaxAttrsNode : public BaseAttrsNode {
public:
    int axis;
    KXC_DECLARE_ATTRS_NODE
};

/*! \brief nn.softmax 属性引用类型。 */
class SoftmaxAttrs : public Attrs {
    KXC_DECLARE_ATTRS_REF(SoftmaxAttrs, SoftmaxAttrsNode)

public:
    static SoftmaxAttrs Create(int axis);
};

KXC_DEFINE_SIMPLE_ATTRS(AddAttrs)

/*! \brief cast 的目标 dtype 编码属性。 */
class CastAttrsNode : public BaseAttrsNode {
public:
    int to;
    KXC_DECLARE_ATTRS_NODE
};
class CastAttrs : public Attrs {
    KXC_DECLARE_ATTRS_REF(CastAttrs, CastAttrsNode)

public:
    static CastAttrs Create(int to = 0);
};

/*! \brief reduce.mean 的归约轴和 keepdims 属性。 */
class ReduceMeanAttrsNode : public BaseAttrsNode {
public:
    Array<int64_t> axes;
    int64_t keepdims = 1;
    KXC_DECLARE_ATTRS_NODE
};
class ReduceMeanAttrs : public Attrs {
    KXC_DECLARE_ATTRS_REF(ReduceMeanAttrs, ReduceMeanAttrsNode)

public:
    static ReduceMeanAttrs Create(Array<int64_t> axes, int64_t keepdims = 1);
};

/*! \brief reshape 的目标形状和 allowzero 语义属性。 */
class ReshapeAttrsNode : public BaseAttrsNode {
public:
    Array<int64_t> newshape;
    int allowzero = 0;
    KXC_DECLARE_ATTRS_NODE
};
class ReshapeAttrs : public Attrs {
    KXC_DECLARE_ATTRS_REF(ReshapeAttrs, ReshapeAttrsNode)

public:
    static ReshapeAttrs Create(Array<int64_t> newshape, int allowzero = 0);
};

/*! \brief transpose 的维度置换属性。 */
class TransposeAttrsNode : public BaseAttrsNode {
public:
    Array<int64_t> perm;
    KXC_DECLARE_ATTRS_NODE
};
class TransposeAttrs : public Attrs {
    KXC_DECLARE_ATTRS_REF(TransposeAttrs, TransposeAttrsNode)

public:
    static TransposeAttrs Create(Array<int64_t> perm);
};

KXC_DEFINE_SIMPLE_ATTRS(ReluAttrs)
KXC_DEFINE_SIMPLE_ATTRS(GlobalAvgPool2DAttrs)

/*! \brief flatten 的起始展平轴属性。 */
class FlattenAttrsNode : public BaseAttrsNode {
public:
    int axis = 1;
    KXC_DECLARE_ATTRS_NODE
};
class FlattenAttrs : public Attrs {
    KXC_DECLARE_ATTRS_REF(FlattenAttrs, FlattenAttrsNode)

public:
    static FlattenAttrs Create(int axis = 1);
};

/*! \brief gemm 的缩放系数和输入转置开关属性。 */
class GemmAttrsNode : public BaseAttrsNode {
public:
    float alpha = 1.0f;
    float beta = 1.0f;
    int transA = 0;
    int transB = 0;
    KXC_DECLARE_ATTRS_NODE
};
class GemmAttrs : public Attrs {
    KXC_DECLARE_ATTRS_REF(GemmAttrs, GemmAttrsNode)

public:
    static GemmAttrs Create(float alpha = 1.0f, float beta = 1.0f, int transA = 0,
                            int transB = 0);
};

/*! \brief 跨虚拟设备拷贝的源/目标设备和同步属性。 */
class DeviceCopyAttrsNode : public BaseAttrsNode {
public:
    VirtualDevice src_virtual_device;
    VirtualDevice dst_virtual_device;
    bool async = false;
    bool in_group = true;
    KXC_DECLARE_ATTRS_NODE
};
class DeviceCopyAttrs : public Attrs {
    KXC_DECLARE_ATTRS_REF(DeviceCopyAttrs, DeviceCopyAttrsNode)

public:
    static DeviceCopyAttrs Create(VirtualDevice src_virtual_device,
                                  VirtualDevice dst_virtual_device, bool async = false,
                                  bool in_group = true);
};

/*! \brief 集合通信算子的通信类型、归约类型和组信息属性。 */
class CollectiveAttrsNode : public BaseAttrsNode {
public:
    std::string kind;
    std::string reduce_kind = "sum";
    bool in_group = true;
    int group_id = 0;
    int root_worker = 0;
    KXC_DECLARE_ATTRS_NODE
};
class CollectiveAttrs : public Attrs {
    KXC_DECLARE_ATTRS_REF(CollectiveAttrs, CollectiveAttrsNode)

public:
    static CollectiveAttrs Create(std::string kind, std::string reduce_kind = "sum",
                                  bool in_group = true, int group_id = 0, int root_worker = 0);
};

}  // namespace relay
}  // namespace kxc
