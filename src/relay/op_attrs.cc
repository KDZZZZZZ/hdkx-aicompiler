/*! \file src/relay/op_attrs.cc
 * \brief 实现 Relay 节点、算子元数据、pass 工具和公共注册。
 */

#include "relay/op.h"

namespace kxc {
namespace relay {

// 构造并持有算子的稳定名称与说明元数据。
Op::Op(std::string name, std::string description) {
    auto* node = new OpNode();
    node->name = std::move(name);
    node->description = std::move(description);
    SetData(node);
}

// 创建卷积属性对象，并将可变长维度参数迁入对象系统容器。
Conv2DAttrs Conv2DAttrs::Create(Array<int64_t> strides, Array<int64_t> padding,
                                Array<int64_t> dilation, int groups, int channels,
                                Array<int64_t> kernel_size, std::string data_layout,
                                std::string kernel_layout, std::string out_layout,
                                std::string out_dtype) {
    auto* node = new Conv2DAttrsNode();
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

// 创建全连接层属性。
DenseAttrs DenseAttrs::Create(int64_t units, std::string out_dtype) {
    auto* node = new DenseAttrsNode();
    node->units = units;
    node->out_dtype = std::move(out_dtype);
    return InternalCreate(node);
}

// 创建二维池化属性对象。
MaxPool2DAttrs MaxPool2DAttrs::Create(Array<int64_t> strides, Array<int64_t> padding,
                                      Array<int64_t> dilation,
                                      Array<int64_t> pool_size, std::string layout,
                                      bool ceil_mode) {
    auto* node = new MaxPool2DAttrsNode();
    node->strides = std::move(strides);
    node->padding = std::move(padding);
    node->dilation = std::move(dilation);
    node->pool_size = std::move(pool_size);
    node->layout = std::move(layout);
    node->ceil_mode = ceil_mode;
    return InternalCreate(node);
}

// 创建 softmax 归约轴属性。
SoftmaxAttrs SoftmaxAttrs::Create(int axis) {
    auto* node = new SoftmaxAttrsNode();
    node->axis = axis;
    return InternalCreate(node);
}

// 创建 batch normalization 数值和开关属性。
BatchNormAttrs BatchNormAttrs::Create(double epsilon, bool center, bool scale) {
    auto* node = new BatchNormAttrsNode();
    node->epsilon = epsilon;
    node->center = center;
    node->scale = scale;
    return InternalCreate(node);
}

// 创建 dtype 转换属性。
CastAttrs CastAttrs::Create(int to) {
    auto* node = new CastAttrsNode();
    node->to = to;
    return InternalCreate(node);
}

// 创建张量拼接轴属性。
ConcatAttrs ConcatAttrs::Create(int axis) {
    auto* node = new ConcatAttrsNode();
    node->axis = axis;
    return InternalCreate(node);
}

// 创建持有 Storage-backed NDArray 的常量属性，保持张量生命周期。
ConstantAttrs ConstantAttrs::Create(runtime::NDArray value) {
    auto* node = new ConstantAttrsNode();
    node->value = std::move(value);
    return InternalCreate(node);
}

// 创建带显式填充值张量的 constant_of_shape 属性。
ConstantOfShapeAttrs ConstantOfShapeAttrs::Create(runtime::NDArray value) {
    auto* node = new ConstantOfShapeAttrsNode();
    node->value = std::move(value);
    return InternalCreate(node);
}

// 创建采用默认填充值的 constant_of_shape 属性。
ConstantOfShapeAttrs ConstantOfShapeAttrs::Create() {
    return InternalCreate(new ConstantOfShapeAttrsNode());
}

// 创建 gather 轴属性。
GatherAttrs GatherAttrs::Create(int axis) {
    auto* node = new GatherAttrsNode();
    node->axis = axis;
    return InternalCreate(node);
}

// 创建均值归约的轴集合和维度保留属性。
ReduceMeanAttrs ReduceMeanAttrs::Create(Array<int64_t> axes, int64_t keepdims) {
    auto* node = new ReduceMeanAttrsNode();
    node->axes = std::move(axes);
    node->keepdims = keepdims;
    return InternalCreate(node);
}

// 创建 reshape 目标形状和零维解释属性。
ReshapeAttrs ReshapeAttrs::Create(Array<int64_t> newshape, int allowzero) {
    auto* node = new ReshapeAttrsNode();
    node->newshape = std::move(newshape);
    node->allowzero = allowzero;
    return InternalCreate(node);
}

// 创建 split 分段长度和分割轴属性。
SplitAttrs SplitAttrs::Create(Array<int64_t> split, int axis) {
    auto* node = new SplitAttrsNode();
    node->split = std::move(split);
    node->axis = axis;
    return InternalCreate(node);
}

// 创建 transpose 轴排列属性。
TransposeAttrs TransposeAttrs::Create(Array<int64_t> perm) {
    auto* node = new TransposeAttrsNode();
    node->perm = std::move(perm);
    return InternalCreate(node);
}

// 创建 flatten 起始轴属性。
FlattenAttrs FlattenAttrs::Create(int axis) {
    auto* node = new FlattenAttrsNode();
    node->axis = axis;
    return InternalCreate(node);
}

// 创建 GEMM 缩放系数和转置标志属性。
GemmAttrs GemmAttrs::Create(float alpha, float beta, int transA, int transB) {
    auto* node = new GemmAttrsNode();
    node->alpha = alpha;
    node->beta = beta;
    node->transA = transA;
    node->transB = transB;
    return InternalCreate(node);
}

// 创建跨虚拟设备复制属性，显式记录源、目标和异步语义。
DeviceCopyAttrs DeviceCopyAttrs::Create(VirtualDevice src_virtual_device,
                                        VirtualDevice dst_virtual_device, bool async,
                                        bool in_group) {
    auto* node = new DeviceCopyAttrsNode();
    node->src_virtual_device = std::move(src_virtual_device);
    node->dst_virtual_device = std::move(dst_virtual_device);
    node->async = async;
    node->in_group = in_group;
    return InternalCreate(node);
}

// 创建 collective 类型、分组和根 worker 属性。
CollectiveAttrs CollectiveAttrs::Create(std::string kind, std::string reduce_kind, bool in_group,
                                        int group_id, int root_worker) {
    auto* node = new CollectiveAttrsNode();
    node->kind = std::move(kind);
    node->reduce_kind = std::move(reduce_kind);
    node->in_group = in_group;
    node->group_id = group_id;
    node->root_worker = root_worker;
    return InternalCreate(node);
}

// 为无字段属性节点统一生成 Create 工厂。
#define KXC_DEFINE_SIMPLE_ATTRS_CREATE(TypeName) \
    TypeName TypeName::Create() { return InternalCreate(new TypeName##Node()); }

KXC_DEFINE_SIMPLE_ATTRS_CREATE(AddAttrs)
KXC_DEFINE_SIMPLE_ATTRS_CREATE(DivAttrs)
KXC_DEFINE_SIMPLE_ATTRS_CREATE(EqualAttrs)
KXC_DEFINE_SIMPLE_ATTRS_CREATE(ErfAttrs)
KXC_DEFINE_SIMPLE_ATTRS_CREATE(ExpandAttrs)
KXC_DEFINE_SIMPLE_ATTRS_CREATE(ReluAttrs)
KXC_DEFINE_SIMPLE_ATTRS_CREATE(GlobalAvgPool2DAttrs)

#undef KXC_DEFINE_SIMPLE_ATTRS_CREATE

}  // namespace relay
}  // namespace kxc
