#include "relay/op.h"

namespace kxc {
namespace relay {

Op::Op(std::string name, std::string description) {
    auto* node = new OpNode();
    node->name = std::move(name);
    node->description = std::move(description);
    SetData(node);
}

Conv2DAttrs Conv2DAttrs::Create(std::vector<int64_t> strides, std::vector<int64_t> padding,
                                std::vector<int64_t> dilation, int groups, int channels,
                                std::vector<int64_t> kernel_size, std::string data_layout,
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

DenseAttrs DenseAttrs::Create(int64_t units, std::string out_dtype) {
    auto* node = new DenseAttrsNode();
    node->units = units;
    node->out_dtype = std::move(out_dtype);
    return InternalCreate(node);
}

MaxPool2DAttrs MaxPool2DAttrs::Create(std::vector<int64_t> strides, std::vector<int64_t> padding,
                                      std::vector<int64_t> dilation,
                                      std::vector<int64_t> pool_size, std::string layout,
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

SoftmaxAttrs SoftmaxAttrs::Create(int axis) {
    auto* node = new SoftmaxAttrsNode();
    node->axis = axis;
    return InternalCreate(node);
}

BatchNormAttrs BatchNormAttrs::Create(double epsilon, bool center, bool scale) {
    auto* node = new BatchNormAttrsNode();
    node->epsilon = epsilon;
    node->center = center;
    node->scale = scale;
    return InternalCreate(node);
}

CastAttrs CastAttrs::Create(int to) {
    auto* node = new CastAttrsNode();
    node->to = to;
    return InternalCreate(node);
}

ConcatAttrs ConcatAttrs::Create(int axis) {
    auto* node = new ConcatAttrsNode();
    node->axis = axis;
    return InternalCreate(node);
}

ConstantAttrs ConstantAttrs::Create(runtime::NDArray value) {
    auto* node = new ConstantAttrsNode();
    node->value = std::move(value);
    return InternalCreate(node);
}

ConstantOfShapeAttrs ConstantOfShapeAttrs::Create(runtime::NDArray value) {
    auto* node = new ConstantOfShapeAttrsNode();
    node->value = std::move(value);
    return InternalCreate(node);
}

ConstantOfShapeAttrs ConstantOfShapeAttrs::Create() {
    return InternalCreate(new ConstantOfShapeAttrsNode());
}

GatherAttrs GatherAttrs::Create(int axis) {
    auto* node = new GatherAttrsNode();
    node->axis = axis;
    return InternalCreate(node);
}

ReduceMeanAttrs ReduceMeanAttrs::Create(std::vector<int64_t> axes, int64_t keepdims) {
    auto* node = new ReduceMeanAttrsNode();
    node->axes = std::move(axes);
    node->keepdims = keepdims;
    return InternalCreate(node);
}

ReshapeAttrs ReshapeAttrs::Create(std::vector<int64_t> newshape, int allowzero) {
    auto* node = new ReshapeAttrsNode();
    node->newshape = std::move(newshape);
    node->allowzero = allowzero;
    return InternalCreate(node);
}

SplitAttrs SplitAttrs::Create(std::vector<int64_t> split, int axis) {
    auto* node = new SplitAttrsNode();
    node->split = std::move(split);
    node->axis = axis;
    return InternalCreate(node);
}

TransposeAttrs TransposeAttrs::Create(std::vector<int64_t> perm) {
    auto* node = new TransposeAttrsNode();
    node->perm = std::move(perm);
    return InternalCreate(node);
}

FlattenAttrs FlattenAttrs::Create(int axis) {
    auto* node = new FlattenAttrsNode();
    node->axis = axis;
    return InternalCreate(node);
}

GemmAttrs GemmAttrs::Create(float alpha, float beta, int transA, int transB) {
    auto* node = new GemmAttrsNode();
    node->alpha = alpha;
    node->beta = beta;
    node->transA = transA;
    node->transB = transB;
    return InternalCreate(node);
}

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
