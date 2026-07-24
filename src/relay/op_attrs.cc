/*! \file src/relay/op_attrs.cc
 * \brief 实现 Relay 算子节点构造和各类 attrs 的对象工厂。
 */

#include "kxc/relay/op.h"
#include "kxc/support/object_registration.h"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace kxc {
namespace relay {

KXC_OBJECT_DEFINE(OpNode)
KXC_OBJECT_DEFINE(BaseAttrsNode)
KXC_OBJECT_DEFINE(Conv2DAttrsNode)
KXC_OBJECT_DEFINE(DenseAttrsNode)
KXC_OBJECT_DEFINE(MaxPool2DAttrsNode)
KXC_OBJECT_DEFINE(SoftmaxAttrsNode)
KXC_OBJECT_DEFINE(LayerNormAttrsNode)
KXC_OBJECT_DEFINE(AddAttrsNode)
KXC_OBJECT_DEFINE(CastAttrsNode)
KXC_OBJECT_DEFINE(ReduceMeanAttrsNode)
KXC_OBJECT_DEFINE(ReshapeAttrsNode)
KXC_OBJECT_DEFINE(TransposeAttrsNode)
KXC_OBJECT_DEFINE(GatherAttrsNode)
KXC_OBJECT_DEFINE(ConcatenateAttrsNode)
KXC_OBJECT_DEFINE(SliceAttrsNode)
KXC_OBJECT_DEFINE(ReluAttrsNode)
KXC_OBJECT_DEFINE(GlobalAvgPool2DAttrsNode)
KXC_OBJECT_DEFINE(FlattenAttrsNode)
KXC_OBJECT_DEFINE(GemmAttrsNode)
KXC_OBJECT_DEFINE(DeviceCopyAttrsNode)
KXC_OBJECT_DEFINE(CollectiveAttrsNode)

namespace {

void AppendLengthDelimited(std::string* output, std::string_view value) {
    *output += std::to_string(value.size());
    output->push_back(':');
    output->append(value.data(), value.size());
}

std::string EncodeIntArray(const Array<int64_t>& values) {
    std::string result = std::to_string(values.size());
    result.push_back('[');
    for (int64_t value : values) {
        AppendLengthDelimited(&result, std::to_string(value));
    }
    result.push_back(']');
    return result;
}

std::string EncodeFloatBits(float value) {
    static_assert(sizeof(float) == sizeof(uint32_t));
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    std::ostringstream stream;
    stream << std::hex << std::setw(8) << std::setfill('0') << bits;
    return stream.str();
}

}  // namespace

void CanonicalAttrWriter::AddEncoded(std::string_view name,
                                     std::string_view type,
                                     const std::string& payload) {
    const std::string owned_name(name);
    if (std::find(field_names_.begin(), field_names_.end(), owned_name) !=
        field_names_.end()) {
        throw std::invalid_argument(
            "Canonical attrs serialization contains a duplicate field: " +
            owned_name);
    }
    field_names_.push_back(owned_name);
    AppendLengthDelimited(&buffer_, name);
    AppendLengthDelimited(&buffer_, type);
    AppendLengthDelimited(&buffer_, payload);
}

void CanonicalAttrWriter::Add(std::string_view name, bool value) {
    AddEncoded(name, "bool", value ? "1" : "0");
}

void CanonicalAttrWriter::Add(std::string_view name, int value) {
    AddEncoded(name, "int", std::to_string(value));
}

void CanonicalAttrWriter::Add(std::string_view name, int64_t value) {
    AddEncoded(name, "int64", std::to_string(value));
}

void CanonicalAttrWriter::Add(std::string_view name, float value) {
    AddEncoded(name, "float32-bits", EncodeFloatBits(value));
}

void CanonicalAttrWriter::Add(std::string_view name,
                              const std::string& value) {
    AddEncoded(name, "string", value);
}

void CanonicalAttrWriter::Add(std::string_view name,
                              const Array<int64_t>& value) {
    AddEncoded(name, "int64-array", EncodeIntArray(value));
}

void CanonicalAttrWriter::Add(std::string_view name,
                              const VirtualDevice& value) {
    AddEncoded(name, "virtual-device",
               SerializeVirtualDeviceLogicalPlacement(value));
}

std::string SerializeVirtualDeviceLogicalPlacement(
    const VirtualDevice& virtual_device) {
    CanonicalAttrWriter writer;
    writer.Add("defined", virtual_device.defined());
    if (!virtual_device.defined()) return writer.Finish();

    const VirtualDeviceNode* node = virtual_device.operator->();
    writer.Add("device_defined", node->device.defined());
    if (node->device.defined()) {
        writer.Add("device_type", static_cast<int>(node->device.device_type()));
        writer.Add("device_id", node->device.device_id());
    } else if (node->target.defined()) {
        const TargetNode* target = node->target.operator->();
        writer.Add("target_kind", target->kind);
        writer.Add("target_device_type", static_cast<int>(target->device_type));
        writer.Add("target_device_id", target->device_id);
    }
    writer.Add("memory_scope", node->memory_scope);
    writer.Add("virtual_device_id", node->virtual_device_id);
    return writer.Finish();
}

std::string CanonicalAttrWriter::Finish() const {
    return buffer_;
}

std::string SerializeAttrs(const Attrs& attrs) {
    if (!attrs.defined()) return "<none>";
    const auto* node = attrs.As<BaseAttrsNode>();
    if (!node) {
        throw std::invalid_argument(
            "Canonical attrs serialization requires BaseAttrsNode");
    }
    CanonicalAttrWriter writer;
    node->SerializeCanonical(writer);
    std::string result = "kxc.attrs.v1";
    AppendLengthDelimited(&result, node->GetTypeKey());
    AppendLengthDelimited(&result, writer.Finish());
    return result;
}

void Conv2DAttrsNode::SerializeCanonical(CanonicalAttrWriter& writer) const {
    writer.Add("strides", strides);
    writer.Add("padding", padding);
    writer.Add("dilation", dilation);
    writer.Add("groups", groups);
    writer.Add("channels", channels);
    writer.Add("kernel_size", kernel_size);
    writer.Add("data_layout", data_layout);
    writer.Add("kernel_layout", kernel_layout);
    writer.Add("out_layout", out_layout);
    writer.Add("out_dtype", out_dtype);
}

void DenseAttrsNode::SerializeCanonical(CanonicalAttrWriter& writer) const {
    writer.Add("units", units);
    writer.Add("out_dtype", out_dtype);
}

void MaxPool2DAttrsNode::SerializeCanonical(CanonicalAttrWriter& writer) const {
    writer.Add("pool_size", pool_size);
    writer.Add("strides", strides);
    writer.Add("padding", padding);
    writer.Add("dilation", dilation);
    writer.Add("layout", layout);
    writer.Add("ceil_mode", ceil_mode);
}

void SoftmaxAttrsNode::SerializeCanonical(CanonicalAttrWriter& writer) const {
    writer.Add("axis", axis);
}

void LayerNormAttrsNode::SerializeCanonical(CanonicalAttrWriter& writer) const {
    writer.Add("axis", axis);
    writer.Add("epsilon", epsilon);
    writer.Add("accumulation_dtype", accumulation_dtype);
}

void CastAttrsNode::SerializeCanonical(CanonicalAttrWriter& writer) const {
    writer.Add("to", to);
}

void ReduceMeanAttrsNode::SerializeCanonical(CanonicalAttrWriter& writer) const {
    writer.Add("axes", axes);
    writer.Add("keepdims", keepdims);
}

void ReshapeAttrsNode::SerializeCanonical(CanonicalAttrWriter& writer) const {
    writer.Add("newshape", newshape);
    writer.Add("allowzero", allowzero);
}

void TransposeAttrsNode::SerializeCanonical(CanonicalAttrWriter& writer) const {
    writer.Add("perm", perm);
}

void GatherAttrsNode::SerializeCanonical(CanonicalAttrWriter& writer) const {
    writer.Add("axis", axis);
}

void ConcatenateAttrsNode::SerializeCanonical(CanonicalAttrWriter& writer) const {
    writer.Add("axis", axis);
}

void SliceAttrsNode::SerializeCanonical(CanonicalAttrWriter& writer) const {
    writer.Add("starts", starts);
    writer.Add("ends", ends);
    writer.Add("axes", axes);
    writer.Add("steps", steps);
}

void FlattenAttrsNode::SerializeCanonical(CanonicalAttrWriter& writer) const {
    writer.Add("axis", axis);
}

void GemmAttrsNode::SerializeCanonical(CanonicalAttrWriter& writer) const {
    writer.Add("alpha", alpha);
    writer.Add("beta", beta);
    writer.Add("transA", transA);
    writer.Add("transB", transB);
}

void DeviceCopyAttrsNode::SerializeCanonical(CanonicalAttrWriter& writer) const {
    writer.Add("src_virtual_device", src_virtual_device);
    writer.Add("dst_virtual_device", dst_virtual_device);
    writer.Add("async", async);
    writer.Add("in_group", in_group);
}

void CollectiveAttrsNode::SerializeCanonical(CanonicalAttrWriter& writer) const {
    writer.Add("kind", kind);
    writer.Add("reduce_kind", reduce_kind);
    writer.Add("in_group", in_group);
    writer.Add("group_id", group_id);
    writer.Add("root_worker", root_worker);
}

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

LayerNormAttrs LayerNormAttrs::Create(int axis, float epsilon,
                                      std::string accumulation_dtype) {
    auto* node = new LayerNormAttrsNode();
    node->axis = axis;
    node->epsilon = epsilon;
    node->accumulation_dtype = std::move(accumulation_dtype);
    return InternalCreate(node);
}

// 创建 dtype 转换属性。
CastAttrs CastAttrs::Create(int to) {
    auto* node = new CastAttrsNode();
    node->to = to;
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

// 创建 transpose 轴排列属性。
TransposeAttrs TransposeAttrs::Create(Array<int64_t> perm) {
    auto* node = new TransposeAttrsNode();
    node->perm = std::move(perm);
    return InternalCreate(node);
}

GatherAttrs GatherAttrs::Create(int axis) {
    auto* node = new GatherAttrsNode();
    node->axis = axis;
    return InternalCreate(node);
}

ConcatenateAttrs ConcatenateAttrs::Create(int axis) {
    auto* node = new ConcatenateAttrsNode();
    node->axis = axis;
    return InternalCreate(node);
}

SliceAttrs SliceAttrs::Create(Array<int64_t> starts, Array<int64_t> ends,
                               Array<int64_t> axes, Array<int64_t> steps) {
    auto* node = new SliceAttrsNode();
    node->starts = std::move(starts);
    node->ends = std::move(ends);
    node->axes = std::move(axes);
    node->steps = std::move(steps);
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
KXC_DEFINE_SIMPLE_ATTRS_CREATE(ReluAttrs)
KXC_DEFINE_SIMPLE_ATTRS_CREATE(GlobalAvgPool2DAttrs)

#undef KXC_DEFINE_SIMPLE_ATTRS_CREATE

}  // namespace relay
}  // namespace kxc
