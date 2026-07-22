/*! \file src/relay/op/nn/convolution.cc
 * \brief 注册 Relay 算子及其 FRelayToTE compute。
 */

#include "kxc/relay/op_macros.h"
#include "kxc/relay/op.h"
#include "kxc/relay/op_attr_types.h"
#include "kxc/relay/type_infer.h"
#include "kxc/te/topi/nn.h"
#include <string>
#include <stdexcept>

namespace kxc {
namespace relay {

// ---------------------------------------------------------------------------
// 1. Define Attribute Structure
// ---------------------------------------------------------------------------
// Conv2D attributes (strides, padding, dilation, etc.)
// Note: Ideally, this definition should be in a header file (e.g., include/relay/attrs/nn.h)
// so that other parts of the system (like Pass) can access it.
// For this example, we assume it's defined in include/relay/op.h or we re-use the one there.
// But for completeness of a standalone op file, we often define local helpers or refer to shared ones.

// Since Conv2DAttrs is already defined in include/relay/op.h, we just use it.
// In a real large project, you would include "include/relay/attrs/nn.h".

namespace {
// 读取二维卷积属性的指定分量，并为省略项提供规范默认值。
int Read2DValue(const Array<int64_t>& v, int idx, int default_value) {
    if (idx < 0 || static_cast<size_t>(idx) >= v.size()) {
        return default_value;
    }
    return static_cast<int>(v[idx]);
}
}

// 将 Relay nn_conv2d 调用转换为 NCHW TE 卷积，并在存在 bias 时附加广播加法。
te::Tensor Conv2DCompute(const Attrs& attrs, const Array<te::Tensor>& inputs, const kxc::Type& out_type) {
    (void)out_type;
    if (inputs.size() < 2 || inputs.size() > 3) {
        throw std::runtime_error("nn_conv2d expects 2 or 3 inputs (data, weight[, bias])");
    }
    auto* p = attrs.As<Conv2DAttrsNode>();
    if (!p) {
        throw std::runtime_error("nn_conv2d expects Conv2DAttrs");
    }
    if (p->data_layout != "NCHW" && !p->data_layout.empty()) {
        throw std::runtime_error("nn_conv2d currently only supports NCHW");
    }

    int stride_h = Read2DValue(p->strides, 0, 1);
    int stride_w = Read2DValue(p->strides, 1, 1);
    int pad_h = Read2DValue(p->padding, 0, 0);
    int pad_w = Read2DValue(p->padding, 1, 0);
    int dilation_h = Read2DValue(p->dilation, 0, 1);
    int dilation_w = Read2DValue(p->dilation, 1, 1);

    if (inputs[0]->shape.size() != 4 || inputs[1]->shape.size() != 4) {
        throw std::runtime_error("nn_conv2d currently expects NCHW/OIHW rank-4 tensors");
    }

    bool has_bias = inputs.size() == 3;
    if (has_bias && inputs[2]->shape.size() != 1) {
        throw std::runtime_error("nn_conv2d bias must be rank-1");
    }
    te::Tensor conv_out = te::topi::conv2d_nchw(
        inputs[0], inputs[1], stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, "T_conv2d");

    if (!has_bias) {
        return conv_out;
    }
    return te::compute(conv_out->shape, [&](const Array<kxc::tir::Var>& axis) {
        return conv_out(axis) + inputs[2](axis[1]);
    }, "T_conv2d_bias_add");
}

// ---------------------------------------------------------------------------
// 注册卷积的属性类型、类型推导和 Relay-to-TE 计算入口。
// ---------------------------------------------------------------------------

KXC_REGISTER_OP(nn_conv2d)
    .describe(R"doc(2D convolution layer (e.g. spatial convolution over images).

This operator computes a 2D convolution of input `data` with `weight`.
The `data` input should have shape `(batch_size, in_channels, height, width)`
if layout is `NCHW`.
)doc")
    .set_input_arity_range(2, 3)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("weight", "Tensor", "The weight tensor.")
    .set_attr<std::string>("TAttrs", "Conv2DAttrs") // Bind to C++ Attribute Struct
    .set_attr<FInferType>("FInferType", Conv2DInferType)
    .set_attr<FRelayToTE>("FRelayToTE", Conv2DCompute);

} // namespace relay
} // namespace kxc

namespace kxc::builtin_anchor {
void RelayConvolutionOps() {}
}  // namespace kxc::builtin_anchor
