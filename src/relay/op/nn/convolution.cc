#include "relay/op_macros.h"
#include "relay/op.h"
#include "relay/op_attr_types.h"
#include "te/te.h"
#include "te/topi/utils.h"
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdio>

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
int Read2DValue(const std::vector<int64_t>& v, int idx, int default_value) {
    if (idx < 0 || static_cast<size_t>(idx) >= v.size()) {
        return default_value;
    }
    return static_cast<int>(v[idx]);
}
}

te::Tensor Conv2DCompute(const Attrs& attrs, const Array<te::Tensor>& inputs, const kxc::Type& out_type) {
    std::fprintf(stderr, "[conv] enter inputs=%zu\n", inputs.size());
    if (inputs.size() < 2 || inputs.size() > 3) {
        throw std::runtime_error("nn_conv2d expects 2 or 3 inputs (data, weight[, bias])");
    }
    auto* p = attrs.As<Conv2DAttrsNode>();
    std::fprintf(stderr, "[conv] attrs parsed=%d\n", p != nullptr ? 1 : 0);
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
    std::fprintf(stderr, "[conv] stride=(%d,%d) pad=(%d,%d) dil=(%d,%d)\n",
                 stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w);

    std::fprintf(stderr, "[conv] rank data=%zu weight=%zu\n",
                 inputs[0]->shape.size(), inputs[1]->shape.size());
    if (inputs[0]->shape.size() != 4 || inputs[1]->shape.size() != 4) {
        throw std::runtime_error("nn_conv2d currently expects NCHW/OIHW rank-4 tensors");
    }
    std::fprintf(stderr, "[conv] input ranks ok\n");

    tir::PrimExpr n = inputs[0]->shape[0];
    tir::PrimExpr c = inputs[0]->shape[1];
    tir::PrimExpr h = inputs[0]->shape[2];
    tir::PrimExpr w = inputs[0]->shape[3];

    tir::PrimExpr o = inputs[1]->shape[0];
    tir::PrimExpr kh = inputs[1]->shape[2];
    tir::PrimExpr kw = inputs[1]->shape[3];
    tir::PrimExpr dil_kh = (kh - 1) * dilation_h + 1;
    tir::PrimExpr dil_kw = (kw - 1) * dilation_w + 1;
    tir::PrimExpr oh = (h + 2 * pad_h - dil_kh) / stride_h + 1;
    tir::PrimExpr ow = (w + 2 * pad_w - dil_kw) / stride_w + 1;
    std::fprintf(stderr, "[conv] output shape expr ready\n");

    te::IterVar rc = te::reduce_axis(0, c, "rc");
    te::IterVar rh = te::reduce_axis(0, kh, "rh");
    te::IterVar rw = te::reduce_axis(0, kw, "rw");

    bool has_bias = inputs.size() == 3;
    if (has_bias && inputs[2]->shape.size() != 1) {
        throw std::runtime_error("nn_conv2d bias must be rank-1");
    }
    std::fprintf(stderr, "[conv] has_bias=%d\n", has_bias ? 1 : 0);

    te::Tensor conv_out = te::compute({n, o, oh, ow}, [&](const Array<kxc::tir::Var>& axis) {
        tir::Var bn = axis[0];
        tir::Var bo = axis[1];
        tir::Var by = axis[2];
        tir::Var bx = axis[3];

        tir::PrimExpr in_y = by * stride_h + tir::PrimExpr(rh->var) * dilation_h - pad_h;
        tir::PrimExpr in_x = bx * stride_w + tir::PrimExpr(rw->var) * dilation_w - pad_w;
        tir::PrimExpr in_bounds = (!(in_y < 0)) && (in_y < h) && (!(in_x < 0)) && (in_x < w);
        tir::PrimExpr in_val = tir::Select(
            in_bounds,
            inputs[0](bn, rc, in_y, in_x),
            te::topi::make_const(inputs[0]->dtype, 0));

        return te::sum(in_val * inputs[1](bo, rc, rh, rw), {rc, rh, rw});
    }, "T_conv2d");
    std::fprintf(stderr, "[conv] compute built\n");
    if (!has_bias) {
        return conv_out;
    }
    return te::compute(conv_out->shape, [&](const Array<kxc::tir::Var>& axis) {
        return conv_out(axis) + inputs[2](axis[1]);
    }, "T_conv2d_bias_add");
}

// ---------------------------------------------------------------------------
// 2. Operator Registration
// ---------------------------------------------------------------------------

KXC_REGISTER_OP(nn_conv2d)
    .describe(R"doc(2D convolution layer (e.g. spatial convolution over images).

This operator computes a 2D convolution of input `data` with `weight`.
The `data` input should have shape `(batch_size, in_channels, height, width)`
if layout is `NCHW`.
)doc")
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("weight", "Tensor", "The weight tensor.")
    .set_attr<std::string>("TAttrs", "Conv2DAttrs") // Bind to C++ Attribute Struct
    .set_attr<FRelayToTE>("FRelayToTE", Conv2DCompute);

} // namespace relay
} // namespace kxc
