#include "relay/op_macros.h"
#include "relay/op.h"
#include "relay/op_attr_types.h"
#include "te/te.h"
#include "te/topi/utils.h"
#include <vector>
#include <string>
#include <stdexcept>

namespace kxc {
namespace relay {

namespace {
int Read2DValue(const std::vector<int64_t>& v, int idx, int default_value) {
    if (idx < 0 || static_cast<size_t>(idx) >= v.size()) {
        return default_value;
    }
    return static_cast<int>(v[idx]);
}

struct Padding2D {
    int top = 0;
    int left = 0;
    int bottom = 0;
    int right = 0;
};

Padding2D ParsePadding2D(const std::vector<int64_t>& padding) {
    if (padding.size() == 2) {
        return {static_cast<int>(padding[0]), static_cast<int>(padding[1]),
                static_cast<int>(padding[0]), static_cast<int>(padding[1])};
    }
    if (padding.size() >= 4) {
        return {static_cast<int>(padding[0]), static_cast<int>(padding[1]),
                static_cast<int>(padding[2]), static_cast<int>(padding[3])};
    }
    return {};
}
}

te::Tensor MaxPool2DCompute(const Attrs& attrs, const Array<te::Tensor>& inputs, const kxc::Type& out_type) {
    if (inputs.size() != 1) {
        throw std::runtime_error("nn_max_pool2d expects exactly 1 input");
    }
    auto* p = attrs.As<MaxPool2DAttrsNode>();
    if (!p) {
        throw std::runtime_error("nn_max_pool2d expects MaxPool2DAttrs");
    }
    if (inputs[0]->shape.size() != 4) {
        throw std::runtime_error("nn_max_pool2d expects NCHW rank-4 input");
    }

    int kh = Read2DValue(p->pool_size, 0, 1);
    int kw = Read2DValue(p->pool_size, 1, 1);
    int sh = Read2DValue(p->strides, 0, 1);
    int sw = Read2DValue(p->strides, 1, 1);
    Padding2D pad = ParsePadding2D(p->padding);

    tir::PrimExpr n = inputs[0]->shape[0];
    tir::PrimExpr c = inputs[0]->shape[1];
    tir::PrimExpr h = inputs[0]->shape[2];
    tir::PrimExpr w = inputs[0]->shape[3];
    tir::PrimExpr oh = (h + pad.top + pad.bottom - kh) / sh + 1;
    tir::PrimExpr ow = (w + pad.left + pad.right - kw) / sw + 1;

    te::IterVar rh = te::reduce_axis(0, kh, "rh");
    te::IterVar rw = te::reduce_axis(0, kw, "rw");

    te::Tensor sum_out = te::compute({n, c, oh, ow}, [&](const Array<kxc::tir::Var>& axis) {
        tir::Var bn = axis[0];
        tir::Var bc = axis[1];
        tir::Var by = axis[2];
        tir::Var bx = axis[3];

        tir::PrimExpr in_y = by * sh + rh - pad.top;
        tir::PrimExpr in_x = bx * sw + rw - pad.left;
        tir::PrimExpr in_bounds = (!(in_y < 0)) && (in_y < h) && (!(in_x < 0)) && (in_x < w);
        tir::PrimExpr val = tir::Select(
            in_bounds,
            inputs[0](bn, bc, in_y, in_x),
            te::topi::make_const(inputs[0]->dtype, -1e30));
        return te::sum(val, {rh, rw});
    }, "T_max_pool2d");
    return sum_out;
}

te::Tensor AvgPool2DCompute(const Attrs& attrs, const Array<te::Tensor>& inputs, const kxc::Type& out_type) {
    if (inputs.size() != 1) {
        throw std::runtime_error("nn_avg_pool2d expects exactly 1 input");
    }
    auto* p = attrs.As<MaxPool2DAttrsNode>();
    if (!p) {
        throw std::runtime_error("nn_avg_pool2d expects MaxPool2DAttrs");
    }
    if (inputs[0]->shape.size() != 4) {
        throw std::runtime_error("nn_avg_pool2d expects NCHW rank-4 input");
    }

    int kh = Read2DValue(p->pool_size, 0, 1);
    int kw = Read2DValue(p->pool_size, 1, 1);
    int sh = Read2DValue(p->strides, 0, 1);
    int sw = Read2DValue(p->strides, 1, 1);
    Padding2D pad = ParsePadding2D(p->padding);

    tir::PrimExpr n = inputs[0]->shape[0];
    tir::PrimExpr c = inputs[0]->shape[1];
    tir::PrimExpr h = inputs[0]->shape[2];
    tir::PrimExpr w = inputs[0]->shape[3];
    tir::PrimExpr oh = (h + pad.top + pad.bottom - kh) / sh + 1;
    tir::PrimExpr ow = (w + pad.left + pad.right - kw) / sw + 1;

    te::IterVar rh = te::reduce_axis(0, kh, "rh");
    te::IterVar rw = te::reduce_axis(0, kw, "rw");

    te::Tensor sum_out = te::compute({n, c, oh, ow}, [&](const Array<kxc::tir::Var>& axis) {
        tir::Var bn = axis[0];
        tir::Var bc = axis[1];
        tir::Var by = axis[2];
        tir::Var bx = axis[3];

        tir::PrimExpr in_y = by * sh + rh - pad.top;
        tir::PrimExpr in_x = bx * sw + rw - pad.left;
        tir::PrimExpr in_bounds = (!(in_y < 0)) && (in_y < h) && (!(in_x < 0)) && (in_x < w);
        tir::PrimExpr val = tir::Select(
            in_bounds,
            inputs[0](bn, bc, in_y, in_x),
            te::topi::make_const(inputs[0]->dtype, 0));
        return te::sum(val, {rh, rw});
    }, "T_avg_pool2d_sum");

    return te::compute(sum_out->shape, [&](const Array<kxc::tir::Var>& axis) {
        return sum_out(axis) / (kh * kw);
    }, "T_avg_pool2d");
}

te::Tensor GlobalAvgPool2DCompute(const Attrs& attrs, const Array<te::Tensor>& inputs, const kxc::Type& out_type) {
    if (inputs.size() != 1) {
        throw std::runtime_error("nn_global_avg_pool2d expects exactly 1 input");
    }
    if (inputs[0]->shape.size() != 4) {
        throw std::runtime_error("nn_global_avg_pool2d expects NCHW rank-4 input");
    }
    tir::PrimExpr n = inputs[0]->shape[0];
    tir::PrimExpr c = inputs[0]->shape[1];
    tir::PrimExpr h = inputs[0]->shape[2];
    tir::PrimExpr w = inputs[0]->shape[3];

    te::IterVar rh = te::reduce_axis(0, h, "rh");
    te::IterVar rw = te::reduce_axis(0, w, "rw");

    te::Tensor sum_out = te::compute({n, c, 1, 1}, [&](const Array<kxc::tir::Var>& axis) {
        tir::Var bn = axis[0];
        tir::Var bc = axis[1];
        return te::sum(inputs[0](bn, bc, rh, rw), {rh, rw});
    }, "T_global_avg_pool2d_sum");

    return te::compute(sum_out->shape, [&](const Array<kxc::tir::Var>& axis) {
        return sum_out(axis) / (h * w);
    }, "T_global_avg_pool2d");
}

// ---------------------------------------------------------------------------
// 1. Operator Registration for Pooling
// ---------------------------------------------------------------------------

KXC_REGISTER_OP(nn_max_pool2d)
    .describe(R"doc(2D max pooling operation.

This operator performs max pooling on the input tensor.
)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<std::string>("TAttrs", "MaxPool2DAttrs")
    .set_attr<FRelayToTE>("FRelayToTE", MaxPool2DCompute);

KXC_REGISTER_OP(nn_avg_pool2d)
    .describe(R"doc(2D average pooling operation.

This operator performs average pooling on the input tensor.
)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    // Reusing MaxPool2DAttrs for AvgPool2D as they share structure (pool_size, strides, padding)
    // In real TVM, they might share a generic Pool2DAttrs.
    .set_attr<std::string>("TAttrs", "MaxPool2DAttrs")
    .set_attr<FRelayToTE>("FRelayToTE", AvgPool2DCompute); 

KXC_REGISTER_OP(nn_global_avg_pool2d)
    .describe(R"doc(Global average pooling operation.

Reduces the spatial dimensions (H, W) to 1x1 by averaging.
)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<std::string>("TAttrs", "GlobalAvgPool2DAttrs")
    .set_attr<FRelayToTE>("FRelayToTE", GlobalAvgPool2DCompute);

} // namespace relay
} // namespace kxc
