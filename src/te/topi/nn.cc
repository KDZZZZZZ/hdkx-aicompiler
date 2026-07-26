/*! \file src/te/topi/nn.cc
 * \brief Implements TOPI nn helpers.
 */

#include "kxc/te/topi/nn.h"

#include <utility>

namespace kxc {
namespace te {
namespace topi {

Tensor relu(const Tensor& x, std::string name , std::string tag ){
    return compute(
        x->shape,
        [&](const Array<tir::Var>& indices) {
            // max(x, 0)
            return Max(x(indices), make_const(x->dtype, 0)); 
            // make_const might need implementation or use 0 casted
        },
        name,
        tag
    );
}

Tensor leaky_relu(const Tensor& x, double alpha, std::string name , std::string tag ){
    return compute(
        x->shape,
        [&](const Array<tir::Var>& indices) {
            PrimExpr val = x(indices);
            return Select(val > make_const(x->dtype, 0), val, val * make_const(x->dtype, alpha));
        },
        name,
        tag
    );
}

Tensor dense(const Tensor& A, const Tensor& B, const Tensor& bias , std::string name , std::string tag ){
    // Check dimensions
    // A: [M, K], B: [N, K]
    // M = A.shape[0], K = A.shape[1]
    // N = B.shape[0]
    
    // We assume 2D for simplicity
    PrimExpr M = A->shape[0];
    PrimExpr K = A->shape[1];
    PrimExpr N = B->shape[0]; // Assuming B is (N, K)
    
    // Reduction axis k
    IterVar k = reduce_axis(0, K, "k");
    
    Tensor matmul = compute(
        {M, N},
        [&](const Array<tir::Var>& indices) {
            tir::Var i = indices[0];
            tir::Var j = indices[1];
            return kxc::te::sum(A(i, k) * B(j, k), {k});
        },
        name + "_matmul",
        tag
    );
    
    if (bias.defined()) {
        // Bias add: [M, N] + [N] (broadcast)
        // Using generic add from broadcast (need to include broadcast.h or re-implement)
        return add(matmul, bias, name, kBroadcast);
    }
    
    return matmul;
}

Tensor matmul(const Tensor& A, const Tensor& B, std::string name , std::string tag ){
    if (!A.defined() || !B.defined() || A->shape.size() < 2 || B->shape.size() < 2) {
        throw std::runtime_error("topi::matmul expects rank >= 2 tensors");
    }

    const size_t a_rank = A->shape.size();
    const size_t b_rank = B->shape.size();
    const PrimExpr M = A->shape[a_rank - 2];
    const PrimExpr K = A->shape[a_rank - 1];
    const PrimExpr rhs_k = B->shape[b_rank - 2];
    const PrimExpr N = B->shape[b_rank - 1];
    int64_t lhs_k_value = 0;
    int64_t rhs_k_value = 0;
    if (GetConstInt(K, &lhs_k_value) && GetConstInt(rhs_k, &rhs_k_value) &&
        lhs_k_value != rhs_k_value) {
        throw std::runtime_error("topi::matmul reduction dimension mismatch");
    }

    Array<PrimExpr> a_batch;
    Array<PrimExpr> b_batch;
    for (size_t i = 0; i + 2 < a_rank; ++i) a_batch.push_back(A->shape[i]);
    for (size_t i = 0; i + 2 < b_rank; ++i) b_batch.push_back(B->shape[i]);
    const Array<PrimExpr> batch_shape = detail::InferBroadcastShape(a_batch, b_batch);
    // Array is a mutable reference type, so copy elements instead of aliasing
    // batch_shape and accidentally appending matrix axes to the broadcast rank.
    Array<PrimExpr> output_shape;
    for (const PrimExpr& dimension : batch_shape) {
        output_shape.push_back(dimension);
    }
    output_shape.push_back(M);
    output_shape.push_back(N);

    const IterVar k = reduce_axis(0, K, "k");
    return compute(
        output_shape,
        [A, B, a_batch, b_batch, batch_shape, k](const Array<tir::Var>& indices) {
            Array<tir::Var> batch_indices;
            for (size_t i = 0; i + 2 < indices.size(); ++i) {
                batch_indices.push_back(indices[i]);
            }
            Array<PrimExpr> a_indices =
                detail::GetBroadcastIndices(batch_indices, a_batch, batch_shape);
            Array<PrimExpr> b_indices =
                detail::GetBroadcastIndices(batch_indices, b_batch, batch_shape);
            a_indices.push_back(indices[indices.size() - 2]);
            a_indices.push_back(AsPrimExpr(k));
            b_indices.push_back(AsPrimExpr(k));
            b_indices.push_back(indices[indices.size() - 1]);
            return kxc::te::sum(A(a_indices) * B(b_indices), {k});
        },
        name,
        tag
    );
}

Tensor conv2d_nchw(const Tensor& data, const Tensor& kernel, AxisPair2D strides,
                   Padding2D padding, AxisPair2D dilation, std::string name,
                   std::string tag) {
    PrimExpr N = data->shape[0];
    PrimExpr C = data->shape[1];
    PrimExpr H = data->shape[2];
    PrimExpr W = data->shape[3];

    PrimExpr O = kernel->shape[0];
    PrimExpr KH = kernel->shape[2];
    PrimExpr KW = kernel->shape[3];

    // 与 Relay 类型推导共用同一公式，保证 compute 出的 shape 与推导出的 TensorType 一致。
    PrimExpr OH = WindowOutputExtent<PrimExpr>(H, KH, padding.top, padding.bottom, strides.h,
                                               dilation.h, /*ceil_mode=*/false);
    PrimExpr OW = WindowOutputExtent<PrimExpr>(W, KW, padding.left, padding.right, strides.w,
                                               dilation.w, /*ceil_mode=*/false);

    // Reduction axes
    IterVar rc = reduce_axis(0, C, "rc");
    IterVar rh = reduce_axis(0, KH, "rh");
    IterVar rw = reduce_axis(0, KW, "rw");

    return compute(
        {N, O, OH, OW},
        [&](const Array<tir::Var>& indices) {
            tir::Var n = indices[0];
            tir::Var o = indices[1];
            tir::Var h = indices[2];
            tir::Var w = indices[3];

            // Input indices
            PrimExpr h_in = PrimExpr(strides.h) * h + PrimExpr(dilation.h) * rh -
                            padding.top;
            PrimExpr w_in = PrimExpr(strides.w) * w + PrimExpr(dilation.w) * rw -
                            padding.left;

            // Pad handling (PaddedInput)
            // Simplified: Assume Select/If logic or data is already padded.
            // If data is not padded, we need check bounds.
            // Logic: if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) data(...) else 0
            
            PrimExpr in_val = Select(
                (h_in >= 0) && (h_in < H) && (w_in >= 0) && (w_in < W),
                data(n, rc, h_in, w_in),
                make_const(data->dtype, 0)
            );
            
            return kxc::te::sum(in_val * kernel(o, rc, rh, rw), {rc, rh, rw});
        },
        name,
        tag
    );
}

Tensor conv2d_nchw(const Tensor& data, const Tensor& kernel, int stride_h, int stride_w,
                   Padding2D padding, int dilation_h, int dilation_w,
                   std::string name, std::string tag) {
    return conv2d_nchw(data, kernel, AxisPair2D{stride_h, stride_w}, padding,
                       AxisPair2D{dilation_h, dilation_w}, std::move(name),
                       std::move(tag));
}

Tensor conv2d_nchw(const Tensor& data, const Tensor& kernel, int stride_h, int stride_w,
                   int pad_h, int pad_w, int dilation_h, int dilation_w,
                   std::string name, std::string tag) {
    return conv2d_nchw(data, kernel, AxisPair2D{stride_h, stride_w},
                       Padding2D{pad_h, pad_w, pad_h, pad_w},
                       AxisPair2D{dilation_h, dilation_w}, std::move(name),
                       std::move(tag));
}

Tensor pool2d(const Tensor& data, AxisPair2D kernel_size, AxisPair2D stride,
              Padding2D padding, AxisPair2D dilation, std::string pool_type,
              bool ceil_mode, std::string name, std::string tag) {
    if (pool_type != "max" && pool_type != "avg") {
        throw std::runtime_error("topi::pool2d only supports pool_type=max/avg");
    }

    // Assuming NCHW
    PrimExpr N = data->shape[0];
    PrimExpr C = data->shape[1];
    PrimExpr H = data->shape[2];
    PrimExpr W = data->shape[3];

    const int64_t KH = kernel_size.h;
    const int64_t KW = kernel_size.w;
    const int64_t SH = stride.h;
    const int64_t SW = stride.w;
    const int64_t DH = dilation.h;
    const int64_t DW = dilation.w;

    // 与 Relay 类型推导共用同一公式和同一 dilation。
    PrimExpr OH = WindowOutputExtent<PrimExpr>(H, KH, padding.top, padding.bottom, SH, DH,
                                               ceil_mode);
    PrimExpr OW = WindowOutputExtent<PrimExpr>(W, KW, padding.left, padding.right, SW, DW,
                                               ceil_mode);

    IterVar rh = reduce_axis(0, KH, "rh");
    IterVar rw = reduce_axis(0, KW, "rw");

    if (pool_type == "max") {
        return compute(
            {N, C, OH, OW},
            [&](const Array<tir::Var>& indices) {
                tir::Var n = indices[0];
                tir::Var c = indices[1];
                tir::Var h = indices[2];
                tir::Var w = indices[3];

                PrimExpr h_in =
                    PrimExpr(SH) * h + PrimExpr(DH) * rh - padding.top;
                PrimExpr w_in =
                    PrimExpr(SW) * w + PrimExpr(DW) * rw - padding.left;

                PrimExpr in_val = Select(
                    (h_in >= 0) && (h_in < H) && (w_in >= 0) && (w_in < W),
                    data(n, c, h_in, w_in),
                    make_const(data->dtype, -1e30));
                return kxc::te::max(in_val, {rh, rw});
            },
            name,
            tag);
    }

    Tensor sum_out = compute(
        {N, C, OH, OW},
        [&](const Array<tir::Var>& indices) {
            tir::Var n = indices[0];
            tir::Var c = indices[1];
            tir::Var h = indices[2];
            tir::Var w = indices[3];

            PrimExpr h_in = PrimExpr(SH) * h + PrimExpr(DH) * rh - padding.top;
            PrimExpr w_in = PrimExpr(SW) * w + PrimExpr(DW) * rw - padding.left;

            PrimExpr in_val = Select(
                (h_in >= 0) && (h_in < H) && (w_in >= 0) && (w_in < W),
                data(n, c, h_in, w_in),
                make_const(data->dtype, 0));
            return kxc::te::sum(in_val, {rh, rw});
        },
        name + "_sum",
        tag);

    return compute(
        sum_out->shape,
        [sum_out, KH, KW](const Array<tir::Var>& indices) {
            return sum_out(indices) / make_const(sum_out->dtype, KH * KW);
        },
        name,
        tag);
}

Tensor pool2d(const Tensor& data, Array<int> kernel_size, Array<int> stride,
              Padding2D padding, Array<int> dilation, std::string pool_type,
              bool ceil_mode, std::string name, std::string tag) {
    if (kernel_size.size() < 2) {
        throw std::runtime_error("topi::pool2d expects kernel_size with at least 2 elements");
    }
    if (stride.size() < 2) {
        throw std::runtime_error("topi::pool2d expects stride with at least 2 elements");
    }
    if (dilation.size() < 2) {
        throw std::runtime_error("topi::pool2d expects dilation with at least 2 elements");
    }
    return pool2d(data, AxisPair2D{kernel_size[0], kernel_size[1]},
                  AxisPair2D{stride[0], stride[1]}, padding,
                  AxisPair2D{dilation[0], dilation[1]}, std::move(pool_type),
                  ceil_mode, std::move(name), std::move(tag));
}

Tensor pool2d(const Tensor& data, Array<int> kernel_size, Array<int> stride,
              Array<int> padding, std::string pool_type, bool ceil_mode,
              std::string name, std::string tag) {
    if (kernel_size.size() < 2) {
        throw std::runtime_error("topi::pool2d expects kernel_size with at least 2 elements");
    }
    if (stride.size() < 2) {
        throw std::runtime_error("topi::pool2d expects stride with at least 2 elements");
    }
    Padding2D expanded_padding;
    if (padding.size() >= 4) {
        expanded_padding =
            Padding2D{padding[0], padding[1], padding[2], padding[3]};
    } else if (padding.size() >= 2) {
        expanded_padding =
            Padding2D{padding[0], padding[1], padding[0], padding[1]};
    }
    return pool2d(data, AxisPair2D{kernel_size[0], kernel_size[1]},
                  AxisPair2D{stride[0], stride[1]}, expanded_padding, AxisPair2D{},
                  std::move(pool_type), ceil_mode, std::move(name), std::move(tag));
}

Tensor global_avg_pool2d(const Tensor& data, std::string name ,
                                std::string tag ){
    PrimExpr N = data->shape[0];
    PrimExpr C = data->shape[1];
    PrimExpr H = data->shape[2];
    PrimExpr W = data->shape[3];

    IterVar rh = reduce_axis(0, H, "rh");
    IterVar rw = reduce_axis(0, W, "rw");

    Tensor sum_out = compute(
        {N, C, 1, 1},
        [&](const Array<tir::Var>& indices) {
            tir::Var n = indices[0];
            tir::Var c = indices[1];
            return kxc::te::sum(data(n, c, rh, rw), {rh, rw});
        },
        name + "_sum",
        tag);

    return compute(
        sum_out->shape,
        [&](const Array<tir::Var>& indices) {
            return sum_out(indices) / (H * W);
        },
        name,
        tag);
}

}  // namespace topi
}  // namespace te
}  // namespace kxc
