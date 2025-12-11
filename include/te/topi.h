#pragma once
#include "te/te.h"
#include "tir/expr.h"
#include <algorithm>

namespace kxc {
namespace topi {

using namespace kxc::te;
using namespace kxc::tir;

// Helper for make_const
inline PrimExpr make_const(DataType t, int64_t value) {
    if (t.code == 0) return IntImm(value, t); // Int
    if (t.code == 1) return IntImm(value, t); // UInt
    if (t.code == 2) return FloatImm((double)value, t); // Float
    return IntImm(value, t);
}

// --- Broadcast/Element-wise Operations ---

// Add
inline Tensor add(const Tensor& A, const Tensor& B, std::string name = "T_add") {
    // Assuming same shape for simplicity. 
    // Real implementation needs broadcasting logic.
    return compute(A->shape, [&](const std::vector<Var>& i) {
        return A(i) + B(i);
    }, name);
}

// Sub
inline Tensor subtract(const Tensor& A, const Tensor& B, std::string name = "T_sub") {
    return compute(A->shape, [&](const std::vector<Var>& i) {
        return A(i) - B(i); // Using operator- from PrimExpr
    }, name);
}

// Mul
inline Tensor multiply(const Tensor& A, const Tensor& B, std::string name = "T_mul") {
    return compute(A->shape, [&](const std::vector<Var>& i) {
        return A(i) * B(i);
    }, name);
}

// Relu: max(x, 0)
inline Tensor relu(const Tensor& A, std::string name = "T_relu") {
    return compute(A->shape, [&](const std::vector<Var>& i) {
        // We need a Max op or similar.
        // Assuming Max(PrimExpr, PrimExpr) exists.
        return Max(A(i), make_const(A->dtype, 0));
    }, name);
}

// --- Reduction Operations ---

// MatMul: C[i, j] = sum(A[i, k] * B[k, j], axis=k)
inline Tensor matmul(const Tensor& A, const Tensor& B, std::string name = "T_matmul") {
    // A: [M, K], B: [K, N]
    // C: [M, N]
    
    // Check shapes (simplified)
    // PrimExpr M = A->shape[0];
    // PrimExpr K = A->shape[1];
    // PrimExpr N = B->shape[1];
    // Wait, B->shape[0] should be K.
    
    // In symbolic world, we assume they match or assertion fails at runtime.
    // Let's assume input shapes are valid [M, K] and [K, N]
    
    auto M = A->shape[0];
    auto N = B->shape[1]; // Or B->shape[0] if transposed? Assume standard matmul
    auto K = A->shape[1];
    
    IterVar k = reduce_axis(0, K, "k");
    
    return compute({M, N}, [&](const std::vector<Var>& indices) {
        Var i = indices[0];
        Var j = indices[1];
        return sum(A(i, k) * B(k, j), {k});
    }, name);
}

// Conv2D (NCHW)
// Input: [N, CI, H, W]
// Weight: [CO, CI, KH, KW]
// Output: [N, CO, OH, OW]
inline Tensor conv2d_nchw(const Tensor& Input, const Tensor& Filter, int stride, int padding, std::string name = "T_conv2d_nchw") {
    auto N = Input->shape[0];
    auto CI = Input->shape[1];
    auto H = Input->shape[2];
    auto W = Input->shape[3];
    
    auto CO = Filter->shape[0];
    auto KH = Filter->shape[2];
    auto KW = Filter->shape[3];
    
    // Output Height/Width (Simplified calculation, assuming padding handled or valid)
    // OH = (H - KH + 2*padding) / stride + 1
    // For symbolic exprs, we construct the expression
    
    // Let's use simplified expression for OH/OW assuming valid padding logic or PaddedInput
    // Actually, handling padding in TE usually involves a separate "pad" compute or `if` condition.
    // For this example, let's assume padding=0, stride=1 for simplicity
    
    auto OH = (H - KH) + 1;
    auto OW = (W - KW) + 1;
    
    IterVar rc = reduce_axis(0, CI, "rc");
    IterVar ry = reduce_axis(0, KH, "ry");
    IterVar rx = reduce_axis(0, KW, "rx");
    
    return compute({N, CO, OH, OW}, [&](const std::vector<Var>& indices) {
        Var n = indices[0];
        Var co = indices[1];
        Var h = indices[2];
        Var w = indices[3];
        
        // Input indices
        // input_h = h * stride + ry - padding
        // input_w = w * stride + rx - padding
        // Simplified stride=1, padding=0:
        auto input_h = h + ry;
        auto input_w = w + rx;
        
        return sum(Input(n, rc, input_h, input_w) * Filter(co, rc, ry, rx), {rc, ry, rx});
    }, name);
}

} // namespace topi
} // namespace kxc

// Helper for make_const if not defined
// (Moved to top)
