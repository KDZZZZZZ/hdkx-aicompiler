#pragma once
#include "te/te.h"
#include "te/topi/tags.h"
#include "te/topi/utils.h"
#include "base/container.h"
#include <algorithm>

namespace kxc {
namespace te {
namespace topi {

namespace detail {
    // Helper to map output indices to input indices for broadcasting
    inline Array<PrimExpr> GetBroadcastIndices(
        const Array<tir::Var>& output_indices,
        const Array<PrimExpr>& input_shape,
        const Array<PrimExpr>& output_shape) {
        
        Array<PrimExpr> input_indices;
        size_t in_ndim = input_shape.size();
        size_t out_ndim = output_shape.size();
        size_t offset = out_ndim - in_ndim;
        
        for (size_t i = 0; i < in_ndim; ++i) {
            // Align from the right
            PrimExpr dim = input_shape[i];
            // If input dim is 1, broadcast (index 0)
            // Otherwise use the corresponding output index
            // We can check if dim is constant 1.
            int64_t dim_val = 0;
            if (GetConstInt(dim, &dim_val) && dim_val == 1) {
                input_indices.push_back(0);
            } else {
                input_indices.push_back(output_indices[i + offset]);
            }
        }
        return input_indices;
    }

    // Helper to infer broadcast shape
    inline Array<PrimExpr> InferBroadcastShape(
        const Array<PrimExpr>& shape1,
        const Array<PrimExpr>& shape2) {
        
        size_t ndim1 = shape1.size();
        size_t ndim2 = shape2.size();
        size_t out_ndim = std::max(ndim1, ndim2);
        Array<PrimExpr> out_shape;
        
        size_t offset1 = out_ndim - ndim1;
        size_t offset2 = out_ndim - ndim2;
        
        for (size_t i = 0; i < out_ndim; ++i) {
            PrimExpr dim1 = (i >= offset1) ? shape1[i - offset1] : 1;
            PrimExpr dim2 = (i >= offset2) ? shape2[i - offset2] : 1;
            
            int64_t v1 = 0, v2 = 0;
            if (GetConstInt(dim1, &v1) && GetConstInt(dim2, &v2)) {
                if (v1 == 1) out_shape.push_back(dim2);
                else if (v2 == 1) out_shape.push_back(dim1);
                else if (v1 == v2) out_shape.push_back(dim1);
                else {
                     out_shape.push_back((v1 > v2) ? dim1 : dim2); 
                }
            } else {
                out_shape.push_back(dim1); 
            }
        }
        return out_shape;
    }
}

inline Tensor broadcast_to(const Tensor& t, const Array<PrimExpr>& output_shape, std::string name = "broadcast_to", std::string tag = kBroadcast) {
    return compute(
        output_shape,
        [&](const Array<tir::Var>& indices) {
            auto input_indices = detail::GetBroadcastIndices(indices, t->shape, output_shape);
            return t(input_indices);
        },
        name,
        tag
    );
}

// Binary Broadcast Ops
#define KXC_TOPI_BINARY_BROADCAST_OP(OpName, ComputeFunc) \
    inline Tensor OpName(const Tensor& A, const Tensor& B, std::string name = #OpName, std::string tag = kBroadcast) { \
        auto output_shape = detail::InferBroadcastShape(A->shape, B->shape); \
        return compute( \
            output_shape, \
            [&](const Array<tir::Var>& indices) { \
                auto a_indices = detail::GetBroadcastIndices(indices, A->shape, output_shape); \
                auto b_indices = detail::GetBroadcastIndices(indices, B->shape, output_shape); \
                return ComputeFunc(A(a_indices), B(b_indices)); \
            }, \
            name, \
            tag \
        ); \
    }

// Define specific ops
// Note: We need the underlying PrimExpr operators (Add, Sub, Mul, etc.) to be defined.
// Assuming te/te.h or tir/expr.h provides operator+ etc. for PrimExpr.

KXC_TOPI_BINARY_BROADCAST_OP(add, std::plus<PrimExpr>())
KXC_TOPI_BINARY_BROADCAST_OP(subtract, std::minus<PrimExpr>())
KXC_TOPI_BINARY_BROADCAST_OP(multiply, std::multiplies<PrimExpr>())
KXC_TOPI_BINARY_BROADCAST_OP(divide, std::divides<PrimExpr>())

inline Tensor maximum(const Tensor& A, const Tensor& B, std::string name = "maximum", std::string tag = kBroadcast) {
     auto output_shape = detail::InferBroadcastShape(A->shape, B->shape); 
        return compute( 
            output_shape, 
            [&](const Array<tir::Var>& indices) { 
                auto idx_a = detail::GetBroadcastIndices(indices, A->shape, output_shape); 
                auto idx_b = detail::GetBroadcastIndices(indices, B->shape, output_shape); 
                // Assuming Max exists in tir or as a function
                // kxc::tir::Max or similar. 
                // If not, we might need to implement it or use Select.
                // Assuming standard Max(PrimExpr, PrimExpr) exists.
                return Max(A(idx_a), B(idx_b)); 
            }, 
            name, 
            tag 
        ); 
}

inline Tensor minimum(const Tensor& A, const Tensor& B, std::string name = "minimum", std::string tag = kBroadcast) {
     auto output_shape = detail::InferBroadcastShape(A->shape, B->shape); 
        return compute( 
            output_shape, 
            [&](const Array<tir::Var>& indices) { 
                auto idx_a = detail::GetBroadcastIndices(indices, A->shape, output_shape); 
                auto idx_b = detail::GetBroadcastIndices(indices, B->shape, output_shape); 
                return Min(A(idx_a), B(idx_b)); 
            }, 
            name, 
            tag 
        ); 
}

} // namespace topi
} // namespace te
} // namespace kxc
