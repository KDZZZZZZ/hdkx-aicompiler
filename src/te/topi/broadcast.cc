/*! \file src/te/topi/broadcast.cc
 * \brief Implements TOPI broadcast helpers.
 */

#include "kxc/te/topi/broadcast.h"

#include <stdexcept>

namespace kxc {
namespace te {
namespace topi {

namespace detail {

Array<PrimExpr> GetBroadcastIndices(
        const Array<tir::Var>& output_indices,
        const Array<PrimExpr>& input_shape,
        const Array<PrimExpr>& output_shape){
        
        Array<PrimExpr> input_indices;
        size_t in_ndim = input_shape.size();
        size_t out_ndim = output_shape.size();
        if (in_ndim > out_ndim) {
            throw std::runtime_error("broadcast input rank exceeds output rank");
        }
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

Array<PrimExpr> InferBroadcastShape(
        const Array<PrimExpr>& shape1,
        const Array<PrimExpr>& shape2){
        
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
                    throw std::runtime_error("incompatible static broadcast dimensions");
                }
            } else if (GetConstInt(dim1, &v1) && v1 == 1) {
                out_shape.push_back(dim2);
            } else if (GetConstInt(dim2, &v2) && v2 == 1) {
                out_shape.push_back(dim1);
            } else {
                // Two dynamic dimensions remain symbolic; no known dimension is invented.
                out_shape.push_back(dim1);
            }
        }
        return out_shape;
}

}  // namespace detail

namespace {

template <typename FCompute>
Tensor BinaryBroadcast(const Tensor& a, const Tensor& b, std::string name,
                       std::string tag, FCompute compute_func) {
    auto output_shape = detail::InferBroadcastShape(a->shape, b->shape);
    return compute(
        output_shape,
        [&](const Array<tir::Var>& indices) {
            auto a_indices =
                detail::GetBroadcastIndices(indices, a->shape, output_shape);
            auto b_indices =
                detail::GetBroadcastIndices(indices, b->shape, output_shape);
            return compute_func(a(a_indices), b(b_indices));
        },
        name, tag);
}

}  // namespace

Tensor broadcast_to(const Tensor& t, const Array<PrimExpr>& output_shape, std::string name , std::string tag ){
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

Tensor add(const Tensor& A, const Tensor& B, std::string name, std::string tag) {
    return BinaryBroadcast(A, B, std::move(name), std::move(tag),
                           std::plus<PrimExpr>());
}

Tensor subtract(const Tensor& A, const Tensor& B, std::string name,
                std::string tag) {
    return BinaryBroadcast(A, B, std::move(name), std::move(tag),
                           std::minus<PrimExpr>());
}

Tensor multiply(const Tensor& A, const Tensor& B, std::string name,
                std::string tag) {
    return BinaryBroadcast(A, B, std::move(name), std::move(tag),
                           std::multiplies<PrimExpr>());
}

Tensor divide(const Tensor& A, const Tensor& B, std::string name,
              std::string tag) {
    return BinaryBroadcast(A, B, std::move(name), std::move(tag),
                           std::divides<PrimExpr>());
}

Tensor power(const Tensor& A, const Tensor& B, std::string name,
             std::string tag) {
    if (!A.defined() || !B.defined()) {
        throw std::runtime_error("power requires defined tensors");
    }
    if (A->dtype != B->dtype) {
        throw std::runtime_error("power input dtypes must match");
    }
    if (A->dtype.code != 2) {
        throw std::runtime_error("power requires floating-point inputs");
    }
    // 直接发射 "pow" TIR 调用：LLVM 分派把它绑定到 llvm.pow 声明（libm powf），
    // 禁止用 exp(b·log a) 覆盖一般 Pow（负底数/精确性会丢失）。
    return BinaryBroadcast(
        A, B, std::move(name), std::move(tag),
        [](const PrimExpr& base, const PrimExpr& exponent) {
            return tir::Call(base.dtype(), "pow", {base, exponent});
        });
}

Tensor where(const Tensor& condition, const Tensor& x, const Tensor& y,
             std::string name, std::string tag) {
    if (!condition.defined() || !x.defined() || !y.defined()) {
        throw std::runtime_error("where requires defined tensors");
    }
    if (condition->dtype != tir::DataType::Bool()) {
        throw std::runtime_error("where condition dtype must be bool");
    }
    if (x->dtype != y->dtype) {
        throw std::runtime_error("where x and y dtype mismatch");
    }
    const auto output_shape = detail::InferBroadcastShape(
        detail::InferBroadcastShape(condition->shape, x->shape), y->shape);
    return compute(
        output_shape,
        [condition, x, y, output_shape](const Array<tir::Var>& indices) {
            return tir::Select(
                condition(detail::GetBroadcastIndices(indices, condition->shape, output_shape)),
                x(detail::GetBroadcastIndices(indices, x->shape, output_shape)),
                y(detail::GetBroadcastIndices(indices, y->shape, output_shape)));
        },
        std::move(name), std::move(tag));
}

Tensor maximum(const Tensor& A, const Tensor& B, std::string name , std::string tag ){
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

Tensor minimum(const Tensor& A, const Tensor& B, std::string name , std::string tag ){
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

}  // namespace topi
}  // namespace te
}  // namespace kxc
