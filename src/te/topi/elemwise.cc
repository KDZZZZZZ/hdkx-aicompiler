/*! \file src/te/topi/elemwise.cc
 * \brief Implements TOPI elemwise helpers.
 */

#include "kxc/te/topi/elemwise.h"

#include "kxc/te/topi/broadcast.h"

#include <stdexcept>

namespace kxc {
namespace te {
namespace topi {

PrimExpr exp(PrimExpr x){
    return kxc::tir::Call(x.dtype(), "exp", {x});
}

PrimExpr log(PrimExpr x){
    return kxc::tir::Call(x.dtype(), "log", {x});
}

PrimExpr sqrt(PrimExpr x){
    return kxc::tir::Call(x.dtype(), "sqrt", {x});
}

PrimExpr floor(PrimExpr x){
    return kxc::tir::Call(x.dtype(), "floor", {x});
}

PrimExpr ceil(PrimExpr x){
    return kxc::tir::Call(x.dtype(), "ceil", {x});
}

PrimExpr sigmoid_expr(PrimExpr x){
    // 1.0f / (1.0f + exp(-x))
    // We need correct type for 1.0
    // Assuming float32 for simplicity or matching x
    DataType dtype = x.dtype();
    PrimExpr one;
    if (dtype.code == 2) { // Float
        if (dtype.bits == 64) one = FloatImm(1.0, dtype);
        else one = FloatImm(1.0f, dtype);
    } else {
        // Fallback for non-float? Usually cast to float.
        // Assuming input is float.
         one = FloatImm(1.0f, DataType::Float(32));
    }
    return one / (one + exp(0 - x)); // 0 - x for negation if unary - not defined
}

namespace {

template <typename FCompute>
Tensor UnaryElemwise(const Tensor& x, std::string name, std::string tag,
                     FCompute compute_expr) {
    return compute(
        x->shape,
        [&](const Array<tir::Var>& indices) {
            return compute_expr(x(indices));
        },
        name, tag);
}

}  // namespace

Tensor exp(const Tensor& x, std::string name, std::string tag) {
    return UnaryElemwise(x, std::move(name), std::move(tag),
                         [](PrimExpr value) { return exp(value); });
}

Tensor log(const Tensor& x, std::string name, std::string tag) {
    return UnaryElemwise(x, std::move(name), std::move(tag),
                         [](PrimExpr value) { return log(value); });
}

Tensor sqrt(const Tensor& x, std::string name, std::string tag) {
    return UnaryElemwise(x, std::move(name), std::move(tag),
                         [](PrimExpr value) { return sqrt(value); });
}

Tensor floor(const Tensor& x, std::string name, std::string tag) {
    return UnaryElemwise(x, std::move(name), std::move(tag),
                         [](PrimExpr value) { return floor(value); });
}

Tensor ceil(const Tensor& x, std::string name, std::string tag) {
    return UnaryElemwise(x, std::move(name), std::move(tag),
                         [](PrimExpr value) { return ceil(value); });
}

Tensor sigmoid(const Tensor& x, std::string name, std::string tag) {
    return UnaryElemwise(x, std::move(name), std::move(tag),
                         [](PrimExpr value) { return sigmoid_expr(value); });
}

Tensor identity(const Tensor& x, std::string name , std::string tag ){
    return compute(
        x->shape,
        [&](const Array<tir::Var>& indices) {
            return x(indices);
        },
        name,
        tag
    );
}

Tensor negative(const Tensor& x, std::string name , std::string tag ){
    return compute(
        x->shape,
        [&](const Array<tir::Var>& indices) {
            return 0 - x(indices); // Assuming 0-x works or use Sub(0, x)
        },
        name,
        tag
    );
}

Tensor clip(const Tensor& x, PrimExpr a_min, PrimExpr a_max, std::string name , std::string tag ){
    return compute(
        x->shape,
        [&](const Array<tir::Var>& indices) {
            return Max(Min(x(indices), a_max), a_min);
        },
        name,
        tag
    );
}

Tensor cast(const Tensor& x, DataType dtype, std::string name , std::string tag ){
    return compute(
        x->shape,
        [&](const Array<tir::Var>& indices) {
            return kxc::tir::Call(dtype, "cast", {x(indices)});
        },
        name,
        tag
    );
}

Tensor equal(const Tensor& A, const Tensor& B, std::string name, std::string tag) {
    if (!A.defined() || !B.defined()) {
        throw std::runtime_error("equal requires defined tensors");
    }
    if (A->dtype != B->dtype) {
        throw std::runtime_error("equal input dtypes must match");
    }
    const auto output_shape = detail::InferBroadcastShape(A->shape, B->shape);
    return compute(
        output_shape,
        [A, B, output_shape](const Array<tir::Var>& indices) {
            return A(detail::GetBroadcastIndices(indices, A->shape, output_shape)) ==
                   B(detail::GetBroadcastIndices(indices, B->shape, output_shape));
        },
        std::move(name), std::move(tag));
}

}  // namespace topi
}  // namespace te
}  // namespace kxc
