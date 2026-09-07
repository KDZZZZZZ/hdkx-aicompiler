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
    // 选定语义：1/(1+exp(-x))，直接调用既有 exp 调用分派（不用 tanh 近似）。
    // 字面量必须携带 x 的 dtype：裸整数 0/1 会把中间表达式的 dtype 定成
    // int32，在 LLVM codegen 的 math-call 分派处失败。仅接受浮点 dtype。
    const DataType dtype = x.dtype();
    if (dtype.code != 2) {
        throw std::runtime_error("sigmoid_expr requires a floating-point dtype");
    }
    const PrimExpr one = FloatImm(1.0, dtype);
    const PrimExpr zero = FloatImm(0.0, dtype);
    return one / (one + exp(zero - x));
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
    // 0 - x：零字面量必须携带 x 的 dtype；裸整数 0 会把 Sub 的 dtype 定成
    // int32 并破坏 lowering 的输出 dtype 合同。IEEE 语义：-0.0 输入得 +0.0，
    // +0.0 输入同样得 +0.0（0-x 的减法语义，非符号位取反）。
    const PrimExpr zero = x->dtype.code == 2
        ? PrimExpr(FloatImm(0.0, x->dtype))
        : PrimExpr(IntImm(0, x->dtype));
    return compute(
        x->shape,
        [&](const Array<tir::Var>& indices) {
            return zero - x(indices);
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
