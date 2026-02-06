#pragma once
#include "te/te.h"
#include "te/topi/tags.h"
#include "tir/expr.h"
#include "base/container.h"

namespace kxc {
namespace te {
namespace topi {

using namespace kxc::tir;

// Helper to create intrinsic calls
inline PrimExpr exp(PrimExpr x) {
    return Call(x.dtype(), "exp", {x});
}

inline PrimExpr log(PrimExpr x) {
    return Call(x.dtype(), "log", {x});
}

inline PrimExpr sqrt(PrimExpr x) {
    return Call(x.dtype(), "sqrt", {x});
}

inline PrimExpr floor(PrimExpr x) {
    return Call(x.dtype(), "floor", {x});
}

inline PrimExpr ceil(PrimExpr x) {
    return Call(x.dtype(), "ceil", {x});
}

// Sigmoid: 1 / (1 + exp(-x))
inline PrimExpr sigmoid_expr(PrimExpr x) {
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

// Unary Elemwise Ops
#define KXC_TOPI_UNARY_ELEMWISE_OP(OpName, ComputeExpr) \
    inline Tensor OpName(const Tensor& x, std::string name = #OpName, std::string tag = kElementWise) { \
        return compute( \
            x->shape, \
            [&](const Array<Var>& indices) { \
                return ComputeExpr(x(indices)); \
            }, \
            name, \
            tag \
        ); \
    }

KXC_TOPI_UNARY_ELEMWISE_OP(exp, exp)
KXC_TOPI_UNARY_ELEMWISE_OP(log, log)
KXC_TOPI_UNARY_ELEMWISE_OP(sqrt, sqrt)
KXC_TOPI_UNARY_ELEMWISE_OP(floor, floor)
KXC_TOPI_UNARY_ELEMWISE_OP(ceil, ceil)
KXC_TOPI_UNARY_ELEMWISE_OP(sigmoid, sigmoid_expr)

// Identity
inline Tensor identity(const Tensor& x, std::string name = "identity", std::string tag = kElementWise) {
    return compute(
        x->shape,
        [&](const Array<Var>& indices) {
            return x(indices);
        },
        name,
        tag
    );
}

// Negative
inline Tensor negative(const Tensor& x, std::string name = "negative", std::string tag = kElementWise) {
    return compute(
        x->shape,
        [&](const Array<Var>& indices) {
            return 0 - x(indices); // Assuming 0-x works or use Sub(0, x)
        },
        name,
        tag
    );
}

// Clip
inline Tensor clip(const Tensor& x, PrimExpr a_min, PrimExpr a_max, std::string name = "clip", std::string tag = kElementWise) {
    return compute(
        x->shape,
        [&](const Array<Var>& indices) {
            return Max(Min(x(indices), a_max), a_min);
        },
        name,
        tag
    );
}

// Cast
inline Tensor cast(const Tensor& x, DataType dtype, std::string name = "cast", std::string tag = kElementWise) {
    return compute(
        x->shape,
        [&](const Array<Var>& indices) {
            return Call(dtype, "cast", {x(indices)});
        },
        name,
        tag
    );
}

} // namespace topi
} // namespace te
} // namespace kxc
