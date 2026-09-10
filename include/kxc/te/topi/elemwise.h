/*! \file include/kxc/te/topi/elemwise.h
 * \brief 定义 TOPI 风格的 tensor compute helper。
 */

#pragma once
#include "kxc/te/te.h"
#include "kxc/te/topi/tags.h"
#include "kxc/tir/expr.h"
#include "kxc/support/container.h"

namespace kxc {
namespace te {
namespace topi {

using namespace kxc::tir;

// Helper to create intrinsic calls
PrimExpr exp(PrimExpr x);

PrimExpr log(PrimExpr x);

PrimExpr sqrt(PrimExpr x);

PrimExpr floor(PrimExpr x);

PrimExpr ceil(PrimExpr x);

// Sigmoid: 1 / (1 + exp(-x))
PrimExpr sigmoid_expr(PrimExpr x);

Tensor exp(const Tensor& x, std::string name = "exp",
           std::string tag = kElementWise);
Tensor log(const Tensor& x, std::string name = "log",
           std::string tag = kElementWise);
Tensor sqrt(const Tensor& x, std::string name = "sqrt",
            std::string tag = kElementWise);
Tensor floor(const Tensor& x, std::string name = "floor",
             std::string tag = kElementWise);
Tensor ceil(const Tensor& x, std::string name = "ceil",
            std::string tag = kElementWise);
Tensor sigmoid(const Tensor& x, std::string name = "sigmoid",
               std::string tag = kElementWise);

// Identity
Tensor identity(const Tensor& x, std::string name = "identity", std::string tag = kElementWise);

// Negative
Tensor negative(const Tensor& x, std::string name = "negative", std::string tag = kElementWise);

// Clip
Tensor clip(const Tensor& x, PrimExpr a_min, PrimExpr a_max, std::string name = "clip", std::string tag = kElementWise);

// Cast
Tensor cast(const Tensor& x, DataType dtype, std::string name = "cast", std::string tag = kElementWise);

// Broadcast numeric equality; produces a Bool tensor via the TIR EQ expression.
Tensor equal(const Tensor& A, const Tensor& B, std::string name = "equal",
             std::string tag = kBroadcast);

} // namespace topi
} // namespace te
} // namespace kxc
