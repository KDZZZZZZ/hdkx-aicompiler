/*! \file include/kxc/te/topi/broadcast.h
 * \brief 定义 TOPI 风格的 tensor compute helper。
 */

#pragma once
#include "kxc/te/te.h"
#include "kxc/te/topi/tags.h"
#include "kxc/te/topi/utils.h"
#include "kxc/support/container.h"
#include <algorithm>

namespace kxc {
namespace te {
namespace topi {

namespace detail {
    // Helper to map output indices to input indices for broadcasting
    Array<PrimExpr> GetBroadcastIndices(
        const Array<tir::Var>& output_indices,
        const Array<PrimExpr>& input_shape,
        const Array<PrimExpr>& output_shape);

    // Helper to infer broadcast shape
    Array<PrimExpr> InferBroadcastShape(
        const Array<PrimExpr>& shape1,
        const Array<PrimExpr>& shape2);
}

Tensor broadcast_to(const Tensor& t, const Array<PrimExpr>& output_shape, std::string name = "broadcast_to", std::string tag = kBroadcast);

Tensor add(const Tensor& A, const Tensor& B, std::string name = "add",
           std::string tag = kBroadcast);
Tensor subtract(const Tensor& A, const Tensor& B,
                std::string name = "subtract", std::string tag = kBroadcast);
Tensor multiply(const Tensor& A, const Tensor& B,
                std::string name = "multiply", std::string tag = kBroadcast);
Tensor divide(const Tensor& A, const Tensor& B, std::string name = "divide",
              std::string tag = kBroadcast);

Tensor maximum(const Tensor& A, const Tensor& B, std::string name = "maximum", std::string tag = kBroadcast);

Tensor minimum(const Tensor& A, const Tensor& B, std::string name = "minimum", std::string tag = kBroadcast);

} // namespace topi
} // namespace te
} // namespace kxc
