/*! \file include/kxc/te/topi/reduction.h
 * \brief 定义 TOPI 风格的 tensor compute helper。
 */

#pragma once
#include "kxc/te/te.h"
#include "kxc/te/topi/tags.h"
#include "kxc/te/topi/utils.h"
#include "kxc/support/container.h"
#include <vector>
#include <set>

namespace kxc {
namespace te {
namespace topi {

// Generic Reduction
// Combiner: function that takes (expr, axis_vars) -> expr (e.g. kxc::te::sum)
using FCombine = std::function<PrimExpr(PrimExpr, Array<IterVar>)>;

Tensor comm_reduce(const Tensor& data, const Array<int>& axis, bool keepdims, FCombine combiner, std::string name = "reduce", std::string tag = kCommReduce);

Tensor sum(const Tensor& data, const Array<int>& axis, bool keepdims = false, std::string name = "sum");

PrimExpr max_reducer(PrimExpr expr, Array<IterVar> axis);

Tensor max(const Tensor& data, const Array<int>& axis, bool keepdims = false, std::string name = "max");

Tensor min(const Tensor& data, const Array<int>& axis, bool keepdims = false, std::string name = "min");

} // namespace topi
} // namespace te
} // namespace kxc
