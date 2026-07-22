/*! \file include/kxc/te/topi/utils.h
 * \brief 定义 TOPI 风格的 tensor compute helper。
 */

#pragma once
#include "kxc/te/te.h"
#include "kxc/support/container.h"
#include <vector>
#include "kxc/tir/expr.h"
#include <algorithm>

namespace kxc {
namespace te {
namespace topi {

using namespace kxc::tir;

// Helper to get real axis indices from input
// Handles negative indices and conversion to size_t
std::vector<size_t> GetRealAxis(size_t ndim, const Array<int>& axis);

// Check if a shape is empty
bool IsScalar(const Array<PrimExpr>& shape);

// Helper to get constant integer from PrimExpr
// Returns true if success, value is stored in out_value
bool GetConstInt(const PrimExpr& expr, int64_t* out_value);

// Make Const Helper
PrimExpr make_const(DataType t, double value);

// Operator overloads for PrimExpr (missing in tir/expr.h)
PrimExpr operator>(PrimExpr a, PrimExpr b);
PrimExpr operator>=(PrimExpr a, PrimExpr b);
PrimExpr operator<=(PrimExpr a, PrimExpr b);

} // namespace topi
} // namespace te
} // namespace kxc
