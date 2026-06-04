/*! \file include/te/topi/utils.h
 * \brief 定义 TOPI 风格的 tensor compute helper。
 */

#pragma once
#include "te/te.h"
#include "base/container.h"
#include <vector>
#include "tir/expr.h"
#include <algorithm>

namespace kxc {
namespace te {
namespace topi {

using namespace kxc::tir;

// Helper to get real axis indices from input
// Handles negative indices and conversion to size_t
inline std::vector<size_t> GetRealAxis(size_t ndim, const Array<int>& axis) {
    std::vector<size_t> real_axis;
    if (axis.empty()) {
        for (size_t i = 0; i < ndim; ++i) real_axis.push_back(i);
    } else {
        for (int a : axis) {
            if (a < 0) a += (int)ndim;
            if (a >= 0 && a < (int)ndim) {
                real_axis.push_back((size_t)a);
            }
        }
        std::sort(real_axis.begin(), real_axis.end());
    }
    return real_axis;
}

// Check if a shape is empty
inline bool IsScalar(const Array<PrimExpr>& shape) {
    return shape.empty();
}

// Helper to get constant integer from PrimExpr
// Returns true if success, value is stored in out_value
inline bool GetConstInt(const PrimExpr& expr, int64_t* out_value) {
    if (auto* ptr = expr.As<IntImmNode>()) {
        *out_value = ptr->value;
        return true;
    }
    return false;
}

// Make Const Helper
inline PrimExpr make_const(DataType t, double value) {
    if (t.code == 2) { // Float
        if (t.bits == 64) return FloatImm(value, t);
        return FloatImm((float)value, t);
    } else if (t.code == 0 || t.code == 1) { // Int or UInt
        return IntImm((int64_t)value, t);
    }
    // Default fallback
    return FloatImm(value, t);
}

// Operator overloads for PrimExpr (missing in tir/expr.h)
inline PrimExpr operator>(PrimExpr a, PrimExpr b) { return kxc::tir::LT(b, a); }
inline PrimExpr operator>=(PrimExpr a, PrimExpr b) { return !kxc::tir::LT(a, b); }
inline PrimExpr operator<=(PrimExpr a, PrimExpr b) { return !kxc::tir::LT(b, a); }

} // namespace topi
} // namespace te
} // namespace kxc
