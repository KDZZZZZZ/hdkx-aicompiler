/*! \file src/te/topi/utils.cc
 * \brief Implements TOPI utils helpers.
 */

#include "kxc/te/topi/utils.h"

namespace kxc {
namespace te {
namespace topi {

std::vector<size_t> GetRealAxis(size_t ndim, const Array<int>& axis){
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

bool IsScalar(const Array<PrimExpr>& shape){
    return shape.empty();
}

bool GetConstInt(const PrimExpr& expr, int64_t* out_value){
    if (auto* ptr = expr.As<IntImmNode>()) {
        *out_value = ptr->value;
        return true;
    }
    return false;
}

PrimExpr make_const(DataType t, double value){
    if (t.code == 2) { // Float
        if (t.bits == 64) return FloatImm(value, t);
        return FloatImm((float)value, t);
    } else if (t.code == 0 || t.code == 1) { // Int or UInt
        return IntImm((int64_t)value, t);
    }
    // Default fallback
    return FloatImm(value, t);
}

PrimExpr operator>(PrimExpr a, PrimExpr b){ return kxc::tir::LT(b, a); }

PrimExpr operator>=(PrimExpr a, PrimExpr b){ return !kxc::tir::LT(a, b); }

PrimExpr operator<=(PrimExpr a, PrimExpr b){ return !kxc::tir::LT(b, a); }

}  // namespace topi
}  // namespace te
}  // namespace kxc
