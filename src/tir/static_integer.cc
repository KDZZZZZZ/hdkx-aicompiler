#include "internal/static_integer.h"

#include <algorithm>
#include <limits>

namespace kxc::tir::internal {
namespace {
bool CheckedAddInt64(int64_t lhs, int64_t rhs, int64_t* result) {
    if ((rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs) ||
        (rhs < 0 && lhs < std::numeric_limits<int64_t>::min() - rhs)) {
        return false;
    }
    *result = lhs + rhs;
    return true;
}

bool CheckedSubInt64(int64_t lhs, int64_t rhs, int64_t* result) {
    if ((rhs > 0 && lhs < std::numeric_limits<int64_t>::min() + rhs) ||
        (rhs < 0 && lhs > std::numeric_limits<int64_t>::max() + rhs)) {
        return false;
    }
    *result = lhs - rhs;
    return true;
}

bool CheckedMulInt64(int64_t lhs, int64_t rhs, int64_t* result) {
    if (lhs == 0 || rhs == 0) {
        *result = 0;
        return true;
    }
    if ((lhs == -1 && rhs == std::numeric_limits<int64_t>::min()) ||
        (rhs == -1 && lhs == std::numeric_limits<int64_t>::min())) {
        return false;
    }
    if (lhs > 0) {
        if ((rhs > 0 && lhs > std::numeric_limits<int64_t>::max() / rhs) ||
            (rhs < 0 && rhs < std::numeric_limits<int64_t>::min() / lhs)) {
            return false;
        }
    } else if ((rhs > 0 && lhs < std::numeric_limits<int64_t>::min() / rhs) ||
               (rhs < 0 && lhs < std::numeric_limits<int64_t>::max() / rhs)) {
        return false;
    }
    *result = lhs * rhs;
    return true;
}

bool EvaluateStaticInt64Impl(const tir::PrimExpr& expression, int64_t* result, size_t depth) {
    if (depth > 256) return false;
    if (const auto* value = expression.As<tir::IntImmNode>()) {
        if (value->dtype.code != 0 && value->dtype.code != 1) return false;
        *result = value->value;
        return true;
    }
    const auto* binary = expression.As<tir::BinaryOpNode>();
    if (!binary) return false;
    int64_t lhs = 0;
    int64_t rhs = 0;
    if (!EvaluateStaticInt64Impl(binary->a, &lhs, depth + 1) ||
        !EvaluateStaticInt64Impl(binary->b, &rhs, depth + 1)) {
        return false;
    }
    if (expression.As<tir::AddNode>()) return CheckedAddInt64(lhs, rhs, result);
    if (expression.As<tir::SubNode>()) return CheckedSubInt64(lhs, rhs, result);
    if (expression.As<tir::MulNode>()) return CheckedMulInt64(lhs, rhs, result);
    if (expression.As<tir::DivNode>()) {
        if (rhs == 0 ||
            (lhs == std::numeric_limits<int64_t>::min() && rhs == -1)) {
            return false;
        }
        *result = lhs / rhs;
        return true;
    }
    if (expression.As<tir::ModNode>()) {
        if (rhs == 0 ||
            (lhs == std::numeric_limits<int64_t>::min() && rhs == -1)) {
            return false;
        }
        *result = lhs % rhs;
        return true;
    }
    if (expression.As<tir::MinNode>()) {
        *result = std::min(lhs, rhs);
        return true;
    }
    if (expression.As<tir::MaxNode>()) {
        *result = std::max(lhs, rhs);
        return true;
    }
    return false;
}

}  // namespace

bool EvaluateStaticInt64(const PrimExpr& expression, int64_t* result) {
    return EvaluateStaticInt64Impl(expression, result, 0);
}

}  // namespace kxc::tir::internal
