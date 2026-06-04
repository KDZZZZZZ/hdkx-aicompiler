/*! \file src/tir/transforms/fold_constant.cc
 * \brief 实现 TIR 优化 pass 和 pipeline。
 */

#include "tir/transforms/fold_constant.h"

#include <cmath>
#include <cstdint>
#include <limits>

#include "base/pass.h"

namespace kxc {
namespace tir {

namespace {

bool IsIntDType(const DataType& dtype) { return dtype.code == 0 || dtype.code == 1; }

bool FitsInDType(int64_t value, const DataType& dtype) {
    if (dtype.lanes != 1) {
        return false;
    }
    if (dtype.code == 0) {
        if (dtype.bits == 0) {
            return false;
        }
        if (dtype.bits >= 64) {
            return true;
        }
        const int64_t min_value = -(static_cast<int64_t>(1) << (dtype.bits - 1));
        const int64_t max_value = (static_cast<int64_t>(1) << (dtype.bits - 1)) - 1;
        return value >= min_value && value <= max_value;
    }
    if (dtype.code == 1) {
        if (dtype.bits == 0) {
            return false;
        }
        if (value < 0) {
            return false;
        }
        if (dtype.bits >= 63) {
            return true;
        }
        const uint64_t max_value = (static_cast<uint64_t>(1) << dtype.bits) - 1;
        return static_cast<uint64_t>(value) <= max_value;
    }
    return false;
}

bool TryGetIntImm(const PrimExpr& expr, int64_t* out) {
    if (!out) {
        return false;
    }
    const auto* imm = expr.As<IntImmNode>();
    if (!imm) {
        return false;
    }
    *out = imm->value;
    return true;
}

bool TryGetFloatImm(const PrimExpr& expr, double* out) {
    if (!out) {
        return false;
    }
    const auto* imm = expr.As<FloatImmNode>();
    if (!imm) {
        return false;
    }
    *out = imm->value;
    return true;
}

bool TryGetNumericConst(const PrimExpr& expr, double* out) {
    if (!out) {
        return false;
    }
    if (const auto* int_imm = expr.As<IntImmNode>()) {
        *out = static_cast<double>(int_imm->value);
        return true;
    }
    if (const auto* float_imm = expr.As<FloatImmNode>()) {
        *out = float_imm->value;
        return true;
    }
    return false;
}

bool TryMakeIntImm(int64_t value, const DataType& dtype, PrimExpr* out) {
    if (!out || !IsIntDType(dtype)) {
        return false;
    }
    if (!FitsInDType(value, dtype)) {
        return false;
    }
    *out = IntImm(value, dtype);
    return true;
}

bool TryMakeFloatImm(double value, const DataType& dtype, PrimExpr* out) {
    if (!out || dtype.code != 2 || dtype.lanes != 1) {
        return false;
    }
    if (!std::isfinite(value)) {
        return false;
    }
    *out = FloatImm(value, dtype);
    return true;
}

bool TryFoldIntAdd(int64_t lhs, int64_t rhs, int64_t* out) {
    const long double result =
        static_cast<long double>(lhs) + static_cast<long double>(rhs);
    if (result < static_cast<long double>(std::numeric_limits<int64_t>::min()) ||
        result > static_cast<long double>(std::numeric_limits<int64_t>::max())) {
        return false;
    }
    *out = static_cast<int64_t>(result);
    return true;
}

bool TryFoldIntSub(int64_t lhs, int64_t rhs, int64_t* out) {
    const long double result =
        static_cast<long double>(lhs) - static_cast<long double>(rhs);
    if (result < static_cast<long double>(std::numeric_limits<int64_t>::min()) ||
        result > static_cast<long double>(std::numeric_limits<int64_t>::max())) {
        return false;
    }
    *out = static_cast<int64_t>(result);
    return true;
}

bool TryFoldIntMul(int64_t lhs, int64_t rhs, int64_t* out) {
    const long double result =
        static_cast<long double>(lhs) * static_cast<long double>(rhs);
    if (result < static_cast<long double>(std::numeric_limits<int64_t>::min()) ||
        result > static_cast<long double>(std::numeric_limits<int64_t>::max())) {
        return false;
    }
    *out = static_cast<int64_t>(result);
    return true;
}

bool TryFoldIntDiv(int64_t lhs, int64_t rhs, int64_t* out) {
    if (rhs == 0) {
        return false;
    }
    if (lhs == std::numeric_limits<int64_t>::min() && rhs == -1) {
        return false;
    }
    *out = lhs / rhs;
    return true;
}

bool TryFoldIntMod(int64_t lhs, int64_t rhs, int64_t* out) {
    if (rhs == 0) {
        return false;
    }
    if (lhs == std::numeric_limits<int64_t>::min() && rhs == -1) {
        return false;
    }
    *out = lhs % rhs;
    return true;
}

class FoldConstantRewriter : public TIRPass {
protected:
    PrimExpr VisitAdd(const AddNode* op, const PrimExpr& ref) override {
        PrimExpr rewritten = TIRPass::VisitAdd(op, ref);
        const auto* add = rewritten.As<AddNode>();
        if (!add) {
            return rewritten;
        }
        PrimExpr folded;
        if (TryFoldArithmetic(add->a, add->b, rewritten.dtype(), &TryFoldIntAdd, '+', &folded)) {
            return folded;
        }
        return rewritten;
    }

    PrimExpr VisitSub(const SubNode* op, const PrimExpr& ref) override {
        PrimExpr rewritten = TIRPass::VisitSub(op, ref);
        const auto* sub = rewritten.As<SubNode>();
        if (!sub) {
            return rewritten;
        }
        PrimExpr folded;
        if (TryFoldArithmetic(sub->a, sub->b, rewritten.dtype(), &TryFoldIntSub, '-', &folded)) {
            return folded;
        }
        return rewritten;
    }

    PrimExpr VisitMul(const MulNode* op, const PrimExpr& ref) override {
        PrimExpr rewritten = TIRPass::VisitMul(op, ref);
        const auto* mul = rewritten.As<MulNode>();
        if (!mul) {
            return rewritten;
        }
        PrimExpr folded;
        if (TryFoldArithmetic(mul->a, mul->b, rewritten.dtype(), &TryFoldIntMul, '*', &folded)) {
            return folded;
        }
        return rewritten;
    }

    PrimExpr VisitDiv(const DivNode* op, const PrimExpr& ref) override {
        PrimExpr rewritten = TIRPass::VisitDiv(op, ref);
        const auto* div = rewritten.As<DivNode>();
        if (!div) {
            return rewritten;
        }
        PrimExpr folded;
        if (TryFoldArithmetic(div->a, div->b, rewritten.dtype(), &TryFoldIntDiv, '/', &folded)) {
            return folded;
        }
        return rewritten;
    }

    PrimExpr VisitMod(const ModNode* op, const PrimExpr& ref) override {
        PrimExpr rewritten = TIRPass::VisitMod(op, ref);
        const auto* mod = rewritten.As<ModNode>();
        if (!mod) {
            return rewritten;
        }
        PrimExpr folded;
        if (TryFoldArithmetic(mod->a, mod->b, rewritten.dtype(), &TryFoldIntMod, '%', &folded)) {
            return folded;
        }
        return rewritten;
    }

    PrimExpr VisitMin(const MinNode* op, const PrimExpr& ref) override {
        PrimExpr rewritten = TIRPass::VisitMin(op, ref);
        const auto* min_node = rewritten.As<MinNode>();
        if (!min_node) {
            return rewritten;
        }
        PrimExpr folded;
        if (TryFoldMinMax(min_node->a, min_node->b, rewritten.dtype(), true, &folded)) {
            return folded;
        }
        return rewritten;
    }

    PrimExpr VisitMax(const MaxNode* op, const PrimExpr& ref) override {
        PrimExpr rewritten = TIRPass::VisitMax(op, ref);
        const auto* max_node = rewritten.As<MaxNode>();
        if (!max_node) {
            return rewritten;
        }
        PrimExpr folded;
        if (TryFoldMinMax(max_node->a, max_node->b, rewritten.dtype(), false, &folded)) {
            return folded;
        }
        return rewritten;
    }

    PrimExpr VisitEQ(const EQNode* op, const PrimExpr& ref) override {
        PrimExpr rewritten = TIRPass::VisitEQ(op, ref);
        const auto* eq = rewritten.As<EQNode>();
        if (!eq) {
            return rewritten;
        }
        PrimExpr folded;
        if (TryFoldCompare(eq->a, eq->b, /*is_lt=*/false, &folded)) {
            return folded;
        }
        return rewritten;
    }

    PrimExpr VisitLT(const LTNode* op, const PrimExpr& ref) override {
        PrimExpr rewritten = TIRPass::VisitLT(op, ref);
        const auto* lt = rewritten.As<LTNode>();
        if (!lt) {
            return rewritten;
        }
        PrimExpr folded;
        if (TryFoldCompare(lt->a, lt->b, /*is_lt=*/true, &folded)) {
            return folded;
        }
        return rewritten;
    }

    PrimExpr VisitAnd(const AndNode* op, const PrimExpr& ref) override {
        PrimExpr rewritten = TIRPass::VisitAnd(op, ref);
        const auto* and_node = rewritten.As<AndNode>();
        if (!and_node) {
            return rewritten;
        }
        int64_t lhs = 0;
        int64_t rhs = 0;
        if (!TryGetIntImm(and_node->a, &lhs) || !TryGetIntImm(and_node->b, &rhs)) {
            return rewritten;
        }
        return IntImm((lhs != 0 && rhs != 0) ? 1 : 0, DataType::Bool());
    }

    PrimExpr VisitOr(const OrNode* op, const PrimExpr& ref) override {
        PrimExpr rewritten = TIRPass::VisitOr(op, ref);
        const auto* or_node = rewritten.As<OrNode>();
        if (!or_node) {
            return rewritten;
        }
        int64_t lhs = 0;
        int64_t rhs = 0;
        if (!TryGetIntImm(or_node->a, &lhs) || !TryGetIntImm(or_node->b, &rhs)) {
            return rewritten;
        }
        return IntImm((lhs != 0 || rhs != 0) ? 1 : 0, DataType::Bool());
    }

    PrimExpr VisitNot(const NotNode* op, const PrimExpr& ref) override {
        PrimExpr rewritten = TIRPass::VisitNot(op, ref);
        const auto* not_node = rewritten.As<NotNode>();
        if (!not_node) {
            return rewritten;
        }
        int64_t value = 0;
        if (!TryGetIntImm(not_node->value, &value)) {
            return rewritten;
        }
        return IntImm(value == 0 ? 1 : 0, DataType::Bool());
    }

    PrimExpr VisitSelect(const SelectNode* op, const PrimExpr& ref) override {
        PrimExpr rewritten = TIRPass::VisitSelect(op, ref);
        const auto* select = rewritten.As<SelectNode>();
        if (!select) {
            return rewritten;
        }
        int64_t cond = 0;
        if (!TryGetIntImm(select->condition, &cond)) {
            return rewritten;
        }
        return cond != 0 ? select->true_value : select->false_value;
    }

private:
    using IntBinaryFoldFn = bool (*)(int64_t lhs, int64_t rhs, int64_t* out);

    bool TryFoldArithmetic(const PrimExpr& lhs, const PrimExpr& rhs, const DataType& out_dtype,
                           IntBinaryFoldFn int_fold, char float_op, PrimExpr* out) const {
        if (!out) {
            return false;
        }
        if (IsIntDType(out_dtype)) {
            int64_t lhs_value = 0;
            int64_t rhs_value = 0;
            int64_t result = 0;
            if (!TryGetIntImm(lhs, &lhs_value) || !TryGetIntImm(rhs, &rhs_value)) {
                return false;
            }
            if (!int_fold(lhs_value, rhs_value, &result)) {
                return false;
            }
            return TryMakeIntImm(result, out_dtype, out);
        }
        if (out_dtype.code == 2) {
            double lhs_value = 0.0;
            double rhs_value = 0.0;
            if (!TryGetFloatImm(lhs, &lhs_value) || !TryGetFloatImm(rhs, &rhs_value)) {
                return false;
            }
            double result = 0.0;
            if (float_op == '+') {
                result = lhs_value + rhs_value;
            } else if (float_op == '-') {
                result = lhs_value - rhs_value;
            } else if (float_op == '*') {
                result = lhs_value * rhs_value;
            } else if (float_op == '/') {
                if (rhs_value == 0.0) {
                    return false;
                }
                result = lhs_value / rhs_value;
            } else if (float_op == '%') {
                if (rhs_value == 0.0) {
                    return false;
                }
                result = std::fmod(lhs_value, rhs_value);
            } else {
                return false;
            }
            return TryMakeFloatImm(result, out_dtype, out);
        }
        return false;
    }

    bool TryFoldMinMax(const PrimExpr& lhs, const PrimExpr& rhs, const DataType& out_dtype,
                       bool is_min, PrimExpr* out) const {
        if (!out) {
            return false;
        }
        if (IsIntDType(out_dtype)) {
            int64_t lhs_value = 0;
            int64_t rhs_value = 0;
            if (!TryGetIntImm(lhs, &lhs_value) || !TryGetIntImm(rhs, &rhs_value)) {
                return false;
            }
            int64_t result = is_min ? (lhs_value < rhs_value ? lhs_value : rhs_value)
                                    : (lhs_value > rhs_value ? lhs_value : rhs_value);
            return TryMakeIntImm(result, out_dtype, out);
        }
        if (out_dtype.code == 2) {
            double lhs_value = 0.0;
            double rhs_value = 0.0;
            if (!TryGetFloatImm(lhs, &lhs_value) || !TryGetFloatImm(rhs, &rhs_value)) {
                return false;
            }
            const double result = is_min ? (lhs_value < rhs_value ? lhs_value : rhs_value)
                                         : (lhs_value > rhs_value ? lhs_value : rhs_value);
            return TryMakeFloatImm(result, out_dtype, out);
        }
        return false;
    }

    bool TryFoldCompare(const PrimExpr& lhs, const PrimExpr& rhs, bool is_lt, PrimExpr* out) const {
        if (!out) {
            return false;
        }
        int64_t lhs_i = 0;
        int64_t rhs_i = 0;
        if (TryGetIntImm(lhs, &lhs_i) && TryGetIntImm(rhs, &rhs_i)) {
            const bool result = is_lt ? (lhs_i < rhs_i) : (lhs_i == rhs_i);
            *out = IntImm(result ? 1 : 0, DataType::Bool());
            return true;
        }
        double lhs_f = 0.0;
        double rhs_f = 0.0;
        if (TryGetNumericConst(lhs, &lhs_f) && TryGetNumericConst(rhs, &rhs_f)) {
            const bool result = is_lt ? (lhs_f < rhs_f) : (lhs_f == rhs_f);
            *out = IntImm(result ? 1 : 0, DataType::Bool());
            return true;
        }
        return false;
    }
};

}  // namespace

PrimFunc FoldConstantPass(const PrimFunc& func) {
    FoldConstantRewriter pass;
    return pass.Mutate(func);
}

}  // namespace tir
}  // namespace kxc
