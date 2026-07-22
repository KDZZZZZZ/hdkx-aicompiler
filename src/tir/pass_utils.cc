/*! \file src/tir/pass_utils.cc
 * \brief 实现 TIR 节点构造、工具函数和 pass 基础能力。
 */

#include "kxc/tir/pass_utils.h"

#include <cstdint>
#include <limits>

namespace kxc {
namespace tir {
namespace pass_utils {

bool TryGetConstValue(const PrimExpr& expr, double* out_value) {
    if (!out_value || !expr.defined()) {
        return false;
    }
    if (const auto* int_imm = expr.As<IntImmNode>()) {
        *out_value = static_cast<double>(int_imm->value);
        return true;
    }
    if (const auto* float_imm = expr.As<FloatImmNode>()) {
        *out_value = float_imm->value;
        return true;
    }
    return false;
}

bool TryGetConstInt64(const PrimExpr& expr, int64_t* out_value) {
    if (!out_value || !expr.defined()) {
        return false;
    }
    if (const auto* int_imm = expr.As<IntImmNode>()) {
        *out_value = int_imm->value;
        return true;
    }
    return false;
}

bool IsConstZero(const PrimExpr& expr) {
    double value = 0.0;
    return TryGetConstValue(expr, &value) && value == 0.0;
}

bool IsConstOne(const PrimExpr& expr) {
    double value = 0.0;
    return TryGetConstValue(expr, &value) && value == 1.0;
}

bool IsNoOpStmt(const Stmt& stmt) {
    if (!stmt.defined()) {
        return true;
    }
    if (const auto* evaluate = stmt.As<EvaluateNode>()) {
        return IsConstZero(evaluate->value);
    }
    if (const auto* seq = stmt.As<SeqStmtNode>()) {
        for (const auto& child : seq->seq) {
            if (!IsNoOpStmt(child)) {
                return false;
            }
        }
        return true;
    }
    return false;
}

bool ContainsFor(const Stmt& stmt) {
    if (!stmt.defined()) {
        return false;
    }
    if (stmt.As<ForNode>()) {
        return true;
    }
    if (const auto* binding = stmt.As<ThreadBindingNode>()) {
        return ContainsFor(binding->body);
    }
    if (const auto* let_stmt = stmt.As<LetStmtNode>()) {
        return ContainsFor(let_stmt->body);
    }
    if (const auto* if_stmt = stmt.As<IfThenElseNode>()) {
        return ContainsFor(if_stmt->then_case) || ContainsFor(if_stmt->else_case);
    }
    if (const auto* alloc = stmt.As<AllocateNode>()) {
        return ContainsFor(alloc->body);
    }
    if (const auto* attr = stmt.As<AttrStmtNode>()) {
        return ContainsFor(attr->body);
    }
    if (const auto* block = stmt.As<BlockNode>()) {
        return ContainsFor(block->init) || ContainsFor(block->body);
    }
    if (const auto* seq = stmt.As<SeqStmtNode>()) {
        for (const auto& child : seq->seq) {
            if (ContainsFor(child)) {
                return true;
            }
        }
    }
    return false;
}

bool IsSimpleStraightLineStmt(const Stmt& stmt) {
    if (!stmt.defined()) {
        return true;
    }
    if (stmt.As<EvaluateNode>() || stmt.As<StoreNode>()) {
        return true;
    }
    if (const auto* binding = stmt.As<ThreadBindingNode>()) {
        return IsSimpleStraightLineStmt(binding->body);
    }
    if (const auto* let_stmt = stmt.As<LetStmtNode>()) {
        return IsSimpleStraightLineStmt(let_stmt->body);
    }
    if (const auto* seq = stmt.As<SeqStmtNode>()) {
        for (const auto& child : seq->seq) {
            if (!IsSimpleStraightLineStmt(child)) {
                return false;
            }
        }
        return true;
    }
    return false;
}

void AppendFlattenedStmt(const Stmt& stmt, Array<Stmt>* out) {
    if (!out || !stmt.defined()) {
        return;
    }
    if (const auto* seq = stmt.As<SeqStmtNode>()) {
        for (const auto& child : seq->seq) {
            AppendFlattenedStmt(child, out);
        }
        return;
    }
    if (IsNoOpStmt(stmt)) {
        return;
    }
    out->push_back(stmt);
}

Stmt MakeSeqStmtOrUnit(const Array<Stmt>& seq) {
    Array<Stmt> flattened;
    for (const auto& stmt : seq) {
        AppendFlattenedStmt(stmt, &flattened);
    }
    if (flattened.empty()) {
        return Stmt();
    }
    if (flattened.size() == 1) {
        return flattened[0];
    }
    return SeqStmt(flattened);
}

bool CanNarrowInt64ToInt32(int64_t value) {
    return value >= static_cast<int64_t>(std::numeric_limits<int32_t>::min()) &&
           value <= static_cast<int64_t>(std::numeric_limits<int32_t>::max());
}

PrimExpr NarrowIntImmToInt32IfPossible(const PrimExpr& expr) {
    if (!expr.defined()) {
        return expr;
    }
    const auto* int_imm = expr.As<IntImmNode>();
    if (!int_imm) {
        return expr;
    }
    if (int_imm->dtype.code != 0 || int_imm->dtype.bits != 64 || int_imm->dtype.lanes != 1) {
        return expr;
    }
    if (!CanNarrowInt64ToInt32(int_imm->value)) {
        return expr;
    }
    return IntImm(int_imm->value, DataType::Int(32));
}

}  // namespace pass_utils
}  // namespace tir
}  // namespace kxc
