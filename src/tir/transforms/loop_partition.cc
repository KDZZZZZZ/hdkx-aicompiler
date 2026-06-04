/*! \file src/tir/transforms/loop_partition.cc
 * \brief 实现 TIR 优化 pass 和 pipeline。
 */

#include "tir/transforms/loop_partition.h"

#include <cstdint>
#include <string>

#include "base/pass.h"
#include "tir/pass_utils.h"

namespace kxc {
namespace tir {

namespace {

PrimExpr CanonicalAdd(const PrimExpr& lhs, const PrimExpr& rhs, const DataType& dtype) {
    if (pass_utils::IsConstZero(lhs)) {
        return rhs;
    }
    if (pass_utils::IsConstZero(rhs)) {
        return lhs;
    }
    int64_t lhs_v = 0;
    int64_t rhs_v = 0;
    if (pass_utils::TryGetConstInt64(lhs, &lhs_v) && pass_utils::TryGetConstInt64(rhs, &rhs_v)) {
        return IntImm(lhs_v + rhs_v, dtype);
    }
    return lhs + rhs;
}

class LoopPartitionRewriter : public TIRPass {
protected:
    Stmt VisitFor(const ForNode* op, const Stmt& ref) override {
        Stmt rewritten = TIRPass::VisitFor(op, ref);
        const auto* for_node = rewritten.As<ForNode>();
        if (!for_node || for_node->for_type != ForType::Serial) {
            return rewritten;
        }
        if (pass_utils::ContainsFor(for_node->body)) {
            return rewritten;
        }

        int64_t extent = 0;
        if (!pass_utils::TryGetConstInt64(for_node->extent, &extent) || extent <= 4) {
            return rewritten;
        }

        const int64_t factor = 4;
        const int64_t main_iters = extent / factor;
        const int64_t tail_iters = extent % factor;
        const DataType loop_dtype = for_node->loop_var->dtype;

        Array<Stmt> seq;

        if (main_iters > 0) {
            Var outer_var(for_node->loop_var->name_hint + "_part_outer", loop_dtype);
            Var inner_var(for_node->loop_var->name_hint + "_part_inner", loop_dtype);
            PrimExpr scaled_outer = outer_var * IntImm(factor, loop_dtype);
            PrimExpr main_base = CanonicalAdd(for_node->min, scaled_outer, loop_dtype);
            PrimExpr bound_value = CanonicalAdd(main_base, inner_var, loop_dtype);
            Stmt inner_body = LetStmt(for_node->loop_var, bound_value, for_node->body);
            Stmt inner_loop = For(inner_var, IntImm(0, loop_dtype), IntImm(factor, loop_dtype),
                                  ForType::Serial, inner_body);
            seq.push_back(For(outer_var, IntImm(0, loop_dtype), IntImm(main_iters, loop_dtype),
                              ForType::Serial, inner_loop));
        }

        if (tail_iters > 0) {
            Var tail_var(for_node->loop_var->name_hint + "_part_tail", loop_dtype);
            PrimExpr tail_base =
                CanonicalAdd(for_node->min, IntImm(main_iters * factor, loop_dtype), loop_dtype);
            PrimExpr bound_value = CanonicalAdd(tail_base, tail_var, loop_dtype);
            Stmt tail_body = LetStmt(for_node->loop_var, bound_value, for_node->body);
            seq.push_back(For(tail_var, IntImm(0, loop_dtype), IntImm(tail_iters, loop_dtype),
                              ForType::Serial, tail_body));
        }

        return pass_utils::MakeSeqStmtOrUnit(seq);
    }
};

}  // namespace

PrimFunc LoopPartitionPass(const PrimFunc& func) {
    LoopPartitionRewriter pass;
    return pass.Mutate(func);
}

}  // namespace tir
}  // namespace kxc
