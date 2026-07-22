/*! \file src/tir/transforms/unroll_loop.cc
 * \brief 实现 TIR 优化 pass 和 pipeline。
 */

#include "kxc/tir/transforms/unroll_loop.h"

#include <cstdint>

#include "kxc/tir/visitor.h"
#include "kxc/tir/pass_utils.h"

namespace kxc {
namespace tir {

namespace {

PrimExpr BuildUnrolledIterValue(const PrimExpr& min, int64_t step, const DataType& dtype) {
    if (step == 0) {
        return min;
    }
    int64_t min_value = 0;
    if (pass_utils::TryGetConstInt64(min, &min_value)) {
        return IntImm(min_value + step, dtype);
    }
    if (pass_utils::IsConstZero(min)) {
        return IntImm(step, dtype);
    }
    return min + IntImm(step, dtype);
}

class UnrollLoopRewriter : public TIRPass {
protected:
    Stmt VisitFor(const ForNode* op, const Stmt& ref) override {
        Stmt rewritten = TIRPass::VisitFor(op, ref);
        const auto* for_node = rewritten.As<ForNode>();
        if (!for_node || for_node->for_type != ForType::Serial) {
            return rewritten;
        }

        int64_t extent = 0;
        if (!pass_utils::TryGetConstInt64(for_node->extent, &extent) || extent < 0 || extent > 8) {
            return rewritten;
        }
        if (extent == 0) {
            return Stmt();
        }

        Array<Stmt> seq;
        const DataType loop_dtype = for_node->loop_var->dtype;
        for (int64_t i = 0; i < extent; ++i) {
            PrimExpr iter_value = BuildUnrolledIterValue(for_node->min, i, loop_dtype);
            seq.push_back(LetStmt(for_node->loop_var, iter_value, for_node->body));
        }
        return pass_utils::MakeSeqStmtOrUnit(seq);
    }
};

}  // namespace

PrimFunc UnrollLoopPass(const PrimFunc& func) {
    UnrollLoopRewriter pass;
    return pass.Mutate(func);
}

}  // namespace tir
}  // namespace kxc
