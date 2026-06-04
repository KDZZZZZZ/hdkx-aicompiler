/*! \file src/tir/transforms/remove_no_op.cc
 * \brief 实现 TIR 优化 pass 和 pipeline。
 */

#include "tir/transforms/remove_no_op.h"

#include "base/pass.h"
#include "tir/pass_utils.h"

namespace kxc {
namespace tir {

namespace {

class RemoveNoOpRewriter : public TIRPass {
protected:
    Stmt VisitEvaluate(const EvaluateNode* op, const Stmt& ref) override {
        Stmt rewritten = TIRPass::VisitEvaluate(op, ref);
        const auto* evaluate = rewritten.As<EvaluateNode>();
        if (!evaluate) {
            return rewritten;
        }
        if (pass_utils::IsConstZero(evaluate->value)) {
            return Stmt();
        }
        return rewritten;
    }

    Stmt VisitSeqStmt(const SeqStmtNode* op, const Stmt& ref) override {
        (void)ref;
        Array<Stmt> flattened;
        for (const auto& stmt : op->seq) {
            pass_utils::AppendFlattenedStmt(Mutate(stmt), &flattened);
        }
        return pass_utils::MakeSeqStmtOrUnit(flattened);
    }

    Stmt VisitFor(const ForNode* op, const Stmt& ref) override {
        Stmt rewritten = TIRPass::VisitFor(op, ref);
        const auto* for_node = rewritten.As<ForNode>();
        if (!for_node) {
            return rewritten;
        }
        if (pass_utils::IsNoOpStmt(for_node->body)) {
            return Stmt();
        }
        return rewritten;
    }
};

}  // namespace

PrimFunc RemoveNoOpPass(const PrimFunc& func) {
    RemoveNoOpRewriter pass;
    return pass.Mutate(func);
}

}  // namespace tir
}  // namespace kxc
