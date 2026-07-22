/*! \file src/relay/transforms/eliminate_dead_let.cc
 * \brief 实现 Relay 优化 pass 及其 pipeline 集成。
 */

#include "kxc/relay/transforms/eliminate_dead_let.h"

#include "kxc/relay/visitor.h"
#include "kxc/relay/pass_utils.h"

namespace kxc {
namespace relay {

namespace {

class EliminateDeadLetRewriter : public RelayPass {
protected:
    Expr VisitLet(const LetNode* op, const Expr& ref) override {
        Expr rewritten = RelayPass::VisitLet(op, ref);
        const auto* let_node = rewritten.As<LetNode>();
        if (!let_node) {
            return rewritten;
        }
        if (pass_utils::HasSideEffect(let_node->value)) {
            return rewritten;
        }
        if (pass_utils::CountVarUses(let_node->body, let_node->var) != 0) {
            return rewritten;
        }
        return pass_utils::CopyVirtualDevice(rewritten, let_node->body);
    }
};

}  // namespace

Function EliminateDeadLetPass(const Function& func) {
    EliminateDeadLetRewriter pass;
    return pass.Mutate(func);
}

}  // namespace relay
}  // namespace kxc
