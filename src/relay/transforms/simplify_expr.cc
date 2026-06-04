/*! \file src/relay/transforms/simplify_expr.cc
 * \brief 实现 Relay 优化 pass 及其 pipeline 集成。
 */

#include "relay/transforms/simplify_expr.h"

#include "base/pass.h"
#include "relay/pass_utils.h"

namespace kxc {
namespace relay {

namespace {

class SimplifyExprRewriter : public RelayPass {
protected:
    Expr VisitCall(const CallNode* op, const Expr& ref) override {
        Expr rewritten = RelayPass::VisitCall(op, ref);
        const auto* call = rewritten.As<CallNode>();
        if (!call || call->args.size() != 2) {
            return rewritten;
        }

        const std::string op_name = pass_utils::GetCallOpName(call);
        const Expr& lhs = call->args[0];
        const Expr& rhs = call->args[1];
        Expr simplified;

        if (op_name == "add") {
            if (pass_utils::IsConstZero(rhs)) simplified = lhs;
            if (!simplified.defined() && pass_utils::IsConstZero(lhs)) simplified = rhs;
        } else if (op_name == "mul" || op_name == "multiply") {
            if (pass_utils::IsConstOne(rhs)) simplified = lhs;
            if (!simplified.defined() && pass_utils::IsConstOne(lhs)) simplified = rhs;
        } else if (op_name == "subtract" || op_name == "sub") {
            if (pass_utils::IsConstZero(rhs)) simplified = lhs;
        } else if (op_name == "divide" || op_name == "div") {
            if (pass_utils::IsConstOne(rhs)) simplified = lhs;
        }

        if (!simplified.defined()) {
            return rewritten;
        }
        return pass_utils::CopyVirtualDevice(rewritten, simplified);
    }
};

}  // namespace

Function SimplifyExprPass(const Function& func) {
    SimplifyExprRewriter pass;
    return pass.Mutate(func);
}

}  // namespace relay
}  // namespace kxc
