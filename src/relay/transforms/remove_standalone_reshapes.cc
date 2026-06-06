/*! \file src/relay/transforms/remove_standalone_reshapes.cc
 * \brief 实现 Relay 优化 pass 及其 pipeline 集成。
 */

#include "relay/transforms/remove_standalone_reshapes.h"

#include <string>

#include "base/pass.h"
#include "relay/pass_utils.h"

namespace kxc {
namespace relay {

namespace {

class RemoveStandaloneReshapesRewriter : public RelayPass {
protected:
    Expr VisitCall(const CallNode* op, const Expr& ref) override {
        Expr rewritten = RelayPass::VisitCall(op, ref);
        const auto* call = rewritten.As<CallNode>();
        if (!call || call->args.empty()) {
            return rewritten;
        }
        if (pass_utils::GetCallOpName(call) != "reshape") {
            return rewritten;
        }

        const auto* inner_call = call->args[0].As<CallNode>();
        if (!inner_call || inner_call->args.empty()) {
            return rewritten;
        }
        if (pass_utils::GetCallOpName(inner_call) != "reshape") {
            return rewritten;
        }

        Array<Expr> folded_args = {inner_call->args[0]};
        if (call->args.size() > 1) {
            folded_args.push_back(call->args[1]);
        }
        Call folded(call->op, folded_args, call->attrs);
        return pass_utils::CopyVirtualDevice(rewritten, folded);
    }
};

}  // namespace

Function RemoveStandaloneReshapesPass(const Function& func) {
    RemoveStandaloneReshapesRewriter pass;
    return pass.Mutate(func);
}

}  // namespace relay
}  // namespace kxc

