/*! \file src/relay/transforms/canonicalize_cast.cc
 * \brief 实现 Relay 优化 pass 及其 pipeline 集成。
 */

#include "relay/transforms/canonicalize_cast.h"

#include <string>

#include "base/pass.h"
#include "relay/op.h"
#include "relay/pass_utils.h"

namespace kxc {
namespace relay {

namespace {

class CanonicalizeCastRewriter : public RelayPass {
protected:
    Expr VisitCall(const CallNode* op, const Expr& ref) override {
        Expr rewritten = RelayPass::VisitCall(op, ref);
        const auto* call = rewritten.As<CallNode>();
        if (!call) {
            return rewritten;
        }
        if (pass_utils::GetCallOpName(call) != "cast" || call->args.empty()) {
            return rewritten;
        }

        const auto* outer_attrs = call->attrs.As<CastAttrsNode>();
        const auto* inner_call = call->args[0].As<CallNode>();
        if (!outer_attrs || !inner_call || inner_call->args.empty()) {
            return rewritten;
        }
        if (pass_utils::GetCallOpName(inner_call) != "cast") {
            return rewritten;
        }
        const auto* inner_attrs = inner_call->attrs.As<CastAttrsNode>();
        if (!inner_attrs) {
            return rewritten;
        }
        if (outer_attrs->to != inner_attrs->to) {
            return rewritten;
        }

        Call folded(call->op, {inner_call->args[0]}, call->attrs);
        return pass_utils::CopyVirtualDevice(rewritten, folded);
    }
};

}  // namespace

Function CanonicalizeCastPass(const Function& func) {
    CanonicalizeCastRewriter pass;
    return pass.Mutate(func);
}

}  // namespace relay
}  // namespace kxc

