/*! \file src/relay/transforms/eliminate_common_subexpr.cc
 * \brief 实现 Relay 优化 pass 及其 pipeline 集成。
 */

#include "kxc/relay/transforms/eliminate_common_subexpr.h"

#include <string>
#include <unordered_map>

#include "kxc/relay/visitor.h"
#include "kxc/relay/pass_utils.h"

namespace kxc {
namespace relay {

namespace {

class EliminateCommonSubexprRewriter : public RelayPass {
protected:
    Expr VisitLet(const LetNode* op, const Expr& ref) override {
        Expr new_value = Mutate(op->value);
        Var new_var = op->var;

        if (!pass_utils::HasSideEffect(new_value)) {
            const std::string key = pass_utils::ExprStructuralKey(new_value);
            auto it = cse_table_.find(key);
            if (it != cse_table_.end()) {
                Expr replaced_body = pass_utils::SubstituteVar(op->body, op->var, it->second);
                Expr new_body = Mutate(replaced_body);
                return pass_utils::CopyVirtualDevice(ref, new_body);
            }

            cse_table_[key] = Expr(new_var);
            Expr new_body = Mutate(op->body);
            cse_table_.erase(key);

            if (new_value.get() == op->value.get() && new_body.get() == op->body.get()) {
                return ref;
            }
            return pass_utils::CopyVirtualDevice(ref, Let(new_var, new_value, new_body));
        }

        Expr new_body = Mutate(op->body);
        if (new_value.get() == op->value.get() && new_body.get() == op->body.get()) {
            return ref;
        }
        return pass_utils::CopyVirtualDevice(ref, Let(new_var, new_value, new_body));
    }

private:
    std::unordered_map<std::string, Expr> cse_table_;
};

}  // namespace

Function EliminateCommonSubexprPass(const Function& func) {
    EliminateCommonSubexprRewriter pass;
    return pass.Mutate(func);
}

}  // namespace relay
}  // namespace kxc
