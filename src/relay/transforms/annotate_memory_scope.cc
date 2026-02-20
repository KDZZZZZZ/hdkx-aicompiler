#include "relay/transforms/annotate_memory_scope.h"

#include <functional>
#include <string>
#include <unordered_set>

#include "relay/pass_utils.h"

namespace kxc {
namespace relay {

namespace {

void VisitRelayExpr(const Expr& expr, const std::function<void(const Expr&)>& f,
                    std::unordered_set<const Object*>* visited) {
    if (!visited || !expr.defined() || visited->count(expr.get()) != 0) {
        return;
    }
    visited->insert(expr.get());
    f(expr);

    if (const auto* fn = expr.As<FunctionNode>()) {
        for (const auto& param : fn->params) {
            VisitRelayExpr(Expr(param), f, visited);
        }
        VisitRelayExpr(fn->body, f, visited);
        return;
    }
    if (const auto* call = expr.As<CallNode>()) {
        VisitRelayExpr(call->op, f, visited);
        for (const auto& arg : call->args) {
            VisitRelayExpr(arg, f, visited);
        }
        return;
    }
    if (const auto* if_node = expr.As<IfNode>()) {
        VisitRelayExpr(if_node->cond, f, visited);
        VisitRelayExpr(if_node->true_branch, f, visited);
        VisitRelayExpr(if_node->false_branch, f, visited);
        return;
    }
    if (const auto* let_node = expr.As<LetNode>()) {
        VisitRelayExpr(Expr(let_node->var), f, visited);
        VisitRelayExpr(let_node->value, f, visited);
        VisitRelayExpr(let_node->body, f, visited);
        return;
    }
    if (const auto* tuple = expr.As<TupleNode>()) {
        for (const auto& field : tuple->fields) {
            VisitRelayExpr(field, f, visited);
        }
        return;
    }
    if (const auto* tuple_get = expr.As<TupleGetItemNode>()) {
        VisitRelayExpr(tuple_get->tuple, f, visited);
    }
}

}  // namespace

Function AnnotateMemoryScopePass(const Function& func) {
    if (!func.defined()) {
        return func;
    }

    std::unordered_set<const Object*> visited;
    VisitRelayExpr(Expr(func),
                   [](const Expr& expr) {
                       if (!expr.defined()) {
                           return;
                       }
                       auto* relay_node = const_cast<RelayNode*>(
                           dynamic_cast<const RelayNode*>(expr.get()));
                       if (!relay_node || !relay_node->virtual_device_.defined()) {
                           return;
                       }
                       if (!relay_node->virtual_device_->memory_scope.empty()) {
                           return;
                       }
                       const std::string scope = expr.As<ConstantNode>() ? "const" : "global";
                       relay_node->virtual_device_ =
                           pass_utils::WithMemoryScope(relay_node->virtual_device_, scope);
                   },
                   &visited);
    return func;
}

}  // namespace relay
}  // namespace kxc
