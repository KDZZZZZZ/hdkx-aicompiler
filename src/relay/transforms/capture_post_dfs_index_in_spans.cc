/*! \file src/relay/transforms/capture_post_dfs_index_in_spans.cc
 * \brief 实现 Relay 优化 pass 及其 pipeline 集成。
 */

#include "kxc/relay/transforms/capture_post_dfs_index_in_spans.h"

#include <algorithm>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "kxc/relay/op.h"

namespace kxc {
namespace relay {

namespace {

std::vector<Expr> GetChildren(const Expr& expr) {
    std::vector<Expr> children;
    if (!expr.defined()) {
        return children;
    }
    if (const auto* fn = expr.As<FunctionNode>()) {
        for (const auto& param : fn->params) {
            children.push_back(Expr(ObjectRef(param)));
        }
        children.push_back(fn->body);
        return children;
    }
    if (const auto* call = expr.As<CallNode>()) {
        for (const auto& arg : call->args) {
            children.push_back(arg);
        }
        return children;
    }
    if (const auto* if_node = expr.As<IfNode>()) {
        children.push_back(if_node->cond);
        children.push_back(if_node->true_branch);
        children.push_back(if_node->false_branch);
        return children;
    }
    if (const auto* while_node = expr.As<WhileNode>()) {
        children.push_back(while_node->initial_state);
        children.push_back(Expr(ObjectRef(while_node->loop_var)));
        children.push_back(while_node->condition);
        children.push_back(while_node->body);
        return children;
    }
    if (const auto* let_node = expr.As<LetNode>()) {
        children.push_back(Expr(ObjectRef(let_node->var)));
        children.push_back(let_node->value);
        children.push_back(let_node->body);
        return children;
    }
    if (const auto* tuple = expr.As<TupleNode>()) {
        for (const auto& field : tuple->fields) {
            children.push_back(field);
        }
        return children;
    }
    if (const auto* tuple_get = expr.As<TupleGetItemNode>()) {
        children.push_back(tuple_get->tuple);
        return children;
    }
    return children;
}

void IntersectInto(std::unordered_set<const Object*>* dst,
                   const std::unordered_set<const Object*>& src) {
    for (auto it = dst->begin(); it != dst->end();) {
        if (src.count(*it) == 0) {
            it = dst->erase(it);
        } else {
            ++it;
        }
    }
}

}  // namespace

Function CapturePostDfsIndexInSpansPass(const Function& func) {
    if (!func.defined()) {
        return func;
    }

    Expr root = Expr(ObjectRef(func));
    std::unordered_set<const Object*> visited;
    std::unordered_map<const Object*, Expr> node_to_expr;
    std::unordered_map<const Object*, std::vector<const Object*>> succ;
    std::unordered_map<const Object*, std::vector<const Object*>> pred;
    std::vector<const Object*> postorder;

    std::function<void(const Expr&)> dfs = [&](const Expr& expr) {
        if (!expr.defined()) {
            return;
        }
        if (expr.As<relay::OpNode>()) {
            return;
        }
        const Object* node = expr.get();
        if (!node || visited.count(node)) {
            return;
        }
        visited.insert(node);
        node_to_expr[node] = expr;

        for (const auto& child : GetChildren(expr)) {
            if (!child.defined() || child.As<relay::OpNode>()) {
                continue;
            }
            succ[node].push_back(child.get());
            pred[child.get()].push_back(node);
            dfs(child);
        }
        postorder.push_back(node);
    };

    dfs(root);
    if (postorder.empty()) {
        return func;
    }

    std::unordered_map<const Object*, int> post_index;
    for (size_t i = 0; i < postorder.size(); ++i) {
        post_index[postorder[i]] = static_cast<int>(i);
    }

    const Object* root_node = root.get();
    std::unordered_set<const Object*> all_nodes(postorder.begin(), postorder.end());
    std::unordered_map<const Object*, std::unordered_set<const Object*>> dominators;

    for (const auto* node : postorder) {
        if (node == root_node) {
            dominators[node] = {root_node};
        } else {
            dominators[node] = all_nodes;
        }
    }

    bool changed = true;
    while (changed) {
        changed = false;
        for (const auto* node : postorder) {
            if (node == root_node) {
                continue;
            }
            std::unordered_set<const Object*> new_dom;
            auto pred_it = pred.find(node);
            if (pred_it == pred.end() || pred_it->second.empty()) {
                new_dom.insert(node);
            } else {
                new_dom = dominators[pred_it->second.front()];
                for (size_t i = 1; i < pred_it->second.size(); ++i) {
                    IntersectInto(&new_dom, dominators[pred_it->second[i]]);
                }
                new_dom.insert(node);
            }

            if (new_dom != dominators[node]) {
                dominators[node] = std::move(new_dom);
                changed = true;
            }
        }
    }

    std::unordered_map<const Object*, int> dominator_post_index;
    for (const auto* node : postorder) {
        if (node == root_node) {
            dominator_post_index[node] = post_index[node];
            continue;
        }
        const auto& dom_set = dominators[node];
        const Object* idom = root_node;
        size_t best_score = 0;
        for (const auto* candidate : dom_set) {
            if (candidate == node) {
                continue;
            }
            size_t score = dominators[candidate].size();
            if (!idom || score > best_score) {
                idom = candidate;
                best_score = score;
            }
        }
        if (!idom) {
            idom = root_node;
        }
        dominator_post_index[node] = post_index[idom];
    }

    for (const auto& kv : node_to_expr) {
        const Object* node = kv.first;
        const Expr& expr = kv.second;
        auto* expr_node = const_cast<ExprNode*>(static_cast<const ExprNode*>(node));
        if (!expr_node || expr_node->span.defined()) {
            continue;
        }
        expr_node->span = Span("pass.capture_post_dfs", post_index[node],
                               dominator_post_index[node]);
    }

    return func;
}

}  // namespace relay
}  // namespace kxc
