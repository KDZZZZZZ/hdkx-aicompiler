/*! \file include/kxc/relay/visitor.h
 * \brief Defines Relay visitor, mutator, and PassContext adapters.
 */

#pragma once

#include <unordered_map>

#include "kxc/pass/context.h"
#include "kxc/relay/op.h"
#include "kxc/relay/relay.h"

namespace kxc {

/*! \brief Relay IR visitor functor，派生类通过重载 Visit* 处理不同节点。 */
template <typename R>
class RelayPassFunctor {
public:
    virtual ~RelayPassFunctor() = default;

    virtual R Visit(const Expr& expr) { return VisitExpr(expr); }

    virtual R VisitExpr(const Expr& expr) {
        if (!expr.defined()) return R();

        if (auto* n = expr.As<ConstantNode>()) return VisitConstant(n, expr);
        if (auto* n = expr.As<VarNode>()) return VisitVar(n, expr);
        if (auto* n = expr.As<relay::OpNode>()) return VisitOp(n, expr);
        if (auto* n = expr.As<CallNode>()) return VisitCall(n, expr);
        if (auto* n = expr.As<FunctionNode>()) return VisitFunction(n, expr);
        if (auto* n = expr.As<IfNode>()) return VisitIf(n, expr);
        if (auto* n = expr.As<WhileNode>()) return VisitWhile(n, expr);
        if (auto* n = expr.As<LetNode>()) return VisitLet(n, expr);
        if (auto* n = expr.As<TupleNode>()) return VisitTuple(n, expr);
        if (auto* n = expr.As<TupleGetItemNode>()) return VisitTupleGetItem(n, expr);

        return VisitDefault(expr);
    }

protected:
    virtual R VisitConstant(const ConstantNode* op, const Expr& ref) { return VisitDefault(ref); }
    virtual R VisitVar(const VarNode* op, const Expr& ref) { return VisitDefault(ref); }
    virtual R VisitOp(const relay::OpNode* op, const Expr& ref) { return VisitDefault(ref); }
    virtual R VisitCall(const CallNode* op, const Expr& ref) { return VisitDefault(ref); }
    virtual R VisitFunction(const FunctionNode* op, const Expr& ref) { return VisitDefault(ref); }
    virtual R VisitIf(const IfNode* op, const Expr& ref) { return VisitDefault(ref); }
    virtual R VisitWhile(const WhileNode* op, const Expr& ref) { return VisitDefault(ref); }
    virtual R VisitLet(const LetNode* op, const Expr& ref) { return VisitDefault(ref); }
    virtual R VisitTuple(const TupleNode* op, const Expr& ref) { return VisitDefault(ref); }
    virtual R VisitTupleGetItem(const TupleGetItemNode* op, const Expr& ref) { return VisitDefault(ref); }

    virtual R VisitDefault(const Expr& expr) {
        (void)expr;
        return R();
    }
};

/*! \brief Relay IR mutator，默认递归重建表达式树。 */
class RelayPass : public RelayPassFunctor<Expr> {
public:
    Expr Mutate(const Expr& expr);
    Function Mutate(const Function& func);
    Var Mutate(const Var& var);

protected:
    Expr VisitConstant(const ConstantNode* op, const Expr& ref) override;
    Expr VisitVar(const VarNode* op, const Expr& ref) override;
    Expr VisitOp(const relay::OpNode* op, const Expr& ref) override;
    Expr VisitCall(const CallNode* op, const Expr& ref) override;
    Expr VisitFunction(const FunctionNode* op, const Expr& ref) override;
    Expr VisitIf(const IfNode* op, const Expr& ref) override;
    Expr VisitWhile(const WhileNode* op, const Expr& ref) override;
    Expr VisitLet(const LetNode* op, const Expr& ref) override;
    Expr VisitTuple(const TupleNode* op, const Expr& ref) override;
    Expr VisitTupleGetItem(const TupleGetItemNode* op, const Expr& ref) override;

private:
    Var MutateToVar(const Var& var);

    // Relay 表达式是 DAG：Transformer 的每个残差都让同一子表达式被多个消费者
    // 引用。没有记忆时，共享节点每被引用一次就重写一遍，重写代价随共享点数量
    // 指数增长，且输出会把 DAG 展开成树，把膨胀继续传给下游阶段。
    //
    // 因此按节点指针记忆内部节点的重写结果：每个子表达式只重写一次，输出保留
    // 原有共享。叶子（Var/Constant/Op）不记忆——它们重写代价是 O(1)，而 Var
    // 还可能在不同作用域下被重命名，记忆会跨作用域串味。
    //
    // 记忆只在一次顶层 Mutate 内有效：depth_ 回到 0 时清空，避免节点释放后
    // 指针复用导致命中陈旧条目。
    std::unordered_map<const Object*, Expr> memo_;
    int depth_{0};
};

namespace relay {

PassContext PassContextFromRelay(const Expr& expr);
PassContext PassContextFromRelay(const Function& func);

}  // namespace relay
}  // namespace kxc
