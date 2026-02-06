#pragma once
#include "relay/relay.h"
#include "base/container.h"
#include <vector>

namespace kxc {

// Visitor Pattern Functor for Relay IR (operating on Expr)
template <typename R>
class RelayPassFunctor {
public:
    virtual ~RelayPassFunctor() = default;
    
    virtual R Visit(const Expr& expr) {
        if (!expr.defined()) return R();
        
        // Dispatch based on runtime type
        if (auto* n = expr.As<ConstantNode>())     return VisitConstant(n, expr);
        if (auto* n = expr.As<VarNode>())          return VisitVar(n, expr);
        if (auto* n = expr.As<CallNode>())         return VisitCall(n, expr);
        if (auto* n = expr.As<FunctionNode>())     return VisitFunction(n, expr);
        if (auto* n = expr.As<IfNode>())           return VisitIf(n, expr);
        if (auto* n = expr.As<LetNode>())          return VisitLet(n, expr);
        if (auto* n = expr.As<TupleNode>())        return VisitTuple(n, expr);
        if (auto* n = expr.As<TupleGetItemNode>()) return VisitTupleGetItem(n, expr);
        
        return VisitDefault(expr);
    }

protected:
    virtual R VisitConstant(const ConstantNode* op, const Expr& ref) { return VisitDefault(ref); }
    virtual R VisitVar(const VarNode* op, const Expr& ref)           { return VisitDefault(ref); }
    virtual R VisitCall(const CallNode* op, const Expr& ref)         { return VisitDefault(ref); }
    virtual R VisitFunction(const FunctionNode* op, const Expr& ref) { return VisitDefault(ref); }
    virtual R VisitIf(const IfNode* op, const Expr& ref)             { return VisitDefault(ref); }
    virtual R VisitLet(const LetNode* op, const Expr& ref)           { return VisitDefault(ref); }
    virtual R VisitTuple(const TupleNode* op, const Expr& ref)       { return VisitDefault(ref); }
    virtual R VisitTupleGetItem(const TupleGetItemNode* op, const Expr& ref) { return VisitDefault(ref); }

    virtual R VisitDefault(const Expr& expr) {
        return R(); 
    }
};

// RelayPass: A default Mutator that traverses the graph and reconstructs it if children change.
class RelayPass : public RelayPassFunctor<Expr> {
public:
    Expr Mutate(const Expr& expr) {
        return Visit(expr);
    }

protected:
    // Leaf nodes: default is to preserve identity
    Expr VisitConstant(const ConstantNode* op, const Expr& ref) override { 
        return ref; 
    }
    
    Expr VisitVar(const VarNode* op, const Expr& ref) override { 
        return ref; 
    }

    Expr VisitCall(const CallNode* op, const Expr& ref) override {
        auto new_op = Mutate(op->op);
        Array<Expr> new_args;
        bool changed = (new_op.get() != op->op.get());
        
        for (const auto& arg : op->args) {
            auto new_arg = Mutate(arg);
            if (new_arg.get() != arg.get()) changed = true;
            new_args.push_back(new_arg);
        }

        if (!changed) return ref;
        return Call(new_op, new_args, op->attrs);
    }

    Expr VisitFunction(const FunctionNode* op, const Expr& ref) override {
        auto new_body = Mutate(op->body);
        if (new_body.get() == op->body.get()) return ref;
        return Function(op->params, new_body);
    }

    Expr VisitIf(const IfNode* op, const Expr& ref) override {
        auto new_cond = Mutate(op->cond);
        auto new_true = Mutate(op->true_branch);
        auto new_false = Mutate(op->false_branch);
        
        if (new_cond.get() == op->cond.get() && 
            new_true.get() == op->true_branch.get() && 
            new_false.get() == op->false_branch.get()) {
            return ref;
        }
        return If(new_cond, new_true, new_false);
    }

    Expr VisitLet(const LetNode* op, const Expr& ref) override {
        auto new_value = Mutate(op->value);
        auto new_body = Mutate(op->body);
        
        if (new_value.get() == op->value.get() && 
            new_body.get() == op->body.get()) {
            return ref;
        }
        return Let(op->var, new_value, new_body);
    }

    Expr VisitTuple(const TupleNode* op, const Expr& ref) override {
        Array<Expr> new_fields;
        bool changed = false;
        for (const auto& field : op->fields) {
            auto new_field = Mutate(field);
            if (new_field.get() != field.get()) changed = true;
            new_fields.push_back(new_field);
        }
        
        if (!changed) return ref;
        return Tuple(new_fields);
    }

    Expr VisitTupleGetItem(const TupleGetItemNode* op, const Expr& ref) override {
        auto new_tuple = Mutate(op->tuple);
        if (new_tuple.get() == op->tuple.get()) return ref;
        return TupleGetItem(new_tuple, op->index);
    }
};

} // namespace kxc
