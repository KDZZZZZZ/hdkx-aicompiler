#pragma once
#include "relay/relay.h"
#include "tir/stmt.h"
#include "base/container.h"
#include <vector>
#include <stdexcept>

namespace kxc {

// Visitor Pattern Functor for Relay IR (operating on Expr)
template <typename R>
class RelayPassFunctor {
public:
    virtual ~RelayPassFunctor() = default;

    virtual R Visit(const Expr& expr) {
        return VisitExpr(expr);
    }

    virtual R VisitExpr(const Expr& expr) {
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
        return VisitExpr(expr);
    }

    Function Mutate(const Function& func) {
        if (!func.defined()) return func;
        Expr out = Mutate(Expr(func));
        if (!out.defined()) return Function();
        if (!out.As<FunctionNode>()) {
            throw std::runtime_error("RelayPass expected Function result when mutating Function");
        }
        return Function(out);
    }

    Var Mutate(const Var& var) {
        return MutateToVar(var);
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
        Array<Var> new_params;
        bool changed = false;
        for (const auto& param : op->params) {
            Var new_param = MutateToVar(param);
            if (new_param.get() != param.get()) changed = true;
            new_params.push_back(new_param);
        }

        auto new_body = Mutate(op->body);
        if (new_body.get() != op->body.get()) changed = true;

        if (!changed) return ref;
        return Function(changed ? new_params : op->params, new_body);
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
        auto new_var = MutateToVar(op->var);
        auto new_value = Mutate(op->value);
        auto new_body = Mutate(op->body);

        if (new_var.get() == op->var.get() &&
            new_value.get() == op->value.get() &&
            new_body.get() == op->body.get()) {
            return ref;
        }
        return Let(new_var, new_value, new_body);
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

private:
    Var MutateToVar(const Var& var) {
        if (!var.defined()) return var;
        Expr new_var = Mutate(Expr(var));
        if (!new_var.defined()) {
            throw std::runtime_error("RelayPass mutated Var into undefined expression");
        }
        if (!new_var.As<VarNode>()) {
            throw std::runtime_error("RelayPass expects variable position to remain Var");
        }
        return Var(new_var);
    }
};

// Visitor Pattern Functor for TIR PrimExpr.
template <typename R>
class TIRExprFunctor {
public:
    virtual ~TIRExprFunctor() = default;

    virtual R VisitExpr(const tir::PrimExpr& expr) {
        if (!expr.defined()) return R();

        if (auto* n = expr.As<tir::IntImmNode>()) return VisitIntImm(n, expr);
        if (auto* n = expr.As<tir::FloatImmNode>()) return VisitFloatImm(n, expr);
        if (auto* n = expr.As<tir::VarNode>()) return VisitVar(n, expr);
        if (auto* n = expr.As<tir::AddNode>()) return VisitAdd(n, expr);
        if (auto* n = expr.As<tir::SubNode>()) return VisitSub(n, expr);
        if (auto* n = expr.As<tir::MulNode>()) return VisitMul(n, expr);
        if (auto* n = expr.As<tir::DivNode>()) return VisitDiv(n, expr);
        if (auto* n = expr.As<tir::ModNode>()) return VisitMod(n, expr);
        if (auto* n = expr.As<tir::MinNode>()) return VisitMin(n, expr);
        if (auto* n = expr.As<tir::MaxNode>()) return VisitMax(n, expr);
        if (auto* n = expr.As<tir::EQNode>()) return VisitEQ(n, expr);
        if (auto* n = expr.As<tir::LTNode>()) return VisitLT(n, expr);
        if (auto* n = expr.As<tir::AndNode>()) return VisitAnd(n, expr);
        if (auto* n = expr.As<tir::OrNode>()) return VisitOr(n, expr);
        if (auto* n = expr.As<tir::NotNode>()) return VisitNot(n, expr);
        if (auto* n = expr.As<tir::LoadNode>()) return VisitLoad(n, expr);
        if (auto* n = expr.As<tir::CallNode>()) return VisitCall(n, expr);
        if (auto* n = expr.As<tir::SelectNode>()) return VisitSelect(n, expr);

        return VisitExprDefault(expr);
    }

protected:
    virtual R VisitIntImm(const tir::IntImmNode* op, const tir::PrimExpr& ref) { return VisitExprDefault(ref); }
    virtual R VisitFloatImm(const tir::FloatImmNode* op, const tir::PrimExpr& ref) { return VisitExprDefault(ref); }
    virtual R VisitVar(const tir::VarNode* op, const tir::PrimExpr& ref) { return VisitExprDefault(ref); }
    virtual R VisitAdd(const tir::AddNode* op, const tir::PrimExpr& ref) { return VisitExprDefault(ref); }
    virtual R VisitSub(const tir::SubNode* op, const tir::PrimExpr& ref) { return VisitExprDefault(ref); }
    virtual R VisitMul(const tir::MulNode* op, const tir::PrimExpr& ref) { return VisitExprDefault(ref); }
    virtual R VisitDiv(const tir::DivNode* op, const tir::PrimExpr& ref) { return VisitExprDefault(ref); }
    virtual R VisitMod(const tir::ModNode* op, const tir::PrimExpr& ref) { return VisitExprDefault(ref); }
    virtual R VisitMin(const tir::MinNode* op, const tir::PrimExpr& ref) { return VisitExprDefault(ref); }
    virtual R VisitMax(const tir::MaxNode* op, const tir::PrimExpr& ref) { return VisitExprDefault(ref); }
    virtual R VisitEQ(const tir::EQNode* op, const tir::PrimExpr& ref) { return VisitExprDefault(ref); }
    virtual R VisitLT(const tir::LTNode* op, const tir::PrimExpr& ref) { return VisitExprDefault(ref); }
    virtual R VisitAnd(const tir::AndNode* op, const tir::PrimExpr& ref) { return VisitExprDefault(ref); }
    virtual R VisitOr(const tir::OrNode* op, const tir::PrimExpr& ref) { return VisitExprDefault(ref); }
    virtual R VisitNot(const tir::NotNode* op, const tir::PrimExpr& ref) { return VisitExprDefault(ref); }
    virtual R VisitLoad(const tir::LoadNode* op, const tir::PrimExpr& ref) { return VisitExprDefault(ref); }
    virtual R VisitCall(const tir::CallNode* op, const tir::PrimExpr& ref) { return VisitExprDefault(ref); }
    virtual R VisitSelect(const tir::SelectNode* op, const tir::PrimExpr& ref) { return VisitExprDefault(ref); }

    virtual R VisitExprDefault(const tir::PrimExpr& expr) { return R(); }
};

// Visitor Pattern Functor for TIR Stmt.
template <typename R>
class TIRStmtFunctor {
public:
    virtual ~TIRStmtFunctor() = default;

    virtual R VisitStmt(const tir::Stmt& stmt) {
        if (!stmt.defined()) return R();

        if (auto* n = stmt.As<tir::LetStmtNode>()) return VisitLetStmt(n, stmt);
        if (auto* n = stmt.As<tir::StoreNode>()) return VisitStore(n, stmt);
        if (auto* n = stmt.As<tir::ForNode>()) return VisitFor(n, stmt);
        if (auto* n = stmt.As<tir::IfThenElseNode>()) return VisitIfThenElse(n, stmt);
        if (auto* n = stmt.As<tir::AllocateNode>()) return VisitAllocate(n, stmt);
        if (auto* n = stmt.As<tir::AttrStmtNode>()) return VisitAttrStmt(n, stmt);
        if (auto* n = stmt.As<tir::BlockNode>()) return VisitBlock(n, stmt);
        if (auto* n = stmt.As<tir::SeqStmtNode>()) return VisitSeqStmt(n, stmt);
        if (auto* n = stmt.As<tir::EvaluateNode>()) return VisitEvaluate(n, stmt);

        return VisitStmtDefault(stmt);
    }

protected:
    virtual R VisitLetStmt(const tir::LetStmtNode* op, const tir::Stmt& ref) { return VisitStmtDefault(ref); }
    virtual R VisitStore(const tir::StoreNode* op, const tir::Stmt& ref) { return VisitStmtDefault(ref); }
    virtual R VisitFor(const tir::ForNode* op, const tir::Stmt& ref) { return VisitStmtDefault(ref); }
    virtual R VisitIfThenElse(const tir::IfThenElseNode* op, const tir::Stmt& ref) { return VisitStmtDefault(ref); }
    virtual R VisitAllocate(const tir::AllocateNode* op, const tir::Stmt& ref) { return VisitStmtDefault(ref); }
    virtual R VisitAttrStmt(const tir::AttrStmtNode* op, const tir::Stmt& ref) { return VisitStmtDefault(ref); }
    virtual R VisitBlock(const tir::BlockNode* op, const tir::Stmt& ref) { return VisitStmtDefault(ref); }
    virtual R VisitSeqStmt(const tir::SeqStmtNode* op, const tir::Stmt& ref) { return VisitStmtDefault(ref); }
    virtual R VisitEvaluate(const tir::EvaluateNode* op, const tir::Stmt& ref) { return VisitStmtDefault(ref); }

    virtual R VisitStmtDefault(const tir::Stmt& stmt) { return R(); }
};

// TIRPass: default mutator for PrimExpr/Stmt/PrimFunc with structural sharing.
class TIRPass : public TIRExprFunctor<tir::PrimExpr>, public TIRStmtFunctor<tir::Stmt> {
public:
    tir::PrimExpr Mutate(const tir::PrimExpr& expr) { return VisitExpr(expr); }
    tir::Stmt Mutate(const tir::Stmt& stmt) { return VisitStmt(stmt); }
    tir::PrimFunc Mutate(const tir::PrimFunc& func) { return VisitPrimFunc(func); }

protected:
    virtual tir::PrimFunc VisitPrimFunc(const tir::PrimFunc& func) {
        if (!func.defined()) return func;

        Array<tir::Var> new_params;
        bool params_changed = false;
        for (const auto& p : func->params) {
            tir::Var np = MutateToVar(p);
            if (np.get() != p.get()) params_changed = true;
            new_params.push_back(np);
        }

        tir::Stmt new_body = Mutate(func->body);
        bool body_changed = (new_body.get() != func->body.get());

        Map<tir::Var, tir::Buffer> new_buffer_map;
        bool buffer_map_changed = false;
        for (const auto& kv : func->buffer_map) {
            tir::Var new_key = MutateToVar(kv.first);
            tir::Buffer new_val = MutateBuffer(kv.second);
            if (new_key.get() != kv.first.get() || new_val.get() != kv.second.get()) {
                buffer_map_changed = true;
            }
            new_buffer_map.Set(new_key, new_val);
        }

        if (!params_changed && !body_changed && !buffer_map_changed) {
            return func;
        }

        return tir::PrimFunc(
            params_changed ? new_params : func->params,
            new_body,
            buffer_map_changed ? new_buffer_map : func->buffer_map,
            func->attrs);
    }

    // PrimExpr leaf nodes.
    tir::PrimExpr VisitIntImm(const tir::IntImmNode* op, const tir::PrimExpr& ref) override { return ref; }
    tir::PrimExpr VisitFloatImm(const tir::FloatImmNode* op, const tir::PrimExpr& ref) override { return ref; }
    tir::PrimExpr VisitVar(const tir::VarNode* op, const tir::PrimExpr& ref) override { return ref; }

#define KXC_TIR_MUTATE_BINARY_EXPR(OpName) \
    tir::PrimExpr Visit##OpName(const tir::OpName##Node* op, const tir::PrimExpr& ref) override { \
        tir::PrimExpr new_a = Mutate(op->a); \
        tir::PrimExpr new_b = Mutate(op->b); \
        if (new_a.get() == op->a.get() && new_b.get() == op->b.get()) return ref; \
        return tir::OpName(new_a, new_b); \
    }

    KXC_TIR_MUTATE_BINARY_EXPR(Add)
    KXC_TIR_MUTATE_BINARY_EXPR(Sub)
    KXC_TIR_MUTATE_BINARY_EXPR(Mul)
    KXC_TIR_MUTATE_BINARY_EXPR(Div)
    KXC_TIR_MUTATE_BINARY_EXPR(Mod)
    KXC_TIR_MUTATE_BINARY_EXPR(Min)
    KXC_TIR_MUTATE_BINARY_EXPR(Max)
    KXC_TIR_MUTATE_BINARY_EXPR(EQ)
    KXC_TIR_MUTATE_BINARY_EXPR(LT)
    KXC_TIR_MUTATE_BINARY_EXPR(And)
    KXC_TIR_MUTATE_BINARY_EXPR(Or)

#undef KXC_TIR_MUTATE_BINARY_EXPR

    tir::PrimExpr VisitNot(const tir::NotNode* op, const tir::PrimExpr& ref) override {
        tir::PrimExpr new_value = Mutate(op->value);
        if (new_value.get() == op->value.get()) return ref;
        return tir::Not(new_value);
    }

    tir::PrimExpr VisitLoad(const tir::LoadNode* op, const tir::PrimExpr& ref) override {
        tir::Var new_buffer_var = MutateToVar(op->buffer_var);
        tir::PrimExpr new_index = Mutate(op->index);
        tir::PrimExpr new_pred = Mutate(op->predicate);
        if (new_buffer_var.get() == op->buffer_var.get() &&
            new_index.get() == op->index.get() &&
            new_pred.get() == op->predicate.get()) {
            return ref;
        }
        return tir::Load(new_buffer_var, new_index, new_pred);
    }

    tir::PrimExpr VisitCall(const tir::CallNode* op, const tir::PrimExpr& ref) override {
        Array<tir::PrimExpr> new_args;
        bool changed = false;
        for (const auto& arg : op->args) {
            tir::PrimExpr new_arg = Mutate(arg);
            if (new_arg.get() != arg.get()) changed = true;
            new_args.push_back(new_arg);
        }
        if (!changed) return ref;
        return tir::Call(ref.dtype(), op->name, new_args);
    }

    tir::PrimExpr VisitSelect(const tir::SelectNode* op, const tir::PrimExpr& ref) override {
        tir::PrimExpr new_cond = Mutate(op->condition);
        tir::PrimExpr new_true = Mutate(op->true_value);
        tir::PrimExpr new_false = Mutate(op->false_value);
        if (new_cond.get() == op->condition.get() &&
            new_true.get() == op->true_value.get() &&
            new_false.get() == op->false_value.get()) {
            return ref;
        }
        return tir::Select(new_cond, new_true, new_false);
    }

    // Stmt nodes.
    tir::Stmt VisitLetStmt(const tir::LetStmtNode* op, const tir::Stmt& ref) override {
        tir::Var new_var = MutateToVar(op->var);
        tir::PrimExpr new_value = Mutate(op->value);
        tir::Stmt new_body = Mutate(op->body);
        if (new_var.get() == op->var.get() &&
            new_value.get() == op->value.get() &&
            new_body.get() == op->body.get()) {
            return ref;
        }
        return tir::LetStmt(new_var, new_value, new_body);
    }

    tir::Stmt VisitStore(const tir::StoreNode* op, const tir::Stmt& ref) override {
        tir::Var new_buffer_var = MutateToVar(op->buffer_var);
        tir::PrimExpr new_value = Mutate(op->value);
        tir::PrimExpr new_index = Mutate(op->index);
        tir::PrimExpr new_pred = Mutate(op->predicate);
        if (new_buffer_var.get() == op->buffer_var.get() &&
            new_value.get() == op->value.get() &&
            new_index.get() == op->index.get() &&
            new_pred.get() == op->predicate.get()) {
            return ref;
        }
        return tir::Store(new_buffer_var, new_value, new_index, new_pred);
    }

    tir::Stmt VisitFor(const tir::ForNode* op, const tir::Stmt& ref) override {
        tir::Var new_loop_var = MutateToVar(op->loop_var);
        tir::PrimExpr new_min = Mutate(op->min);
        tir::PrimExpr new_extent = Mutate(op->extent);
        tir::Stmt new_body = Mutate(op->body);
        if (new_loop_var.get() == op->loop_var.get() &&
            new_min.get() == op->min.get() &&
            new_extent.get() == op->extent.get() &&
            new_body.get() == op->body.get()) {
            return ref;
        }
        return tir::For(new_loop_var, new_min, new_extent, op->for_type, new_body);
    }

    tir::Stmt VisitIfThenElse(const tir::IfThenElseNode* op, const tir::Stmt& ref) override {
        tir::PrimExpr new_cond = Mutate(op->condition);
        tir::Stmt new_then = Mutate(op->then_case);
        tir::Stmt new_else = Mutate(op->else_case);
        if (new_cond.get() == op->condition.get() &&
            new_then.get() == op->then_case.get() &&
            new_else.get() == op->else_case.get()) {
            return ref;
        }
        return tir::IfThenElse(new_cond, new_then, new_else);
    }

    tir::Stmt VisitAllocate(const tir::AllocateNode* op, const tir::Stmt& ref) override {
        tir::Var new_buffer_var = MutateToVar(op->buffer_var);
        Array<tir::PrimExpr> new_extents;
        bool extents_changed = false;
        for (const auto& extent : op->extents) {
            tir::PrimExpr new_extent = Mutate(extent);
            if (new_extent.get() != extent.get()) extents_changed = true;
            new_extents.push_back(new_extent);
        }
        tir::PrimExpr new_cond = Mutate(op->condition);
        tir::Stmt new_body = Mutate(op->body);
        if (new_buffer_var.get() == op->buffer_var.get() &&
            !extents_changed &&
            new_cond.get() == op->condition.get() &&
            new_body.get() == op->body.get()) {
            return ref;
        }
        return tir::Allocate(
            new_buffer_var,
            op->dtype,
            extents_changed ? new_extents : op->extents,
            new_cond,
            new_body);
    }

    tir::Stmt VisitAttrStmt(const tir::AttrStmtNode* op, const tir::Stmt& ref) override {
        tir::PrimExpr new_value = Mutate(op->value);
        tir::Stmt new_body = Mutate(op->body);
        if (new_value.get() == op->value.get() && new_body.get() == op->body.get()) {
            return ref;
        }
        return tir::AttrStmt(op->node, op->attr_key, new_value, new_body);
    }

    tir::Stmt VisitBlock(const tir::BlockNode* op, const tir::Stmt& ref) override {
        Array<tir::IterVar> new_iter_vars;
        bool iter_vars_changed = false;
        for (const auto& iv : op->iter_vars) {
            tir::IterVar new_iv = MutateIterVar(iv);
            if (new_iv.get() != iv.get()) iter_vars_changed = true;
            new_iter_vars.push_back(new_iv);
        }

        Array<tir::BufferRegion> new_reads;
        bool reads_changed = false;
        for (const auto& r : op->reads) {
            tir::BufferRegion nr = MutateBufferRegion(r);
            if (nr.get() != r.get()) reads_changed = true;
            new_reads.push_back(nr);
        }

        Array<tir::BufferRegion> new_writes;
        bool writes_changed = false;
        for (const auto& w : op->writes) {
            tir::BufferRegion nw = MutateBufferRegion(w);
            if (nw.get() != w.get()) writes_changed = true;
            new_writes.push_back(nw);
        }

        tir::Stmt new_body = Mutate(op->body);
        tir::Stmt new_init = Mutate(op->init);

        if (!iter_vars_changed && !reads_changed && !writes_changed &&
            new_body.get() == op->body.get() &&
            new_init.get() == op->init.get()) {
            return ref;
        }

        return tir::Block(
            iter_vars_changed ? new_iter_vars : op->iter_vars,
            reads_changed ? new_reads : op->reads,
            writes_changed ? new_writes : op->writes,
            op->name_hint,
            new_body,
            new_init);
    }

    tir::Stmt VisitSeqStmt(const tir::SeqStmtNode* op, const tir::Stmt& ref) override {
        Array<tir::Stmt> new_seq;
        bool changed = false;
        for (const auto& s : op->seq) {
            tir::Stmt ns = Mutate(s);
            if (ns.get() != s.get()) changed = true;
            new_seq.push_back(ns);
        }
        if (!changed) return ref;
        return tir::SeqStmt(new_seq);
    }

    tir::Stmt VisitEvaluate(const tir::EvaluateNode* op, const tir::Stmt& ref) override {
        tir::PrimExpr new_value = Mutate(op->value);
        if (new_value.get() == op->value.get()) return ref;
        return tir::Evaluate(new_value);
    }

private:
    tir::Var MutateToVar(const tir::Var& var) {
        if (!var.defined()) return var;
        tir::PrimExpr new_var_expr = Mutate(tir::PrimExpr(var));
        if (!new_var_expr.defined()) {
            throw std::runtime_error("TIRPass mutated Var into undefined expression");
        }
        auto* var_node = new_var_expr.As<tir::VarNode>();
        if (!var_node) {
            throw std::runtime_error("TIRPass expects variable position to remain tir::Var");
        }
        return tir::Var(new_var_expr);
    }

    tir::Range MutateRange(const tir::Range& range) {
        if (!range.defined()) return range;
        tir::PrimExpr new_min = Mutate(range->min);
        tir::PrimExpr new_extent = Mutate(range->extent);
        if (new_min.get() == range->min.get() && new_extent.get() == range->extent.get()) {
            return range;
        }
        return tir::Range(new_min, new_extent);
    }

    tir::IterVar MutateIterVar(const tir::IterVar& iv) {
        if (!iv.defined()) return iv;
        tir::Range new_dom = MutateRange(iv->dom);
        tir::Var new_var = MutateToVar(iv->var);
        if (new_dom.get() == iv->dom.get() && new_var.get() == iv->var.get()) {
            return iv;
        }
        return tir::IterVar(new_dom, new_var, iv->iter_type, iv->thread_tag);
    }

    tir::Buffer MutateBuffer(const tir::Buffer& buffer) {
        if (!buffer.defined()) return buffer;

        tir::Var new_data = MutateToVar(buffer->data);

        Array<tir::PrimExpr> new_shape;
        bool shape_changed = false;
        for (const auto& s : buffer->shape) {
            tir::PrimExpr ns = Mutate(s);
            if (ns.get() != s.get()) shape_changed = true;
            new_shape.push_back(ns);
        }

        Array<tir::PrimExpr> new_strides;
        bool strides_changed = false;
        for (const auto& s : buffer->strides) {
            tir::PrimExpr ns = Mutate(s);
            if (ns.get() != s.get()) strides_changed = true;
            new_strides.push_back(ns);
        }

        tir::PrimExpr new_elem_offset = Mutate(buffer->elem_offset);

        if (new_data.get() == buffer->data.get() &&
            !shape_changed &&
            !strides_changed &&
            new_elem_offset.get() == buffer->elem_offset.get()) {
            return buffer;
        }

        return tir::Buffer(
            new_data,
            buffer->dtype,
            shape_changed ? new_shape : buffer->shape,
            strides_changed ? new_strides : buffer->strides,
            new_elem_offset,
            buffer->name,
            buffer->data_alignment,
            buffer->offset_factor);
    }

    tir::BufferRegion MutateBufferRegion(const tir::BufferRegion& region) {
        if (!region.defined()) return region;
        tir::Buffer new_buffer = MutateBuffer(region->buffer);
        Array<tir::Range> new_region;
        bool region_changed = false;
        for (const auto& r : region->region) {
            tir::Range nr = MutateRange(r);
            if (nr.get() != r.get()) region_changed = true;
            new_region.push_back(nr);
        }
        if (new_buffer.get() == region->buffer.get() && !region_changed) {
            return region;
        }
        return tir::BufferRegion(new_buffer, region_changed ? new_region : region->region);
    }
};

} // namespace kxc
