/*! \file include/kxc/tir/visitor.h
 * \brief Defines TIR visitors, mutator, and PassContext adapters.
 */

#pragma once

#include "kxc/pass/context.h"
#include "kxc/tir/stmt.h"

namespace kxc {

/*! \brief TIR PrimExpr visitor functor。 */
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

    virtual R VisitExprDefault(const tir::PrimExpr& expr) {
        (void)expr;
        return R();
    }
};

/*! \brief TIR Stmt visitor functor。 */
template <typename R>
class TIRStmtFunctor {
public:
    virtual ~TIRStmtFunctor() = default;

    virtual R VisitStmt(const tir::Stmt& stmt) {
        if (!stmt.defined()) return R();

        if (auto* n = stmt.As<tir::LetStmtNode>()) return VisitLetStmt(n, stmt);
        if (auto* n = stmt.As<tir::StoreNode>()) return VisitStore(n, stmt);
        if (auto* n = stmt.As<tir::ForNode>()) return VisitFor(n, stmt);
        if (auto* n = stmt.As<tir::ThreadBindingNode>()) return VisitThreadBinding(n, stmt);
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
    virtual R VisitThreadBinding(const tir::ThreadBindingNode* op, const tir::Stmt& ref) {
        return VisitStmtDefault(ref);
    }
    virtual R VisitIfThenElse(const tir::IfThenElseNode* op, const tir::Stmt& ref) { return VisitStmtDefault(ref); }
    virtual R VisitAllocate(const tir::AllocateNode* op, const tir::Stmt& ref) { return VisitStmtDefault(ref); }
    virtual R VisitAttrStmt(const tir::AttrStmtNode* op, const tir::Stmt& ref) { return VisitStmtDefault(ref); }
    virtual R VisitBlock(const tir::BlockNode* op, const tir::Stmt& ref) { return VisitStmtDefault(ref); }
    virtual R VisitSeqStmt(const tir::SeqStmtNode* op, const tir::Stmt& ref) { return VisitStmtDefault(ref); }
    virtual R VisitEvaluate(const tir::EvaluateNode* op, const tir::Stmt& ref) { return VisitStmtDefault(ref); }

    virtual R VisitStmtDefault(const tir::Stmt& stmt) {
        (void)stmt;
        return R();
    }
};

/*! \brief TIR mutator，默认递归重建 PrimExpr/Stmt/PrimFunc。 */
class TIRPass : public TIRExprFunctor<tir::PrimExpr>, public TIRStmtFunctor<tir::Stmt> {
public:
    tir::PrimExpr Mutate(const tir::PrimExpr& expr);
    tir::Stmt Mutate(const tir::Stmt& stmt);
    tir::PrimFunc Mutate(const tir::PrimFunc& func);

protected:
    virtual tir::PrimFunc VisitPrimFunc(const tir::PrimFunc& func);

    tir::PrimExpr VisitIntImm(const tir::IntImmNode* op, const tir::PrimExpr& ref) override;
    tir::PrimExpr VisitFloatImm(const tir::FloatImmNode* op, const tir::PrimExpr& ref) override;
    tir::PrimExpr VisitVar(const tir::VarNode* op, const tir::PrimExpr& ref) override;
    tir::PrimExpr VisitAdd(const tir::AddNode* op, const tir::PrimExpr& ref) override;
    tir::PrimExpr VisitSub(const tir::SubNode* op, const tir::PrimExpr& ref) override;
    tir::PrimExpr VisitMul(const tir::MulNode* op, const tir::PrimExpr& ref) override;
    tir::PrimExpr VisitDiv(const tir::DivNode* op, const tir::PrimExpr& ref) override;
    tir::PrimExpr VisitMod(const tir::ModNode* op, const tir::PrimExpr& ref) override;
    tir::PrimExpr VisitMin(const tir::MinNode* op, const tir::PrimExpr& ref) override;
    tir::PrimExpr VisitMax(const tir::MaxNode* op, const tir::PrimExpr& ref) override;
    tir::PrimExpr VisitEQ(const tir::EQNode* op, const tir::PrimExpr& ref) override;
    tir::PrimExpr VisitLT(const tir::LTNode* op, const tir::PrimExpr& ref) override;
    tir::PrimExpr VisitAnd(const tir::AndNode* op, const tir::PrimExpr& ref) override;
    tir::PrimExpr VisitOr(const tir::OrNode* op, const tir::PrimExpr& ref) override;
    tir::PrimExpr VisitNot(const tir::NotNode* op, const tir::PrimExpr& ref) override;
    tir::PrimExpr VisitLoad(const tir::LoadNode* op, const tir::PrimExpr& ref) override;
    tir::PrimExpr VisitCall(const tir::CallNode* op, const tir::PrimExpr& ref) override;
    tir::PrimExpr VisitSelect(const tir::SelectNode* op, const tir::PrimExpr& ref) override;

    tir::Stmt VisitLetStmt(const tir::LetStmtNode* op, const tir::Stmt& ref) override;
    tir::Stmt VisitStore(const tir::StoreNode* op, const tir::Stmt& ref) override;
    tir::Stmt VisitFor(const tir::ForNode* op, const tir::Stmt& ref) override;
    tir::Stmt VisitThreadBinding(const tir::ThreadBindingNode* op,
                                 const tir::Stmt& ref) override;
    tir::Stmt VisitIfThenElse(const tir::IfThenElseNode* op, const tir::Stmt& ref) override;
    tir::Stmt VisitAllocate(const tir::AllocateNode* op, const tir::Stmt& ref) override;
    tir::Stmt VisitAttrStmt(const tir::AttrStmtNode* op, const tir::Stmt& ref) override;
    tir::Stmt VisitBlock(const tir::BlockNode* op, const tir::Stmt& ref) override;
    tir::Stmt VisitSeqStmt(const tir::SeqStmtNode* op, const tir::Stmt& ref) override;
    tir::Stmt VisitEvaluate(const tir::EvaluateNode* op, const tir::Stmt& ref) override;

private:
    tir::Var MutateToVar(const tir::Var& var);
    tir::Range MutateRange(const tir::Range& range);
    tir::IterVar MutateIterVar(const tir::IterVar& iv);
    tir::Buffer MutateBuffer(const tir::Buffer& buffer);
    tir::BufferRegion MutateBufferRegion(const tir::BufferRegion& region);
};

namespace tir {

PassContext PassContextFromTIR(const PrimFunc& func);
Map<String, ObjectRef> AttachPassContextAttrs(
    const Map<String, ObjectRef>& attrs, const PassContext& pass_ctx);

}  // namespace tir
}  // namespace kxc
