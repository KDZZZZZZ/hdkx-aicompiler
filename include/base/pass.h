/*! \file include/base/pass.h
 * \brief 定义基础对象系统、容器、设备、NDArray、Target、PassContext 和 profiling 公共类型。
 */

#pragma once

#include <stdexcept>
#include <memory>
#include <string>
#include <vector>

#include "base/container.h"
#include "base/disco_placement.h"
#include "relay/op.h"
#include "relay/relay.h"
#include "tir/stmt.h"

namespace kxc {

/*! \brief 编译 pass 的上下文，携带 target、虚拟设备和 Disco placement 信息。 */
class PassContext {
public:
    PassContext() = default;

    bool defined() const { return defined_; }
    bool is_multi_device() const { return is_multi_device_; }
    const VirtualDevice& primary_virtual_device() const { return primary_virtual_device_; }
    const Array<VirtualDevice>& virtual_devices() const { return virtual_devices_; }
    const Target& default_target() const { return default_target_; }
    const ObjectRef& default_device_obj() const { return default_device_obj_; }
    const DiscoPlacement& disco_placement() const { return disco_placement_; }
    bool has_disco_placement() const { return disco_placement_.defined(); }

    std::string ToString() const;

    /*! \brief 返回线程本地当前 PassContext；未设置时返回默认上下文。 */
    static PassContext Current();
    /*! \brief 从 Relay 表达式推导 PassContext。 */
    static PassContext FromRelay(const Expr& expr);
    /*! \brief 从 Relay 函数推导 PassContext。 */
    static PassContext FromRelay(const Function& func);
    /*! \brief 从 TIR PrimFunc 推导 PassContext。 */
    static PassContext FromTIR(const tir::PrimFunc& func);
    /*! \brief 在已有上下文上附加 Disco placement。 */
    static PassContext WithDiscoPlacement(const PassContext& base_ctx,
                                          const DiscoPlacement& disco_placement);

    /*! \brief RAII 作用域，临时安装当前线程的 PassContext。 */
    class Scope {
    public:
        explicit Scope(const PassContext& pass_ctx);
        ~Scope();

        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;

    private:
        bool active_{false};
        std::unique_ptr<PassContext> previous_;
    };

private:
    bool defined_{false};
    bool is_multi_device_{false};
    VirtualDevice primary_virtual_device_;
    Array<VirtualDevice> virtual_devices_;
    Target default_target_;
    ObjectRef default_device_obj_;
    DiscoPlacement disco_placement_;

    static PassContext BuildFromVirtualDevices(const Array<VirtualDevice>& virtual_devices);
    static void SetCurrent(const PassContext& pass_ctx);
    static void ClearCurrent();
};

/*! \brief 将 PassContext 的关键字段附加到 attrs 中。 */
Map<String, ObjectRef> AttachPassContextAttrs(const Map<String, ObjectRef>& attrs,
                                              const PassContext& pass_ctx);

/*! \brief 从 Relay 表达式构建 Disco placement pass 结果。 */
PassContext BuildDiscoPlacementPass(const Expr& expr);
/*! \brief 从 Relay 函数构建 Disco placement pass 结果。 */
PassContext BuildDiscoPlacementPass(const Function& func);

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
    Expr VisitLet(const LetNode* op, const Expr& ref) override;
    Expr VisitTuple(const TupleNode* op, const Expr& ref) override;
    Expr VisitTupleGetItem(const TupleGetItemNode* op, const Expr& ref) override;

private:
    Var MutateToVar(const Var& var);
};

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

}  // namespace kxc
