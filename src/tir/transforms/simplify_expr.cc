/*! \file src/tir/transforms/simplify_expr.cc
 * \brief 实现 TIR 优化 pass 和 pipeline。
 */

#include "tir/transforms/simplify_expr.h"

#include "base/pass.h"
#include "tir/pass_utils.h"

namespace kxc {
namespace tir {

namespace {

class SimplifyExprRewriter : public TIRPass {
protected:
    PrimExpr VisitAdd(const AddNode* op, const PrimExpr& ref) override {
        PrimExpr rewritten = TIRPass::VisitAdd(op, ref);
        const auto* add = rewritten.As<AddNode>();
        if (!add) {
            return rewritten;
        }
        if (pass_utils::IsConstZero(add->b)) {
            return add->a;
        }
        if (pass_utils::IsConstZero(add->a)) {
            return add->b;
        }
        return rewritten;
    }

    PrimExpr VisitSub(const SubNode* op, const PrimExpr& ref) override {
        PrimExpr rewritten = TIRPass::VisitSub(op, ref);
        const auto* sub = rewritten.As<SubNode>();
        if (!sub) {
            return rewritten;
        }
        if (pass_utils::IsConstZero(sub->b)) {
            return sub->a;
        }
        return rewritten;
    }

    PrimExpr VisitMul(const MulNode* op, const PrimExpr& ref) override {
        PrimExpr rewritten = TIRPass::VisitMul(op, ref);
        const auto* mul = rewritten.As<MulNode>();
        if (!mul) {
            return rewritten;
        }
        if (pass_utils::IsConstOne(mul->b)) {
            return mul->a;
        }
        if (pass_utils::IsConstOne(mul->a)) {
            return mul->b;
        }
        return rewritten;
    }

    PrimExpr VisitDiv(const DivNode* op, const PrimExpr& ref) override {
        PrimExpr rewritten = TIRPass::VisitDiv(op, ref);
        const auto* div = rewritten.As<DivNode>();
        if (!div) {
            return rewritten;
        }
        if (pass_utils::IsConstOne(div->b)) {
            return div->a;
        }
        return rewritten;
    }
};

}  // namespace

PrimFunc SimplifyExprPass(const PrimFunc& func) {
    SimplifyExprRewriter pass;
    return pass.Mutate(func);
}

}  // namespace tir
}  // namespace kxc
