/*! \file src/tir/transforms/convert_for_loops_serial.cc
 * \brief 实现 TIR 优化 pass 和 pipeline。
 */

#include "kxc/tir/transforms/convert_for_loops_serial.h"

#include "kxc/tir/visitor.h"

namespace kxc {
namespace tir {

namespace {

class ConvertForLoopsSerialRewriter : public TIRPass {
protected:
    Stmt VisitFor(const ForNode* op, const Stmt& ref) override {
        Stmt rewritten = TIRPass::VisitFor(op, ref);
        const auto* for_node = rewritten.As<ForNode>();
        if (!for_node) {
            return rewritten;
        }
        if (for_node->for_type == ForType::Serial) {
            return rewritten;
        }
        return For(for_node->loop_var, for_node->min, for_node->extent, ForType::Serial,
                   for_node->body);
    }
};

}  // namespace

PrimFunc ConvertForLoopsSerialPass(const PrimFunc& func) {
    ConvertForLoopsSerialRewriter pass;
    return pass.Mutate(func);
}

}  // namespace tir
}  // namespace kxc
