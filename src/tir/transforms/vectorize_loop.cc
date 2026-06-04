/*! \file src/tir/transforms/vectorize_loop.cc
 * \brief 实现 TIR 优化 pass 和 pipeline。
 */

#include "tir/transforms/vectorize_loop.h"

#include <cstdint>

#include "base/pass.h"
#include "tir/pass_utils.h"

namespace kxc {
namespace tir {

namespace {

class VectorizeLoopRewriter : public TIRPass {
protected:
    Stmt VisitFor(const ForNode* op, const Stmt& ref) override {
        Stmt rewritten = TIRPass::VisitFor(op, ref);
        const auto* for_node = rewritten.As<ForNode>();
        if (!for_node || for_node->for_type != ForType::Serial) {
            return rewritten;
        }

        int64_t extent = 0;
        if (!pass_utils::TryGetConstInt64(for_node->extent, &extent) || extent <= 0 ||
            (extent % 4) != 0) {
            return rewritten;
        }
        if (pass_utils::ContainsFor(for_node->body)) {
            return rewritten;
        }
        if (!pass_utils::IsSimpleStraightLineStmt(for_node->body)) {
            return rewritten;
        }

        return For(for_node->loop_var, for_node->min, for_node->extent, ForType::Vectorized,
                   for_node->body);
    }
};

}  // namespace

PrimFunc VectorizeLoopPass(const PrimFunc& func) {
    VectorizeLoopRewriter pass;
    return pass.Mutate(func);
}

}  // namespace tir
}  // namespace kxc

