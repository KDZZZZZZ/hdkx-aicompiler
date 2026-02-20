#include "relay/transforms/fold_tuple_get_item.h"

#include "base/pass.h"
#include "relay/pass_utils.h"

namespace kxc {
namespace relay {

namespace {

class FoldTupleGetItemRewriter : public RelayPass {
protected:
    Expr VisitTupleGetItem(const TupleGetItemNode* op, const Expr& ref) override {
        Expr rewritten = RelayPass::VisitTupleGetItem(op, ref);
        const auto* tuple_get = rewritten.As<TupleGetItemNode>();
        if (!tuple_get) {
            return rewritten;
        }
        const auto* tuple = tuple_get->tuple.As<TupleNode>();
        if (!tuple) {
            return rewritten;
        }
        if (tuple_get->index < 0 ||
            static_cast<size_t>(tuple_get->index) >= tuple->fields.size()) {
            return rewritten;
        }
        return pass_utils::CopyVirtualDevice(rewritten, tuple->fields[tuple_get->index]);
    }
};

}  // namespace

Function FoldTupleGetItemPass(const Function& func) {
    FoldTupleGetItemRewriter pass;
    return pass.Mutate(func);
}

}  // namespace relay
}  // namespace kxc
