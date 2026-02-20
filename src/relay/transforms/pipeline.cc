#include "relay/transforms/pipeline.h"

#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "base/packedfunc.h"
#include "base/registry.h"
#include "relay/transforms/annotate_memory_scope.h"
#include "relay/transforms/canonicalize_cast.h"
#include "relay/transforms/capture_post_dfs_index_in_spans.h"
#include "relay/transforms/eliminate_dead_let.h"
#include "relay/transforms/eliminate_common_subexpr.h"
#include "relay/transforms/fold_constant.h"
#include "relay/transforms/fold_tuple_get_item.h"
#include "relay/transforms/remove_standalone_reshapes.h"
#include "relay/transforms/simplify_expr.h"

namespace kxc {
namespace relay {

namespace {

using RelayPassFunc = std::function<Function(const Function&)>;

const std::unordered_map<std::string, RelayPassFunc>& GetRelayPassTable() {
    static const std::unordered_map<std::string, RelayPassFunc> table = {
        {"fold_tuple_get_item", FoldTupleGetItemPass},
        {"fold_constant", FoldConstantPass},
        {"simplify_expr", SimplifyExprPass},
        {"canonicalize_cast", CanonicalizeCastPass},
        {"remove_standalone_reshapes", RemoveStandaloneReshapesPass},
        {"eliminate_common_subexpr", EliminateCommonSubexprPass},
        {"eliminate_dead_let", EliminateDeadLetPass},
        {"annotate_memory_scope", AnnotateMemoryScopePass},
        {"capture_post_dfs_index_in_spans", CapturePostDfsIndexInSpansPass},
    };
    return table;
}

Array<String> GetDefaultPassOrder() {
    return {String("fold_tuple_get_item"), String("fold_constant"), String("simplify_expr"),
            String("canonicalize_cast"), String("remove_standalone_reshapes"),
            String("eliminate_common_subexpr"), String("eliminate_dead_let"),
            String("annotate_memory_scope"),
            String("capture_post_dfs_index_in_spans")};
}

Function RunSinglePass(const Function& func, const std::string& pass_name) {
    const auto& pass_table = GetRelayPassTable();
    auto it = pass_table.find(pass_name);
    if (it == pass_table.end()) {
        throw std::runtime_error("Unknown Relay pass in pipeline: " + pass_name);
    }
    return it->second(func);
}

}  // namespace

Function RunRelayPassPipeline(const Function& func, const Array<String>& pass_names) {
    if (!func.defined()) {
        throw std::runtime_error("RunRelayPassPipeline expects a defined Function");
    }

    Function current = func;
    for (const auto& pass_name_obj : pass_names) {
        const std::string pass_name = pass_name_obj;
        if (pass_name == "optimize_default") {
            for (const auto& default_name : GetDefaultPassOrder()) {
                current = RunSinglePass(current, static_cast<std::string>(default_name));
            }
            continue;
        }
        current = RunSinglePass(current, pass_name);
    }
    return current;
}

KXC_REGISTER_GLOBAL("kxc.relay.transform.run_pipeline")
    .set_body(ToPackedFunc([](Function func, Array<String> pass_names) -> ObjectRef {
        return ObjectRef(RunRelayPassPipeline(func, pass_names));
    }));

KXC_REGISTER_GLOBAL("kxc.relay.transform.fold_tuple_get_item")
    .set_body(ToPackedFunc([](Function func) -> ObjectRef {
        return ObjectRef(FoldTupleGetItemPass(func));
    }));

KXC_REGISTER_GLOBAL("kxc.relay.transform.simplify_expr")
    .set_body(ToPackedFunc([](Function func) -> ObjectRef {
        return ObjectRef(SimplifyExprPass(func));
    }));

KXC_REGISTER_GLOBAL("kxc.relay.transform.eliminate_dead_let")
    .set_body(ToPackedFunc([](Function func) -> ObjectRef {
        return ObjectRef(EliminateDeadLetPass(func));
    }));

KXC_REGISTER_GLOBAL("kxc.relay.transform.fold_constant")
    .set_body(ToPackedFunc([](Function func) -> ObjectRef {
        return ObjectRef(FoldConstantPass(func));
    }));

KXC_REGISTER_GLOBAL("kxc.relay.transform.canonicalize_cast")
    .set_body(ToPackedFunc([](Function func) -> ObjectRef {
        return ObjectRef(CanonicalizeCastPass(func));
    }));

KXC_REGISTER_GLOBAL("kxc.relay.transform.remove_standalone_reshapes")
    .set_body(ToPackedFunc([](Function func) -> ObjectRef {
        return ObjectRef(RemoveStandaloneReshapesPass(func));
    }));

KXC_REGISTER_GLOBAL("kxc.relay.transform.eliminate_common_subexpr")
    .set_body(ToPackedFunc([](Function func) -> ObjectRef {
        return ObjectRef(EliminateCommonSubexprPass(func));
    }));

KXC_REGISTER_GLOBAL("kxc.relay.transform.capture_post_dfs_index_in_spans")
    .set_body(ToPackedFunc([](Function func) -> ObjectRef {
        return ObjectRef(CapturePostDfsIndexInSpansPass(func));
    }));

KXC_REGISTER_GLOBAL("kxc.relay.transform.annotate_memory_scope")
    .set_body(ToPackedFunc([](Function func) -> ObjectRef {
        return ObjectRef(AnnotateMemoryScopePass(func));
    }));

}  // namespace relay
}  // namespace kxc
