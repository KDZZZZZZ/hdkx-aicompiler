/*! \file src/relay/transforms/pipeline.cc
 * \brief Implements Relay pass pipeline integration and PassSpec binding.
 */

#include "kxc/relay/transforms/pipeline.h"

#include <functional>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kxc/ffi/packed_func.h"
#include "kxc/ffi/registration.h"
#include "kxc/pass/pass.h"
#include "kxc/profiling/profiling.h"
#include "kxc/relay/pass/print_ir.h"
#include "kxc/relay/transforms/annotate_memory_scope.h"
#include "kxc/relay/transforms/canonicalize_cast.h"
#include "kxc/relay/transforms/capture_post_dfs_index_in_spans.h"
#include "kxc/relay/transforms/eliminate_common_subexpr.h"
#include "kxc/relay/transforms/eliminate_dead_let.h"
#include "kxc/relay/transforms/fold_constant.h"
#include "kxc/relay/transforms/fold_tuple_get_item.h"
#include "kxc/relay/transforms/infer_type.h"
#include "kxc/relay/transforms/remove_standalone_reshapes.h"
#include "kxc/relay/transforms/simplify_expr.h"

namespace kxc {
namespace relay {

namespace {

using RelayPassFunc = std::function<Function(const Function&)>;

struct RelayPassBinding {
    const char* name;
    const char* implementation_key;
    RelayPassFunc function;
    bool in_default_pipeline;
    bool idempotent;
};

std::string SanitizeArtifactName(const std::string& pass_name) {
    std::string out;
    out.reserve(pass_name.size());
    for (char ch : pass_name) {
        if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
            (ch >= '0' && ch <= '9') || ch == '_' || ch == '-') {
            out.push_back(ch);
        } else {
            out.push_back('_');
        }
    }
    return out;
}

Function RunSinglePass(const Function& func, const std::string& pass_name);

const std::vector<RelayPassBinding>& GetRelayPassBindings() {
    static const std::vector<RelayPassBinding> bindings = {
        {"fold_tuple_get_item", "kxc.relay.transform.fold_tuple_get_item", FoldTupleGetItemPass,
         true, true},
        {"fold_constant", "kxc.relay.transform.fold_constant", FoldConstantPass, true, true},
        {"simplify_expr", "kxc.relay.transform.simplify_expr", SimplifyExprPass, true, true},
        {"canonicalize_cast", "kxc.relay.transform.canonicalize_cast", CanonicalizeCastPass, true,
         true},
        {"remove_standalone_reshapes", "kxc.relay.transform.remove_standalone_reshapes",
         RemoveStandaloneReshapesPass, true, true},
        {"eliminate_common_subexpr", "kxc.relay.transform.eliminate_common_subexpr",
         EliminateCommonSubexprPass, false, true},
        {"eliminate_dead_let", "kxc.relay.transform.eliminate_dead_let", EliminateDeadLetPass,
         true, true},
        {"annotate_memory_scope", "kxc.relay.transform.annotate_memory_scope",
         AnnotateMemoryScopePass, true, true},
        {"capture_post_dfs_index_in_spans",
         "kxc.relay.transform.capture_post_dfs_index_in_spans",
         CapturePostDfsIndexInSpansPass, true, true},
        {"infer_type", "kxc.relay.transform.infer_type", InferTypePass, true, true},
    };
    return bindings;
}

PassSpec MakeRelayPassSpec(const RelayPassBinding& binding) {
    PassSpec spec;
    spec.name = String(binding.name);
    spec.schema_version = 1;
    spec.dialect = IRDialect::kRelay;
    spec.scope = PassScope::kGraph;
    spec.phase = String("relay_optimize");
    spec.opt_level = binding.in_default_pipeline ? 1 : 3;
    spec.produced_invariants = binding.name == std::string("infer_type")
                                   ? Array<String>{String("checked_type")}
                                   : Array<String>{};
    spec.may_change_ir = true;
    spec.deterministic = true;
    spec.idempotent = binding.idempotent;
    spec.thread_safe = false;
    spec.target_dependent = false;
    spec.implementation_key = String(binding.implementation_key);
    return spec;
}

const std::unordered_map<std::string, RelayPassFunc>& GetRelayImplementationTable() {
    static const std::unordered_map<std::string, RelayPassFunc> table = [] {
        std::unordered_map<std::string, RelayPassFunc> out;
        for (const RelayPassBinding& binding : GetRelayPassBindings()) {
            out.emplace(binding.implementation_key, binding.function);
        }
        return out;
    }();
    return table;
}

void EnsureRelayPassSpecsRegistered() {
    static std::once_flag once;
    std::call_once(once, [] {
        Array<PassSpec> specs;
        for (const RelayPassBinding& binding : GetRelayPassBindings()) {
            PassSpec spec = MakeRelayPassSpec(binding);
            specs.push_back(spec);
            PassRegistry::Global().Register(std::move(spec));
        }
        ValidatePassSpecs(specs);
    });
}

Array<String> GetDefaultPassOrder() {
    return {String("fold_tuple_get_item"), String("fold_constant"), String("simplify_expr"),
            String("canonicalize_cast"), String("remove_standalone_reshapes"),
            String("eliminate_dead_let"), String("annotate_memory_scope"),
            String("capture_post_dfs_index_in_spans"), String("infer_type")};
}

Function RunInstrumentedPass(const Function& func, const std::string& pass_name) {
    auto profile_context = profiling::CurrentContext();
    profiling::EventSpec spec;
    spec.component = "relay_pass";
    spec.event_type = "run_pass";
    spec.pass_name = pass_name;
    profiling::ScopedSpan span(profile_context, std::move(spec));

    const std::string before_text = relay::pass::ToText(func);
    const std::string before_hash = profiling::HashText(before_text);
    span.AddField("ir_before_hash", before_hash);
    span.AddMetric("ir_before_bytes", static_cast<double>(before_text.size()));

    try {
        Function updated = RunSinglePass(func, pass_name);
        const std::string after_text = relay::pass::ToText(updated);
        const std::string after_hash = profiling::HashText(after_text);
        const bool changed = before_hash != after_hash;
        span.AddField("ir_after_hash", after_hash);
        span.AddField("ir_changed", changed ? "true" : "false");
        span.AddMetric("ir_after_bytes", static_cast<double>(after_text.size()));

        if (profiling::ShouldCaptureIR(profile_context, changed, false)) {
            const std::string prefix = profiling::CurrentRunId() + "/relay/" +
                                       SanitizeArtifactName(pass_name);
            profile_context->WriteArtifact(prefix + ".before.relay.txt", before_text);
            profile_context->WriteArtifact(prefix + ".after.relay.txt", after_text);
        }
        return updated;
    } catch (const std::exception& e) {
        span.SetStatus("error");
        span.SetMessage(e.what());
        if (profile_context) {
            const std::string prefix = profiling::CurrentRunId() + "/relay/" +
                                       SanitizeArtifactName(pass_name);
            if (profiling::ShouldCaptureIR(profile_context, true, true)) {
                profile_context->WriteArtifact(prefix + ".failed.before.relay.txt", before_text);
            }
            profile_context->RecordLog(profiling::LogSeverity::kError, "relay_pass", e.what(),
                                       profiling::MakeFields({
                                           {"pass_name", pass_name},
                                           {"ir_before_hash", before_hash},
                                       }));
        }
        throw;
    }
}

Function RunSinglePass(const Function& func, const std::string& pass_name) {
    EnsureRelayPassSpecsRegistered();
    const PassSpec& spec = PassRegistry::Global().Get(IRDialect::kRelay, String(pass_name));
    ValidatePassSpecForPipeline(spec, IRDialect::kRelay, PassScope::kGraph, "relay_optimize");

    const std::string implementation_key = static_cast<std::string>(spec.implementation_key);
    const auto& pass_table = GetRelayImplementationTable();
    auto it = pass_table.find(implementation_key);
    if (it == pass_table.end()) {
        throw std::runtime_error("Relay pass " + pass_name +
                                 " has no implementation binding: " + implementation_key);
    }
    return it->second(func);
}

}  // namespace

Function RunRelayPassPipeline(const Function& func, const Array<String>& pass_names) {
    EnsureRelayPassSpecsRegistered();
    if (!func.defined()) {
        throw std::runtime_error("RunRelayPassPipeline expects a defined Function");
    }

    profiling::EventSpec pipeline_spec;
    pipeline_spec.component = "relay_pipeline";
    pipeline_spec.event_type = "run_pipeline";
    profiling::ScopedSpan pipeline_span(profiling::CurrentContext(), std::move(pipeline_spec));

    Function current = func;
    for (const auto& pass_name_obj : pass_names) {
        const std::string pass_name = pass_name_obj;
        if (pass_name == "optimize_default") {
            for (const auto& default_name : GetDefaultPassOrder()) {
                current = RunInstrumentedPass(current, static_cast<std::string>(default_name));
            }
            continue;
        }
        current = RunInstrumentedPass(current, pass_name);
    }
    return current;
}

Array<String> RelayDefaultPassOrder() {
    return GetDefaultPassOrder();
}

Array<PassSpec> RelayRegisteredPassSpecs() {
    EnsureRelayPassSpecsRegistered();
    std::vector<PassSpec> specs;
    for (const RelayPassBinding& binding : GetRelayPassBindings()) {
        specs.push_back(PassRegistry::Global().Get(IRDialect::kRelay, String(binding.name)));
    }
    return Array<PassSpec>(std::move(specs));
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

KXC_REGISTER_GLOBAL("kxc.relay.transform.infer_type")
    .set_body(ToPackedFunc([](Function func) -> ObjectRef {
        return ObjectRef(InferTypePass(func));
    }));

}  // namespace relay
}  // namespace kxc

namespace kxc::builtin_anchor {
void RelayPasses() {}
}  // namespace kxc::builtin_anchor
