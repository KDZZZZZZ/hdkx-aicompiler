/*! \file src/tir/transforms/pipeline.cc
 * \brief Implements TIR pass pipeline integration and PassSpec binding.
 */

#include "kxc/tir/transforms/pipeline.h"

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
#include "kxc/tir/pass/print_ir.h"
#include "kxc/tir/transforms/bind_cuda_threads.h"
#include "kxc/tir/transforms/convert_for_loops_serial.h"
#include "kxc/tir/transforms/fold_constant.h"
#include "kxc/tir/transforms/force_narrow_index_to_i32.h"
#include "kxc/tir/transforms/loop_partition.h"
#include "kxc/tir/transforms/remove_no_op.h"
#include "kxc/tir/transforms/simplify_expr.h"
#include "kxc/tir/transforms/unroll_loop.h"
#include "kxc/tir/transforms/vectorize_loop.h"
#include "kxc/tir/visitor.h"

namespace kxc {
namespace tir {

namespace {

using TIRPassFunc = std::function<PrimFunc(const PrimFunc&)>;

struct TIRPassBinding {
    const char* name;
    const char* implementation_key;
    TIRPassFunc function;
    bool in_default_pipeline;
    bool idempotent;
    bool target_dependent;
    const char* phase;
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

std::string PrimFuncToText(const PrimFunc& func) {
    std::ostringstream os;
    tir::pass::DumpPrimFunc(func, os);
    return os.str();
}

PrimFunc RunSinglePass(const PrimFunc& func, const std::string& pass_name);

PrimFunc BindCudaThreadsPipelinePass(const PrimFunc& func) {
    PassContext pass_ctx = PassContext::Current();
    if (!pass_ctx.defined()) pass_ctx = PassContextFromTIR(func);
    if (!pass_ctx.defined() || !pass_ctx.default_target().defined()) {
        throw std::runtime_error("bind_cuda_threads requires a Target in PassContext");
    }
    return BindCudaThreads(func, pass_ctx.default_target()).prim_func();
}

const std::vector<TIRPassBinding>& GetTIRPassBindings() {
    static const std::vector<TIRPassBinding> bindings = {
        {"fold_constant", "kxc.tir.transform.fold_constant", FoldConstantPass, true, true, false,
         "tir_optimize"},
        {"simplify_expr", "kxc.tir.transform.simplify_expr", SimplifyExprPass, true, true, false,
         "tir_optimize"},
        {"force_narrow_index_to_i32", "kxc.tir.transform.force_narrow_index_to_i32",
         ForceNarrowIndexToI32Pass, true, true, false, "tir_optimize"},
        {"loop_partition", "kxc.tir.transform.loop_partition", LoopPartitionPass, true, true,
         false, "tir_optimize"},
        {"unroll_loop", "kxc.tir.transform.unroll_loop", UnrollLoopPass, true, true, false,
         "tir_optimize"},
        {"vectorize_loop", "kxc.tir.transform.vectorize_loop", VectorizeLoopPass, true, true,
         false, "tir_optimize"},
        {"remove_no_op", "kxc.tir.transform.remove_no_op", RemoveNoOpPass, true, true, false,
         "tir_optimize"},
        {"convert_for_loops_serial", "kxc.tir.transform.convert_for_loops_serial",
         ConvertForLoopsSerialPass, true, true, false, "tir_optimize"},
        {"bind_cuda_threads", "kxc.tir.transform.bind_cuda_threads", BindCudaThreadsPipelinePass,
         false, false, true, "tir_schedule"},
    };
    return bindings;
}

PassSpec MakeTIRPassSpec(const TIRPassBinding& binding) {
    PassSpec spec;
    spec.name = String(binding.name);
    spec.schema_version = 1;
    spec.dialect = IRDialect::kTIR;
    spec.scope = PassScope::kPrimFunc;
    spec.phase = String(binding.phase);
    spec.opt_level = binding.in_default_pipeline ? 1 : 3;
    spec.may_change_ir = true;
    spec.deterministic = true;
    spec.idempotent = binding.idempotent;
    spec.thread_safe = false;
    spec.target_dependent = binding.target_dependent;
    spec.implementation_key = String(binding.implementation_key);
    return spec;
}

const std::unordered_map<std::string, TIRPassFunc>& GetTIRImplementationTable() {
    static const std::unordered_map<std::string, TIRPassFunc> table = [] {
        std::unordered_map<std::string, TIRPassFunc> out;
        for (const TIRPassBinding& binding : GetTIRPassBindings()) {
            out.emplace(binding.implementation_key, binding.function);
        }
        return out;
    }();
    return table;
}

void EnsureTIRPassSpecsRegistered() {
    static std::once_flag once;
    std::call_once(once, [] {
        Array<PassSpec> specs;
        for (const TIRPassBinding& binding : GetTIRPassBindings()) {
            PassSpec spec = MakeTIRPassSpec(binding);
            specs.push_back(spec);
            PassRegistry::Global().Register(std::move(spec));
        }
        ValidatePassSpecs(specs);
    });
}

Array<String> GetDefaultPassOrder() {
    return {String("fold_constant"), String("simplify_expr"), String("force_narrow_index_to_i32"),
            String("convert_for_loops_serial"), String("loop_partition"),
            String("unroll_loop"), String("vectorize_loop"), String("remove_no_op")};
}

PrimFunc RunInstrumentedPass(const PrimFunc& func, const std::string& pass_name) {
    auto profile_context = profiling::CurrentContext();
    profiling::EventSpec spec;
    spec.component = "tir_pass";
    spec.event_type = "run_pass";
    spec.pass_name = pass_name;
    profiling::ScopedSpan span(profile_context, std::move(spec));

    const std::string before_text = PrimFuncToText(func);
    const std::string before_hash = profiling::HashText(before_text);
    span.AddField("ir_before_hash", before_hash);
    span.AddMetric("ir_before_bytes", static_cast<double>(before_text.size()));

    try {
        PrimFunc updated = RunSinglePass(func, pass_name);
        const std::string after_text = PrimFuncToText(updated);
        const std::string after_hash = profiling::HashText(after_text);
        const bool changed = before_hash != after_hash;
        span.AddField("ir_after_hash", after_hash);
        span.AddField("ir_changed", changed ? "true" : "false");
        span.AddMetric("ir_after_bytes", static_cast<double>(after_text.size()));

        if (profiling::ShouldCaptureIR(profile_context, changed, false)) {
            const std::string prefix = profiling::CurrentRunId() + "/tir/" +
                                       SanitizeArtifactName(pass_name);
            profile_context->WriteArtifact(prefix + ".before.tir.txt", before_text);
            profile_context->WriteArtifact(prefix + ".after.tir.txt", after_text);
        }
        return updated;
    } catch (const std::exception& e) {
        span.SetStatus("error");
        span.SetMessage(e.what());
        if (profile_context) {
            const std::string prefix = profiling::CurrentRunId() + "/tir/" +
                                       SanitizeArtifactName(pass_name);
            if (profiling::ShouldCaptureIR(profile_context, true, true)) {
                profile_context->WriteArtifact(prefix + ".failed.before.tir.txt", before_text);
            }
            profile_context->RecordLog(profiling::LogSeverity::kError, "tir_pass", e.what(),
                                       profiling::MakeFields({
                                           {"pass_name", pass_name},
                                           {"ir_before_hash", before_hash},
                                       }));
        }
        throw;
    }
}

PrimFunc RunSinglePass(const PrimFunc& func, const std::string& pass_name) {
    EnsureTIRPassSpecsRegistered();
    const PassSpec& spec = PassRegistry::Global().Get(IRDialect::kTIR, String(pass_name));
    const std::string phase = static_cast<std::string>(spec.phase);
    ValidatePassSpecForPipeline(spec, IRDialect::kTIR, PassScope::kPrimFunc, phase);

    const std::string implementation_key = static_cast<std::string>(spec.implementation_key);
    const auto& pass_table = GetTIRImplementationTable();
    auto it = pass_table.find(implementation_key);
    if (it == pass_table.end()) {
        throw std::runtime_error("TIR pass " + pass_name +
                                 " has no implementation binding: " + implementation_key);
    }
    return it->second(func);
}

}  // namespace

PrimFunc RunTIRPassPipeline(const PrimFunc& func, const Array<String>& pass_names) {
    EnsureTIRPassSpecsRegistered();
    if (!func.defined()) {
        throw std::runtime_error("RunTIRPassPipeline expects a defined PrimFunc");
    }

    profiling::EventSpec pipeline_spec;
    pipeline_spec.component = "tir_pipeline";
    pipeline_spec.event_type = "run_pipeline";
    profiling::ScopedSpan pipeline_span(profiling::CurrentContext(), std::move(pipeline_spec));

    PrimFunc current = func;
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

Array<String> TIRDefaultPassOrder() {
    return GetDefaultPassOrder();
}

Array<PassSpec> TIRRegisteredPassSpecs() {
    EnsureTIRPassSpecsRegistered();
    std::vector<PassSpec> specs;
    for (const TIRPassBinding& binding : GetTIRPassBindings()) {
        specs.push_back(PassRegistry::Global().Get(IRDialect::kTIR, String(binding.name)));
    }
    return Array<PassSpec>(std::move(specs));
}

KXC_REGISTER_GLOBAL("kxc.tir.transform.run_pipeline")
    .set_body(ToPackedFunc([](ObjectRef func_ref, ObjectRef pass_names_ref) -> ObjectRef {
        PrimFunc func(func_ref.get());
        Array<String> pass_names(pass_names_ref);
        return ObjectRef(RunTIRPassPipeline(func, pass_names));
    }));

KXC_REGISTER_GLOBAL("kxc.tir.transform.simplify_expr")
    .set_body(ToPackedFunc([](ObjectRef func_ref) -> ObjectRef {
        PrimFunc func(func_ref.get());
        return ObjectRef(SimplifyExprPass(func));
    }));

KXC_REGISTER_GLOBAL("kxc.tir.transform.fold_constant")
    .set_body(ToPackedFunc([](ObjectRef func_ref) -> ObjectRef {
        PrimFunc func(func_ref.get());
        return ObjectRef(FoldConstantPass(func));
    }));

KXC_REGISTER_GLOBAL("kxc.tir.transform.remove_no_op")
    .set_body(ToPackedFunc([](ObjectRef func_ref) -> ObjectRef {
        PrimFunc func(func_ref.get());
        return ObjectRef(RemoveNoOpPass(func));
    }));

KXC_REGISTER_GLOBAL("kxc.tir.transform.convert_for_loops_serial")
    .set_body(ToPackedFunc([](ObjectRef func_ref) -> ObjectRef {
        PrimFunc func(func_ref.get());
        return ObjectRef(ConvertForLoopsSerialPass(func));
    }));

KXC_REGISTER_GLOBAL("kxc.tir.transform.force_narrow_index_to_i32")
    .set_body(ToPackedFunc([](ObjectRef func_ref) -> ObjectRef {
        PrimFunc func(func_ref.get());
        return ObjectRef(ForceNarrowIndexToI32Pass(func));
    }));

KXC_REGISTER_GLOBAL("kxc.tir.transform.loop_partition")
    .set_body(ToPackedFunc([](ObjectRef func_ref) -> ObjectRef {
        PrimFunc func(func_ref.get());
        return ObjectRef(LoopPartitionPass(func));
    }));

KXC_REGISTER_GLOBAL("kxc.tir.transform.unroll_loop")
    .set_body(ToPackedFunc([](ObjectRef func_ref) -> ObjectRef {
        PrimFunc func(func_ref.get());
        return ObjectRef(UnrollLoopPass(func));
    }));

KXC_REGISTER_GLOBAL("kxc.tir.transform.vectorize_loop")
    .set_body(ToPackedFunc([](ObjectRef func_ref) -> ObjectRef {
        PrimFunc func(func_ref.get());
        return ObjectRef(VectorizeLoopPass(func));
    }));

KXC_REGISTER_GLOBAL("kxc.tir.transform.bind_cuda_threads")
    .set_body(ToPackedFunc([](ObjectRef func_ref) -> ObjectRef {
        PrimFunc func(func_ref.get());
        return ObjectRef(BindCudaThreadsPipelinePass(func));
    }));

}  // namespace tir
}  // namespace kxc

namespace kxc::builtin_anchor {
void TirPasses() {}
}  // namespace kxc::builtin_anchor
