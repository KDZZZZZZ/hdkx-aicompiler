/*! \file src/tir/transforms/pipeline.cc
 * \brief 实现 TIR 优化 pass 和 pipeline。
 */

#include "tir/transforms/pipeline.h"

#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "base/profiling.h"
#include "base/packedfunc.h"
#include "base/registry.h"
#include "tir/pass/print_ir.h"
#include "tir/transforms/convert_for_loops_serial.h"
#include "tir/transforms/fold_constant.h"
#include "tir/transforms/force_narrow_index_to_i32.h"
#include "tir/transforms/loop_partition.h"
#include "tir/transforms/remove_no_op.h"
#include "tir/transforms/simplify_expr.h"
#include "tir/transforms/unroll_loop.h"
#include "tir/transforms/vectorize_loop.h"

namespace kxc {
namespace tir {

namespace {

using TIRPassFunc = std::function<PrimFunc(const PrimFunc&)>;

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

const std::unordered_map<std::string, TIRPassFunc>& GetTIRPassTable() {
    static const std::unordered_map<std::string, TIRPassFunc> table = {
        {"fold_constant", FoldConstantPass},
        {"simplify_expr", SimplifyExprPass},
        {"force_narrow_index_to_i32", ForceNarrowIndexToI32Pass},
        {"loop_partition", LoopPartitionPass},
        {"unroll_loop", UnrollLoopPass},
        {"vectorize_loop", VectorizeLoopPass},
        {"remove_no_op", RemoveNoOpPass},
        {"convert_for_loops_serial", ConvertForLoopsSerialPass},
    };
    return table;
}

Array<String> GetDefaultPassOrder() {
    return {String("fold_constant"), String("simplify_expr"), String("force_narrow_index_to_i32"),
            String("convert_for_loops_serial"), String("loop_partition"),
            String("unroll_loop"), String("vectorize_loop"), String("remove_no_op")};
}

PrimFunc RunSinglePass(const PrimFunc& func, const std::string& pass_name) {
    const auto& pass_table = GetTIRPassTable();
    auto it = pass_table.find(pass_name);
    if (it == pass_table.end()) {
        throw std::runtime_error("Unknown TIR pass in pipeline: " + pass_name);
    }
    return it->second(func);
}

}  // namespace

PrimFunc RunTIRPassPipeline(const PrimFunc& func, const Array<String>& pass_names) {
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

}  // namespace tir
}  // namespace kxc
