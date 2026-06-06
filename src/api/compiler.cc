/*! \file src/api/compiler.cc
 * \brief 实现编译配置默认值、编译入口和 CompiledModule 执行逻辑。
 */

#include "api/compiler.h"

#include <fstream>
#include <stdexcept>

#include "base/profiling.h"
#include "codegen/codegen_c.h"
#include "relay/transforms/infer_type.h"
#include "relay/transforms/lower.h"
#include "relay/transforms/pipeline.h"
#include "runtime/runtime_session.h"
#include "tir/transforms/pipeline.h"

#if KXC_USE_LLVM
#include <llvm/IR/LLVMContext.h>

#include "codegen/codegen_llvm.h"
#include "codegen/llvm_jit.h"
#endif

namespace kxc {
namespace api {

namespace {

std::shared_ptr<profiling::ProfileContext> MaybeCreateProfileContext(const CompileConfig& config) {
    if (profiling::CurrentContext()) {
        return profiling::CurrentContext();
    }
    if (!config.defined() || !config->profile_options.enabled) {
        return nullptr;
    }
    return profiling::ProfileContext::Create(config->profile_options);
}

profiling::EventSpec MakeEvent(const std::string& component, const std::string& event_type) {
    profiling::EventSpec spec;
    spec.component = component;
    spec.event_type = event_type;
    return spec;
}

}  // namespace

CompiledModule Compiler::Compile(Function func, CompileConfig config) {
    auto profile_context = MaybeCreateProfileContext(config);
    const std::string run_id = profile_context ? profile_context->NextRunId("compile") : "";
    profiling::ActivationScope activation(profile_context, run_id);
    profiling::ScopedSpan compile_span(profile_context, MakeEvent("compiler", "compile_module"),
                                       run_id);
    if (config.defined()) {
        compile_span.AddField("compile_mode", std::to_string(static_cast<int>(config->mode)));
        compile_span.AddField("target_kind", config->target.defined() ? config->target->kind : "");
        compile_span.AddField("backend", std::to_string(static_cast<int>(config->backend)));
        compile_span.AddMetric("opt_level", static_cast<double>(config->opt_level));
    }

    if (config->mode == CompileMode::kAdaptive) {
        auto session = std::make_shared<runtime::RuntimeSession>(func, config, profile_context);
        for (const auto& shape : config->suggested_input_shapes) {
            std::vector<std::vector<int64_t>> shapes_vec;
            std::vector<int64_t> single_shape;
            for (const auto& dim : shape) {
                single_shape.push_back(dim);
            }
            shapes_vec.push_back(single_shape);
            session->WarmUp(shapes_vec);
        }
        if (profile_context) {
            profile_context->RecordLog(profiling::LogSeverity::kInfo, "compiler",
                                       "Adaptive session initialized", {}, {}, run_id);
            profile_context->Flush();
        }
        return CompiledModule(config, func, session, profile_context);
    }

    Array<String> relay_passes;
    if (config->mode == CompileMode::kAOT) {
        relay_passes = {String("optimize_default")};
    } else {
        relay_passes = {String("fold_constant"), String("simplify_expr")};
    }
    Function typed_func = relay::InferTypePass(func);
    Function optimized_func = relay::RunRelayPassPipeline(typed_func, relay_passes);
    optimized_func = relay::InferTypePass(optimized_func);
    tir::PrimFunc prim_func = relay::LowerToTIR(optimized_func);

    Array<String> passes;
    if (config->mode == CompileMode::kAOT) {
        passes = {String("optimize_default")};
    } else {
        passes = {String("fold_constant"), String("simplify_expr")};
    }
    prim_func = RunTIRPassPipeline(prim_func, passes);

    codegen::CompiledKernel kernel;

#if KXC_USE_LLVM
    if (config->backend == codegen::CodeGenBackend::kLLVM) {
        profiling::ScopedSpan codegen_span(profile_context, MakeEvent("codegen", "llvm_jit_compile"),
                                           run_id);
        auto llvm_ctx = std::make_unique<llvm::LLVMContext>();
        codegen::CodeGenLLVM codegen(*llvm_ctx);
        codegen.AddFunction(prim_func, "main");
        auto module = codegen.TakeModule();

        codegen::LLVMJITEngine jit;
        kernel = jit.Compile(std::move(module), std::move(llvm_ctx), "main",
                             config->opt_level);
        codegen_span.AddField("kernel_symbol", "main");
        codegen_span.SetMessage("LLVM JIT compilation completed");
        codegen_span.AddMetric("opt_level", static_cast<double>(config->opt_level));
    } else
#endif
    {
        compile_span.SetStatus("error");
        compile_span.SetMessage("Unsupported backend");
        throw std::runtime_error(
            "C backend compilation not fully implemented. Use LLVM backend.");
    }

    if (profile_context) {
        profile_context->Flush();
    }
    return CompiledModule(config, prim_func, kernel, profile_context);
}

CompiledModule::CompiledModule(CompileConfig config, tir::PrimFunc prim_func,
                               codegen::CompiledKernel kernel,
                               std::shared_ptr<profiling::ProfileContext> profile_context)
    : config_(config), prim_func_(prim_func), kernel_(kernel),
      profile_context_(std::move(profile_context)) {
    codegen::CodeGenC codegen_c;
    c_source_ = codegen_c.Generate(prim_func, "main");
}

CompiledModule::CompiledModule(CompileConfig config, Function relay_func,
                               std::shared_ptr<runtime::RuntimeSession> session,
                               std::shared_ptr<profiling::ProfileContext> profile_context)
    : config_(config), relay_func_(relay_func), session_(std::move(session)),
      profile_context_(std::move(profile_context)) {}

void CompiledModule::Run(const std::vector<void*>& packed_args) {
    if (session_) {
        throw std::runtime_error(
            "Adaptive mode requires Run(args, input_shapes). Use the overload with input_shapes.");
    }
    if (!kernel_.IsReady()) {
        throw std::runtime_error("CompiledModule: kernel not ready");
    }
    const std::string run_id = profile_context_ ? profile_context_->NextRunId("execute") : "";
    profiling::ActivationScope activation(profile_context_, run_id);
    profiling::ScopedSpan span(profile_context_, MakeEvent("runtime", "compiled_module_run"),
                               run_id);
    if (kernel_.IsReady() && kernel_->kernel_name.size()) {
        span.AddField("kernel_symbol", kernel_->kernel_name);
    }
    kernel_(packed_args);
    if (profile_context_) {
        profile_context_->Flush();
    }
}

void CompiledModule::Run(const std::vector<void*>& packed_args,
                         const std::vector<std::vector<int64_t>>& input_shapes) {
    const std::string run_id =
        profile_context_ ? profile_context_->NextRunId(session_ ? "adaptive_run" : "execute") : "";
    profiling::ActivationScope activation(profile_context_, run_id);
    profiling::EventSpec spec = MakeEvent("runtime",
                                          session_ ? "adaptive_module_run" : "compiled_module_run");
    spec.shape_signature = profiling::ShapeSignatureToString(input_shapes);
    if (!session_ && kernel_.IsReady()) {
        spec.kernel_symbol = kernel_->kernel_name;
    }
    profiling::ScopedSpan span(profile_context_, std::move(spec), run_id);
    if (session_) {
        session_->Run(packed_args, input_shapes);
        if (profile_context_) {
            profile_context_->Flush();
        }
        return;
    }
    if (!kernel_.IsReady()) {
        throw std::runtime_error("CompiledModule: kernel not ready");
    }
    kernel_(packed_args);
    if (profile_context_) {
        profile_context_->Flush();
    }
}

void CompiledModule::WarmUp(const std::vector<std::vector<int64_t>>& input_shapes) {
    const std::string run_id = profile_context_ ? profile_context_->NextRunId("warmup") : "";
    profiling::ActivationScope activation(profile_context_, run_id);
    profiling::EventSpec spec = MakeEvent("runtime", "module_warmup");
    spec.shape_signature = profiling::ShapeSignatureToString(input_shapes);
    profiling::ScopedSpan span(profile_context_, std::move(spec), run_id);
    if (session_) {
        session_->WarmUp(input_shapes);
    }
    if (profile_context_) {
        profile_context_->Flush();
    }
}

void CompiledModule::WaitAll() {
    const std::string run_id = profile_context_ ? profile_context_->NextRunId("waitall") : "";
    profiling::ActivationScope activation(profile_context_, run_id);
    profiling::ScopedSpan span(profile_context_, MakeEvent("runtime", "module_wait_all"), run_id);
    if (session_) {
        session_->WaitAll();
    }
    if (profile_context_) {
        profile_context_->Flush();
    }
}

bool CompiledModule::IsReady() const {
    if (session_) {
        return true;
    }
    return kernel_.IsReady();
}

std::string CompiledModule::GetStatus() const {
    if (session_) {
        return session_->GetStatus();
    }
    if (!kernel_.IsReady()) {
        return "not_ready";
    }
    if (config_->mode == CompileMode::kAOT) {
        return "aot_compiled";
    }
    if (config_->mode == CompileMode::kJIT) {
        return "jit_compiled";
    }
    return "compiled";
}

void CompiledModule::SaveCSource(const std::string& path) const {
    if (c_source_.empty() && prim_func_.defined()) {
        codegen::CodeGenC codegen_c;
        const_cast<std::string&>(c_source_) = codegen_c.Generate(prim_func_, "main");
    }
    std::ofstream ofs(path);
    if (!ofs) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    ofs << c_source_;
}

std::string CompiledModule::GetProfileBundlePath() const {
    return profile_context_ ? profile_context_->bundle_dir() : "";
}

}  // namespace api
}  // namespace kxc
