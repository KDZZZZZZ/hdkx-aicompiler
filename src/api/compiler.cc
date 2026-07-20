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
#include "tir/transforms/pipeline.h"

#if KXC_USE_LLVM
#include <llvm/IR/LLVMContext.h>

#include "codegen/codegen_llvm.h"
#include "codegen/llvm_jit.h"
#endif

namespace kxc {
namespace api {

namespace {

constexpr const char* kKernelEntrySymbol = "kxc_kernel_main";

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
    // 配置验证必须先于任何字段读取，避免错误 ObjectRef 被解释为 CompileConfigNode。
    config.Validate();
    auto profile_context = MaybeCreateProfileContext(config);
    const std::string run_id = profile_context ? profile_context->NextRunId("compile") : "";
    profiling::ActivationScope activation(profile_context, run_id);
    profiling::ScopedSpan compile_span(profile_context, MakeEvent("compiler", "compile_module"),
                                       run_id);
    compile_span.AddField("target_kind", config->target->kind);
    compile_span.AddField("target_device_type",
                          std::to_string(static_cast<int>(config->target->device_type)));
    compile_span.AddMetric("opt_level", static_cast<double>(config->opt_level));

    // Task 8 将把 pass 策略拆成显式阶段；当前按优化等级保持原两档行为。
    Array<String> relay_passes;
    if (config->opt_level >= 2) {
        relay_passes = {String("optimize_default")};
    } else {
        relay_passes = {String("fold_constant"), String("simplify_expr")};
    }
    Function typed_func = relay::InferTypePass(func);
    Function optimized_func = relay::RunRelayPassPipeline(typed_func, relay_passes);
    optimized_func = relay::InferTypePass(optimized_func);
    relay::LoweredFunction lowered = relay::LowerToTIR(optimized_func);
    tir::PrimFunc prim_func = lowered->prim_func;

    Array<String> passes;
    if (config->opt_level >= 2) {
        passes = {String("optimize_default")};
    } else {
        passes = {String("fold_constant"), String("simplify_expr")};
    }
    prim_func = RunTIRPassPipeline(prim_func, passes);

    codegen::CompiledKernel kernel{ObjectRef()};

#if KXC_USE_LLVM
    // LLVM 是 llvm/cpu Target 的唯一执行后端，不能由独立配置字段覆盖。
    if (config->target->kind == "llvm" && config->target->device_type == kCPU) {
        profiling::ScopedSpan codegen_span(profile_context, MakeEvent("codegen", "llvm_jit_compile"),
                                           run_id);
        auto llvm_ctx = std::make_unique<llvm::LLVMContext>();
        codegen::CodeGenLLVM codegen(*llvm_ctx);
        codegen.AddFunction(prim_func, kKernelEntrySymbol);
        auto module = codegen.TakeModule();

        codegen::LLVMJITEngine jit;
        kernel = jit.Compile(std::move(module), std::move(llvm_ctx), kKernelEntrySymbol,
                             config->opt_level);
        codegen_span.AddField("kernel_symbol", kKernelEntrySymbol);
        codegen_span.SetMessage("LLVM JIT compilation completed");
        codegen_span.AddMetric("opt_level", static_cast<double>(config->opt_level));
    } else
#else
    if (config->target->kind == "llvm" && config->target->device_type == kCPU) {
        compile_span.SetStatus("error");
        compile_span.SetMessage("LLVM backend is disabled in this build");
        throw std::runtime_error(
            "Compiler target 'llvm' requires a build with KXC_ENABLE_LLVM=ON");
    } else
#endif
    {
        // CUDA Target 已被配置层识别，但真实 CUDA codegen 在后续任务实现前明确拒绝。
        compile_span.SetStatus("error");
        compile_span.SetMessage("CUDA backend is not implemented");
        throw std::runtime_error(
            "Compiler target 'cuda' is recognized but CUDA codegen is not implemented");
    }

    if (profile_context) {
        profile_context->Flush();
    }
    return CompiledModule(config, prim_func, kernel, lowered.constants(), profile_context);
}

CompiledModule::CompiledModule(CompileConfig config, tir::PrimFunc prim_func,
                               codegen::CompiledKernel kernel,
                               Array<relay::ConstantBinding> constants,
                               std::shared_ptr<profiling::ProfileContext> profile_context)
    : config_(config), prim_func_(prim_func), kernel_(kernel),
      constants_(std::move(constants)),
      profile_context_(std::move(profile_context)) {
    codegen::CSourceEmitter emitter;
    c_source_ = emitter.Generate(prim_func, "main");
}

// 返回独立 Array，防止调用方通过共享容器别名改写模块内常量顺序。
Array<relay::ConstantBinding> CompiledModule::GetConstants() const {
    Array<relay::ConstantBinding> result;
    for (const auto& binding : constants_) result.push_back(binding);
    return result;
}

void CompiledModule::Run(const std::vector<void*>& packed_args) {
    (void)packed_args;
    if (!kernel_.IsReady()) {
        throw std::runtime_error("CompiledModule: kernel not ready");
    }
    // 新 CompiledKernel 只接受已校验的 NDArray；Task 6 会删除本裸指针入口并接通 Launch。
    throw std::runtime_error(
        "CompiledModule::Run uses the removed packed-pointer ABI; use the typed launch API");
}

bool CompiledModule::IsReady() const {
    return kernel_.IsReady();
}

std::string CompiledModule::GetStatus() const {
    if (!kernel_.IsReady()) {
        return "not_ready";
    }
    return "compiled";
}

void CompiledModule::SaveCSource(const std::string& path) const {
    if (c_source_.empty() && prim_func_.defined()) {
        codegen::CSourceEmitter emitter;
        const_cast<std::string&>(c_source_) = emitter.Generate(prim_func_, "main");
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
