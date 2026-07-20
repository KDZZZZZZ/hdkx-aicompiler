/*! \file src/api/compiler.cc
 * \brief 实现编译配置默认值、编译入口和 CompiledModule 执行逻辑。
 */

#include "api/compiler.h"

#include <stdexcept>

#include "base/profiling.h"
#include "codegen/kernel_signature.h"
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

    // 常量表以 key 索引并强持有 payload；签名与 Launch 使用同一批对象。
    Map<String, runtime::NDArray> constants;
    for (const auto& binding : lowered.constants()) {
        constants.Set(binding->key, binding->value);
    }
    const String symbol(kKernelEntrySymbol);
    codegen::KernelSignature signature = codegen::BuildKernelSignature(
        prim_func, constants, config->target, symbol);
    const codegen::CodeGenBackend backend =
        config->target->kind == "llvm" ? codegen::CodeGenBackend::kLLVM
                                       : codegen::CodeGenBackend::kCUDA;
    codegen::KernelLaunchMetadata metadata(
        Device(config->target->device_type, config->target->device_id),
        backend);

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
        codegen::CompiledKernel kernel = jit.Compile(
            std::move(module), std::move(llvm_ctx), signature, metadata,
            config->opt_level);
        codegen_span.AddField("kernel_symbol", kKernelEntrySymbol);
        codegen_span.SetMessage("LLVM JIT compilation completed");
        codegen_span.AddMetric("opt_level", static_cast<double>(config->opt_level));
        if (profile_context) profile_context->Flush();
        return CompiledModule(config->target, prim_func, signature, metadata,
                              constants, kernel, profile_context);
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

}

}  // namespace api
}  // namespace kxc
