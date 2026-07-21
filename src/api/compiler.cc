/*! \file src/api/compiler.cc
 * \brief 实现 Target 驱动、显式分阶段的 Relay 到后端编译管线。
 */

#include "api/compiler.h"

#include <cctype>
#include <sstream>
#include <stdexcept>
#include <string>

#include "api/compile_result.h"
#include "base/pass.h"
#include "base/profiling.h"
#include "codegen/kernel_signature.h"
#include "relay/pass/print_ir.h"
#include "relay/transforms/infer_type.h"
#include "relay/transforms/lower.h"
#include "relay/transforms/pipeline.h"
#include "tir/pass/print_ir.h"
#include "tir/transforms/bind_cuda_threads.h"
#include "tir/transforms/pipeline.h"

#if KXC_USE_LLVM
#include <llvm/IR/LLVMContext.h>

#include "codegen/codegen_llvm.h"
#include "codegen/llvm_jit.h"
#endif

#if KXC_USE_CUDA
#include "codegen/codegen_cuda.h"
#include "codegen/cuda_module.h"
#endif

namespace kxc::api {
namespace {

// 已有 profiling 作用域优先复用；否则只在配置启用时创建 bundle。
std::shared_ptr<profiling::ProfileContext> MaybeCreateProfileContext(
    const CompileConfig& config) {
    if (profiling::CurrentContext()) return profiling::CurrentContext();
    if (!config->profile_options.enabled) return nullptr;
    return profiling::ProfileContext::Create(config->profile_options);
}

// Compiler 阶段事件统一携带 Target、device 和优化等级，便于跨后端比较。
profiling::EventSpec MakeStageEvent(const char* stage,
                                    const CompileConfig& config) {
    profiling::EventSpec spec;
    spec.component = "compiler";
    spec.event_type = "compile_stage";
    spec.pass_name = stage;
    spec.fields = profiling::MakeFields({
        {"stage", stage},
        {"target_kind", config->target->kind},
        {"device_type",
         std::to_string(static_cast<int>(config->target->device_type))},
        {"device_id", std::to_string(config->target->device_id)},
        {"opt_level", std::to_string(config->opt_level)},
    });
    return spec;
}

// 将当前阶段唯一 IR、symbol 和 backend 写入阶段 span，不维护第二份编译状态。
void AddResultFields(profiling::ScopedSpan* span, const CompileResult& result) {
    const CompileStage stage = result.stage();
    if (stage == CompileStage::kRelayOptimized) {
        const std::string text = relay::pass::ToText(result.optimized_relay());
        span->AddField("ir_hash", profiling::HashText(text));
        span->AddMetric("ir_bytes", static_cast<double>(text.size()));
    }
    if (static_cast<int>(stage) >= static_cast<int>(CompileStage::kLowered)) {
        const tir::PrimFunc function =
            stage == CompileStage::kLowered ? result.lowered_tir()
                                             : result.optimized_tir();
        std::ostringstream stream;
        tir::pass::DumpPrimFunc(function, stream);
        const std::string text = stream.str();
        span->AddField("ir_hash", profiling::HashText(text));
        span->AddMetric("ir_bytes", static_cast<double>(text.size()));
    }
    if (static_cast<int>(stage) >=
        static_cast<int>(CompileStage::kSignatureBuilt)) {
        span->AddField("symbol", std::string(result.signature()->symbol));
    }
    if (stage == CompileStage::kBackendCompiled) {
        span->AddField(
            "backend",
            result.launch_metadata()->backend == codegen::CodeGenBackend::kLLVM
                ? "llvm"
                : "cuda");
    }
}

// 所有阶段共享错误包装和 profiling 状态，失败信息明确指出责任阶段。
template <typename Fn>
CompileResult RunStage(const char* stage, const CompileConfig& config, Fn&& fn) {
    profiling::ScopedSpan span(profiling::CurrentContext(),
                               MakeStageEvent(stage, config));
    try {
        CompileResult result = fn();
        AddResultFields(&span, result);
        return result;
    } catch (const std::exception& error) {
        span.SetStatus("error");
        span.SetMessage(error.what());
        throw std::runtime_error(std::string("Compiler stage '") + stage +
                                 "' failed: " + error.what());
    }
}

// global_symbol 必须是后端可直接导出的 C 标识符，禁止隐式改名造成 lookup 漂移。
String ReadKernelSymbol(const tir::PrimFunc& function) {
    const String key("global_symbol");
    if (!function->attrs.count(key)) {
        throw std::invalid_argument("optimized PrimFunc is missing global_symbol");
    }
    const auto* value = function->attrs.at(key).As<StringObj>();
    if (!value || value->data.empty()) {
        throw std::invalid_argument("PrimFunc global_symbol must be a non-empty String");
    }
    const std::string& symbol = value->data;
    const auto is_head = [](unsigned char ch) {
        return std::isalpha(ch) != 0 || ch == '_';
    };
    const auto is_tail = [](unsigned char ch) {
        return std::isalnum(ch) != 0 || ch == '_';
    };
    if (!is_head(static_cast<unsigned char>(symbol[0]))) {
        throw std::invalid_argument("PrimFunc global_symbol is not a C identifier");
    }
    for (size_t i = 1; i < symbol.size(); ++i) {
        if (!is_tail(static_cast<unsigned char>(symbol[i]))) {
            throw std::invalid_argument("PrimFunc global_symbol is not a C identifier");
        }
    }
    return String(symbol);
}

// Validate 阶段建立状态机起点；配置和 Relay 节点错误不会进入 pass。
CompileResult ValidateInput(Function function, const CompileConfig& config) {
    config.Validate();
    return CompileResult::Validate(config->target, std::move(function));
}

// Relay 优化阶段先后执行强制类型推导、等级策略和最终类型确认。
CompileResult OptimizeRelay(const CompileResult& input,
                            const CompileConfig& config) {
    Function typed = relay::InferTypePass(input.validated_relay());
    Function optimized = relay::RunRelayPassPipeline(
        typed, Compiler::RelayPassPolicy(config->opt_level));
    optimized = relay::InferTypePass(optimized);
    return input.AfterRelayOptimization(std::move(optimized));
}

// Lower 阶段原子发布 PrimFunc 与常量 payload，后续不得重新扫描 Relay 常量。
CompileResult Lower(const CompileResult& input) {
    relay::LoweredFunction lowered = relay::LowerToTIR(input.optimized_relay());
    Map<String, runtime::NDArray> constants;
    const Device target_device(input.target()->device_type,
                               input.target()->device_id);
    for (const auto& binding : lowered.constants()) {
        runtime::NDArray value = binding->value;
#if KXC_USE_CUDA
        // 常量是最终内核参数，必须在签名冻结前与 Target 位于同一设备。
        // 模块持有迁移后的唯一 payload，Launch 再按 constant_key 注入该对象。
        if (value.device() != target_device) value = value.CopyTo(target_device);
#else
        // CUDA-off 构建保留 lowering payload，后端阶段负责返回精确 feature 错误。
        if (target_device.device_type() != kCUDA &&
            value.device() != target_device) {
            value = value.CopyTo(target_device);
        }
#endif
        constants.Set(binding->key, std::move(value));
    }
    return input.AfterLowering(lowered->prim_func, constants);
}

// TIR 阶段只保留优化后的唯一 PrimFunc 事实。
CompileResult OptimizeTIR(const CompileResult& input,
                          const CompileConfig& config) {
    Array<String> passes = Compiler::TIRPassPolicy(config->opt_level);
    if (passes.empty()) {
        // 静态 KernelSignature 要求输出 extent 已化为 IntImm；这是 ABI 正确性步骤。
        passes = {String("fold_constant"), String("simplify_expr")};
    }
    tir::PrimFunc optimized = RunTIRPassPipeline(input.lowered_tir(), passes);
#if KXC_USE_CUDA
    if (input.target()->kind == "cuda" && input.target()->device_type == kCUDA) {
        // CUDA thread binding 是后端 ABI 的一部分，必须在 Signature 冻结前完成。
        optimized = tir::BindCudaThreads(optimized, input.target()).prim_func();
    }
#endif
    return input.AfterTIROptimization(std::move(optimized));
}

// Signature 阶段从最终 TIR、保活常量和显式 Target 构造公共 ABI。
CompileResult BuildSignature(const CompileResult& input) {
    const String symbol = ReadKernelSymbol(input.optimized_tir());
    codegen::KernelSignature signature = codegen::BuildKernelSignature(
        input.optimized_tir(), input.constants(), input.target(), symbol);
    return input.AfterSignature(std::move(signature));
}

// Backend 阶段只按 Target dispatch，并一次性发布 metadata 与 executable。
CompileResult BuildBackend(const CompileResult& input,
                           const CompileConfig& config) {
    const Target target = input.target();
    const Device device(target->device_type, target->device_id);
    if (target->kind == "llvm" && target->device_type == kCPU) {
#if KXC_USE_LLVM
        codegen::KernelLaunchMetadata metadata(
            device, codegen::CodeGenBackend::kLLVM);
        auto llvm_context = std::make_unique<llvm::LLVMContext>();
        codegen::CodeGenLLVM codegen(*llvm_context);
        codegen.AddFunction(input.optimized_tir(), input.signature()->symbol);
        codegen::LLVMJITEngine jit;
        codegen::CompiledKernel kernel = jit.Compile(
            codegen.TakeModule(), std::move(llvm_context), input.signature(),
            metadata, config->opt_level);
        return input.AfterBackend(metadata, kernel);
#else
        throw std::runtime_error(
            "Compiler target 'llvm' requires a build with KXC_ENABLE_LLVM=ON");
#endif
    }
    if (target->kind == "cuda" && target->device_type == kCUDA) {
#if KXC_USE_CUDA
        codegen::KernelLaunchMetadata metadata =
            tir::GetCudaLaunchMetadata(input.optimized_tir());
        codegen::CodeGenCUDA emitter;
        const std::string source = emitter.Generate(
            input.optimized_tir(), std::string(input.signature()->symbol));
        if (target->attrs.compute_version_major <= 0 ||
            target->attrs.compute_version_minor < 0) {
            throw std::runtime_error(
                "CUDA Target has no usable compute capability");
        }
        codegen::CUDACompileOptions options;
        options.architecture =
            "compute_" + std::to_string(target->attrs.compute_version_major) +
            std::to_string(target->attrs.compute_version_minor);
        options.source_name = std::string(input.signature()->symbol) + ".cu";
        codegen::CompiledKernel kernel = codegen::CUDAModule::Compile(
            source, input.signature(), metadata, options);
        return input.AfterBackend(metadata, kernel);
#else
        throw std::runtime_error(
            "Compiler target 'cuda' requires a build with KXC_ENABLE_CUDA=ON");
#endif
    }
    throw std::runtime_error("Compiler Target has no matching backend");
}

// Assemble 阶段只转移已经完成 Backend 状态的对象，不再推导任何契约。
CompiledModule AssembleModule(
    const CompileResult& result,
    std::shared_ptr<profiling::ProfileContext> profile_context) {
    result.ValidateState();
    if (result.stage() != CompileStage::kBackendCompiled) {
        throw std::logic_error("AssembleModule requires backend_compiled state");
    }
    return CompiledModule(result.target(), result.optimized_tir(),
                          result.signature(), result.launch_metadata(),
                          result.constants(), result.kernel(),
                          std::move(profile_context));
}

}  // namespace

// 每个等级是前一级的确定超集；InferType 属于强制阶段，不放入本策略。
Array<String> Compiler::RelayPassPolicy(int opt_level) {
    if (opt_level < 0 || opt_level > 3) {
        throw std::invalid_argument("Relay pass policy requires opt_level 0..3");
    }
    if (opt_level == 0) return {};
    if (opt_level == 1) {
        return {String("fold_tuple_get_item"), String("fold_constant"),
                String("simplify_expr")};
    }
    if (opt_level == 2) {
        return {String("fold_tuple_get_item"), String("fold_constant"),
                String("simplify_expr"), String("canonicalize_cast"),
                String("remove_standalone_reshapes"),
                String("eliminate_dead_let")};
    }
    return {String("optimize_default")};
}

// TIR 等级逐步加入索引规范化、循环串行化和完整默认优化。
Array<String> Compiler::TIRPassPolicy(int opt_level) {
    if (opt_level < 0 || opt_level > 3) {
        throw std::invalid_argument("TIR pass policy requires opt_level 0..3");
    }
    if (opt_level == 0) return {};
    if (opt_level == 1) {
        return {String("fold_constant"), String("simplify_expr")};
    }
    if (opt_level == 2) {
        return {String("fold_constant"), String("simplify_expr"),
                String("force_narrow_index_to_i32"),
                String("convert_for_loops_serial")};
    }
    return {String("optimize_default")};
}

// 顶层入口只负责建立 profiling/PassContext 作用域并按固定顺序调用七个阶段。
CompiledModule Compiler::Compile(Function function, CompileConfig config) {
    config.Validate();
    auto profile_context = MaybeCreateProfileContext(config);
    const std::string run_id =
        profile_context ? profile_context->NextRunId("compile") : "";
    profiling::ActivationScope activation(profile_context, run_id);
    profiling::ScopedSpan compile_span(
        profile_context, MakeStageEvent("compile", config), run_id);

    // Relay placement 缺失时写入 config Target，存在时只允许完全相同身份。
    const PassContext pass_context = PassContext::MergeTarget(
        PassContext::FromRelay(function), config->target);
    PassContext::Scope pass_scope(pass_context);

    CompileResult result = RunStage(
        "validate", config, [&] { return ValidateInput(function, config); });
    result = RunStage("optimize_relay", config,
                      [&] { return OptimizeRelay(result, config); });
    result = RunStage("lower", config, [&] { return Lower(result); });
    result = RunStage("optimize_tir", config,
                      [&] { return OptimizeTIR(result, config); });
    result = RunStage("build_signature", config,
                      [&] { return BuildSignature(result); });
    result = RunStage("build_backend", config,
                      [&] { return BuildBackend(result, config); });

    profiling::ScopedSpan assemble_span(
        profile_context, MakeStageEvent("assemble", config), run_id);
    CompiledModule module = AssembleModule(result, profile_context);
    AddResultFields(&assemble_span, result);
    if (profile_context) profile_context->Flush();
    return module;
}

}  // namespace kxc::api
