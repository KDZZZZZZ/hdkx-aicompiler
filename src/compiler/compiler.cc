/*! \file src/compiler/compiler.cc
 * \brief Implements the target-driven per-operator compiler pipeline.
 */

#include "kxc/compiler/compiler.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "internal/compile_state.h"
#include "internal/kernel_abi_builder.h"
#include "internal/lowered_graph.h"
#include "internal/primitive_cache.h"
#include "../runtime/internal/compiled_module_node.h"
#include "../runtime/internal/memory_plan.h"
#include "kxc/pass/context.h"
#include "kxc/profiling/profiling.h"
#include "kxc/relay/pass/print_ir.h"
#include "kxc/relay/transforms/infer_type.h"
#include "kxc/relay/transforms/pipeline.h"
#include "kxc/relay/visitor.h"
#include "kxc/tir/pass/print_ir.h"
#include "kxc/tir/transforms/bind_cuda_threads.h"
#include "kxc/tir/transforms/pipeline.h"

#if KXC_USE_LLVM
#include <llvm/IR/LLVMContext.h>

#include "../codegen/llvm/internal/codegen_llvm.h"
#include "../codegen/llvm/internal/llvm_jit.h"
#endif

#if KXC_USE_CUDA
#include "../codegen/cuda/internal/codegen_cuda.h"
#include "../codegen/cuda/internal/cuda_module.h"
#endif

namespace kxc::api {
namespace {

std::shared_ptr<profiling::ProfileContext> MaybeCreateProfileContext(
    const CompileConfig& config) {
    if (profiling::CurrentContext()) return profiling::CurrentContext();
    if (!config->profile_options.enabled) return nullptr;
    return profiling::ProfileContext::Create(config->profile_options);
}

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

std::string PrimitiveContext(const PrimitiveCompileState& primitive) {
    return "unit " + std::to_string(primitive.unit_id) + " ('" +
           std::string(primitive.symbol) + "', " +
           std::string(primitive.operator_identity) + ")";
}

uint64_t PlannedStorageBytes(const runtime::ExecutablePlan& plan) {
    std::unordered_map<int64_t, uint64_t> bytes_by_storage;
    for (const auto& value : plan.values()) {
        uint64_t elements = 1;
        bool dynamic = false;
        for (int64_t dimension : value.shape()) {
            if (dimension < 0) {
                dynamic = true;
                break;
            }
            if (dimension != 0 &&
                elements > std::numeric_limits<uint64_t>::max() /
                               static_cast<uint64_t>(dimension)) {
                return std::numeric_limits<uint64_t>::max();
            }
            elements *= static_cast<uint64_t>(dimension);
        }
        if (dynamic) continue;
        const uint64_t element_bytes =
            static_cast<uint64_t>(value->dtype.bits / 8) *
            static_cast<uint64_t>(value->dtype.lanes);
        if (element_bytes != 0 &&
            elements > std::numeric_limits<uint64_t>::max() / element_bytes) {
            return std::numeric_limits<uint64_t>::max();
        }
        bytes_by_storage[value->storage_id] =
            std::max(bytes_by_storage[value->storage_id],
                     elements * element_bytes);
    }
    uint64_t total = 0;
    for (const auto& item : bytes_by_storage) {
        if (total > std::numeric_limits<uint64_t>::max() - item.second) {
            return std::numeric_limits<uint64_t>::max();
        }
        total += item.second;
    }
    return total;
}

void AddResultFields(profiling::ScopedSpan* span, const CompileResult& result) {
    const CompileStage stage = result.stage();
    if (stage == CompileStage::kRelayOptimized) {
        const std::string text = relay::pass::ToText(result.optimized_relay());
        span->AddField("ir_hash", profiling::HashText(text));
        span->AddMetric("ir_bytes", static_cast<double>(text.size()));
        return;
    }
    if (static_cast<int>(stage) < static_cast<int>(CompileStage::kLowered)) {
        return;
    }
    const std::vector<PrimitiveCompileState> primitives = result.primitives();
    span->AddMetric("primitive_count", static_cast<double>(primitives.size()));
    std::unordered_set<int64_t> storage_ids;
    for (const auto& value : result.plan().values()) {
        storage_ids.insert(value->storage_id);
    }
    span->AddMetric("value_count",
                    static_cast<double>(result.plan().values().size()));
    span->AddMetric("storage_slot_count",
                    static_cast<double>(storage_ids.size()));
    span->AddMetric(
        "storage_reuse_count",
        static_cast<double>(result.plan().values().size() -
                            storage_ids.size()));
    span->AddMetric("planned_peak_storage_bytes",
                    static_cast<double>(PlannedStorageBytes(result.plan())));
    size_t cache_hits = 0;
    for (const PrimitiveCompileState& primitive : primitives) {
        const std::string prefix = "unit." + std::to_string(primitive.unit_id) + ".";
        std::ostringstream stream;
        tir::pass::DumpPrimFunc(primitive.tir, stream);
        const std::string text = stream.str();
        span->AddField(prefix + "symbol", std::string(primitive.symbol));
        span->AddField(prefix + "operator",
                       std::string(primitive.operator_identity));
        span->AddField(prefix + "ir_hash", profiling::HashText(text));
        span->AddMetric(prefix + "ir_bytes", static_cast<double>(text.size()));
        if (primitive.launch_metadata) {
            span->AddField(
                prefix + "backend",
                (*primitive.launch_metadata)->backend ==
                        codegen::CodeGenBackend::kLLVM
                    ? "llvm"
                    : "cuda");
            span->AddMetric(prefix + "cache_hit",
                            primitive.cache_hit ? 1.0 : 0.0);
            if (primitive.cache_hit) ++cache_hits;
        }
    }
    if (stage == CompileStage::kBackendCompiled) {
        span->AddMetric("cache_hits", static_cast<double>(cache_hits));
        span->AddMetric(
            "cache_hit_rate",
            primitives.empty()
                ? 0.0
                : static_cast<double>(cache_hits) /
                      static_cast<double>(primitives.size()));
    }
}

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

template <typename Fn>
auto RunPrimitiveStage(const char* stage,
                       const PrimitiveCompileState& primitive, Fn&& fn)
    -> decltype(fn()) {
    try {
        return fn();
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string(stage) + " failed for " +
                                 PrimitiveContext(primitive) + ": " +
                                 error.what());
    }
}

String ReadKernelSymbol(const tir::PrimFunc& function) {
    const String key("global_symbol");
    if (!function->attrs.count(key)) {
        throw std::invalid_argument("optimized PrimFunc is missing global_symbol");
    }
    const auto* value = function->attrs.at(key).As<StringObj>();
    if (!value || value->data.empty()) {
        throw std::invalid_argument(
            "PrimFunc global_symbol must be a non-empty String");
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
            throw std::invalid_argument(
                "PrimFunc global_symbol is not a C identifier");
        }
    }
    return String(symbol);
}

tir::PrimFunc RefreshStructuralHash(const tir::PrimFunc& function) {
    const String structural_hash_key("kxc.structural_hash");
    Map<String, ObjectRef> attrs;
    for (const auto& item : function->attrs) {
        if (!(item.first == structural_hash_key)) {
            attrs.Set(item.first, item.second);
        }
    }
    tir::PrimFunc canonical(function->params, function->body,
                            function->buffer_map, attrs);
    std::ostringstream stream;
    tir::pass::DumpPrimFunc(canonical, stream);
    attrs.Set(structural_hash_key,
              String(profiling::HashText(stream.str())));
    return tir::PrimFunc(function->params, function->body,
                         function->buffer_map, std::move(attrs));
}

Map<String, runtime::NDArray> PlaceConstants(
    const Map<String, runtime::NDArray>& source, const Target& target) {
    Map<String, runtime::NDArray> result;
    const Device target_device(target->device_type, target->device_id);
    for (const auto& item : source) {
        runtime::NDArray value = item.second;
#if KXC_USE_CUDA
        if (value.device() != target_device) value = value.CopyTo(target_device);
#else
        if (target_device.device_type() != kCUDA &&
            value.device() != target_device) {
            value = value.CopyTo(target_device);
        }
#endif
        result.Set(item.first, std::move(value));
    }
    return result;
}

CompileResult ValidateInput(Function function, const CompileConfig& config) {
    config.Validate();
    return CompileResult::Validate(config->target, std::move(function));
}

CompileResult OptimizeRelay(const CompileResult& input,
                            const CompileConfig& config) {
    Function typed = relay::InferTypePass(input.validated_relay());
    Function optimized = relay::RunRelayPassPipeline(
        typed, Compiler::RelayPassPolicy(config->opt_level));
    optimized = relay::InferTypePass(optimized);
    return input.AfterRelayOptimization(std::move(optimized));
}

CompileResult LowerOperators(const CompileResult& input) {
    const Device device(input.target()->device_type, input.target()->device_id);
    internal::LoweredGraph lowered =
        internal::LowerGraph(input.optimized_relay(), device);
    std::vector<PrimitiveCompileState> primitives;
    primitives.reserve(lowered.primitives.size());
    for (const internal::LoweredPrimitive& source : lowered.primitives) {
        PrimitiveCompileState primitive;
        primitive.unit_id = source.unit_id;
        primitive.symbol = source.symbol;
        primitive.operator_identity = source.operator_identity;
        primitive.structural_hash = source.structural_hash;
        primitive.tir = source.lowered->prim_func;
        primitives.push_back(std::move(primitive));
    }
    return input.AfterLowering(
        std::move(primitives), runtime::internal::PlanMemory(lowered.plan),
        PlaceConstants(lowered.constants, input.target()));
}

CompileResult OptimizeTIR(const CompileResult& input,
                          const CompileConfig& config) {
    Array<String> passes =
        Compiler::TIRPassPolicy(config->opt_level, input.target());
    if (passes.empty()) {
        passes = {String("fold_constant"), String("simplify_expr")};
    }
    std::vector<tir::PrimFunc> optimized;
    for (const PrimitiveCompileState& primitive : input.primitives()) {
        optimized.push_back(RunPrimitiveStage(
            "optimize_tir", primitive, [&] {
                tir::PrimFunc result =
                    RunTIRPassPipeline(primitive.tir, passes);
#if KXC_USE_CUDA
                if (input.target()->kind == "cuda" &&
                    input.target()->device_type == kCUDA) {
                    result = tir::BindCudaThreads(result, input.target()).prim_func();
                }
#endif
                return RefreshStructuralHash(result);
            }));
    }
    return input.AfterTIROptimization(std::move(optimized));
}

const char* BackendVersion(const Target& target) {
    if (target->kind == "llvm" && target->device_type == kCPU) {
        return "llvm-orc-v1";
    }
    if (target->kind == "cuda" && target->device_type == kCUDA) {
        return "cuda-nvrtc-driver-v1";
    }
    throw std::invalid_argument("Primitive cache target has no backend version");
}

CompileResult BuildSignatures(const CompileResult& input,
                              const CompileConfig& config) {
    std::vector<codegen::KernelSignature> signatures;
    std::vector<bool> cache_hits;
    for (const PrimitiveCompileState& primitive : input.primitives()) {
        const std::string cache_key = internal::BuildPrimitiveCacheKey(
            primitive.structural_hash, input.target(), config->opt_level,
            BackendVersion(input.target()));
        const auto cached = internal::LookupPrimitiveCache(cache_key);
        if (cached) {
            if (!(cached->signature->symbol == primitive.symbol) ||
                !cached->kernel.IsReady()) {
                throw std::logic_error(
                    PrimitiveContext(primitive) +
                    " cache entry does not match its stable symbol");
            }
            signatures.push_back(cached->signature);
            cache_hits.push_back(true);
            continue;
        }
        signatures.push_back(RunPrimitiveStage(
            "build_signature", primitive, [&] {
                const String symbol = ReadKernelSymbol(primitive.tir);
                if (!(symbol == primitive.symbol)) {
                    throw std::invalid_argument(
                        "PrimFunc global_symbol drifted from unit symbol");
                }
                return codegen::BuildKernelSignature(
                    primitive.tir, input.constants(), input.target(), symbol);
            }));
        cache_hits.push_back(false);
    }
    return input.AfterSignatures(std::move(signatures), std::move(cache_hits));
}

CompileResult BuildBackends(const CompileResult& input,
                            const CompileConfig& config) {
    const Target target = input.target();
    const Device device(target->device_type, target->device_id);
    const std::vector<PrimitiveCompileState> primitives = input.primitives();
    std::vector<std::optional<codegen::KernelLaunchMetadata>> metadata_slots(
        primitives.size());
    std::vector<std::optional<codegen::CompiledKernel>> kernel_slots(
        primitives.size());
    std::vector<std::string> cache_keys;
    std::vector<size_t> misses;
    cache_keys.reserve(primitives.size());
    for (size_t i = 0; i < primitives.size(); ++i) {
        const PrimitiveCompileState& primitive = primitives[i];
        if (!primitive.signature) {
            throw std::logic_error(
                PrimitiveContext(primitive) + " has no signature");
        }
        cache_keys.push_back(internal::BuildPrimitiveCacheKey(
            primitive.structural_hash, target, config->opt_level,
            BackendVersion(target)));
        if (!primitive.cache_hit) {
            misses.push_back(i);
            continue;
        }
        const auto cached = internal::PeekPrimitiveCache(cache_keys.back());
        if (!cached || cached->signature.get() != primitive.signature->get() ||
            !cached->kernel.IsReady()) {
            throw std::logic_error(
                PrimitiveContext(primitive) +
                " cache entry disappeared or changed during compilation");
        }
        metadata_slots[i] = cached->launch_metadata;
        kernel_slots[i] = cached->kernel;
    }

    if (!misses.empty() && target->kind == "llvm" &&
        target->device_type == kCPU) {
#if KXC_USE_LLVM
        auto llvm_context = std::make_unique<llvm::LLVMContext>();
        codegen::CodeGenLLVM codegen(*llvm_context);
        std::vector<std::pair<tir::PrimFunc, std::string>> functions;
        std::vector<codegen::KernelSignature> signatures;
        std::vector<codegen::KernelLaunchMetadata> metadata;
        functions.reserve(misses.size());
        signatures.reserve(misses.size());
        metadata.reserve(misses.size());
        for (size_t index : misses) {
            const PrimitiveCompileState& primitive = primitives[index];
            functions.emplace_back(primitive.tir,
                                   std::string(primitive.symbol));
            signatures.push_back(*primitive.signature);
            metadata.emplace_back(
                device, codegen::CodeGenBackend::kLLVM);
        }
        codegen.AddFunctions(functions);
        codegen::LLVMJITEngine jit;
        std::vector<codegen::CompiledKernel> compiled = jit.CompileMany(
            codegen.TakeModule(), std::move(llvm_context), signatures,
            metadata, config->opt_level);
        for (size_t i = 0; i < misses.size(); ++i) {
            const size_t index = misses[i];
            metadata_slots[index] = metadata[i];
            kernel_slots[index] = compiled[i];
        }
#else
        throw std::runtime_error(
            "Compiler target 'llvm' requires a build with KXC_ENABLE_LLVM=ON");
#endif
    } else if (!misses.empty() && target->kind == "cuda" &&
               target->device_type == kCUDA) {
#if KXC_USE_CUDA
        std::vector<std::pair<tir::PrimFunc, std::string>> functions;
        std::vector<codegen::KernelSignature> signatures;
        std::vector<codegen::KernelLaunchMetadata> metadata;
        functions.reserve(misses.size());
        signatures.reserve(misses.size());
        metadata.reserve(misses.size());
        for (size_t index : misses) {
            const PrimitiveCompileState& primitive = primitives[index];
            const tir::CudaLaunchConfig launch_config =
                tir::GetCudaLaunchConfig(primitive.tir);
            functions.emplace_back(primitive.tir,
                                   std::string(primitive.symbol));
            signatures.push_back(*primitive.signature);
            metadata.emplace_back(
                device, codegen::CodeGenBackend::kCUDA,
                codegen::Dim3{launch_config.grid_x, launch_config.grid_y,
                              launch_config.grid_z},
                codegen::Dim3{launch_config.block_x, launch_config.block_y,
                              launch_config.block_z},
                launch_config.dynamic_shared_memory_bytes);
        }
        if (target->attrs.compute_version_major <= 0 ||
            target->attrs.compute_version_minor < 0) {
            throw std::runtime_error(
                "CUDA Target has no usable compute capability");
        }
        codegen::CodeGenCUDA emitter;
        const std::string source = emitter.GenerateModule(functions);
        codegen::CUDACompileOptions options;
        options.architecture =
            "compute_" +
            std::to_string(target->attrs.compute_version_major) +
            std::to_string(target->attrs.compute_version_minor);
        options.source_name = "kxc_operator_module.cu";
        std::vector<codegen::CompiledKernel> compiled =
            codegen::CUDAModule::CompileMany(source, signatures, metadata,
                                             options);
        for (size_t i = 0; i < misses.size(); ++i) {
            const size_t index = misses[i];
            metadata_slots[index] = metadata[i];
            kernel_slots[index] = compiled[i];
        }
#else
        throw std::runtime_error(
            "Compiler target 'cuda' requires a build with KXC_ENABLE_CUDA=ON");
#endif
    } else if (!misses.empty()) {
        throw std::runtime_error("Compiler Target has no matching backend");
    }

    std::vector<codegen::KernelLaunchMetadata> metadata;
    std::vector<codegen::CompiledKernel> kernels;
    metadata.reserve(primitives.size());
    kernels.reserve(primitives.size());
    for (size_t i = 0; i < primitives.size(); ++i) {
        if (!metadata_slots[i] || !kernel_slots[i]) {
            throw std::logic_error(
                PrimitiveContext(primitives[i]) +
                " backend batch did not produce an executable");
        }
        if (!primitives[i].cache_hit) {
            internal::StorePrimitiveCache(
                cache_keys[i],
                internal::CachedPrimitive{*primitives[i].signature,
                                          *metadata_slots[i],
                                          *kernel_slots[i]});
        }
        metadata.push_back(*metadata_slots[i]);
        kernels.push_back(*kernel_slots[i]);
    }
    return input.AfterBackends(std::move(metadata), std::move(kernels));
}

CompiledModule AssembleModule(
    const CompileResult& result,
    std::shared_ptr<profiling::ProfileContext> profile_context) {
    result.ValidateState();
    if (result.stage() != CompileStage::kBackendCompiled) {
        throw std::logic_error("AssembleModule requires backend_compiled state");
    }
    std::vector<internal::CompiledModuleEntry> entries;
    for (const PrimitiveCompileState& primitive : result.primitives()) {
        entries.push_back(internal::CompiledModuleEntry{
            primitive.tir, *primitive.signature, *primitive.launch_metadata,
            *primitive.kernel});
    }
    return internal::BuildCompiledModule(
        result.target(), std::move(entries), result.constants(),
        std::move(profile_context));
}

CompiledGraph CompilePipeline(Function function, CompileConfig config) {
    config.Validate();
    auto profile_context = MaybeCreateProfileContext(config);
    const std::string run_id =
        profile_context ? profile_context->NextRunId("compile") : "";
    profiling::ActivationScope activation(profile_context, run_id);
    profiling::ScopedSpan compile_span(
        profile_context, MakeStageEvent("compile", config), run_id);

    const PassContext pass_context = PassContext::MergeTarget(
        relay::PassContextFromRelay(function), config->target);
    PassContext::Scope pass_scope(pass_context);

    CompileResult result = RunStage(
        "validate", config, [&] { return ValidateInput(function, config); });
    result = RunStage("optimize_relay", config,
                      [&] { return OptimizeRelay(result, config); });
    result = RunStage("lower", config, [&] { return LowerOperators(result); });
    result = RunStage("optimize_tir", config,
                      [&] { return OptimizeTIR(result, config); });
    result = RunStage("build_signature", config,
                      [&] { return BuildSignatures(result, config); });
    result = RunStage("build_backend", config,
                      [&] { return BuildBackends(result, config); });

    profiling::ScopedSpan assemble_span(
        profile_context, MakeStageEvent("assemble", config), run_id);
    CompiledModule module = AssembleModule(result, profile_context);
    AddResultFields(&assemble_span, result);
    if (profile_context) profile_context->Flush();
    return CompiledGraph{std::move(module), result.plan()};
}

}  // namespace

Array<String> Compiler::RelayPassPolicy(int opt_level) {
    if (opt_level < 0 || opt_level > 3) {
        throw std::invalid_argument(
            "Relay pass policy requires opt_level 0..3");
    }
    if (opt_level == 0) return {};
    if (opt_level == 1) {
        return {String("fold_tuple_get_item"), String("fold_constant"),
                String("simplify_expr")};
    }
    return {String("fold_tuple_get_item"), String("fold_constant"),
            String("simplify_expr"), String("canonicalize_cast"),
            String("remove_standalone_reshapes"),
            String("eliminate_dead_let")};
}

Array<String> Compiler::TIRPassPolicy(int opt_level, const Target& target) {
    if (opt_level < 0 || opt_level > 3) {
        throw std::invalid_argument(
            "TIR pass policy requires opt_level 0..3");
    }
    if (!target.defined() || !target.As<TargetNode>()) {
        throw std::invalid_argument(
            "TIR pass policy requires a defined Target");
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
    if (target->kind == "cuda" && target->device_type == kCUDA) {
        return {String("fold_constant"), String("simplify_expr"),
                String("force_narrow_index_to_i32"), String("remove_no_op")};
    }
    return {String("fold_constant"), String("simplify_expr"),
            String("force_narrow_index_to_i32"),
            String("convert_for_loops_serial"), String("loop_partition"),
            String("unroll_loop"), String("vectorize_loop"),
            String("remove_no_op")};
}

CompiledGraph Compiler::Compile(Function function, CompileConfig config) {
    return CompilePipeline(std::move(function), std::move(config));
}
}  // namespace kxc::api
