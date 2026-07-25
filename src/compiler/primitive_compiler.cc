/*! \file src/compiler/primitive_compiler.cc
 * \brief One production primitive compiler shared by all Relay topologies.
 */

#include "internal/primitive_compiler.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <memory>
#include <optional>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "internal/kernel_abi_builder.h"
#include "internal/kernel_abi_equivalence.h"
#include "internal/lowered_graph.h"
#include "internal/primitive_cache.h"
#include "../runtime/internal/compiled_module_node.h"
#include "kxc/compiler/pipeline.h"
#include "kxc/tir/printer/print_ir.h"
#include "kxc/tir/transforms/bind_cuda_threads.h"

#if KXC_USE_LLVM
#include <llvm/IR/LLVMContext.h>

#include "../codegen/llvm/internal/codegen_llvm.h"
#include "../codegen/llvm/internal/llvm_jit.h"
#endif

#if KXC_USE_CUDA
#include "../codegen/cuda/internal/codegen_cuda.h"
#include "../codegen/cuda/internal/cuda_module.h"
#endif

namespace kxc::api::internal {
namespace {

struct PrimitiveWork final {
    const PrimitiveUnit* unit{nullptr};
    tir::PrimFunc tir;
    codegen::KernelSignature signature;
    PrimitiveArtifactKey artifact_key;
    PrimitiveCacheLease lease;
    PrimitiveArtifactPin pin;
    std::optional<codegen::KernelLaunchMetadata> launch_metadata;
    std::optional<codegen::CompiledKernel> kernel;
};

std::string Context(const PrimitiveUnit& unit) {
    return "primitive unit " + std::to_string(unit.id) + " ('" +
           std::string(unit.symbol) + "', " + unit.call.spec.name + ")";
}

template <typename Fn>
auto RunUnitPhase(const char* phase, const PrimitiveUnit& unit, Fn&& fn)
    -> decltype(fn()) {
    try {
        return fn();
    } catch (const std::exception& error) {
        throw std::runtime_error(
            std::string(phase) + " failed for " + Context(unit) + ": " +
            error.what());
    }
}

String ReadKernelSymbol(const tir::PrimFunc& function) {
    const String key("global_symbol");
    if (!function->attrs.count(key)) {
        throw std::invalid_argument(
            "optimized PrimFunc is missing global_symbol");
    }
    const auto* value = function->attrs.at(key).As<StringObj>();
    if (!value || value->data.empty()) {
        throw std::invalid_argument(
            "PrimFunc global_symbol must be a non-empty String");
    }
    const std::string& symbol = value->data;
    const auto is_head = [](const unsigned char ch) {
        return std::isalpha(ch) != 0 || ch == '_';
    };
    const auto is_tail = [](const unsigned char ch) {
        return std::isalnum(ch) != 0 || ch == '_';
    };
    if (!is_head(static_cast<unsigned char>(symbol[0]))) {
        throw std::invalid_argument(
            "PrimFunc global_symbol is not a C identifier");
    }
    for (std::size_t index = 1; index < symbol.size(); ++index) {
        if (!is_tail(static_cast<unsigned char>(symbol[index]))) {
            throw std::invalid_argument(
                "PrimFunc global_symbol is not a C identifier");
        }
    }
    return String(symbol);
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

uint64_t AccountedTIRBytes(const tir::PrimFunc& function) {
    std::ostringstream out;
    tir::printer::DumpPrimFunc(function, out);
    return std::max<uint64_t>(1, static_cast<uint64_t>(out.str().size()));
}

codegen::CompiledKernel RelocateCachedKernel(
    const PrimitiveArtifactPin& pin,
    const codegen::KernelSignature& current_signature,
    const std::string& context) {
    const CachedPrimitive& artifact = pin.artifact();
    if (!SamePhysicalKernelAbi(artifact.signature, current_signature)) {
        throw std::logic_error(context +
                               " cached artifact physical ABI changed");
    }
    if (!artifact.kernel.IsReady() || !artifact.kernel->launcher) {
        throw std::logic_error(context +
                               " cached artifact is not executable");
    }
    return codegen::CompiledKernel(
        current_signature, artifact.launch_metadata,
        artifact.kernel->launcher);
}

class PrimitiveOwnerGuard final {
public:
    explicit PrimitiveOwnerGuard(std::vector<PrimitiveWork>* work)
        : work_(work) {}

    ~PrimitiveOwnerGuard() {
        if (dismissed_ || work_ == nullptr) return;
        for (const PrimitiveWork& item : *work_) {
            if (item.lease.access() != PrimitiveCacheAccess::kOwner) continue;
            try {
                FailPrimitiveCacheLease(
                    item.lease, PrimitiveFailureCategory::kCompile,
                    "compile owner abandoned before publishing a validated artifact",
                    std::chrono::seconds(1));
            } catch (...) {
            }
        }
    }

    void Dismiss() noexcept { dismissed_ = true; }

private:
    std::vector<PrimitiveWork>* work_{nullptr};
    bool dismissed_{false};
};

PrimitiveArtifactPin RequirePrimitivePin(
    const PrimitiveCacheLease& lease, const std::string& context) {
    try {
        return WaitPrimitiveCacheLease(lease);
    } catch (const std::exception& error) {
        throw std::runtime_error(
            context + " primitive cache transaction failed: " + error.what());
    }
}

void CompileBackendMisses(std::vector<PrimitiveWork>* work,
                          const std::vector<std::size_t>& misses,
                          const CompileConfig& config) {
    if (misses.empty()) return;
    const Target target = config->target;
    const Device device(target->device_type, target->device_id);
    if (target->kind == "llvm" && target->device_type == kCPU) {
#if KXC_USE_LLVM
        auto llvm_context = std::make_unique<llvm::LLVMContext>();
        codegen::CodeGenLLVM codegen(*llvm_context);
        std::vector<std::pair<tir::PrimFunc, std::string>> functions;
        std::vector<codegen::KernelSignature> signatures;
        std::vector<codegen::KernelLaunchMetadata> metadata;
        functions.reserve(misses.size());
        signatures.reserve(misses.size());
        metadata.reserve(misses.size());
        for (const std::size_t index : misses) {
            const PrimitiveWork& item = work->at(index);
            functions.emplace_back(item.tir,
                                   std::string(item.unit->symbol));
            signatures.push_back(item.signature);
            metadata.emplace_back(device, codegen::CodeGenBackend::kLLVM);
        }
        codegen.AddFunctions(functions);
        codegen::LLVMJITEngine jit;
        std::vector<codegen::CompiledKernel> compiled = jit.CompileMany(
            codegen.TakeModule(), std::move(llvm_context), signatures,
            metadata, config->opt_level);
        for (std::size_t offset = 0; offset < misses.size(); ++offset) {
            PrimitiveWork& item = work->at(misses[offset]);
            item.launch_metadata = metadata[offset];
            item.kernel = compiled[offset];
        }
#else
        throw std::runtime_error(
            "Compiler target 'llvm' requires a build with KXC_ENABLE_LLVM=ON");
#endif
        return;
    }
    if (target->kind == "cuda" && target->device_type == kCUDA) {
#if KXC_USE_CUDA
        std::vector<std::pair<tir::PrimFunc, std::string>> functions;
        std::vector<codegen::KernelSignature> signatures;
        std::vector<codegen::KernelLaunchMetadata> metadata;
        functions.reserve(misses.size());
        signatures.reserve(misses.size());
        metadata.reserve(misses.size());
        for (const std::size_t index : misses) {
            const PrimitiveWork& item = work->at(index);
            const tir::CudaLaunchConfig launch_config =
                tir::GetCudaLaunchConfig(item.tir);
            functions.emplace_back(item.tir,
                                   std::string(item.unit->symbol));
            signatures.push_back(item.signature);
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
        for (std::size_t offset = 0; offset < misses.size(); ++offset) {
            PrimitiveWork& item = work->at(misses[offset]);
            item.launch_metadata = metadata[offset];
            item.kernel = compiled[offset];
        }
#else
        throw std::runtime_error(
            "Compiler target 'cuda' requires a build with KXC_ENABLE_CUDA=ON");
#endif
        return;
    }
    throw std::runtime_error("Compiler Target has no matching backend");
}

}  // namespace

CompiledPrimitiveBatch CompilePrimitiveUnits(
    const std::vector<PrimitiveUnit>& units,
    const std::vector<LogicalValueContract>& values,
    const CompileConfig& config,
    const CompilerExecutionContract& contract,
    const std::vector<PrimitiveUnitId>& requested_unit_ids) {
    if (units.empty()) {
        throw std::invalid_argument(
            "CompilePrimitiveUnits requires at least one PrimitiveUnit");
    }
    for (std::size_t index = 0; index < units.size(); ++index) {
        if (units[index].id != static_cast<PrimitiveUnitId>(index)) {
            throw std::invalid_argument(
                "CompilePrimitiveUnits requires dense ordered unit ids");
        }
    }
    for (std::size_t index = 0; index < requested_unit_ids.size(); ++index) {
        const PrimitiveUnitId id = requested_unit_ids[index];
        if (id < 0 || static_cast<std::size_t>(id) >= units.size()) {
            throw std::invalid_argument(
                "CompilePrimitiveUnits requested unit id is out of range");
        }
        if (index != 0 && id <= requested_unit_ids[index - 1]) {
            throw std::invalid_argument(
                "CompilePrimitiveUnits requested unit ids must be strictly increasing");
        }
    }
    config.Validate();
    std::vector<std::pair<const PrimitiveUnit*, tir::PrimFunc>> lowered_units;
    lowered_units.reserve(requested_unit_ids.size());
    Map<String, runtime::NDArray> constants;
    for (const PrimitiveUnitId id : requested_unit_ids) {
        const PrimitiveUnit& unit = units[static_cast<std::size_t>(id)];
        ValidatePrimitiveUnit(unit, values);
        relay::LoweredFunction lowered = RunUnitPhase(
            "per_unit_lowering", unit,
            [&] { return LowerPrimitiveUnit(values, unit); });
        for (const auto& binding : lowered.constants()) {
            if (constants.count(binding->key) &&
                constants.at(binding->key).get() != binding->value.get()) {
                throw std::invalid_argument(
                    "Primitive constant key resolved to different payloads");
            }
            constants.Set(binding->key, binding->value);
        }
        tir::PrimFunc optimized = RunUnitPhase(
            "optimize_tir", unit,
            [&] {
                return PipelineExecutor::ExecuteTIR(
                    contract.tir_pipeline, lowered->prim_func,
                    config->target);
            });
        const String symbol = ReadKernelSymbol(optimized);
        if (!(symbol == unit.symbol)) {
            throw std::invalid_argument(
                Context(unit) + " PrimFunc symbol drifted");
        }
        lowered_units.emplace_back(&unit, std::move(optimized));
    }
    constants = PlaceConstants(constants, config->target);
    std::vector<PrimitiveWork> work;
    work.reserve(lowered_units.size());
    for (auto& lowered : lowered_units) {
        const PrimitiveUnit& unit = *lowered.first;
        codegen::KernelSignature signature = RunUnitPhase(
            "build_signature", unit,
            [&] {
                return codegen::BuildKernelSignature(
                    lowered.second, constants, config->target, unit.symbol);
            });
        PrimitiveArtifactKey artifact_key = BuildPrimitiveArtifactKey(
            unit.semantic_key, config->target, contract.canonical_bytes,
            contract.schedule_version.c_str(),
            contract.backend_version.c_str());
        work.push_back(PrimitiveWork{
            &unit, std::move(lowered.second), std::move(signature),
            std::move(artifact_key), PrimitiveCacheLease(),
            PrimitiveArtifactPin(), std::nullopt, std::nullopt});
    }

    std::vector<std::size_t> misses;
    PrimitiveOwnerGuard owner_guard(&work);
    for (std::size_t index = 0; index < work.size(); ++index) {
        PrimitiveWork& item = work[index];
        item.lease = AcquirePrimitiveCache(item.artifact_key);
        switch (item.lease.access()) {
            case PrimitiveCacheAccess::kOwner:
                misses.push_back(index);
                break;
            case PrimitiveCacheAccess::kHit:
                item.pin = item.lease.pin();
                break;
            case PrimitiveCacheAccess::kFailed:
            case PrimitiveCacheAccess::kRejected:
                (void)RequirePrimitivePin(item.lease, Context(*item.unit));
                break;
            case PrimitiveCacheAccess::kWait:
                break;
        }
    }

    CompileBackendMisses(&work, misses, config);
    for (const std::size_t index : misses) {
        PrimitiveWork& item = work[index];
        if (!item.launch_metadata || !item.kernel) {
            throw std::logic_error(
                Context(*item.unit) +
                " backend batch did not produce an executable");
        }
        item.pin = PublishPrimitiveCacheLease(
            item.lease,
            CachedPrimitive{
                item.signature, *item.launch_metadata, *item.kernel,
                AccountedTIRBytes(item.tir), "CompilePrimitiveUnits",
                "signature+backend-validated"});
    }

    CompiledPrimitiveBatch result;
    result.constants = std::move(constants);
    result.primitives.reserve(work.size());
    for (PrimitiveWork& item : work) {
        if (!item.pin.defined()) {
            item.pin =
                RequirePrimitivePin(item.lease, Context(*item.unit));
        }
        result.primitives.push_back(CompiledPrimitive{
            item.unit->id, std::move(item.tir), std::move(item.pin),
            item.lease.access() != PrimitiveCacheAccess::kOwner});
    }
    owner_guard.Dismiss();
    return result;
}

CompiledPrimitiveBatch CompilePrimitiveUnits(
    const std::vector<PrimitiveUnit>& units,
    const std::vector<LogicalValueContract>& values,
    const CompileConfig& config,
    const CompilerExecutionContract& contract) {
    std::vector<PrimitiveUnitId> requested_unit_ids(units.size());
    std::iota(requested_unit_ids.begin(), requested_unit_ids.end(), 0);
    return CompilePrimitiveUnits(
        units, values, config, contract, requested_unit_ids);
}

CompiledModule AssemblePrimitiveModule(
    const CompiledPrimitiveBatch& batch,
    const std::vector<PrimitiveUnit>& units, const Target& target,
    std::shared_ptr<profiling::ProfileContext> profile_context) {
    if (batch.primitives.empty()) {
        throw std::invalid_argument(
            "AssemblePrimitiveModule requires compiled primitives");
    }
    std::vector<CompiledModuleEntry> entries;
    entries.reserve(batch.primitives.size());
    for (const CompiledPrimitive& primitive : batch.primitives) {
        if (primitive.unit_id < 0 ||
            static_cast<std::size_t>(primitive.unit_id) >= units.size()) {
            throw std::invalid_argument(
                "AssemblePrimitiveModule primitive unit id is out of range");
        }
        const PrimitiveUnit& unit =
            units[static_cast<std::size_t>(primitive.unit_id)];
        if (unit.id != primitive.unit_id || !primitive.pin.defined()) {
            throw std::invalid_argument(
                "AssemblePrimitiveModule primitive artifact is invalid");
        }
        const CachedPrimitive& artifact = primitive.pin.artifact();
        const codegen::KernelSignature signature(
            unit.symbol, artifact.signature.arguments());
        entries.push_back(CompiledModuleEntry{
            signature, artifact.launch_metadata,
            RelocateCachedKernel(primitive.pin, signature, Context(unit))});
    }
    return BuildCompiledModule(
        target, std::move(entries), batch.constants,
        std::move(profile_context));
}

}  // namespace kxc::api::internal
