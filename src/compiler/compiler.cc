/*! \file src/compiler/compiler.cc
 * \brief Implements the target-driven per-operator compiler pipeline.
 */

#include "kxc/compiler/compiler.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "internal/compile_state.h"
#include "internal/compiled_graph_access.h"
#include "internal/execution_contract.h"
#include "internal/identity_private.h"
#include "internal/kernel_abi_equivalence.h"
#include "internal/lowered_graph.h"
#include "internal/primitive_cache.h"
#include "internal/primitive_compiler.h"
#include "internal/relay_program.h"
#include "../runtime/internal/compiled_module_node.h"
#include "kxc/compiler/pipeline.h"
#include "../runtime/internal/memory_plan.h"
#include "kxc/pass/context.h"
#include "kxc/profiling/profiling.h"
#include "kxc/runtime/session.h"
#include "support/canonical.h"
#include "support/hash.h"
#include "kxc/relay/printer/print_ir.h"
#include "kxc/relay/visitor.h"
#include "kxc/tir/printer/print_ir.h"

namespace kxc::api {

struct CompiledGraph::State final {
    CompiledModule module;
    runtime::ExecutablePlan plan;
    std::shared_ptr<const std::vector<ArtifactPin>> artifact_pins;
    GraphSemanticKey graph_semantic_key;
};

namespace {

void ValidateCompiledGraphCandidate(
    const CompiledModule& module, const runtime::ExecutablePlan& plan,
    const std::vector<ArtifactPin>& pins,
    const GraphSemanticKey& graph_semantic_key) {
    if (!module.defined() || !module.IsReady() || !plan.defined() ||
        !graph_semantic_key.defined()) {
        throw std::invalid_argument("CompiledGraph requires defined ready module, plan, and graph key");
    }
    plan.Validate();
    const auto* module_node = module.As<CompiledModuleNode>();
    const auto* target = module_node ? module_node->target_.As<TargetNode>() : nullptr;
    if (!target) {
        throw std::invalid_argument("CompiledGraph requires a valid module target");
    }
    const std::string module_target_capability_fingerprint =
        internal::BuildTargetCapabilityFingerprint(module_node->target_);
    const codegen::CodeGenBackend expected_backend =
        target->kind == "llvm" && target->device_type == kCPU
            ? codegen::CodeGenBackend::kLLVM
            : target->kind == "cuda" && target->device_type == kCUDA
                  ? codegen::CodeGenBackend::kCUDA
                  : throw std::invalid_argument(
                        "CompiledGraph requires a supported target/backend");
    const Array<runtime::KernelCall> calls = plan.calls();
    if (module.entry_count() != calls.size() || pins.size() != calls.size()) {
        throw std::invalid_argument("CompiledGraph requires one module entry and pin per plan call");
    }
    // Reuse the runtime's existing module/plan ABI validator transiently.  It
    // retains no pins and is discarded before the immutable graph is published.
    (void)runtime::RuntimeSession(module, plan);
    std::unordered_set<std::string> call_symbols;
    for (size_t index = 0; index < calls.size(); ++index) {
        const runtime::KernelCall& call = calls[index];
        if (!call_symbols.emplace(std::string(call->symbol)).second ||
            !pins[index].defined() || !module.HasFunction(call->symbol)) {
            throw std::invalid_argument("CompiledGraph plan call lacks its defined module entry or pin");
        }
        const internal::PrimitiveArtifactPin pin =
            internal::ArtifactPinAccess::Unwrap(pins[index]);
        const internal::CachedPrimitive& cached = pin.artifact();
        const ArtifactRecord& record = pins[index].record();
        const auto entry = module_node->entries_.find(std::string(call->symbol));
        const codegen::KernelSignature signature = module.signature(call->symbol);
        const codegen::KernelLaunchMetadata metadata = module.launch_metadata(call->symbol);
        signature.Validate();
        metadata.Validate();
        const std::string signature_bytes = signature.CanonicalBytes();
        const std::string metadata_bytes = metadata.CanonicalBytes();
        if (!internal::SamePhysicalKernelAbi(cached.signature, signature)) {
            throw std::invalid_argument(
                "CompiledGraph cached and relocated physical ABIs differ");
        }
        if (entry == module_node->entries_.end() || !cached.kernel.IsReady() ||
            std::string(signature->symbol) != std::string(call->symbol) ||
            record.artifact_key != pin.key() ||
            record.executable_token != "primitive-v1:" + pin.key().canonical_bytes() ||
            record.signature_digest != support::HashText(cached.signature.CanonicalBytes()) ||
            record.launch_metadata_digest != support::HashText(cached.launch_metadata.CanonicalBytes()) ||
            record.provenance != cached.provenance ||
            record.byte_size != cached.accounted_bytes ||
            record.validation_record != cached.validation_record ||
            pin.key().target_capability_fingerprint() !=
                module_target_capability_fingerprint ||
            cached.kernel.signature().CanonicalBytes() !=
                cached.signature.CanonicalBytes() ||
            entry->second.signature.CanonicalBytes() != signature_bytes ||
            cached.launch_metadata.CanonicalBytes() != metadata_bytes ||
            cached.kernel.launch_metadata().CanonicalBytes() != metadata_bytes ||
            entry->second.launch_metadata.CanonicalBytes() != metadata_bytes ||
            entry->second.executable.signature().CanonicalBytes() != signature_bytes ||
            entry->second.executable.launch_metadata().CanonicalBytes() != metadata_bytes ||
            cached.kernel->launcher != entry->second.executable->launcher ||
            metadata->device.device_type() != target->device_type ||
            metadata->device.device_id() != target->device_id ||
            metadata->backend != expected_backend) {
            throw std::invalid_argument("CompiledGraph pin, module, and target contracts differ");
        }
    }
}

}  // namespace

CompiledGraph::CompiledGraph(std::shared_ptr<const State> state)
    : state_(std::move(state)) {}

CompiledGraph internal::CompiledGraphAccess::Create(
    CompiledModule module, runtime::ExecutablePlan plan,
    std::vector<ArtifactPin> artifact_pins, GraphSemanticKey graph_semantic_key) {
    ValidateCompiledGraphCandidate(module, plan, artifact_pins, graph_semantic_key);
    auto state = std::make_shared<CompiledGraph::State>(CompiledGraph::State{
        std::move(module), std::move(plan),
        std::make_shared<const std::vector<ArtifactPin>>(std::move(artifact_pins)),
        std::move(graph_semantic_key)});
    return CompiledGraph(std::move(state));
}

bool CompiledGraph::defined() const noexcept { return static_cast<bool>(state_); }
const CompiledModule& CompiledGraph::module() const {
    if (!state_) throw std::logic_error("CompiledGraph is undefined");
    return state_->module;
}
const runtime::ExecutablePlan& CompiledGraph::plan() const {
    if (!state_) throw std::logic_error("CompiledGraph is undefined");
    return state_->plan;
}
const std::vector<ArtifactPin>& CompiledGraph::artifact_pins() const {
    if (!state_) throw std::logic_error("CompiledGraph is undefined");
    return *state_->artifact_pins;
}
const GraphSemanticKey& CompiledGraph::graph_semantic_key() const {
    if (!state_) throw std::logic_error("CompiledGraph is undefined");
    return state_->graph_semantic_key;
}
namespace {

void AppendPipelineIdentityField(std::string* canonical,
                                 const std::string& name,
                                 const std::string& value) {
    support::CanonicalBytesEncoder field;
    field.Field(name, value);
    *canonical += std::move(field).Take();
}

bool SameTargetSnapshot(const Target& left, const Target& right) {
    if (!left.defined() || !right.defined()) return false;
    return internal::CanonicalTargetSnapshot(left) ==
           internal::CanonicalTargetSnapshot(right);
}

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
        const std::string text = relay::printer::ToText(result.optimized_relay());
        span->AddField("ir_hash", support::HashText(text));
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
        tir::printer::DumpPrimFunc(primitive.tir, stream);
        const std::string text = stream.str();
        span->AddField(prefix + "symbol", std::string(primitive.symbol));
        span->AddField(prefix + "operator",
                       std::string(primitive.operator_identity));
        span->AddField(prefix + "ir_hash", support::HashText(text));
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

NormalizedPipeline ResolveRelayPipeline(
    const CompileConfig& config,
    const Array<String>& required_control_capabilities = {}) {
    PipelineRequest request;
    request.dialect = IRDialect::kRelay;
    request.requested_scope = PassScope::kGraph;
    request.target = config->target;
    request.opt_level = config->opt_level;
    request.named_pipeline = String("compiler");
    request.required_control_capabilities =
        required_control_capabilities;
    return PipelineResolver::Resolve(request);
}

NormalizedPipeline ResolveTIRPipeline(const CompileConfig& config) {
    PipelineRequest request;
    request.dialect = IRDialect::kTIR;
    request.requested_scope = PassScope::kPrimFunc;
    request.target = config->target;
    request.opt_level = config->opt_level;
    request.named_pipeline = String("compiler");
    return PipelineResolver::Resolve(request);
}

CompileResult CompilePreparedPrimitiveUnits(
    const CompileResult& input, const internal::PreparedStaticGraph& prepared,
    const CompileConfig& config,
    const internal::CompilerExecutionContract& contract) {
    const Device device(input.target()->device_type, input.target()->device_id);
    if (prepared.device != device ||
        !SameTargetSnapshot(prepared.target, input.target())) {
        throw std::invalid_argument(
            "CompilePreparedPrimitiveUnits requires prepared Device/Target identity unchanged");
    }
    internal::CompiledPrimitiveBatch batch =
        internal::CompilePrimitiveUnits(
            prepared.partitioned.units,
            prepared.partitioned.value_graph.values, config, contract);
    std::vector<PrimitiveCompileState> primitives;
    std::vector<tir::PrimFunc> optimized_tir;
    std::vector<codegen::KernelSignature> signatures;
    std::vector<codegen::KernelLaunchMetadata> metadata;
    std::vector<codegen::CompiledKernel> kernels;
    std::vector<bool> cache_hits;
    std::vector<ArtifactPin> pins;
    primitives.reserve(batch.primitives.size());
    optimized_tir.reserve(batch.primitives.size());
    signatures.reserve(batch.primitives.size());
    metadata.reserve(batch.primitives.size());
    kernels.reserve(batch.primitives.size());
    cache_hits.reserve(batch.primitives.size());
    pins.reserve(batch.primitives.size());
    for (const internal::CompiledPrimitive& compiled : batch.primitives) {
        const internal::PrimitiveUnit& unit =
            prepared.partitioned.units.at(
                static_cast<std::size_t>(compiled.unit_id));
        const internal::CachedPrimitive& artifact = compiled.pin.artifact();
        const codegen::KernelSignature signature(
            unit.symbol, artifact.signature.arguments());
        if (!internal::SamePhysicalKernelAbi(artifact.signature, signature) ||
            !artifact.kernel.IsReady() || !artifact.kernel->launcher) {
            throw std::logic_error(
                "compiled primitive cache artifact cannot be relocated");
        }
        const codegen::CompiledKernel kernel(
            signature, artifact.launch_metadata, artifact.kernel->launcher);
        PrimitiveCompileState primitive;
        primitive.unit_id = compiled.unit_id;
        primitive.symbol = unit.symbol;
        primitive.operator_identity =
            String(unit.call.spec.name + "@v" +
                   std::to_string(unit.call.spec.schema_version));
        primitive.semantic_key = unit.semantic_key;
        primitive.tir = compiled.diagnostic_tir;
        primitives.push_back(std::move(primitive));
        optimized_tir.push_back(compiled.diagnostic_tir);
        signatures.push_back(signature);
        metadata.push_back(artifact.launch_metadata);
        kernels.push_back(kernel);
        cache_hits.push_back(compiled.cache_hit);
        pins.push_back(internal::ArtifactPinAccess::Wrap(compiled.pin));
    }
    CompileResult result = input.AfterLowering(
        std::move(primitives),
        runtime::internal::PlanMemory(
            internal::BuildStaticExecutablePlan(prepared)),
        batch.constants);
    result = result.AfterTIROptimization(std::move(optimized_tir));
    result = result.AfterSignatures(std::move(signatures));
    return result.AfterBackends(
        std::move(metadata), std::move(kernels), std::move(cache_hits),
        std::move(pins));
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
            *primitive.signature, *primitive.launch_metadata, *primitive.kernel});
    }
    return internal::BuildCompiledModule(
        result.target(), std::move(entries), result.constants(),
        std::move(profile_context));
}

CompiledGraph CompilePipeline(
    Function function, CompileConfig config,
    const internal::ControlFlowPolicy& policy,
    const internal::CompilerExecutionContract* expected_contract = nullptr) {
    config.Validate();
    const GraphSemanticKey graph_semantic_key =
        Compiler::BuildGraphSemanticKey(function);
    auto profile_context = MaybeCreateProfileContext(config);
    const std::string run_id =
        profile_context ? profile_context->NextRunId("compile") : "";
    profiling::ActivationScope activation(profile_context, run_id);
    profiling::ScopedSpan compile_span(
        profile_context, MakeStageEvent("compile", config), run_id);

    const PassContext pass_context = PassContext::MergeTarget(
        relay::PassContextFromRelay(function), config->target);
    PassContext::Scope pass_scope(pass_context);

    internal::PreparedRelayProgram prepared = [&] {
        profiling::ScopedSpan span(
            profiling::CurrentContext(),
            MakeStageEvent("prepare_relay", config));
        try {
            return internal::PrepareRelayProgram(function, config, policy);
        } catch (const std::exception& error) {
            span.SetStatus("error");
            span.SetMessage(error.what());
            throw std::runtime_error(
                std::string("Compiler stage 'prepare_relay' failed: ") +
                error.what());
        }
    }();
    const internal::PreparedProgramPlan program_plan =
        internal::PlanRelayProgram(prepared);
    if (!std::holds_alternative<internal::PreparedStaticPlan>(program_plan)) {
        throw std::logic_error(
            "Compiler::Compile cannot publish a structured-control plan");
    }
    const internal::CompilerExecutionContract& contract =
        prepared.execution_contract();
    if (expected_contract &&
        expected_contract->canonical_bytes != contract.canonical_bytes) {
        throw std::invalid_argument(
            "Compiler execution contract does not match prepared Relay program");
    }
    CompileResult result =
        CompileResult::Validate(config->target, prepared.typed_anf())
            .AfterRelayOptimization(prepared.typed_anf());
    const Device device(result.target()->device_type,
                        result.target()->device_id);
    internal::PreparedStaticGraph static_graph =
        internal::PrepareStaticGraph(
            result.optimized_relay(), device, result.target(),
            String(contract.fingerprint));
    result = RunStage(
        "compile_primitives", config,
        [&] {
            return CompilePreparedPrimitiveUnits(
                result, static_graph, config, contract);
        });

    profiling::ScopedSpan assemble_span(
        profile_context, MakeStageEvent("assemble", config), run_id);
    CompiledModule module = AssembleModule(result, profile_context);
    runtime::ExecutablePlan plan = result.plan();
    std::vector<ArtifactPin> artifact_pins = result.artifact_pins();
    AddResultFields(&assemble_span, result);
    if (profile_context) profile_context->Flush();
    return internal::CompiledGraphAccess::Create(
        std::move(module), std::move(plan), std::move(artifact_pins),
        graph_semantic_key);
}

}  // namespace

Target internal::CloneTargetSnapshot(const Target& target) {
    const auto* source = target.As<TargetNode>();
    if (!source) {
        throw std::invalid_argument(
            "CloneTargetSnapshot requires a defined TargetNode");
    }
    auto* copy = new TargetNode();
    copy->kind = source->kind;
    copy->device_type = source->device_type;
    copy->device_id = source->device_id;
    copy->attrs = source->attrs;
    return Target(ObjectRef(copy));
}

std::string internal::CanonicalTargetSnapshot(const Target& target) {
    const auto* node = target.As<TargetNode>();
    if (!node) {
        throw std::invalid_argument(
            "CanonicalTargetSnapshot requires a defined TargetNode");
    }
    std::string canonical;
    AppendPipelineIdentityField(&canonical, "kind", "target-snapshot-v1");
    AppendPipelineIdentityField(&canonical, "target_kind", node->kind);
    AppendPipelineIdentityField(
        &canonical, "device_type",
        std::to_string(static_cast<int>(node->device_type)));
    AppendPipelineIdentityField(&canonical, "device_id",
                                std::to_string(node->device_id));
    const DeviceAttributes& attrs = node->attrs;
    const auto append_integer = [&canonical](const char* name, int64_t value) {
        AppendPipelineIdentityField(&canonical, name, std::to_string(value));
    };
    append_integer("exists", attrs.exists);
    append_integer("max_threads_per_block", attrs.max_threads_per_block);
    append_integer("warp_size", attrs.warp_size);
    append_integer("max_shared_memory_per_block",
                   attrs.max_shared_memory_per_block);
    AppendPipelineIdentityField(&canonical, "compute_version",
                                attrs.compute_version);
    AppendPipelineIdentityField(&canonical, "device_name", attrs.device_name);
    append_integer("max_clock_rate_khz", attrs.max_clock_rate_khz);
    append_integer("max_registers_per_block", attrs.max_registers_per_block);
    append_integer("api_version", attrs.api_version);
    append_integer("driver_version", attrs.driver_version);
    append_integer("l2_cache_size_bytes", attrs.l2_cache_size_bytes);
    append_integer("total_global_memory", attrs.total_global_memory);
    // available_global_memory is a volatile observation, not a codegen
    // capability. It is deliberately excluded from reusable identity.
    append_integer("max_shared_memory_per_multiprocessor",
                   attrs.max_shared_memory_per_multiprocessor);
    append_integer("max_registers_per_multiprocessor",
                   attrs.max_registers_per_multiprocessor);
    append_integer("max_threads_per_multiprocessor",
                   attrs.max_threads_per_multiprocessor);
    append_integer("compute_version_major", attrs.compute_version_major);
    append_integer("compute_version_minor", attrs.compute_version_minor);
    append_integer("multi_processor_count", attrs.multi_processor_count);
    AppendPipelineIdentityField(&canonical, "arch", attrs.arch);
    return canonical;
}

internal::CompilerExecutionContract
internal::ResolveCompilerExecutionContract(const CompileConfig& config) {
    return ResolveCompilerExecutionContract(config, {});
}

internal::CompilerExecutionContract
internal::ResolveCompilerExecutionContract(
    const CompileConfig& config,
    const Array<String>& required_relay_control_capabilities) {
    config.Validate();
    CompilerExecutionContract contract;
    contract.relay_pipeline = ResolveRelayPipeline(
        config, required_relay_control_capabilities);
    contract.tir_pipeline = ResolveTIRPipeline(config);
    contract.schedule_version = "per-unit-schedule-v2";
    contract.backend_version = BackendVersion(config->target);
    AppendPipelineIdentityField(&contract.canonical_bytes, "kind",
                                "compiler-execution-plan-v3");
    AppendPipelineIdentityField(
        &contract.canonical_bytes, "relay",
        std::string(contract.relay_pipeline.canonical_bytes));
    AppendPipelineIdentityField(&contract.canonical_bytes, "lowering",
                                "per-unit-boundary-lowering-v1");
    AppendPipelineIdentityField(
        &contract.canonical_bytes, "tir",
        std::string(contract.tir_pipeline.canonical_bytes));
    AppendPipelineIdentityField(&contract.canonical_bytes, "kernel_abi",
                                "kernel-abi-v1");
    AppendPipelineIdentityField(&contract.canonical_bytes, "schedule",
                                contract.schedule_version);
    AppendPipelineIdentityField(&contract.canonical_bytes, "backend",
                                contract.backend_version);
    contract.fingerprint = support::HashText(contract.canonical_bytes);
    return contract;
}

void internal::ProbeCompilerExecution(
    Function function, CompileConfig config,
    const CompilerExecutionContract& contract) {
    (void)CompilePipeline(
        std::move(function), std::move(config),
        ControlFlowPolicy::StaticOnly(), &contract);
}

internal::PreparedCompilerGraph internal::PrepareCompilerGraph(
    Function function, CompileConfig config,
    const CompilerExecutionContract& contract) {
    config.Validate();
    const GraphSemanticKey graph_semantic_key =
        Compiler::BuildGraphSemanticKey(function);
    auto profile_context = MaybeCreateProfileContext(config);
    const std::string run_id =
        profile_context ? profile_context->NextRunId("shape_exact") : "";
    profiling::ActivationScope activation(profile_context, run_id);
    PreparedRelayProgram prepared = PrepareRelayProgram(
        std::move(function), config, ControlFlowPolicy::StaticOnly());
    if (prepared.execution_contract().canonical_bytes !=
        contract.canonical_bytes) {
        throw std::invalid_argument(
            "PrepareCompilerGraph execution contract does not match "
            "prepared Relay program");
    }
    CompileResult result =
        CompileResult::Validate(config->target, prepared.typed_anf())
            .AfterRelayOptimization(prepared.typed_anf());
    size_t capability_boundary_checks = 1;
    size_t relay_graph_pipelines = 1;
    const Device device(result.target()->device_type, result.target()->device_id);
    profiling::ScopedSpan prepare_span(
        profile_context, MakeStageEvent("prepare_graph", config), run_id);
    PreparedStaticGraph graph;
    try {
        graph = PrepareStaticGraph(result.optimized_relay(), device,
                                   result.target(),
                                   String(contract.fingerprint));
    } catch (const std::exception& error) {
        prepare_span.SetStatus("error");
        prepare_span.SetMessage(error.what());
        throw std::runtime_error(
            std::string("Compiler stage 'prepare_graph' failed: ") +
            error.what());
    }
    capability_boundary_checks += graph.capability_boundary_checks;
    const size_t value_graph_builds = graph.value_graph_builds;
    const size_t partitions = graph.partitions;
    return PreparedCompilerGraph{
        graph_semantic_key, result, std::move(graph), result.target(),
        contract.canonical_bytes,
        std::move(profile_context), run_id, relay_graph_pipelines,
        capability_boundary_checks, value_graph_builds, partitions};
}

CompiledGraph internal::FinishCompilerGraph(
    const PreparedCompilerGraph& prepared, CompileConfig config,
    const CompilerExecutionContract& contract) {
    config.Validate();
    if (prepared.execution_contract_canonical != contract.canonical_bytes ||
        !SameTargetSnapshot(prepared.target, config->target) ||
        !SameTargetSnapshot(prepared.optimized.target(), config->target) ||
        !SameTargetSnapshot(prepared.graph.target, config->target) ||
        prepared.graph.device !=
            Device(config->target->device_type, config->target->device_id) ||
        std::string(prepared.graph.pipeline_fingerprint) !=
            contract.fingerprint) {
        throw std::invalid_argument(
            "FinishCompilerGraph requires the prepared target and execution contract unchanged");
    }
    const PassContext pass_context = PassContext::MergeTarget(
        relay::PassContextFromRelay(prepared.optimized.optimized_relay()), config->target);
    PassContext::Scope pass_scope(pass_context);
    auto profile_context = prepared.profile_context;
    const std::string& run_id = prepared.profile_run_id;
    profiling::ActivationScope activation(profile_context, run_id);
    CompileResult result = RunStage(
        "compile_primitives", config,
        [&] {
            return CompilePreparedPrimitiveUnits(
                prepared.optimized, prepared.graph, config, contract);
        });
    profiling::ScopedSpan assemble_span(profile_context, MakeStageEvent("assemble", config), run_id);
    CompiledModule module = AssembleModule(result, profile_context);
    runtime::ExecutablePlan plan = result.plan();
    std::vector<ArtifactPin> artifact_pins = result.artifact_pins();
    AddResultFields(&assemble_span, result);
    if (profile_context) profile_context->Flush();
    return internal::CompiledGraphAccess::Create(
        std::move(module), std::move(plan), std::move(artifact_pins),
        prepared.graph_semantic_key);
}

GraphSemanticKey Compiler::BuildGraphSemanticKey(
    const Function& function) {
    if (!function.defined()) {
        throw std::invalid_argument(
            "graph semantic identity requires a defined Function");
    }
    return internal::BuildGraphSemanticKey(function);
}

CompiledGraph Compiler::Compile(Function function, CompileConfig config) {
    return CompilePipeline(
        std::move(function), std::move(config),
        internal::ControlFlowPolicy::StaticOnly());
}
}  // namespace kxc::api
