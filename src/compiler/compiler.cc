/*! \file src/compiler/compiler.cc
 * \brief Implements the target-driven per-operator compiler pipeline.
 */

#include "kxc/compiler/compiler.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

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
#include "kxc/pass/context.h"
#include "kxc/profiling/profiling.h"
#include "kxc/runtime/session.h"
#include "support/canonical.h"
#include "support/hash.h"
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

void AddPrimitiveBatchFields(
    profiling::ScopedSpan* span,
    const internal::PreparedCompilerGraph& prepared,
    const internal::CompiledPrimitiveBatch& batch) {
    const runtime::ExecutablePlan plan =
        internal::BuildStaticExecutablePlan(prepared.graph);
    span->AddMetric("primitive_count",
                    static_cast<double>(batch.primitives.size()));
    span->AddMetric("value_count", static_cast<double>(plan.values().size()));
    std::unordered_set<int64_t> storage_ids;
    for (const auto& value : plan.values()) storage_ids.insert(value->storage_id);
    span->AddMetric("storage_slot_count",
                    static_cast<double>(storage_ids.size()));
    span->AddMetric("storage_reuse_count",
                    static_cast<double>(plan.values().size() - storage_ids.size()));
    span->AddMetric("planned_peak_storage_bytes",
                    static_cast<double>(PlannedStorageBytes(plan)));
    size_t cache_hits = 0;
    for (const internal::CompiledPrimitive& primitive : batch.primitives) {
        const internal::PrimitiveUnit& unit = prepared.graph.partitioned.units.at(
            static_cast<std::size_t>(primitive.unit_id));
        const std::string prefix = "unit." +
            std::to_string(primitive.unit_id) + ".";
        std::ostringstream stream;
        tir::printer::DumpPrimFunc(primitive.diagnostic_tir, stream);
        const std::string text = stream.str();
        const internal::CachedPrimitive& artifact = primitive.pin.artifact();
        span->AddField(prefix + "symbol", std::string(unit.symbol));
        span->AddField(
            prefix + "operator", std::string(unit.call.spec.name) + "@v" +
                std::to_string(unit.call.spec.schema_version));
        span->AddField(prefix + "ir_hash", support::HashText(text));
        span->AddMetric(prefix + "ir_bytes",
                        static_cast<double>(text.size()));
        span->AddField(prefix + "backend",
                       artifact.launch_metadata->backend ==
                               codegen::CodeGenBackend::kLLVM
                           ? "llvm"
                           : "cuda");
        span->AddMetric(prefix + "cache_hit", primitive.cache_hit ? 1.0 : 0.0);
        if (primitive.cache_hit) ++cache_hits;
    }
    span->AddMetric("cache_hits", static_cast<double>(cache_hits));
    span->AddMetric(
        "cache_hit_rate", batch.primitives.empty()
                              ? 0.0
                              : static_cast<double>(cache_hits) /
                                    static_cast<double>(batch.primitives.size()));
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

CompiledGraph CompilePipeline(
    Function function, CompileConfig config,
    const internal::CompilerExecutionContract* expected_contract = nullptr) {
    config.Validate();
    auto profile_context = MaybeCreateProfileContext(config);
    const std::string run_id =
        profile_context ? profile_context->NextRunId("compile") : "";
    profiling::ActivationScope activation(profile_context, run_id);
    profiling::ScopedSpan compile_span(
        profile_context, MakeStageEvent("compile", config), run_id);
    const internal::CompilerExecutionContract contract =
        internal::ResolveCompilerExecutionContract(config);
    if (expected_contract &&
        expected_contract->canonical_bytes != contract.canonical_bytes) {
        throw std::invalid_argument(
            "Compiler execution contract does not match prepared Relay program");
    }
    const internal::PreparedCompilerGraph prepared =
        internal::PrepareCompilerGraph(function, config, contract);
    internal::CompiledPrimitiveBatch batch;
    {
        const PassContext pass_context = PassContext::MergeTarget(
            relay::PassContextFromRelay(
                prepared.optimized.optimized_relay()), config->target);
        PassContext::Scope pass_scope(pass_context);
        profiling::ScopedSpan primitive_span(
            profile_context, MakeStageEvent("compile_primitives", config), run_id);
        try {
            batch = internal::CompilePrimitiveUnits(
                prepared.graph.partitioned.units,
                prepared.graph.partitioned.value_graph.values, config, contract);
            AddPrimitiveBatchFields(&primitive_span, prepared, batch);
        } catch (const std::exception& error) {
            primitive_span.SetStatus("error");
            primitive_span.SetMessage(error.what());
            throw std::runtime_error(
                std::string("Compiler stage 'compile_primitives' failed: ") +
                error.what());
        }
    }
    std::vector<internal::PrimitiveArtifactPin> pins;
    pins.reserve(batch.primitives.size());
    for (const internal::CompiledPrimitive& primitive : batch.primitives) {
        pins.push_back(primitive.pin);
    }
    profiling::ScopedSpan assemble_span(
        profile_context, MakeStageEvent("assemble", config), run_id);
    CompiledGraph result = internal::AssembleCompiledGraph(
        prepared, pins, batch.constants);
    AddPrimitiveBatchFields(&assemble_span, prepared, batch);
    if (profile_context) profile_context->Flush();
    return result;
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
    (void)CompilePipeline(std::move(function), std::move(config), &contract);
}

internal::PreparedCompilerGraph internal::PrepareCompilerGraph(
    Function function, CompileConfig config,
    const CompilerExecutionContract& contract) {
    config.Validate();
    const GraphSemanticKey graph_semantic_key =
        Compiler::BuildGraphSemanticKey(function);
    auto profile_context = MaybeCreateProfileContext(config);
    const std::string run_id =
        profile_context && profiling::CurrentContext() == profile_context &&
                !profiling::CurrentRunId().empty()
            ? profiling::CurrentRunId()
            : profile_context ? profile_context->NextRunId("shape_exact") : "";
    profiling::ActivationScope activation(profile_context, run_id);
    PreparedRelayProgram prepared = [&] {
        profiling::ScopedSpan span(
            profile_context, MakeStageEvent("prepare_relay", config), run_id);
        try {
            return PrepareRelayProgram(
                std::move(function), config, ControlFlowPolicy::StaticOnly());
        } catch (const std::exception& error) {
            span.SetStatus("error");
            span.SetMessage(error.what());
            throw std::runtime_error(
                std::string("Compiler stage 'prepare_relay' failed: ") +
                error.what());
        }
    }();
    if (prepared.residual_profile().requires_control_topology()) {
        throw std::logic_error(
            "Compiler::Compile cannot publish a structured-control plan");
    }
    if (prepared.execution_contract().canonical_bytes !=
        contract.canonical_bytes) {
        throw std::invalid_argument(
            "PrepareCompilerGraph execution contract does not match "
            "prepared Relay program");
    }
    size_t capability_boundary_checks =
        prepared.capability_boundary_checks();
    size_t relay_graph_pipelines = 1;
    const Target target = prepared.target();
    const Device device(target->device_type, target->device_id);
    profiling::ScopedSpan prepare_span(
        profile_context, MakeStageEvent("prepare_graph", config), run_id);
    PreparedStaticGraph graph;
    try {
        graph = PrepareStaticGraph(prepared.typed_anf(), device, target,
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
        graph_semantic_key, std::move(graph), target,
        contract.canonical_bytes,
        std::move(profile_context), run_id, relay_graph_pipelines,
        capability_boundary_checks, value_graph_builds, partitions};
}

CompiledGraph internal::AssembleCompiledGraph(
    const PreparedCompilerGraph& prepared,
    const std::vector<PrimitiveArtifactPin>& ordered_pins,
    const Map<String, runtime::NDArray>& constants) {
    const std::vector<PrimitiveUnit>& units = prepared.graph.partitioned.units;
    if (ordered_pins.size() != units.size()) {
        throw std::invalid_argument(
            "AssembleCompiledGraph requires one pin per PrimitiveUnit");
    }
    if (!prepared.target.defined() || !prepared.graph.target.defined() ||
        CanonicalTargetSnapshot(prepared.target) !=
            CanonicalTargetSnapshot(prepared.graph.target)) {
        throw std::invalid_argument(
            "AssembleCompiledGraph prepared targets differ");
    }
    const std::string target_fingerprint =
        BuildTargetCapabilityFingerprint(prepared.target);
    std::vector<CompiledModuleEntry> entries;
    std::vector<ArtifactPin> public_pins;
    entries.reserve(ordered_pins.size());
    public_pins.reserve(ordered_pins.size());
    for (std::size_t index = 0; index < ordered_pins.size(); ++index) {
        const PrimitiveUnit& unit = units[index];
        const PrimitiveArtifactPin& pin = ordered_pins[index];
        if (unit.id != static_cast<PrimitiveUnitId>(index) ||
            !pin.defined() ||
            pin.key().unit_semantic_key() != unit.semantic_key ||
            pin.key().target_capability_fingerprint() != target_fingerprint) {
            throw std::invalid_argument(
                "AssembleCompiledGraph ordered pin does not match its PrimitiveUnit");
        }
        const CachedPrimitive& artifact = pin.artifact();
        const codegen::KernelSignature signature(
            unit.symbol, artifact.signature.arguments());
        if (!SamePhysicalKernelAbi(artifact.signature, signature) ||
            !artifact.kernel.IsReady() || !artifact.kernel->launcher) {
            throw std::invalid_argument(
                "AssembleCompiledGraph pin artifact cannot be relocated");
        }
        entries.push_back(CompiledModuleEntry{
            signature, artifact.launch_metadata,
            codegen::CompiledKernel(
                signature, artifact.launch_metadata, artifact.kernel->launcher)});
        public_pins.push_back(ArtifactPinAccess::Wrap(pin));
    }
    return CompiledGraphAccess::Create(
        BuildCompiledModule(prepared.target, std::move(entries), constants,
                            prepared.profile_context),
        BuildStaticExecutablePlan(prepared.graph), std::move(public_pins),
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
    return CompilePipeline(std::move(function), std::move(config));
}
}  // namespace kxc::api
