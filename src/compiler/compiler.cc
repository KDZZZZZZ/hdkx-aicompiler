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
#include "control_flow/internal_lowering.h"
#include "internal/dynamic_shape_contract.h"
#include "internal/execution_contract.h"
#include "internal/identity_private.h"
#include "internal/kernel_abi_equivalence.h"
#include "internal/lowered_graph.h"
#include "internal/primitive_cache.h"
#include "internal/primitive_compiler.h"
#include "internal/relay_program.h"
#include "internal/relay_snapshot.h"
#include "internal/te_to_tir.h"
#include "runtime/internal/compiled_module_node.h"
#include "runtime/internal/module_invocation_contract.h"
#include "kxc/compiler/experimental_identity.h"
#include "kxc/compiler/pipeline.h"
#include "kxc/pass/context.h"
#include "kxc/profiling/profiling.h"
#include "kxc/runtime/session.h"
#include "support/canonical.h"
#include "support/hash.h"
#include "kxc/relay/visitor.h"
#include "kxc/tir/printer/print_ir.h"

#ifndef KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH
#define KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH 0
#endif
#ifndef KXC_USE_LLVM
#define KXC_USE_LLVM 0
#endif

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
    // Validate through the runtime owner without allocating unused state while
    // compiling or publishing a graph. Actual sessions allocate their own state.
    runtime::RuntimeSession::Validate(module, plan);
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
CompiledGraph CompiledGraph::BindStateOutputs(
    std::vector<runtime::StateOutputBinding> bindings, double state_fill) const {
    return internal::CompiledGraphAccess::Create(module(),
        plan().BindStateOutputs(std::move(bindings), state_fill), artifact_pins(), graph_semantic_key());
}
CompiledGraph CompiledGraph::BindBoundedStateOutputs(
    std::vector<runtime::StateOutputBinding> bindings,
    std::vector<Array<int64_t>> physical_shapes, double state_fill) const {
    return internal::CompiledGraphAccess::Create(module(),
        plan().BindBoundedStateOutputs(std::move(bindings), std::move(physical_shapes), state_fill),
        artifact_pins(), graph_semantic_key());
}
CompiledGraph CompiledGraph::BindRequestBatching(int64_t max_batch_size) const {
    return internal::CompiledGraphAccess::Create(module(), plan().BindRequestBatching(max_batch_size),
        artifact_pins(), graph_semantic_key());
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
            prefix + "operator", internal::PrimitiveUnitOperatorIdentity(unit));
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
        return "cuda-nvrtc-driver-v8";
    }
    throw std::invalid_argument("Primitive cache target has no backend version");
}

}  // namespace

internal::CompilerExecutionContract internal::ResolveBoundedExecutionContract(
    const CompileConfig& config) {
    internal::CompilerExecutionContract contract =
        internal::ResolveCompilerExecutionContract(config);
    contract.schedule_policy =
        relay::internal::kBoundedDynamicTESchedulePolicy;
    AppendPipelineIdentityField(
        &contract.canonical_bytes, "bounded_dynamic_graph_gate",
        "KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH");
    AppendPipelineIdentityField(
        &contract.canonical_bytes, "bounded_schedule_policy",
        contract.schedule_policy);
    AppendPipelineIdentityField(
        &contract.canonical_bytes, "bounded_dynamic_graph_version",
        std::to_string(internal::kBoundedDynamicGraphVersion));
    AppendPipelineIdentityField(
        &contract.canonical_bytes, "bounded_applicability_version",
        std::to_string(experimental::restricted_symbolic_shape::v1::
                           kBoundedCompileApplicabilityVersion));
    AppendPipelineIdentityField(
        &contract.canonical_bytes, "bounded_preparation_version",
        std::to_string(internal::kBoundedCompilePreparationVersion));
    AppendPipelineIdentityField(
        &contract.canonical_bytes, "dynamic_unit_shape_contract_version",
        std::to_string(internal::kDynamicUnitShapeContractVersion));
    AppendPipelineIdentityField(
        &contract.canonical_bytes, "dynamic_plan_mode",
        std::to_string(static_cast<std::uint8_t>(
            runtime::ExecutablePlanMode::kDynamicFreshOutputV1)));
    AppendPipelineIdentityField(
        &contract.canonical_bytes, "module_invocation_abi_version",
        std::to_string(ModuleInvocationContract::kAbiVersion));
    contract.fingerprint = support::HashText(contract.canonical_bytes);
    return contract;
}

namespace {

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
                prepared.graph.partitioned.value_graph.function),
            config->target);
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
    profiling::ScopedSpan assemble_span(
        profile_context, MakeStageEvent("assemble", config), run_id);
    CompiledGraph result = internal::AssembleCompiledGraph(
        prepared, batch);
    AddPrimitiveBatchFields(&assemble_span, prepared, batch);
    if (profile_context) profile_context->Flush();
    return result;
}

}  // namespace


std::string internal::CanonicalTargetSnapshot(const Target& target) {
    return target.CanonicalBytes();
}

internal::CompilerExecutionContract
internal::ResolveCompilerExecutionContract(const CompileConfig& config) {
    return ResolveCompilerExecutionContract(config, {});
}

internal::CompilerExecutionContract
internal::ResolveCompilerExecutionContract(
    const CompileConfig& config,
    const Array<String>& required_relay_control_capabilities) {
    CompilerExecutionContract contract;
    contract.relay_pipeline = ResolveRelayPipeline(
        config, required_relay_control_capabilities);
    contract.tir_pipeline = ResolveTIRPipeline(config);
    contract.schedule_policy =
        relay::internal::kDefaultTESchedulePolicy;
    contract.backend_version = BackendVersion(config->target);
    AppendPipelineIdentityField(&contract.canonical_bytes, "kind",
                                "compiler-execution-plan-v5");
    AppendPipelineIdentityField(
        &contract.canonical_bytes, "relay",
        std::string(contract.relay_pipeline.canonical_bytes));
    AppendPipelineIdentityField(&contract.canonical_bytes, "lowering",
                                "static-te-program-lowering-v1");
    AppendPipelineIdentityField(&contract.canonical_bytes, "partition",
        config->opt_level == 3 && config->target->kind == "llvm" &&
            config->target->device_type == kCPU && config->target->device_id == 0
            ? "static-add-sqrt-v1" : "single-call-v1");
    AppendPipelineIdentityField(
        &contract.canonical_bytes, "tir",
        std::string(contract.tir_pipeline.canonical_bytes));
    AppendPipelineIdentityField(
        &contract.canonical_bytes, "kernel_abi",
        "kernel-abi-v" + std::to_string(codegen::kKernelAbiVersion));
    AppendPipelineIdentityField(&contract.canonical_bytes, "schedule_policy",
                                contract.schedule_policy);
    AppendPipelineIdentityField(&contract.canonical_bytes, "backend",
                                contract.backend_version);
    contract.fingerprint = support::HashText(contract.canonical_bytes);
    return contract;
}

internal::PreparedCompilerGraph internal::PrepareCompilerGraph(
    Function function, CompileConfig config,
    const CompilerExecutionContract& contract) {
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
            String(contract.fingerprint), config->opt_level == 3 && target->kind == "llvm" &&
                target->device_type == kCPU && target->device_id == 0);
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

#if KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH
internal::PreparedCompilerGraph internal::PrepareCompilerGraph(
    const BoundedCompilePreparation& preparation,
    const CompilerExecutionContract& contract) {
    const auto& request = preparation.request();
    const auto config = request.compile_config();
    if (contract.canonical_bytes != ResolveBoundedExecutionContract(config).canonical_bytes) {
        throw std::invalid_argument("bounded preparation execution contract differs from its request");
    }
    auto context = MaybeCreateProfileContext(config);
    const std::string run_id = context ? context->NextRunId("compile_bounded") : "";
    PreparedStaticGraph graph;
    graph.partitioned = preparation.partitioned_graph();
    graph.device = Device(config->target->device_type, config->target->device_id);
    graph.target = config->target;
    graph.pipeline_fingerprint = String(contract.fingerprint);
    PreparedCompilerGraph prepared{request.graph_template().key(), std::move(graph),
        config->target, contract.canonical_bytes, context, run_id};
    FreezePreparedOperators(&prepared);
    return prepared;
}
#endif

CompiledGraph internal::AssembleCompiledGraph(
    const PreparedCompilerGraph& prepared,
    const CompiledPrimitiveBatch& batch) {
    return AssembleCompiledGraph(
        prepared, batch, BuildStaticExecutablePlan(prepared.graph));
}

CompiledGraph internal::AssembleCompiledGraph(
    const PreparedCompilerGraph& prepared,
    const CompiledPrimitiveBatch& batch,
    runtime::ExecutablePlan plan) {
    const std::vector<PrimitiveUnit>& units = prepared.graph.partitioned.units;
    if (batch.primitives.size() != units.size()) {
        throw std::invalid_argument(
            "AssembleCompiledGraph requires one compiled artifact per PrimitiveUnit");
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
    entries.reserve(batch.primitives.size());
    public_pins.reserve(batch.primitives.size());
    for (std::size_t index = 0; index < batch.primitives.size(); ++index) {
        const PrimitiveUnit& unit = units[index];
        const CompiledPrimitive& primitive = batch.primitives[index];
        const PrimitiveArtifactPin& pin = primitive.pin;
        if (unit.id != static_cast<PrimitiveUnitId>(index) ||
            primitive.unit_id != unit.id ||
            !pin.defined() ||
            pin.key().unit_semantic_key() != unit.semantic_key ||
            pin.key().target_capability_fingerprint() != target_fingerprint) {
            throw std::invalid_argument(
                "AssembleCompiledGraph compiled artifact does not match its PrimitiveUnit");
        }
        if (!primitive.current_signature.defined() ||
            primitive.current_signature->symbol != unit.symbol) {
            throw std::invalid_argument(
                "AssembleCompiledGraph compiled signature does not match its PrimitiveUnit");
        }
        const CachedPrimitive& artifact = pin.artifact();
        if (!SamePhysicalKernelAbi(
                artifact.signature, primitive.current_signature) ||
            !artifact.kernel.IsReady() || !artifact.kernel->launcher) {
            throw std::invalid_argument(
                "AssembleCompiledGraph pin artifact cannot be relocated");
        }
        entries.push_back(CompiledModuleEntry{
            primitive.current_signature, artifact.launch_metadata,
            codegen::CompiledKernel(
                primitive.current_signature, artifact.launch_metadata,
                artifact.kernel->launcher),
            primitive.invocation_contract});
        public_pins.push_back(ArtifactPinAccess::Wrap(pin));
    }
    return CompiledGraphAccess::Create(
        BuildCompiledModule(prepared.target, std::move(entries), batch.constants,
                            prepared.profile_context),
        std::move(plan), std::move(public_pins),
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

namespace internal {

CompiledGraph CompileStructuredPipeline(Function function, CompileConfig config) {
    config.Validate();
    auto profile_context = MaybeCreateProfileContext(config);
    const std::string run_id =
        profile_context ? profile_context->NextRunId("compile") : "";
    profiling::ActivationScope activation(profile_context, run_id);
    profiling::ScopedSpan compile_span(
        profile_context, MakeStageEvent("compile", config), run_id);
    const GraphSemanticKey graph_semantic_key =
        Compiler::BuildGraphSemanticKey(function);
    PreparedRelayProgram prepared = PrepareRelayProgram(
        std::move(function), config, ControlFlowPolicy::NativeExact());
    if (!prepared.residual_profile().requires_control_topology()) {
        throw std::invalid_argument(
            "CompileStructuredPipeline requires residual control topology");
    }
    if (config->target->kind != "llvm" || config->target->device_type != kCPU ||
        config->target->device_id != 0) {
        throw std::invalid_argument(
            "CompileStructuredPipeline requires the LLVM CPU:0 backend");
    }
    ControlPlanLowering lowered =
        LowerPreparedRelayToControlPlanWithSidecar(prepared);
    for (const auto& value : lowered.plan.values) {
        if (value.device != Device::CPU()) {
            throw std::invalid_argument(
                "CompileStructuredPipeline requires static CPU:0 values");
        }
    }
    for (const auto& region : lowered.plan.regions) {
        for (const auto& task : region.tasks) {
            if (task.device != Device::CPU() || task.stream != "default") {
                throw std::invalid_argument(
                    "CompileStructuredPipeline requires CPU:0/default stream tasks");
            }
        }
    }

    const PassContext pass_context = PassContext::MergeTarget(
        relay::PassContextFromRelay(prepared.typed_anf()), config->target);
    PassContext::Scope pass_scope(pass_context);
    profiling::ScopedSpan primitive_span(
        profile_context, MakeStageEvent("compile_primitives", config), run_id);
    CompiledPrimitiveBatch batch = internal::CompilePrimitiveUnits(
        lowered.primitive_units, lowered.plan.values, config,
        prepared.execution_contract());
    if (batch.primitives.empty()) {
        throw std::invalid_argument(
            "CompileStructuredPipeline requires at least one real branch kernel");
    }
    const size_t primitive_count = batch.primitives.size();
    primitive_span.AddMetric("primitive_count",
                             static_cast<double>(primitive_count));

    profiling::ScopedSpan assemble_span(
        profile_context, MakeStageEvent("assemble", config), run_id);
    runtime::ExecutablePlan plan =
        BuildStructuredExecutablePlan(lowered.plan, lowered.primitive_units);
    CompiledModule module = internal::AssemblePrimitiveModule(
        batch, lowered.primitive_units, config->target, profile_context);
    std::vector<ArtifactPin> pins;
    pins.reserve(batch.primitives.size());
    for (const internal::CompiledPrimitive& primitive : batch.primitives) {
        pins.push_back(ArtifactPinAccess::Wrap(primitive.pin));
    }
    assemble_span.AddMetric("primitive_count",
                            static_cast<double>(primitive_count));
    if (profile_context) profile_context->Flush();
    return CompiledGraphAccess::Create(std::move(module), std::move(plan),
                                       std::move(pins), graph_semantic_key);
}

}  // namespace internal

CompiledGraph Compiler::Compile(Function function, CompileConfig config) {
#if KXC_ENABLE_CONTROL_RUNTIME
    if (function.defined() &&
        internal::ProfileRelayControlCapabilities(function)
            .requires_control_topology()) {
        return internal::CompileStructuredPipeline(std::move(function),
                                                   std::move(config));
    }
#endif
    return CompilePipeline(std::move(function), std::move(config));
}

CompiledGraph Compiler::CompileBounded(
    const experimental::restricted_symbolic_shape::v1::BoundedCompileRequest&
        request) {
#if !KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH
    (void)request;
    throw std::runtime_error(
        "Compiler::CompileBounded is disabled; configure with "
        "-DKXC_ENABLE_BOUNDED_DYNAMIC_GRAPH=ON");
#elif !KXC_USE_LLVM && !KXC_USE_CUDA
    (void)request;
    throw std::runtime_error(
        "Compiler::CompileBounded requires an available LLVM or CUDA backend");
#else
    const internal::BoundedCompilePreparation preparation =
        internal::PrepareBoundedCompile(request);
    if (preparation.version() !=
            internal::kBoundedCompilePreparationVersion ||
        request.applicability_version() !=
            experimental::restricted_symbolic_shape::v1::
                kBoundedCompileApplicabilityVersion) {
        throw std::invalid_argument(
            "Compiler::CompileBounded received an unsupported bounded contract version");
    }

    runtime::ExecutablePlan plan =
        internal::BuildDynamicExecutablePlan(preparation);
    CompileConfig config = request.compile_config();
    config.Validate();
    const internal::CompilerExecutionContract contract =
        internal::ResolveBoundedExecutionContract(config);
    const auto prepared = internal::PrepareCompilerGraph(preparation, contract);
    const auto& profile_context = prepared.profile_context;
    const auto& run_id = prepared.profile_run_id;
    profiling::ActivationScope activation(profile_context, run_id);
    profiling::ScopedSpan compile_span(
        profile_context, MakeStageEvent("compile_bounded", config), run_id);

    internal::CompiledPrimitiveBatch batch;
    {
        const PassContext pass_context = PassContext::MergeTarget(
            relay::PassContextFromRelay(
                prepared.graph.partitioned.value_graph.function),
            config->target);
        PassContext::Scope pass_scope(pass_context);
        profiling::ScopedSpan primitive_span(
            profile_context,
            MakeStageEvent("compile_bounded_primitives", config), run_id);
        batch = internal::CompilePrimitiveUnits(
            preparation, prepared.graph.partitioned, config, contract);
        primitive_span.AddMetric(
            "primitive_count", static_cast<double>(batch.primitives.size()));
    }

    CompiledGraph result = internal::AssembleCompiledGraph(
        prepared, batch, std::move(plan));
    (void)BuildPlanAbiFingerprint(result);
    compile_span.AddMetric(
        "primitive_count", static_cast<double>(batch.primitives.size()));
    if (profile_context) profile_context->Flush();
    return result;
#endif
}
}  // namespace kxc::api
