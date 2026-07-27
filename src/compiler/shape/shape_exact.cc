#include "kxc/compiler/shape_exact.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../internal/execution_contract.h"
#include "../internal/lowered_graph.h"
#include "../internal/primitive_cache.h"
#include "../internal/primitive_compiler.h"
#include "../internal/relay_snapshot.h"
#include "../internal/te_to_tir.h"
#include "runtime/internal/memory_plan.h"
#include "support/hash.h"
#include "kxc/pass/context.h"
#include "kxc/profiling/profiling.h"
#include "kxc/relay/visitor.h"
#include "kxc/tir/printer/print_ir.h"
#include "kxc/runtime/device_api.h"

#ifndef KXC_ENABLE_SHAPE_PRODUCTION_EXACT
#define KXC_ENABLE_SHAPE_PRODUCTION_EXACT 0
#endif
#ifndef KXC_USE_LLVM
#define KXC_USE_LLVM 0
#endif
#ifndef KXC_USE_CUDA
#define KXC_USE_CUDA 0
#endif

namespace kxc::api::experimental::shape_exact::v1 {
namespace {
namespace shape =
    kxc::api::experimental::shape_specialization::v1;

[[noreturn]] void Reject(const std::string& message) {
    throw std::invalid_argument("ProductionExactShapeAdapter: " + message);
}

void RequireEnabled() {
#if !KXC_ENABLE_SHAPE_PRODUCTION_EXACT
    throw std::runtime_error(
        "ProductionExactShapeAdapter is disabled; configure with "
        "-DKXC_ENABLE_SHAPE_PRODUCTION_EXACT=ON");
#endif
}

void RequireBackendAvailable(const Target& target) {
#if !KXC_USE_LLVM
    if (target->kind == "llvm" && target->device_type == kCPU) {
        Reject("LLVM exact assembly requires KXC_ENABLE_LLVM=ON");
    }
#endif
#if !KXC_USE_CUDA
    if (target->kind == "cuda" && target->device_type == kCUDA) {
        Reject("CUDA exact assembly requires KXC_ENABLE_CUDA=ON");
    }
#endif
}

uint64_t PlannedStorageBytes(const runtime::ExecutablePlan& plan) {
    std::unordered_map<int64_t, uint64_t> bytes_by_storage;
    for (const auto& value : plan.values()) {
        uint64_t elements = 1;
        for (int64_t dimension : value.shape()) {
            if (dimension < 0) return std::numeric_limits<uint64_t>::max();
            if (dimension != 0 &&
                elements > std::numeric_limits<uint64_t>::max() /
                               static_cast<uint64_t>(dimension)) {
                return std::numeric_limits<uint64_t>::max();
            }
            elements *= static_cast<uint64_t>(dimension);
        }
        const uint64_t element_bytes =
            static_cast<uint64_t>(value->dtype.bits / 8) *
            static_cast<uint64_t>(value->dtype.lanes);
        if (element_bytes != 0 &&
            elements > std::numeric_limits<uint64_t>::max() / element_bytes) {
            return std::numeric_limits<uint64_t>::max();
        }
        bytes_by_storage[value->storage_id] = std::max(
            bytes_by_storage[value->storage_id], elements * element_bytes);
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

shape::TensorShapeContract ValueContract(const internal::ValueInfo& value) {
    const auto* type = value.checked_type.As<TensorTypeNode>();
    if (!type) Reject("prepared value has no TensorType");
    std::vector<shape::DimExpr> dimensions;
    dimensions.reserve(type->shape.size());
    for (int64_t extent : type->shape) {
        if (extent < 0) Reject("negative or legacy -1 dimension is unsupported");
        dimensions.push_back(shape::DimExpr::Const(extent));
    }
    return {shape::LogicalShape(dimensions),
            shape::PhysicalCapacity(dimensions),
            shape::ValidExtent(dimensions)};
}

std::string ValueName(int64_t id) { return "value." + std::to_string(id); }

shape::GraphTemplate BuildTemplate(
    const internal::PreparedCompilerGraph& prepared) {
    if (prepared.graph.partitioned.units.empty()) {
        Reject("a production exact plan requires at least one ordinary compute unit");
    }
    const auto& partitioned = prepared.graph.partitioned;
    const auto& graph = partitioned.value_graph;
    std::vector<shape::NamedTensorContract> inputs;
    std::vector<shape::NamedTensorContract> outputs;
    for (const auto& value : graph.values) {
        shape::NamedTensorContract named{ValueName(value.id),
                                         ValueContract(value)};
        if (value.origin == internal::ValueOrigin::kParameter ||
            value.origin == internal::ValueOrigin::kConstant) {
            inputs.push_back(std::move(named));
        } else {
            outputs.push_back(std::move(named));
        }
    }
    std::vector<shape::UnitSkeleton> units;
    units.reserve(partitioned.units.size());
    for (const auto& unit : partitioned.units) {
        shape::UnitSkeleton skeleton{shape::GraphLocalCallLocator(
            ValueName(unit.output_value_ids[0])),
            unit.semantic_key, {}, {}};
        for (int64_t id : unit.boundary_input_value_ids) {
            skeleton.input_value_names.push_back(ValueName(id));
        }
        for (int64_t id : unit.output_value_ids) skeleton.output_value_names.push_back(ValueName(id));
        units.push_back(std::move(skeleton));
    }
    return shape::GraphTemplate(
        prepared.graph_semantic_key,
        shape::ShapeProgram({}, std::move(inputs), std::move(outputs)),
        std::move(units));
}

bool ExactContract(const shape::ConcreteTensorShapeContract& contract) {
    return ContractDefect(contract) == nullptr && IsExactContract(contract);
}

bool SameContract(const shape::ConcreteTensorShapeContract& a,
                  const shape::ConcreteTensorShapeContract& b) {
    return a == b && ExactContract(a);
}

bool SameIds(const Array<int64_t>& actual, const Array<int64_t>& expected) {
    if (actual.size() != expected.size()) return false;
    for (size_t i = 0; i < actual.size(); ++i) if (actual[i] != expected[i]) return false;
    return true;
}

const shape::ConcreteTensorShapeContract& ProfileValue(
    const shape::ExactOracle& oracle, const std::string& name) {
    return oracle.profile().Value(name).contract;
}

void VerifyOracle(const shape::GraphTemplate& graph, const shape::ExactOracle& oracle) {
    if (oracle.profile().key().graph_semantic_key() != graph.key() ||
        oracle.profile().policy_id() != "exact" ||
        oracle.profile().shape_abi_version() !=
            shape::kShapeProfileAbiVersion ||
        !oracle.profile().bindings().bindings().empty()) {
        Reject("oracle is not the empty exact profile of this prepared template");
    }
    const shape::ExactOracle rebuilt = shape::InstantiateExactProfile(
        graph, oracle.profile().bindings());
    if (!(rebuilt.profile().key() == oracle.profile().key()) ||
        rebuilt.profile().values().size() != oracle.profile().values().size()) {
        Reject("oracle content does not exactly match the prepared template");
    }
    for (size_t i = 0; i < rebuilt.profile().values().size(); ++i) {
        if (rebuilt.profile().values()[i].name != oracle.profile().values()[i].name ||
            !SameContract(rebuilt.profile().values()[i].contract,
                          oracle.profile().values()[i].contract)) {
            Reject("oracle value contract does not exactly match the prepared template");
        }
    }
}

bool SameDType(const DLDataType& left, const DLDataType& right) {
    return left.code == right.code && left.bits == right.bits &&
           left.lanes == right.lanes;
}

void VerifyArg(const codegen::KernelArgSpec& arg, codegen::KernelArgRole role,
                const shape::ConcreteTensorShapeContract& contract,
                const TensorTypeNode* expected_type,
                const Target& target) {
    if (!expected_type) Reject("compiled signature source is not a tensor");
    const auto* node = arg.operator->();
    const DLDataType expected_dtype =
        runtime::DataTypeFromString(expected_type->dtype);
    if (!ExactContract(contract) || node->role != role ||
        !SameDType(node->dtype, expected_dtype) ||
        node->device.device_type() != target->device_type ||
        node->device.device_id() != target->device_id) {
        Reject("compiled signature ABI does not match exact shape contract");
    }
    const Array<int64_t> shape = arg.shape();
    if (shape.size() != contract.logical.size() ||
        shape.size() != expected_type->shape.size()) {
        Reject("compiled signature rank mismatch");
    }
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] != contract.logical[i] ||
            shape[i] != expected_type->shape[i]) {
            Reject("compiled signature extent mismatch");
        }
    }
}

void VerifyVariant(
    const shape::GraphTemplate& graph,
    const std::vector<shape::UnitSpecializationRequest>& requests,
    const internal::PreparedCompilerGraph& prepared,
    const CompileConfig& config,
    const internal::CompilerExecutionContract& contract,
    const shape::ExactOracle& oracle, const CompiledGraph& compiled) {
    const auto& partitioned = prepared.graph.partitioned;
    // values()/calls() 每次调用都深拷贝出临时 Array；元素引用必须指向
    // 本地持有的数组，否则悬垂。
    const Array<runtime::ValueSpec> plan_values = compiled.plan().values();
    const Array<runtime::KernelCall> plan_calls = compiled.plan().calls();
    if (requests.size() != partitioned.units.size() ||
        plan_calls.size() != requests.size() ||
        compiled.artifact_pins().size() != requests.size() ||
        compiled.module().entry_count() != requests.size() ||
        plan_values.size() != partitioned.value_graph.values.size()) {
        Reject("compiled module/plan/pin cardinality does not match exact requests");
    }
    for (size_t i = 0; i < plan_values.size(); ++i) {
        const runtime::ValueSpec& value = plan_values[i];
        const auto& source = partitioned.value_graph.values[i];
        const auto& exact = ProfileValue(oracle, ValueName(source.id));
        const auto* expected_type = source.checked_type.As<TensorTypeNode>();
        if (!expected_type) Reject("compiled plan source is not a tensor");
        const DLDataType expected_dtype =
            runtime::DataTypeFromString(expected_type->dtype);
        const bool is_graph_output =
            std::find(partitioned.output_value_ids.begin(),
                      partitioned.output_value_ids.end(),
                      source.id) != partitioned.output_value_ids.end();
        if (value->value_id != source.id ||
            value->is_input !=
                (source.origin == internal::ValueOrigin::kParameter) ||
            value->is_constant !=
                (source.origin == internal::ValueOrigin::kConstant) ||
            value->is_output != is_graph_output ||
            !ExactContract(exact) ||
            !SameDType(value->dtype, expected_dtype) ||
            value->device.device_type() != config->target->device_type ||
            value->device.device_id() != config->target->device_id ||
            value.shape().size() != exact.logical.size()) {
            Reject("compiled plan value does not match the exact profile ABI");
        }
        const Array<int64_t> shape = value.shape();
        for (size_t axis = 0; axis < shape.size(); ++axis) {
            if (shape[axis] != exact.logical[axis]) {
                Reject("compiled plan value extent does not match the exact profile");
            }
        }
    }
    for (size_t i = 0; i < requests.size(); ++i) {
        const auto& request = requests[i];
        const auto& unit = partitioned.units[i];
        if (request.ordered_call_index != i ||
            request.call_locator.value() != ValueName(unit.output_value_ids[0]) ||
            !(request.shape_profile_key == oracle.profile().key()) ||
            request.unit_semantic_key != unit.semantic_key ||
            !shape::MatchesExactSignatureDigest(
                request.signature_digest, request.ordered_inputs,
                request.ordered_outputs) ||
            request.ordered_inputs.size() !=
                unit.boundary_input_value_ids.size() ||
            request.ordered_outputs.size() != unit.output_value_ids.size()) {
            Reject("exact request routing, profile, artifact, or semantic identity drifted");
        }
        for (size_t j = 0; j < request.ordered_inputs.size(); ++j) {
            const auto& value = request.ordered_inputs[j];
            if (!ExactContract(value) ||
                !SameContract(value, ProfileValue(
                    oracle,
                    ValueName(unit.boundary_input_value_ids[j])))) {
                Reject("exact request input contract drifted");
            }
        }
        for (size_t j = 0; j < request.ordered_outputs.size(); ++j) {
            const auto& value = request.ordered_outputs[j];
            if (!ExactContract(value) ||
                !SameContract(value, ProfileValue(
                    oracle, ValueName(unit.output_value_ids[j])))) {
                Reject("exact request output contract drifted");
            }
        }
        const runtime::KernelCall& call = plan_calls[i];
        if (!(call->symbol == unit.symbol) || !compiled.module().HasFunction(call->symbol) ||
            !SameIds(call.input_value_ids(),
                     unit.boundary_input_value_ids) ||
            !SameIds(call.output_value_ids(), unit.output_value_ids)) {
            Reject("compiled plan routing does not match prepared partition");
        }
        const codegen::KernelSignature signature = compiled.module().signature(call->symbol);
        const Array<codegen::KernelArgSpec> args = signature.arguments();
        if (args.size() != unit.boundary_input_value_ids.size() +
                               unit.output_value_ids.size()) {
            Reject("compiled signature arity does not match exact request");
        }
        for (size_t j = 0; j < unit.boundary_input_value_ids.size(); ++j) {
            const auto& value = partitioned.value_graph.values[
                static_cast<size_t>(unit.boundary_input_value_ids[j])];
            VerifyArg(args[j], value.origin == internal::ValueOrigin::kConstant
                          ? codegen::KernelArgRole::kConstant : codegen::KernelArgRole::kInput,
                      request.ordered_inputs[j],
                      value.checked_type.As<TensorTypeNode>(),
                      config->target);
        }
        for (size_t j = 0; j < unit.output_value_ids.size(); ++j) {
            const auto& value = partitioned.value_graph.values[
                static_cast<size_t>(unit.output_value_ids[j])];
            VerifyArg(args[unit.boundary_input_value_ids.size() + j],
                      codegen::KernelArgRole::kOutput,
                      request.ordered_outputs[j],
                      value.checked_type.As<TensorTypeNode>(),
                      config->target);
        }
        const codegen::KernelLaunchMetadata metadata = compiled.module().launch_metadata(call->symbol);
        if (metadata->device.device_type() != config->target->device_type ||
            metadata->device.device_id() != config->target->device_id ||
            (config->target->kind == "llvm" && metadata->backend != codegen::CodeGenBackend::kLLVM) ||
            (config->target->kind == "cuda" && metadata->backend != codegen::CodeGenBackend::kCUDA)) {
            Reject("compiled launch metadata does not match target/backend");
        }
        const relay::LoweredFunction expected_lowered =
            internal::LowerPrimitiveUnit(
                partitioned.value_graph.values, unit, config->target);
        const std::string expected_schedule =
            relay::internal::GetTEScheduleContract(
                expected_lowered->prim_func);
        const PrimitiveArtifactKey expected =
            internal::BuildPrimitiveArtifactKey(
                unit.semantic_key, config->target, contract.canonical_bytes,
                expected_schedule.c_str(), contract.backend_version.c_str());
        const ArtifactPin& public_pin = compiled.artifact_pins()[i];
        if (!public_pin.defined()) {
            Reject("production artifact pin is undefined");
        }
        const internal::PrimitiveArtifactPin primitive_pin =
            internal::ArtifactPinAccess::Unwrap(public_pin);
        const ArtifactRecord& record = public_pin.record();
        if (!(record.artifact_key == expected) ||
            !(primitive_pin.key() == expected) ||
            record.signature_digest != support::HashText(
                primitive_pin.artifact().signature.CanonicalBytes()) ||
            record.launch_metadata_digest != support::HashText(
                primitive_pin.artifact().launch_metadata.CanonicalBytes()) ||
            metadata.CanonicalBytes() !=
                primitive_pin.artifact().launch_metadata.CanonicalBytes()) {
            Reject("production artifact pin does not retain the real full artifact key/signature/launch metadata");
        }
    }
}

}  // namespace

struct PreparedGraphTemplate::Impl final {
    Impl(CompileConfig config, internal::CompilerExecutionContract contract,
         internal::PreparedCompilerGraph prepared, shape::GraphTemplate graph,
         ShapeExactPreparationCounters counters)
        : config(std::move(config)), contract(std::move(contract)),
          prepared(std::move(prepared)), graph(std::move(graph)),
          counters(counters) {}
    CompileConfig config;
    internal::CompilerExecutionContract contract;
    internal::PreparedCompilerGraph prepared;
    shape::GraphTemplate graph;
    ShapeExactPreparationCounters counters;
};

struct ExactPlanVariant::Impl final {
    CompiledGraph graph;
    ShapeProfileKey profile;
    PlanVariantKey key;
    DispatchKey dispatch;
};

PreparedGraphTemplate::PreparedGraphTemplate() = default;
PreparedGraphTemplate::PreparedGraphTemplate(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}
PreparedGraphTemplate::~PreparedGraphTemplate() = default;
PreparedGraphTemplate::PreparedGraphTemplate(const PreparedGraphTemplate&) = default;
PreparedGraphTemplate& PreparedGraphTemplate::operator=(const PreparedGraphTemplate&) = default;
PreparedGraphTemplate::PreparedGraphTemplate(PreparedGraphTemplate&&) noexcept = default;
PreparedGraphTemplate& PreparedGraphTemplate::operator=(PreparedGraphTemplate&&) noexcept = default;
const shape::GraphTemplate& PreparedGraphTemplate::graph_template() const {
    if (!impl_) Reject("prepared template is undefined");
    return impl_->graph;
}
const ShapeExactPreparationCounters& PreparedGraphTemplate::counters() const {
    if (!impl_) Reject("prepared template is undefined");
    return impl_->counters;
}
size_t PreparedGraphTemplate::unit_count() const { return graph_template().ordered_units().size(); }

ExactPlanVariant::ExactPlanVariant() = default;
ExactPlanVariant::ExactPlanVariant(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}
ExactPlanVariant::~ExactPlanVariant() = default;
ExactPlanVariant::ExactPlanVariant(const ExactPlanVariant&) = default;
ExactPlanVariant& ExactPlanVariant::operator=(const ExactPlanVariant&) = default;
ExactPlanVariant::ExactPlanVariant(ExactPlanVariant&&) noexcept = default;
ExactPlanVariant& ExactPlanVariant::operator=(ExactPlanVariant&&) noexcept = default;
const CompiledModule& ExactPlanVariant::module() const { if (!impl_) Reject("exact plan variant is undefined"); return impl_->graph.module(); }
const runtime::ExecutablePlan& ExactPlanVariant::plan() const { if (!impl_) Reject("exact plan variant is undefined"); return impl_->graph.plan(); }
const ShapeProfileKey& ExactPlanVariant::shape_profile_key() const { if (!impl_) Reject("exact plan variant is undefined"); return impl_->profile; }
const PlanVariantKey& ExactPlanVariant::plan_variant_key() const { if (!impl_) Reject("exact plan variant is undefined"); return impl_->key; }
const DispatchKey& ExactPlanVariant::dispatch_key() const { if (!impl_) Reject("exact plan variant is undefined"); return impl_->dispatch; }
const std::vector<ArtifactPin>& ExactPlanVariant::artifact_pins() const { if (!impl_) Reject("exact plan variant is undefined"); return impl_->graph.artifact_pins(); }

bool ProductionExactShapeAdapter::IsEnabled() noexcept {
#if KXC_ENABLE_SHAPE_PRODUCTION_EXACT
    return true;
#else
    return false;
#endif
}

PreparedGraphTemplate ProductionExactShapeAdapter::PrepareGraphTemplate(
    Function function, CompileConfig config) {
    RequireEnabled();
    const Function relay_snapshot = internal::CloneRelaySnapshot(function);
    config.Validate();
    ShapeExactPreparationCounters counters;
    internal::CompilerExecutionContract contract =
        internal::ResolveCompilerExecutionContract(config);
    ++counters.execution_contract_resolutions;
    internal::PreparedCompilerGraph prepared = internal::PrepareCompilerGraph(
        relay_snapshot, config, contract);
    // Capability/registry identity checks have completed.  Detach every
    // lowering descriptor before the prepared graph becomes observable so
    // Assemble never rereads a caller-accessible registry OpNode.
    internal::FreezePreparedOperators(&prepared);
    counters.relay_graph_pipelines = prepared.relay_graph_pipelines;
    counters.capability_boundary_checks =
        prepared.capability_boundary_checks;
    counters.value_graph_builds = prepared.value_graph_builds;
    counters.partitions = prepared.partitions;
    shape::GraphTemplate graph = BuildTemplate(prepared);
    return PreparedGraphTemplate(
        std::make_shared<PreparedGraphTemplate::Impl>(
            std::move(config), std::move(contract), std::move(prepared),
            std::move(graph), counters));
}

shape::ExactOracle ProductionExactShapeAdapter::InstantiateExactProfile(
    const PreparedGraphTemplate& prepared, const shape::BindingSet& bindings) {
    RequireEnabled();
    if (!bindings.bindings().empty()) {
        Reject("Relay is concrete-only; non-empty bindings and multi-profile exact requests are unsupported");
    }
    return shape::InstantiateExactProfile(prepared.graph_template(), bindings);
}

ExactPlanVariant ProductionExactShapeAdapter::AssembleExactPlan(
    const PreparedGraphTemplate& prepared, const shape::ExactOracle& oracle) {
    RequireEnabled();
    if (!prepared.impl_) Reject("prepared template is undefined");
    VerifyOracle(prepared.impl_->graph, oracle);
    const std::vector<shape::UnitSpecializationRequest> requests =
        shape::MakeExactSpecializationRequests(prepared.impl_->graph, oracle);
    const internal::PreparedCompilerGraph& compiler_prepared =
        prepared.impl_->prepared;
    const CompileConfig& config = prepared.impl_->config;
    const internal::CompilerExecutionContract& contract =
        prepared.impl_->contract;
    RequireBackendAvailable(config->target);
    if (compiler_prepared.execution_contract_canonical !=
            contract.canonical_bytes ||
        internal::CanonicalTargetSnapshot(compiler_prepared.target) !=
            internal::CanonicalTargetSnapshot(config->target) ||
        internal::CanonicalTargetSnapshot(compiler_prepared.graph.target) !=
            internal::CanonicalTargetSnapshot(config->target) ||
        compiler_prepared.graph.device !=
            Device(config->target->device_type, config->target->device_id) ||
        std::string(compiler_prepared.graph.pipeline_fingerprint) !=
            contract.fingerprint) {
        Reject("prepared target or execution contract changed before assembly");
    }
    const auto stage_event = [&config](const char* stage) {
        profiling::EventSpec event;
        event.component = "compiler";
        event.event_type = "compile_stage";
        event.pass_name = stage;
        event.fields = profiling::MakeFields({
            {"stage", stage},
            {"target_kind", config->target->kind},
            {"device_type",
             std::to_string(static_cast<int>(config->target->device_type))},
            {"device_id", std::to_string(config->target->device_id)},
            {"opt_level", std::to_string(config->opt_level)},
        });
        return event;
    };
    const auto profile_context = compiler_prepared.profile_context;
    const std::string& run_id = compiler_prepared.profile_run_id;
    profiling::ActivationScope activation(profile_context, run_id);
    internal::CompiledPrimitiveBatch batch;
    {
        const PassContext pass_context = PassContext::MergeTarget(
            relay::PassContextFromRelay(
                compiler_prepared.graph.partitioned.value_graph.function),
            config->target);
        PassContext::Scope pass_scope(pass_context);
        profiling::ScopedSpan compile_span(
            profile_context, stage_event("compile_primitives"), run_id);
        try {
            batch = internal::CompilePrimitiveUnits(
                compiler_prepared.graph.partitioned.units,
                compiler_prepared.graph.partitioned.value_graph.values,
                config, contract);
            AddPrimitiveBatchFields(&compile_span, compiler_prepared, batch);
        } catch (const std::exception& error) {
            compile_span.SetStatus("error");
            compile_span.SetMessage(error.what());
            throw std::runtime_error(
                std::string("Compiler stage 'compile_primitives' failed: ") +
                error.what());
        }
    }
    profiling::ScopedSpan assemble_span(
        profile_context, stage_event("assemble"), run_id);
    CompiledGraph compiled = internal::AssembleCompiledGraph(
        compiler_prepared, batch);
    AddPrimitiveBatchFields(&assemble_span, compiler_prepared, batch);
    if (profile_context) profile_context->Flush();
    VerifyVariant(prepared.impl_->graph, requests, compiler_prepared, config,
                  contract, oracle, compiled);
    std::vector<OrderedArtifactSelectionIdentity> selections;
    selections.reserve(compiled.plan().calls().size());
    for (size_t i = 0; i < compiled.plan().calls().size(); ++i) {
        const ArtifactRecord& record =
            compiled.artifact_pins()[i].record();
        selections.push_back(OrderedArtifactSelectionIdentity{
            i, std::string(compiled.plan().calls()[i]->symbol),
            record.artifact_key, 0});
    }
    auto impl = std::make_shared<ExactPlanVariant::Impl>(ExactPlanVariant::Impl{
        std::move(compiled), oracle.profile().key(),
        BuildPlanVariantKey(
            prepared.impl_->graph.key(), oracle.profile().key(),
            selections, runtime::internal::kStaticMemoryPlanVersion),
        BuildStaticExactDispatchKey(prepared.impl_->graph.key(),
                                    oracle.profile().key())});
    return ExactPlanVariant(std::move(impl));
}

}  // namespace kxc::api::experimental::shape_exact::v1
