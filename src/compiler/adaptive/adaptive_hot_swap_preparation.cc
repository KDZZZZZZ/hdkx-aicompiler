/*! \file src/compiler/adaptive/adaptive_hot_swap_preparation.cc
 * \brief Static and bounded candidate preparation for adaptive compilers.
 */

#include "kxc/compiler/adaptive_hot_swap_preparation.h"

#if KXC_ENABLE_ADAPTIVE_HOT_SWAP

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "../internal/kernel_abi_equivalence.h"
#include "../internal/dynamic_shape_contract.h"
#include "../internal/primitive_cache.h"
#include "runtime/internal/compiled_module_node.h"
#include "runtime/internal/memory_plan.h"

namespace kxc::api::adaptive::hot_swap::preparation {
namespace {

void RequireStaticShape(const Array<int64_t>& shape, const char* context) {
    for (int64_t dimension : shape) {
        if (dimension < 0) {
            throw std::invalid_argument(std::string(context) +
                                        " requires static exact dimensions");
        }
    }
}

const Target& ModuleTarget(const CompiledModule& module) {
    const auto* node = module.As<CompiledModuleNode>();
    if (!node) {
        throw std::invalid_argument(
            "adaptive candidate has an invalid CompiledModule");
    }
    return node->target_;
}

struct VerifiedGraphArtifacts final {
    std::vector<OrderedArtifactIdentity> identities;
};

bool IsBounded(const runtime::ExecutablePlan& plan) {
    return plan.mode() == runtime::ExecutablePlanMode::kDynamicFreshOutputV1 ||
        plan.mode() == runtime::ExecutablePlanMode::kBoundedStatefulExternalV1;
}
ShapeProfileKey Profile(const GraphSemanticKey& key, const runtime::ExecutablePlan& plan) {
    return IsBounded(plan) ? BuildBoundedShapeProfileKey(key, plan)
                           : BuildStaticExactShapeProfileKey(key, plan);
}
DispatchKey Dispatch(const GraphSemanticKey& key, const ShapeProfileKey& profile, bool bounded) {
    return bounded ? BuildBoundedDispatchKey(key, profile) : BuildStaticExactDispatchKey(key, profile);
}

VerifiedGraphArtifacts VerifyGraphArtifacts(
    const CompiledGraph& graph, const CompileConfig& config,
    const GraphSemanticKey& graph_semantic_key) {
    if (!graph.defined() || graph.graph_semantic_key() != graph_semantic_key ||
        internal::BuildTargetCapabilityFingerprint(ModuleTarget(graph.module())) !=
            internal::BuildTargetCapabilityFingerprint(config->target)) {
        throw std::invalid_argument("adaptive candidate differs from its request");
    }
    const Array<runtime::KernelCall> calls = graph.plan().calls();
    const auto& pins = graph.artifact_pins();
    if (calls.empty() || pins.size() != calls.size()) {
        throw std::invalid_argument("adaptive candidate requires one artifact pin per ordered plan call");
    }
    const bool bounded = IsBounded(graph.plan());
    if ((!KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH && bounded) ||
        !((graph.plan().mode() == runtime::ExecutablePlanMode::kStatic &&
           graph.plan().state_value_ids().empty()) ||
          graph.plan().mode() == runtime::ExecutablePlanMode::kStaticStatefulExternalV1 || bounded)) {
        throw std::invalid_argument(
            "adaptive replacement requires a static or enabled bounded plan");
    }
    VerifiedGraphArtifacts verified;
    verified.identities.reserve(calls.size());
    for (size_t index = 0; index < calls.size(); ++index) {
        const auto pin = internal::ArtifactPinAccess::Unwrap(pins[index]);
        const auto signature = graph.module().signature(calls[index]->symbol);
        if (!bounded) for (const auto& argument : signature.arguments()) {
            RequireStaticShape(argument.shape(), "adaptive KernelSignature");
        }
        verified.identities.push_back(OrderedArtifactIdentity{
            index, std::string(calls[index]->symbol), pin.key()});
    }
    if (!bounded) for (const auto& value : graph.plan().values()) {
        RequireStaticShape(value.shape(), "adaptive ExecutablePlan");
    }
    return verified;
}

void ValidateCandidate(const ProductionCompileRequest& request,
                       const CompiledGraph& graph) {
    const VerifiedGraphArtifacts candidate = VerifyGraphArtifacts(
        graph, request.config(), request.graph_semantic_key());
    if (candidate.identities.size() !=
            request.baseline_graph().artifact_pins().size() ||
        candidate.identities.size() != request.ordered_artifacts().size()) {
        throw std::invalid_argument(
            "adaptive candidate primitive mapping differs from the verified baseline");
    }
    for (size_t index = 0; index < candidate.identities.size(); ++index) {
        const OrderedArtifactIdentity& expected = request.ordered_artifacts()[index];
        const OrderedArtifactIdentity& actual = candidate.identities[index];
        const internal::PrimitiveArtifactPin expected_pin =
            internal::ArtifactPinAccess::Unwrap(
                request.baseline_graph().artifact_pins()[index]);
        const internal::PrimitiveArtifactPin actual_pin =
            internal::ArtifactPinAccess::Unwrap(graph.artifact_pins()[index]);
        if (actual.call_index != expected.call_index ||
            actual.link_symbol != expected.link_symbol ||
            !internal::SamePhysicalKernelAbi(
                actual_pin.artifact().signature,
                expected_pin.artifact().signature) ||
            actual_pin.artifact().launch_metadata.CanonicalBytes() !=
                expected_pin.artifact().launch_metadata.CanonicalBytes()) {
            throw std::invalid_argument(
                "adaptive candidate changed a verified callable ABI");
        }
    }
    const ShapeProfileKey candidate_profile = Profile(
        request.graph_semantic_key(), graph.plan());
    if (candidate_profile != request.shape_profile_key() ||
        Dispatch(request.graph_semantic_key(), candidate_profile,
                 IsBounded(graph.plan())) != request.dispatch_key()) {
        throw std::invalid_argument(
            "adaptive candidate dispatch differs from the request");
    }
    if (BuildPlanAbiFingerprint(graph) != request.plan_abi()) {
        throw std::invalid_argument(
            "adaptive candidate Plan ABI differs from the request");
    }
}

PlanVariantKey BuildSelectionPlanKey(
    const ProductionCompileRequest& request,
    const std::vector<OrderedArtifactIdentity>& selected_artifacts) {
    std::vector<OrderedArtifactSelectionIdentity> selections;
    selections.reserve(selected_artifacts.size());
    for (const auto& artifact : selected_artifacts) {
        selections.push_back(OrderedArtifactSelectionIdentity{
            artifact.call_index, artifact.link_symbol, artifact.artifact_key, 0});
    }
    return BuildPlanVariantKey(request.graph_semantic_key(),
                               request.shape_profile_key(), selections,
        request.baseline_graph().plan().request_batching()
            ? runtime::internal::kRequestBatchingMemoryPlanVersion
            :
        request.baseline_graph().plan().mode() == runtime::ExecutablePlanMode::kBoundedStatefulExternalV1
            ? runtime::internal::kBoundedExternalStatefulMemoryPlanVersion
            : request.baseline_graph().plan().mode() == runtime::ExecutablePlanMode::kDynamicFreshOutputV1
            ? runtime::internal::kDynamicFreshOutputMemoryPlanVersion
            :
        request.baseline_graph().plan().mode() == runtime::ExecutablePlanMode::kStaticStatefulExternalV1
            ? runtime::internal::kStaticExternalStatefulMemoryPlanVersion
            : runtime::internal::kStaticMemoryPlanVersion);
}

}  // namespace

ProductionCompileRequest::ProductionCompileRequest(
    Function graph, CompileConfig config, CompiledGraph baseline_graph,
    std::vector<std::int64_t> requested_unit_ids)
    : graph_(std::move(graph)), config_(std::move(config)),
      baseline_graph_(std::move(baseline_graph)),
      requested_unit_ids_(std::move(requested_unit_ids)) {
    Initialize();
}

#if KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH
ProductionCompileRequest::ProductionCompileRequest(
    experimental::restricted_symbolic_shape::v1::BoundedCompileRequest bounded,
    CompiledGraph baseline_graph, std::vector<std::int64_t> requested_unit_ids)
    : graph_(bounded.representative()), config_(bounded.compile_config()),
      baseline_graph_(std::move(baseline_graph)),
      requested_unit_ids_(std::move(requested_unit_ids)) {
    bounded_ = std::make_shared<const internal::BoundedCompilePreparation>(
        internal::PrepareBoundedCompile(bounded));
    Initialize();
}
#endif

void ProductionCompileRequest::Initialize() {
    config_.Validate();
    graph_semantic_key_ = Compiler::BuildGraphSemanticKey(graph_);
    const VerifiedGraphArtifacts baseline = VerifyGraphArtifacts(
        baseline_graph_, config_, graph_semantic_key_);
    if (bool(bounded_) != IsBounded(baseline_graph_.plan())) {
        throw std::invalid_argument("bounded adaptive baseline requires adapter-minted compilation authority");
    }
    ordered_artifacts_ = baseline.identities;
    std::sort(requested_unit_ids_.begin(), requested_unit_ids_.end());
    if (requested_unit_ids_.empty()) {
        throw std::invalid_argument(
            "adaptive compile request requires replacement unit ids");
    }
    for (size_t index = 0; index < requested_unit_ids_.size(); ++index) {
        const std::int64_t id = requested_unit_ids_[index];
        if (id < 0 || static_cast<size_t>(id) >= ordered_artifacts_.size() ||
            (index > 0 && requested_unit_ids_[index - 1] == id)) {
            throw std::invalid_argument(
                "adaptive compile request has invalid replacement unit ids");
        }
    }
    shape_profile_key_ = Profile(
        graph_semantic_key_, baseline_graph_.plan());
#if KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH
    if (bounded_) {
        const auto expected = internal::BuildDynamicExecutablePlan(*bounded_);
        if (Profile(graph_semantic_key_, expected) != shape_profile_key_) {
            throw std::invalid_argument("bounded adaptive baseline differs from its input applicability authority");
        }
        const auto actual_calls = baseline_graph_.plan().calls();
        const auto expected_calls = expected.calls();
        const auto same_ids = [](const Array<int64_t>& a, const Array<int64_t>& b) {
            return std::equal(a.begin(), a.end(), b.begin(), b.end());
        };
        const auto bindings = baseline_graph_.plan().state_output_bindings();
        const auto original_reads = [&](const Array<int64_t>& ids) {
            Array<int64_t> original;
            for (int64_t id : ids) {
                for (const auto& binding : bindings) {
                    if (id == binding.input_value_id) { id = binding.state_value_id; break; }
                }
                original.push_back(id);
            }
            return original;
        };
        if (actual_calls.size() != expected_calls.size() ||
            !same_ids(original_reads(baseline_graph_.plan().input_value_ids()), expected.input_value_ids())) {
            throw std::invalid_argument("bounded adaptive baseline changed the authorized graph boundary");
        }
        for (size_t i = 0; i < actual_calls.size(); ++i) {
            if (!(actual_calls[i]->symbol == expected_calls[i]->symbol) ||
                !same_ids(original_reads(actual_calls[i].input_value_ids()), expected_calls[i].input_value_ids()) ||
                !same_ids(actual_calls[i].output_value_ids(), expected_calls[i].output_value_ids())) {
                throw std::invalid_argument("bounded adaptive baseline changed the authorized call wiring");
            }
        }
    }
#endif
    dispatch_key_ = Dispatch(graph_semantic_key_, shape_profile_key_, bool(bounded_));
    plan_abi_ = BuildPlanAbiFingerprint(baseline_graph_);
    Validate();
}

const Function& ProductionCompileRequest::graph() const noexcept { return graph_; }
CompileConfig ProductionCompileRequest::config() const { return config_; }
const GraphSemanticKey& ProductionCompileRequest::graph_semantic_key() const noexcept { return graph_semantic_key_; }
const ShapeProfileKey& ProductionCompileRequest::shape_profile_key() const noexcept { return shape_profile_key_; }
const DispatchKey& ProductionCompileRequest::dispatch_key() const noexcept { return dispatch_key_; }
const PlanAbiFingerprint& ProductionCompileRequest::plan_abi() const noexcept { return plan_abi_; }
const std::vector<OrderedArtifactIdentity>& ProductionCompileRequest::ordered_artifacts() const noexcept { return ordered_artifacts_; }
const CompiledGraph& ProductionCompileRequest::baseline_graph() const noexcept { return baseline_graph_; }
const std::vector<std::int64_t>& ProductionCompileRequest::requested_unit_ids() const noexcept { return requested_unit_ids_; }
const experimental::restricted_symbolic_shape::v1::BoundedCompileRequest*
ProductionCompileRequest::bounded_request() const noexcept {
#if KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH
    return bounded_ ? &bounded_->request() : nullptr;
#else
    return nullptr;
#endif
}
const internal::BoundedCompilePreparation*
ProductionCompileRequest::bounded_preparation() const noexcept { return bounded_.get(); }

void ProductionCompileRequest::Validate() const {
    if (!graph_.defined() || !graph_semantic_key_.defined() ||
        !shape_profile_key_.defined() || !dispatch_key_.defined() ||
        !plan_abi_.defined() || !baseline_graph_.defined() ||
        ordered_artifacts_.empty() || requested_unit_ids_.empty() ||
        ordered_artifacts_.size() != baseline_graph_.artifact_pins().size()) {
        throw std::invalid_argument(
            "adaptive compile request has an undefined typed contract");
    }
    if (Compiler::BuildGraphSemanticKey(graph_) != graph_semantic_key_ ||
        shape_profile_key_.graph_semantic_key() != graph_semantic_key_) {
        throw std::invalid_argument(
            "adaptive compile request graph identity is inconsistent");
    }
    const VerifiedGraphArtifacts baseline = VerifyGraphArtifacts(
        baseline_graph_, config_, graph_semantic_key_);
    if (baseline.identities != ordered_artifacts_) {
        throw std::invalid_argument(
            "adaptive compile request baseline artifact mapping is inconsistent");
    }
    for (size_t index = 0; index < requested_unit_ids_.size(); ++index) {
        const std::int64_t id = requested_unit_ids_[index];
        if (id < 0 || static_cast<size_t>(id) >= ordered_artifacts_.size() ||
            (index > 0 && requested_unit_ids_[index - 1] >= id)) {
            throw std::invalid_argument(
                "adaptive compile request has non-canonical replacement unit ids");
        }
    }
    for (const OrderedArtifactIdentity& artifact : ordered_artifacts_) {
        (void)artifact.CanonicalBytes();
    }
}

ProductionExecutionRequest::ProductionExecutionRequest(
    DispatchKey dispatch_key, PlanAbiFingerprint plan_abi)
    : dispatch_key_(std::move(dispatch_key)), plan_abi_(std::move(plan_abi)) {
    Validate();
}
const DispatchKey& ProductionExecutionRequest::dispatch_key() const noexcept { return dispatch_key_; }
const PlanAbiFingerprint& ProductionExecutionRequest::plan_abi() const noexcept { return plan_abi_; }
void ProductionExecutionRequest::Validate() const {
    if (!dispatch_key_.defined() || !plan_abi_.defined()) {
        throw std::invalid_argument(
            "adaptive execution requires explicit dispatch and Plan ABI identities");
    }
}

PreparedCandidate::PreparedCandidate(
    CompiledGraph graph, std::shared_ptr<const runtime::RuntimeSession> session,
    PlanVariantKey selection_plan_key,
    std::vector<OrderedArtifactIdentity> selected_artifacts,
    std::string validation_receipt)
    : graph_(std::move(graph)), session_(std::move(session)),
      selection_plan_key_(std::move(selection_plan_key)),
      selected_artifacts_(std::move(selected_artifacts)),
      validation_receipt_(std::move(validation_receipt)) {
    const bool stateful = !graph_.plan().state_value_ids().empty();
    if ((stateful ? bool(session_) : (!session_ || !session_->defined())) ||
        !selection_plan_key_.defined() ||
        selected_artifacts_.empty() || validation_receipt_.empty()) {
        throw std::invalid_argument(
            "prepared adaptive candidate requires appropriate session ownership, selection, and receipt");
    }
}
const CompiledGraph& PreparedCandidate::compiled_graph() const noexcept { return graph_; }
const std::shared_ptr<const runtime::RuntimeSession>& PreparedCandidate::session() const noexcept { return session_; }
const PlanVariantKey& PreparedCandidate::selection_plan_key() const noexcept { return selection_plan_key_; }
const std::vector<OrderedArtifactIdentity>& PreparedCandidate::selected_artifacts() const noexcept { return selected_artifacts_; }
const std::string& PreparedCandidate::validation_receipt() const noexcept { return validation_receipt_; }

std::shared_ptr<const PreparedCandidate> PrepareCandidate(
    const ProductionCompileRequest& request, CompiledGraph graph,
    std::string validation_receipt) {
    request.Validate();
    if (validation_receipt.empty()) {
        throw std::invalid_argument("adaptive candidate requires injected validation receipt");
    }
    const VerifiedGraphArtifacts verified = VerifyGraphArtifacts(
        graph, request.config(), request.graph_semantic_key());
    ValidateCandidate(request, graph);
    runtime::RuntimeSession::Validate(graph.module(), graph.plan());
    std::shared_ptr<const runtime::RuntimeSession> session;
    if (graph.plan().state_value_ids().empty()) {
        session = std::make_shared<const runtime::RuntimeSession>(graph.module(), graph.plan());
    }
    return std::shared_ptr<const PreparedCandidate>(new PreparedCandidate(
        std::move(graph), std::move(session),
        BuildSelectionPlanKey(request, verified.identities), verified.identities,
        std::move(validation_receipt)));
}

}  // namespace kxc::api::adaptive::hot_swap::preparation

#endif  // KXC_ENABLE_ADAPTIVE_HOT_SWAP
