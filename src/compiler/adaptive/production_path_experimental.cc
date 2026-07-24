/*! \file src/compiler/adaptive/production_path_experimental.cc
 * \brief Static-exact candidate preparation for production adaptive compilers.
 */

#include "kxc/compiler/adaptive_production_experimental.h"

#if KXC_ENABLE_EXPERIMENTAL_ADAPTIVE_PRODUCTION

#include <stdexcept>
#include <utility>

#include "../internal/kernel_abi_equivalence.h"
#include "../internal/primitive_cache.h"
#include "../../runtime/internal/compiled_module_node.h"
#include "../../runtime/internal/memory_plan.h"

namespace kxc::api::adaptive::experimental::production_path {
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

Target CloneTarget(const Target& source) {
    if (!source.defined() || !source.As<TargetNode>()) {
        throw std::invalid_argument(
            "adaptive compile request requires a valid Target snapshot");
    }
    const TargetNode* source_node = source.operator->();
    auto* node = new TargetNode();
    node->kind = source_node->kind;
    node->device_type = source_node->device_type;
    node->device_id = source_node->device_id;
    node->attrs = source_node->attrs;
    return Target(ObjectRef(node));
}

CompileConfig CloneCompileConfig(const CompileConfig& source) {
    source.Validate();
    auto* node = new CompileConfigNode();
    node->opt_level = source->opt_level;
    node->target = CloneTarget(source->target);
    node->profile_options = source->profile_options;
    CompileConfig snapshot{ObjectRef(node)};
    snapshot.Validate();
    return snapshot;
}

struct VerifiedGraphArtifacts final {
    std::vector<OrderedArtifactIdentity> identities;
    std::vector<ArtifactPin> pins;
};

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
    if (calls.empty()) {
        throw std::invalid_argument("adaptive candidate requires ordered plan calls");
    }
    VerifiedGraphArtifacts verified;
    verified.identities.reserve(calls.size());
    verified.pins.reserve(calls.size());
    for (size_t index = 0; index < calls.size(); ++index) {
        const auto pin = internal::ArtifactPinAccess::Unwrap(pins[index]);
        const auto signature = graph.module().signature(calls[index]->symbol);
        for (const auto& argument : signature.arguments()) {
            RequireStaticShape(argument.shape(), "adaptive KernelSignature");
        }
        verified.identities.push_back(OrderedArtifactIdentity{
            index, std::string(calls[index]->symbol), pin.key()});
        verified.pins.push_back(pins[index]);
    }
    for (const auto& value : graph.plan().values()) {
        RequireStaticShape(value.shape(), "adaptive ExecutablePlan");
    }
    return verified;
}

void ValidateCandidate(const ProductionCompileRequest& request,
                       const CompiledGraph& graph) {
    const VerifiedGraphArtifacts candidate = VerifyGraphArtifacts(
        graph, request.config(), request.graph_semantic_key());
    if (candidate.pins.size() != request.verified_artifact_pins().size() ||
        candidate.identities.size() != request.ordered_artifacts().size()) {
        throw std::invalid_argument(
            "adaptive candidate primitive mapping differs from the verified baseline");
    }
    for (size_t index = 0; index < candidate.pins.size(); ++index) {
        const OrderedArtifactIdentity& expected = request.ordered_artifacts()[index];
        const OrderedArtifactIdentity& actual = candidate.identities[index];
        const internal::PrimitiveArtifactPin expected_pin =
            internal::ArtifactPinAccess::Unwrap(
                request.verified_artifact_pins()[index]);
        const internal::PrimitiveArtifactPin actual_pin =
            internal::ArtifactPinAccess::Unwrap(candidate.pins[index]);
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
    const ShapeProfileKey candidate_profile = BuildStaticExactShapeProfileKey(
        request.graph_semantic_key(), graph.plan());
    if (candidate_profile != request.shape_profile_key() ||
        BuildStaticExactDispatchKey(request.graph_semantic_key(),
                                    candidate_profile) != request.dispatch_key()) {
        throw std::invalid_argument(
            "adaptive candidate dispatch differs from the exact request");
    }
    if (BuildPlanAbiFingerprint(graph.module(), graph.plan(),
                                candidate.identities) != request.plan_abi()) {
        throw std::invalid_argument(
            "adaptive candidate Plan ABI differs from the exact request");
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
                               runtime::internal::kStaticMemoryPlanVersion);
}

}  // namespace

ProductionCompileRequest::ProductionCompileRequest(
    Function graph, CompileConfig config, const CompiledGraph& expected_contract)
    : graph_(std::move(graph)), config_(CloneCompileConfig(config)) {
    graph_semantic_key_ = Compiler::BuildGraphSemanticKey(graph_);
    const VerifiedGraphArtifacts baseline = VerifyGraphArtifacts(
        expected_contract, config_, graph_semantic_key_);
    ordered_artifacts_ = baseline.identities;
    verified_artifact_pins_ = baseline.pins;
    shape_profile_key_ = BuildStaticExactShapeProfileKey(
        graph_semantic_key_, expected_contract.plan());
    dispatch_key_ = BuildStaticExactDispatchKey(graph_semantic_key_,
                                                shape_profile_key_);
    plan_abi_ = BuildPlanAbiFingerprint(expected_contract.module(),
                                        expected_contract.plan(),
                                        ordered_artifacts_);
    Validate();
}

const Function& ProductionCompileRequest::graph() const noexcept { return graph_; }
CompileConfig ProductionCompileRequest::config() const { return CloneCompileConfig(config_); }
const GraphSemanticKey& ProductionCompileRequest::graph_semantic_key() const noexcept { return graph_semantic_key_; }
const ShapeProfileKey& ProductionCompileRequest::shape_profile_key() const noexcept { return shape_profile_key_; }
const DispatchKey& ProductionCompileRequest::dispatch_key() const noexcept { return dispatch_key_; }
const PlanAbiFingerprint& ProductionCompileRequest::plan_abi() const noexcept { return plan_abi_; }
const std::vector<OrderedArtifactIdentity>& ProductionCompileRequest::ordered_artifacts() const noexcept { return ordered_artifacts_; }
const std::vector<ArtifactPin>& ProductionCompileRequest::verified_artifact_pins() const noexcept { return verified_artifact_pins_; }

void ProductionCompileRequest::Validate() const {
    if (!graph_.defined() || !graph_semantic_key_.defined() ||
        !shape_profile_key_.defined() || !dispatch_key_.defined() ||
        !plan_abi_.defined() || ordered_artifacts_.empty() ||
        ordered_artifacts_.size() != verified_artifact_pins_.size()) {
        throw std::invalid_argument(
            "adaptive compile request has an undefined typed contract");
    }
    config_.Validate();
    if (Compiler::BuildGraphSemanticKey(graph_) != graph_semantic_key_ ||
        shape_profile_key_.graph_semantic_key() != graph_semantic_key_) {
        throw std::invalid_argument(
            "adaptive compile request graph identity is inconsistent");
    }
    for (size_t index = 0; index < ordered_artifacts_.size(); ++index) {
        const internal::PrimitiveArtifactPin pin =
            internal::ArtifactPinAccess::Unwrap(verified_artifact_pins_[index]);
        if (ordered_artifacts_[index].call_index != index ||
            ordered_artifacts_[index].artifact_key != pin.key()) {
            throw std::invalid_argument(
                "adaptive compile request baseline artifact mapping is inconsistent");
        }
        (void)ordered_artifacts_[index].CanonicalBytes();
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
            "adaptive execution requires exact dispatch and Plan ABI identities");
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
    if (!session_ || !session_->defined() || !selection_plan_key_.defined() ||
        selected_artifacts_.empty() || validation_receipt_.empty()) {
        throw std::invalid_argument(
            "prepared adaptive candidate requires session, selection, and receipt");
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
    auto session = std::make_shared<const runtime::RuntimeSession>(
        graph.module(), graph.plan());
    return std::shared_ptr<const PreparedCandidate>(new PreparedCandidate(
        std::move(graph), std::move(session),
        BuildSelectionPlanKey(request, verified.identities), verified.identities,
        std::move(validation_receipt)));
}

}  // namespace kxc::api::adaptive::experimental::production_path

#endif  // KXC_ENABLE_EXPERIMENTAL_ADAPTIVE_PRODUCTION
