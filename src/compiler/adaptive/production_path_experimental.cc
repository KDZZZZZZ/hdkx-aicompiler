/*! \file src/compiler/adaptive/production_path_experimental.cc
 * \brief Experimental static-exact adapter for a production compiler path.
 */

#include "kxc/compiler/adaptive_production_experimental.h"

#if KXC_ENABLE_EXPERIMENTAL_ADAPTIVE_PRODUCTION

#include <algorithm>
#include <atomic>
#include <future>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "../internal/primitive_cache.h"
#include "../../runtime/internal/compiled_module_node.h"
#include "../../runtime/internal/memory_plan.h"

namespace kxc::api::adaptive::experimental::production_path {
namespace {

void AppendField(std::string* out, const std::string& name,
                 const std::string& value) {
    *out += std::to_string(name.size()) + ":" + name + "=" +
            std::to_string(value.size()) + ":" + value + ";";
}

std::string CompileFlightKey(const ProductionCompileRequest& request) {
    std::string key;
    AppendField(&key, "kind", "adaptive-production-flight-v1");
    AppendField(&key, "graph",
                request.graph_semantic_key().canonical_bytes());
    AppendField(&key, "shape_profile",
                request.shape_profile_key().canonical_bytes());
    AppendField(&key, "dispatch", request.dispatch_key().canonical_bytes());
    AppendField(&key, "plan_abi", request.plan_abi().canonical_bytes());
    for (const OrderedArtifactIdentity& artifact :
         request.ordered_artifacts()) {
        AppendField(&key, "requested_primitive",
                    artifact.CanonicalBytes());
    }
    return key;
}

std::string SlotKey(const DispatchKey& dispatch,
                    const PlanAbiFingerprint& plan_abi) {
    std::string key;
    AppendField(&key, "kind", "adaptive-production-slot-v1");
    AppendField(&key, "dispatch", dispatch.canonical_bytes());
    AppendField(&key, "plan_abi", plan_abi.canonical_bytes());
    return key;
}

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
        const auto pin = internal::ProductionArtifactAccess::Pin(pins[index]);
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

/*! \brief Structural gate over a verified baseline and pinned cache objects.
 *
 * A replacement may use distinct selected artifacts and launcher objects. Its
 * represented callable ABI is the signature plus launch target/backend/metadata;
 * the candidate's own pin/module/public-record consistency remains strict. This
 * is structural compatibility, not cryptographic provenance or attestation.
 */
class ProductionValidationAuthority final {
public:
    static void Validate(const ProductionCompileRequest& request,
                         const CompiledGraph& graph) {
        const VerifiedGraphArtifacts candidate = VerifyGraphArtifacts(
            graph, request.config(), request.graph_semantic_key());
        if (candidate.pins.size() != request.verified_artifact_pins().size() ||
            candidate.identities.size() != request.ordered_artifacts().size()) {
            throw std::invalid_argument(
                "adaptive candidate primitive mapping differs from the verified baseline");
        }
        // Selection identity is deliberately independent from Plan ABI. A
        // candidate may choose a different immutable PrimitiveArtifactKey, but every
        // selected pin must retain the baseline's complete typed contract.
        for (size_t index = 0; index < candidate.pins.size(); ++index) {
            const OrderedArtifactIdentity& expected = request.ordered_artifacts()[index];
            const OrderedArtifactIdentity& actual = candidate.identities[index];
            const internal::PrimitiveArtifactPin expected_pin =
                internal::ProductionArtifactAccess::Pin(
                    request.verified_artifact_pins()[index]);
            const internal::PrimitiveArtifactPin actual_pin =
                internal::ProductionArtifactAccess::Pin(candidate.pins[index]);
            if (actual.call_index != expected.call_index ||
                actual.link_symbol != expected.link_symbol ||
                actual_pin.artifact().signature.CanonicalBytes() !=
                    expected_pin.artifact().signature.CanonicalBytes() ||
                actual_pin.artifact().launch_metadata.CanonicalBytes() !=
                    expected_pin.artifact().launch_metadata.CanonicalBytes()) {
                throw std::invalid_argument(
                    "adaptive candidate changed a verified callable ABI");
            }
        }
        const ShapeProfileKey candidate_profile =
            BuildStaticExactShapeProfileKey(
                request.graph_semantic_key(), graph.plan());
        if (candidate_profile != request.shape_profile_key() ||
            BuildStaticExactDispatchKey(request.graph_semantic_key(),
                                        candidate_profile) !=
            request.dispatch_key()) {
            throw std::invalid_argument(
                "adaptive candidate dispatch differs from the exact request");
        }
        if (BuildPlanAbiFingerprint(graph.module(), graph.plan(),
                                    candidate.identities) !=
            request.plan_abi()) {
            throw std::invalid_argument(
                "adaptive candidate Plan ABI differs from the exact request");
        }
    }
};

PlanVariantKey BuildSelectionPlanKey(
    const ProductionCompileRequest& request,
    const std::vector<OrderedArtifactIdentity>& selected_artifacts) {
    std::vector<OrderedArtifactSelectionIdentity> selections;
    selections.reserve(selected_artifacts.size());
    for (const auto& artifact : selected_artifacts) {
        selections.push_back(OrderedArtifactSelectionIdentity{
            artifact.call_index, artifact.link_symbol,
            artifact.artifact_key, 0});
    }
    return BuildPlanVariantKey(
        request.graph_semantic_key(), request.shape_profile_key(),
        selections, runtime::internal::kStaticMemoryPlanVersion);
}

AdaptiveControllerEvent EventFor(
    AdaptiveControllerEventKind kind,
    const ProductionCompileRequest& request) {
    AdaptiveControllerEvent event;
    event.kind = kind;
    event.graph_semantic_key_digest =
        request.graph_semantic_key().digest();
    event.dispatch_key_digest = request.dispatch_key().digest();
    event.plan_abi_digest = request.plan_abi().digest();
    return event;
}

AdaptiveControllerEvent EventFor(
    AdaptiveControllerEventKind kind,
    const FrozenPlanVariant& variant) {
    AdaptiveControllerEvent event;
    event.kind = kind;
    event.generation = variant.generation();
    event.graph_semantic_key_digest =
        variant.compiled_graph().graph_semantic_key().digest();
    event.selection_plan_key_digest =
        variant.artifact_lease().selection_plan_key().digest();
    event.dispatch_key_digest = variant.dispatch_key().digest();
    event.plan_abi_digest = variant.plan_abi().digest();
    event.plan_variant_digest = variant.key().digest();
    return event;
}

struct RunRetention final {
    ArtifactLease artifact_lease;
    std::shared_ptr<const FrozenPlanVariant> frozen_variant;
};

}  // namespace

ProductionCompileRequest::ProductionCompileRequest(
    Function graph, CompileConfig config,
    const CompiledGraph& expected_contract)
    : graph_(std::move(graph)),
      config_(CloneCompileConfig(config)) {
    graph_semantic_key_ = Compiler::BuildGraphSemanticKey(graph_);
    const VerifiedGraphArtifacts baseline = VerifyGraphArtifacts(
        expected_contract, config_, graph_semantic_key_);
    ordered_artifacts_ = baseline.identities;
    verified_artifact_pins_ = baseline.pins;
    shape_profile_key_ = BuildStaticExactShapeProfileKey(
        graph_semantic_key_, expected_contract.plan());
    dispatch_key_ = BuildStaticExactDispatchKey(
        graph_semantic_key_, shape_profile_key_);
    plan_abi_ = BuildPlanAbiFingerprint(
        expected_contract.module(), expected_contract.plan(),
        ordered_artifacts_);
    Validate();
}

const Function& ProductionCompileRequest::graph() const noexcept {
    return graph_;
}

CompileConfig ProductionCompileRequest::config() const {
    return CloneCompileConfig(config_);
}

const GraphSemanticKey&
ProductionCompileRequest::graph_semantic_key() const noexcept {
    return graph_semantic_key_;
}

const ShapeProfileKey&
ProductionCompileRequest::shape_profile_key() const noexcept {
    return shape_profile_key_;
}

const DispatchKey& ProductionCompileRequest::dispatch_key() const noexcept {
    return dispatch_key_;
}

const PlanAbiFingerprint& ProductionCompileRequest::plan_abi() const noexcept {
    return plan_abi_;
}

const std::vector<OrderedArtifactIdentity>&
ProductionCompileRequest::ordered_artifacts() const noexcept {
    return ordered_artifacts_;
}

const std::vector<ArtifactPin>&
ProductionCompileRequest::verified_artifact_pins() const noexcept {
    return verified_artifact_pins_;
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
    if (!session_ || !session_->defined() ||
        !selection_plan_key_.defined() || selected_artifacts_.empty() ||
        validation_receipt_.empty()) {
        throw std::invalid_argument("prepared adaptive candidate requires session, selection, and receipt");
    }
}

const CompiledGraph& PreparedCandidate::compiled_graph() const noexcept { return graph_; }
const std::shared_ptr<const runtime::RuntimeSession>& PreparedCandidate::session() const noexcept { return session_; }
const PlanVariantKey&
PreparedCandidate::selection_plan_key() const noexcept {
    return selection_plan_key_;
}
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
    ProductionValidationAuthority::Validate(request, graph);
    auto session = std::make_shared<const runtime::RuntimeSession>(
        graph.module(), graph.plan());
    const PlanVariantKey selection_plan_key = BuildSelectionPlanKey(
        request, verified.identities);
    return std::shared_ptr<const PreparedCandidate>(new PreparedCandidate(
        std::move(graph), std::move(session), selection_plan_key,
        verified.identities, std::move(validation_receipt)));
}

void ProductionCompileRequest::Validate() const {
    if (!graph_.defined() || !graph_semantic_key_.defined() ||
        !shape_profile_key_.defined() ||
        !dispatch_key_.defined() || !plan_abi_.defined() ||
        ordered_artifacts_.empty() ||
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
            internal::ProductionArtifactAccess::Pin(
                verified_artifact_pins_[index]);
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
    : dispatch_key_(std::move(dispatch_key)),
      plan_abi_(std::move(plan_abi)) {
    Validate();
}

const DispatchKey& ProductionExecutionRequest::dispatch_key() const noexcept {
    return dispatch_key_;
}

const PlanAbiFingerprint& ProductionExecutionRequest::plan_abi() const noexcept {
    return plan_abi_;
}

void ProductionExecutionRequest::Validate() const {
    if (!dispatch_key_.defined() || !plan_abi_.defined()) {
        throw std::invalid_argument(
            "adaptive execution requires exact dispatch and Plan ABI identities");
    }
}

ArtifactLease::ArtifactLease(uint64_t generation,
                             PlanVariantKey selection_plan_key,
                             CompiledGraph graph)
    : generation_(generation),
      selection_plan_key_(std::move(selection_plan_key)),
      graph_(std::move(graph)) {
    if (generation_ == 0 || !selection_plan_key_.defined() || !graph_.defined() ||
        graph_.artifact_pins().empty() ||
        std::any_of(graph_.artifact_pins().begin(), graph_.artifact_pins().end(),
                    [](const ArtifactPin& pin) { return !pin.defined(); })) {
        throw std::invalid_argument(
            "adaptive ArtifactLease requires a generation and complete pins");
    }
}

bool ArtifactLease::valid() const noexcept {
    return generation_ != 0 && selection_plan_key_.defined() && graph_.defined() &&
           !graph_.artifact_pins().empty();
}

uint64_t ArtifactLease::generation() const noexcept {
    return generation_;
}

const PlanVariantKey&
ArtifactLease::selection_plan_key() const noexcept {
    return selection_plan_key_;
}

const std::vector<ArtifactPin>& ArtifactLease::pins() const noexcept {
    return graph_.artifact_pins();
}

FrozenPlanVariant::FrozenPlanVariant(
    uint64_t generation, PlanVariantKey key, DispatchKey dispatch_key,
    PlanAbiFingerprint plan_abi, ArtifactLease artifact_lease,
    CompiledGraph compiled_graph,
    std::shared_ptr<const runtime::RuntimeSession> session)
    : generation_(generation),
      key_(std::move(key)),
      dispatch_key_(std::move(dispatch_key)),
      plan_abi_(std::move(plan_abi)),
      artifact_lease_(std::move(artifact_lease)),
      compiled_graph_(std::move(compiled_graph)),
      session_(std::move(session)) {
    if (generation_ == 0 || !key_.defined() || !dispatch_key_.defined() ||
        !plan_abi_.defined() || !artifact_lease_.valid() || !session_ ||
        !session_->defined()) {
        throw std::invalid_argument(
            "FrozenPlanVariant requires a complete immutable session bundle");
    }
}

uint64_t FrozenPlanVariant::generation() const noexcept {
    return generation_;
}

const PlanVariantKey& FrozenPlanVariant::key() const noexcept {
    return key_;
}

const DispatchKey& FrozenPlanVariant::dispatch_key() const noexcept {
    return dispatch_key_;
}

const PlanAbiFingerprint& FrozenPlanVariant::plan_abi() const noexcept {
    return plan_abi_;
}

const ArtifactLease& FrozenPlanVariant::artifact_lease() const noexcept {
    return artifact_lease_;
}

const CompiledGraph& FrozenPlanVariant::compiled_graph() const noexcept {
    return compiled_graph_;
}

const std::shared_ptr<const runtime::RuntimeSession>&
FrozenPlanVariant::session() const noexcept {
    return session_;
}

std::shared_ptr<const FrozenPlanVariant> FreezePreparedCandidate(
    uint64_t generation, DispatchKey dispatch_key, PlanAbiFingerprint plan_abi,
    std::shared_ptr<const PreparedCandidate> candidate) {
    if (generation == 0 || !candidate || candidate->validation_receipt().empty() ||
        !dispatch_key.defined() || !plan_abi.defined()) {
        throw std::invalid_argument("freeze requires generation, route, ABI, and prepared receipt");
    }
    std::vector<OrderedArtifactSelectionIdentity> artifacts;
    for (const auto& identity : candidate->selected_artifacts()) {
        artifacts.push_back(OrderedArtifactSelectionIdentity{
            identity.call_index, identity.link_symbol,
            identity.artifact_key, generation});
    }
    PlanVariantKey key = BuildPlanVariantKey(
        candidate->compiled_graph().graph_semantic_key(),
        candidate->selection_plan_key().shape_profile_key(),
        artifacts, runtime::internal::kStaticMemoryPlanVersion);
    ArtifactLease pins(generation, candidate->selection_plan_key(),
                       candidate->compiled_graph());
    return std::shared_ptr<const FrozenPlanVariant>(new FrozenPlanVariant(
        generation, std::move(key), std::move(dispatch_key), std::move(plan_abi),
        std::move(pins), candidate->compiled_graph(), candidate->session()));
}

AdministrativeQuarantineRequest::AdministrativeQuarantineRequest(
    std::shared_ptr<const FrozenPlanVariant> variant, std::string reason)
    : variant_(std::move(variant)), reason_(std::move(reason)) {
    if (!variant_ || reason_.empty()) {
        throw std::invalid_argument(
            "administrative quarantine requires a frozen variant and reason");
    }
}

AdministrativeQuarantineRequest
AdministrativeQuarantineRequest::ForTrustedControlPlane(
    std::shared_ptr<const FrozenPlanVariant> variant,
    std::string reason) {
    return AdministrativeQuarantineRequest(std::move(variant),
                                           std::move(reason));
}

const std::shared_ptr<const FrozenPlanVariant>&
AdministrativeQuarantineRequest::variant() const noexcept {
    return variant_;
}

const std::string&
AdministrativeQuarantineRequest::reason() const noexcept {
    return reason_;
}

namespace {

class ObserverScope final {
public:
    explicit ObserverScope(std::atomic<size_t>& active_callbacks) noexcept
        : active_callbacks_(active_callbacks) {
        active_callbacks_.fetch_add(1, std::memory_order_acq_rel);
    }

    ~ObserverScope() {
        active_callbacks_.fetch_sub(1, std::memory_order_release);
    }

    ObserverScope(const ObserverScope&) = delete;
    ObserverScope& operator=(const ObserverScope&) = delete;

private:
    std::atomic<size_t>& active_callbacks_;
};

}  // namespace

class AdaptiveController::State final {
public:
    struct Slot final {
        std::shared_ptr<const FrozenPlanVariant> current;
        std::vector<std::shared_ptr<const FrozenPlanVariant>> history;
        std::unordered_set<std::string> quarantined_plans;
        size_t active_compiles{0};
    };

    State(std::shared_ptr<ProductionPathCompilerAdapter> compiler_value,
          AdaptiveControllerOptions options_value)
        : compiler(std::move(compiler_value)),
          options(std::move(options_value)) {
        if (options.max_in_flight_compiles == 0 || options.max_slots == 0 ||
            options.max_discoverable_generations < 2) {
            throw std::invalid_argument(
                "adaptive controller bounds are invalid");
        }
    }

    void RejectObserverReentry() const {
        if (active_observer_callbacks.load(std::memory_order_acquire) != 0) {
            throw std::logic_error(
                "adaptive controller API called while an observer callback is active");
        }
    }

    void Emit(const AdaptiveControllerEvent& event) const noexcept {
        if (!options.observer) return;
        try {
            ObserverScope observer_scope(active_observer_callbacks);
            options.observer(event);
        } catch (...) {
            // Observability cannot alter validation, publication, or routing.
        }
    }

    template <typename Builder>
    void EmitBestEffort(Builder&& builder) const noexcept {
        try {
            Emit(builder());
        } catch (...) {
            // Event construction is observability too, never routing authority.
        }
    }

    std::shared_ptr<ProductionPathCompilerAdapter> compiler;
    const AdaptiveControllerOptions options;
    mutable std::mutex mutex;
    std::unordered_map<std::string, std::shared_ptr<Slot>> slots;
    std::unordered_map<
        std::string,
        std::shared_future<std::shared_ptr<const FrozenPlanVariant>>>
        flights;
    mutable std::atomic<size_t> active_observer_callbacks{0};
    uint64_t next_generation{1};
    uint64_t compile_requests{0};
    uint64_t merged_compiles{0};
    uint64_t published{0};
    uint64_t acquired{0};
    uint64_t run_requests{0};
    uint64_t rejected{0};
    uint64_t quarantined{0};
};

AdaptiveController::AdaptiveController(
    std::shared_ptr<ProductionPathCompilerAdapter> compiler,
    AdaptiveControllerOptions options)
    : state_(std::make_shared<State>(std::move(compiler),
                                     std::move(options))) {}

AdaptiveController::~AdaptiveController() = default;

std::shared_ptr<const FrozenPlanVariant>
AdaptiveController::CompileAndPublish(
    const ProductionCompileRequest& request) {
    state_->RejectObserverReentry();
    request.Validate();
    const std::string flight_key = CompileFlightKey(request);
    const std::string slot_key =
        SlotKey(request.dispatch_key(), request.plan_abi());
    std::shared_ptr<State::Slot> slot;
    std::shared_future<std::shared_ptr<const FrozenPlanVariant>> flight;
    std::shared_ptr<
        std::promise<std::shared_ptr<const FrozenPlanVariant>>>
        promise;
    bool owner = false;
    bool created_slot = false;
    AdaptiveControllerEvent initial =
        EventFor(AdaptiveControllerEventKind::kCompileStarted, request);
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        ++state_->compile_requests;
        const auto found_slot = state_->slots.find(slot_key);
        if (found_slot == state_->slots.end()) {
            if (state_->slots.size() >= state_->options.max_slots) {
                ++state_->rejected;
                initial.kind = AdaptiveControllerEventKind::kRejected;
                initial.diagnostic = "adaptive exact slot budget is full";
            } else {
                slot = std::make_shared<State::Slot>();
                state_->slots.emplace(slot_key, slot);
                created_slot = true;
            }
        } else {
            slot = found_slot->second;
        }
        if (slot) {
            const auto existing = state_->flights.find(flight_key);
            if (existing != state_->flights.end()) {
                flight = existing->second;
                ++state_->merged_compiles;
                initial.kind = AdaptiveControllerEventKind::kCompileMerged;
            } else if (state_->flights.size() >=
                       state_->options.max_in_flight_compiles) {
                ++state_->rejected;
                initial.kind = AdaptiveControllerEventKind::kRejected;
                initial.diagnostic =
                    "adaptive whole-plan compile budget is full";
            } else {
                promise = std::make_shared<std::promise<
                    std::shared_ptr<const FrozenPlanVariant>>>();
                flight = promise->get_future().share();
                state_->flights.emplace(flight_key, flight);
                ++slot->active_compiles;
                owner = true;
            }
            initial.in_flight_compiles = state_->flights.size();
        }
        if (created_slot && !owner && !initial.diagnostic.empty() &&
            slot->active_compiles == 0 && !slot->current) {
            state_->slots.erase(slot_key);
            slot.reset();
        }
    }
    state_->EmitBestEffort([&initial] { return initial; });
    if (!initial.diagnostic.empty()) {
        throw std::runtime_error(initial.diagnostic);
    }
    if (!owner) return flight.get();

    try {
        CompiledGraph graph = state_->compiler
                                  ? state_->compiler->Compile(request)
                                  : Compiler::Compile(request.graph(),
                                                      request.config());
        const auto prepared = PrepareCandidate(
            request, std::move(graph), "legacy-controller-local-validation");
        state_->EmitBestEffort([&request] {
            return EventFor(AdaptiveControllerEventKind::kValidated, request);
        });

        std::shared_ptr<const FrozenPlanVariant> variant;
        uint64_t predecessor = 0;
        size_t remaining_flights = 0;
        {
            std::lock_guard<std::mutex> lock(state_->mutex);
            if (state_->next_generation ==
                std::numeric_limits<uint64_t>::max()) {
                throw std::overflow_error(
                    "adaptive generation space is exhausted");
            }
            if (slot->quarantined_plans.count(
                    prepared->selection_plan_key().canonical_bytes()) != 0) {
                throw std::invalid_argument(
                    "adaptive selected plan identity is quarantined");
            }
            const uint64_t generation = state_->next_generation;
            predecessor = slot->current ? slot->current->generation() : 0;
            variant = FreezePreparedCandidate(
                generation, request.dispatch_key(), request.plan_abi(), prepared);
            // All potentially allocating authoritative work precedes the head
            // handoff. Observer payloads are built only after publication.
            slot->history.push_back(variant);
            slot->current = variant;
            while (slot->history.size() >
                   state_->options.max_discoverable_generations) {
                slot->history.erase(slot->history.begin());
            }
            ++state_->next_generation;
            ++state_->published;
            --slot->active_compiles;
            state_->flights.erase(flight_key);
            remaining_flights = state_->flights.size();
        }
        promise->set_value(variant);
        state_->EmitBestEffort([&variant, predecessor, remaining_flights] {
            AdaptiveControllerEvent published =
                EventFor(AdaptiveControllerEventKind::kPublished, *variant);
            published.predecessor_generation = predecessor;
            published.in_flight_compiles = remaining_flights;
            return published;
        });
        return variant;
    } catch (...) {
        const std::exception_ptr error = std::current_exception();
        size_t remaining_flights = 0;
        {
            std::lock_guard<std::mutex> lock(state_->mutex);
            state_->flights.erase(flight_key);
            if (slot->active_compiles != 0) --slot->active_compiles;
            if (slot->active_compiles == 0 && !slot->current &&
                slot->history.empty()) {
                const auto found = state_->slots.find(slot_key);
                if (found != state_->slots.end() && found->second == slot) {
                    state_->slots.erase(found);
                }
            }
            ++state_->rejected;
            remaining_flights = state_->flights.size();
        }
        // Resolve every merged waiter before best-effort telemetry allocation.
        promise->set_exception(error);
        state_->EmitBestEffort([&request, error, remaining_flights] {
            AdaptiveControllerEvent rejected =
                EventFor(AdaptiveControllerEventKind::kRejected, request);
            rejected.in_flight_compiles = remaining_flights;
            try {
                std::rethrow_exception(error);
            } catch (const std::exception& failure) {
                rejected.diagnostic = failure.what();
            } catch (...) {
                rejected.diagnostic = "adaptive compilation failed";
            }
            return rejected;
        });
        std::rethrow_exception(error);
    }
}

std::shared_ptr<const FrozenPlanVariant> AdaptiveController::Acquire(
    const ProductionExecutionRequest& request) const {
    state_->RejectObserverReentry();
    request.Validate();
    std::shared_ptr<const FrozenPlanVariant> variant;
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        const auto found = state_->slots.find(
            SlotKey(request.dispatch_key(), request.plan_abi()));
        if (found == state_->slots.end() || !found->second->current) {
            ++state_->rejected;
        } else {
            variant = found->second->current;
            if (found->second->quarantined_plans.count(
                    variant->artifact_lease().selection_plan_key().canonical_bytes()) !=
                0) {
                variant.reset();
                ++state_->rejected;
            } else {
                ++state_->acquired;
            }
        }
    }
    if (!variant) {
        state_->EmitBestEffort([&request] {
            AdaptiveControllerEvent event;
            event.kind = AdaptiveControllerEventKind::kRejected;
            event.dispatch_key_digest = request.dispatch_key().digest();
            event.plan_abi_digest = request.plan_abi().digest();
            event.diagnostic = "no exact published adaptive generation";
            return event;
        });
        throw std::out_of_range("no exact published adaptive generation");
    }
    state_->EmitBestEffort([&variant] {
        return EventFor(AdaptiveControllerEventKind::kAcquired, *variant);
    });
    return variant;
}

AdaptiveRunAsyncResult AdaptiveController::RunAsync(
    const ProductionExecutionRequest& request,
    const Array<runtime::NDArray>& inputs,
    const DeviceStream& stream) const {
    state_->RejectObserverReentry();
    // The strong local snapshot is frozen before input validation or launch.
    const std::shared_ptr<const FrozenPlanVariant> variant = Acquire(request);
    try {
        runtime::RunAsyncResult result =
            variant->session()->RunAsync(inputs, stream);
        auto retention = std::make_shared<RunRetention>(
            RunRetention{variant->artifact_lease(), variant});
        result.completion.RetainDependencies({}, std::move(retention));
        {
            std::lock_guard<std::mutex> lock(state_->mutex);
            ++state_->run_requests;
        }
        state_->EmitBestEffort([&variant] {
            return EventFor(AdaptiveControllerEventKind::kRunSubmitted,
                            *variant);
        });
        return AdaptiveRunAsyncResult{
            std::move(result.outputs), std::move(result.completion),
            variant->artifact_lease(), variant};
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(state_->mutex);
            ++state_->rejected;
        }
        state_->EmitBestEffort([&variant] {
            AdaptiveControllerEvent rejected =
                EventFor(AdaptiveControllerEventKind::kRejected, *variant);
            rejected.diagnostic = "adaptive RunAsync request was rejected";
            return rejected;
        });
        throw;
    }
}

AdaptiveHandoffResult AdaptiveController::RollbackAdministrative(
    const AdministrativeQuarantineRequest& quarantine,
    uint64_t target_generation) {
    state_->RejectObserverReentry();
    const std::shared_ptr<const FrozenPlanVariant>& regressed =
        quarantine.variant();
    if (!regressed || quarantine.reason().empty()) {
        return {false, 0, 0,
                "rollback requires a trusted administrative quarantine"};
    }

    std::shared_ptr<const FrozenPlanVariant> target;
    AdaptiveHandoffResult result;
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        const auto found = state_->slots.find(
            SlotKey(regressed->dispatch_key(), regressed->plan_abi()));
        if (found == state_->slots.end() ||
            found->second->current != regressed) {
            return {false, 0, regressed->generation(),
                    "regression does not identify the current exact generation"};
        }
        const auto& history = found->second->history;
        const auto retained = std::find(history.begin(), history.end(),
                                        regressed);
        if (retained == history.end()) {
            return {false, 0, regressed->generation(),
                    "regressed generation is not retained"};
        }
        if (found->second->quarantined_plans.count(
                regressed->artifact_lease().selection_plan_key().canonical_bytes()) !=
            0) {
            return {false, 0, regressed->generation(),
                    "regressed whole-plan identity is already quarantined"};
        }

        for (auto it = history.rbegin(); it != history.rend(); ++it) {
            const bool requested = target_generation == 0 ||
                                   (*it)->generation() == target_generation;
            if (!requested || (*it)->generation() >= regressed->generation() ||
                found->second->quarantined_plans.count(
                    (*it)->artifact_lease().selection_plan_key().canonical_bytes()) !=
                    0) {
                continue;
            }
            target = *it;
            break;
        }
        if (!target) {
            return {false, 0, regressed->generation(),
                    "rollback target is not a retained admissible predecessor"};
        }

        result = {true, target->generation(), regressed->generation(),
                  "trusted administrative quarantine applied; future routing "
                  "rolled back"};

        // Result allocation and quarantine insertion precede the noexcept head
        // handoff. Observer payloads are built after routing changes.
        found->second->quarantined_plans.insert(
            regressed->artifact_lease().selection_plan_key().canonical_bytes());
        found->second->current = target;
        ++state_->quarantined;
    }
    state_->EmitBestEffort([&regressed, &target, &quarantine] {
        AdaptiveControllerEvent event = EventFor(
            AdaptiveControllerEventKind::kQuarantined, *regressed);
        event.predecessor_generation = target->generation();
        event.diagnostic = quarantine.reason();
        return event;
    });
    state_->EmitBestEffort([&regressed, &target, &quarantine] {
        AdaptiveControllerEvent event = EventFor(
            AdaptiveControllerEventKind::kRolledBack, *target);
        event.predecessor_generation = regressed->generation();
        event.diagnostic = quarantine.reason();
        return event;
    });
    return result;
}

AdaptiveControllerSnapshot AdaptiveController::Snapshot() const {
    state_->RejectObserverReentry();
    std::lock_guard<std::mutex> lock(state_->mutex);
    size_t retained = 0;
    for (const auto& slot : state_->slots) {
        retained += slot.second->history.size();
    }
    return AdaptiveControllerSnapshot{
        state_->next_generation,
        state_->compile_requests,
        state_->merged_compiles,
        state_->published,
        state_->acquired,
        state_->run_requests,
        state_->rejected,
        state_->quarantined,
        retained,
        state_->flights.size()};
}

}  // namespace kxc::api::adaptive::experimental::production_path

#endif  // KXC_ENABLE_EXPERIMENTAL_ADAPTIVE_PRODUCTION
