/*! \file src/compiler/adaptive/kernel_slot.cc
 * \brief Immutable generation publication, canary, rollback, and plan leases.
 */

#include "kxc/compiler/adaptive.h"

#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <utility>

namespace kxc::api::adaptive {

ArtifactLease::ArtifactLease(
    Generation generation, std::shared_ptr<const KernelArtifact> artifact)
    : generation_(generation), artifact_(std::move(artifact)) {
    if (generation_ == 0 || !artifact_) {
        throw std::invalid_argument(
            "artifact lease requires a generation and artifact");
    }
}

namespace {

struct VariantRecord final {
    VariantRecord(Generation generation_value,
                  Generation predecessor_generation_value,
                  std::shared_ptr<const KernelArtifact> artifact_value)
        : generation(generation_value),
          predecessor_generation(predecessor_generation_value),
          artifact(std::move(artifact_value)) {}

    const Generation generation;
    const Generation predecessor_generation;
    const std::shared_ptr<const KernelArtifact> artifact;
};

std::uint64_t StableRouteHash(const std::string& value) noexcept {
    // FNV-1a is used only for deterministic routing, never for key equality.
    std::uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char byte : value) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}

AdaptiveEvent SlotEvent(const KernelSlotKey& slot_key,
                        const PlanAbiFingerprint& abi,
                        const KernelArtifact& artifact,
                        AdaptiveEventKind kind, Generation generation,
                        Generation predecessor) {
    AdaptiveEvent event;
    event.kind = kind;
    event.generation = generation;
    event.predecessor_generation = predecessor;
    event.slot_key = slot_key.canonical();
    event.artifact_key = artifact.key().canonical();
    event.dispatch_key = artifact.applicability().canonical();
    event.abi_fingerprint = abi.canonical();
    return event;
}

}  // namespace

class KernelSlot::State final {
public:
    State(KernelSlotKey slot_key_value,
          PlanAbiFingerprint required_abi_value,
          KernelSlotOptions options_value)
        : slot_key(std::move(slot_key_value)),
          required_abi(std::move(required_abi_value)),
          options(std::move(options_value)) {
        if (options.max_dispatches == 0 ||
            options.max_retained_generations < 2) {
            throw std::invalid_argument(
                "KernelSlot dispatch bound must be non-zero and generation retention must be at least two");
        }
    }

    PublishResult Publish(std::shared_ptr<const KernelArtifact> candidate,
                          PublicationMode mode, CanaryPolicy canary,
                          bool executable_ready) {
        PublishResult result;
        AdaptiveEvent event;
        bool should_emit = false;
        {
            std::lock_guard<std::mutex> lock(mutex);
            const std::string validation_error =
                ValidateCandidate(candidate, mode, canary,
                                  executable_ready);
            if (!validation_error.empty()) {
                result.diagnostic = validation_error;
                return result;
            }
            if (last_generation == std::numeric_limits<Generation>::max()) {
                result.diagnostic = "kernel slot generation is exhausted";
                return result;
            }

            const DispatchKey& dispatch = candidate->applicability();
            if (stable_heads.count(dispatch) == 0 &&
                stable_heads.size() >= options.max_dispatches) {
                result.diagnostic =
                    "kernel slot exact dispatch budget is full";
                return result;
            }
            if (!MakeRecordSpace()) {
                result.diagnostic =
                    "kernel slot generation retention budget is full";
                return result;
            }
            const auto prior = stable_heads.find(dispatch);
            const Generation predecessor =
                prior == stable_heads.end() ? 0 : prior->second;
            const Generation generation = ++last_generation;
            records.emplace(
                generation,
                std::make_shared<const VariantRecord>(generation, predecessor,
                                                      candidate));
            if (mode == PublicationMode::kStable) {
                stable_heads.insert_or_assign(dispatch, generation);
                canary_heads.erase(dispatch);
                canary_policies.erase(dispatch);
                rollback_eligible.insert(generation);
            } else {
                canary_heads.insert_or_assign(dispatch, generation);
                canary_policies.insert_or_assign(dispatch, canary);
            }

            result.published = true;
            result.generation = generation;
            result.predecessor_generation = predecessor;
            event = SlotEvent(slot_key, required_abi, *candidate,
                              AdaptiveEventKind::kPublished, generation,
                              predecessor);
            event.diagnostic = mode == PublicationMode::kStable
                                   ? "stable"
                                   : "canary";
            should_emit = true;
        }
        if (should_emit) Emit(event);
        return result;
    }

    ArtifactLease Acquire(const DispatchKey& dispatch_key,
                          const PlanAbiFingerprint& requested_abi,
                          RoutingContext routing) const {
        std::shared_ptr<const VariantRecord> selected;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (requested_abi != required_abi) return {};
            const auto stable = stable_heads.find(dispatch_key);
            if (stable == stable_heads.end()) return {};
            Generation selected_generation = stable->second;

            const auto canary = canary_heads.find(dispatch_key);
            const auto policy = canary_policies.find(dispatch_key);
            if (routing.allow_canary && !routing.stable_request_key.empty() &&
                canary != canary_heads.end() &&
                policy != canary_policies.end() &&
                StableRouteHash(routing.stable_request_key) % 10000ULL <
                    policy->second.basis_points) {
                selected_generation = canary->second;
            }
            const auto record = records.find(selected_generation);
            if (record == records.end()) return {};
            selected = record->second;
        }

        ArtifactLease lease(selected->generation, selected->artifact);
        AdaptiveEvent event = SlotEvent(
            slot_key, required_abi, *selected->artifact,
            AdaptiveEventKind::kAcquired, selected->generation,
            selected->predecessor_generation);
        Emit(event);
        return lease;
    }

    SlotActionResult PromoteCanary(const DispatchKey& dispatch_key,
                                   Generation generation,
                                   std::string health_evidence) {
        if (health_evidence.empty()) {
            return SlotActionResult{false, generation, 0,
                                    "canary promotion requires health evidence"};
        }
        AdaptiveEvent event;
        SlotActionResult result;
        bool should_emit = false;
        {
            std::lock_guard<std::mutex> lock(mutex);
            const auto canary = canary_heads.find(dispatch_key);
            if (canary == canary_heads.end() || canary->second != generation) {
                return SlotActionResult{false, generation, 0,
                                        "generation is not the active canary"};
            }
            const auto record = records.find(generation);
            if (record == records.end()) {
                return SlotActionResult{false, generation, 0,
                                        "canary generation is unavailable"};
            }
            const auto stable = stable_heads.find(dispatch_key);
            const Generation predecessor =
                stable == stable_heads.end() ? 0 : stable->second;
            stable_heads.insert_or_assign(dispatch_key, generation);
            canary_heads.erase(canary);
            canary_policies.erase(dispatch_key);
            rollback_eligible.insert(generation);
            result = SlotActionResult{true, generation, predecessor, ""};
            event = SlotEvent(slot_key, required_abi,
                              *record->second->artifact,
                              AdaptiveEventKind::kPromoted, generation,
                              predecessor);
            event.diagnostic = std::move(health_evidence);
            should_emit = true;
        }
        if (should_emit) Emit(event);
        return result;
    }

    SlotActionResult WithdrawCanary(const DispatchKey& dispatch_key,
                                    Generation generation,
                                    std::string reason) {
        if (reason.empty()) {
            return SlotActionResult{false, generation, 0,
                                    "canary withdrawal requires a reason"};
        }
        AdaptiveEvent event;
        SlotActionResult result;
        bool should_emit = false;
        {
            std::lock_guard<std::mutex> lock(mutex);
            const auto canary = canary_heads.find(dispatch_key);
            if (canary == canary_heads.end() || canary->second != generation) {
                return SlotActionResult{false, generation, 0,
                                        "generation is not the active canary"};
            }
            const auto record = records.find(generation);
            const auto stable = stable_heads.find(dispatch_key);
            const Generation predecessor =
                stable == stable_heads.end() ? 0 : stable->second;
            canary_heads.erase(canary);
            canary_policies.erase(dispatch_key);
            rollback_eligible.erase(generation);
            result = SlotActionResult{true, generation, predecessor, ""};
            event = SlotEvent(slot_key, required_abi,
                              *record->second->artifact,
                              AdaptiveEventKind::kWithdrawn, generation,
                              predecessor);
            event.diagnostic = std::move(reason);
            should_emit = true;
        }
        if (should_emit) Emit(event);
        return result;
    }

    SlotActionResult Rollback(const DispatchKey& dispatch_key,
                              Generation generation, std::string reason) {
        if (reason.empty()) {
            return SlotActionResult{false, generation, 0,
                                    "rollback requires a reason"};
        }
        AdaptiveEvent event;
        SlotActionResult result;
        bool should_emit = false;
        {
            std::lock_guard<std::mutex> lock(mutex);
            const auto target = records.find(generation);
            if (target == records.end() ||
                rollback_eligible.count(generation) == 0 ||
                target->second->artifact->applicability() != dispatch_key) {
                return SlotActionResult{
                    false, generation, 0,
                    "rollback generation is not a retained healthy stable"};
            }
            const auto stable = stable_heads.find(dispatch_key);
            const Generation predecessor =
                stable == stable_heads.end() ? 0 : stable->second;
            if (predecessor == generation) {
                return SlotActionResult{false, generation, predecessor,
                                        "rollback generation is already stable"};
            }
            stable_heads.insert_or_assign(dispatch_key, generation);
            canary_heads.erase(dispatch_key);
            canary_policies.erase(dispatch_key);
            result = SlotActionResult{true, generation, predecessor, ""};
            event = SlotEvent(slot_key, required_abi,
                              *target->second->artifact,
                              AdaptiveEventKind::kRolledBack, generation,
                              predecessor);
            event.diagnostic = std::move(reason);
            should_emit = true;
        }
        if (should_emit) Emit(event);
        return result;
    }

    KernelSlotSnapshot Snapshot() const {
        std::lock_guard<std::mutex> lock(mutex);
        KernelSlotSnapshot result(slot_key, required_abi);
        result.last_generation = last_generation;
        result.record_count = records.size();
        result.stable_dispatches = stable_heads.size();
        result.canary_dispatches = canary_heads.size();
        return result;
    }

private:
    std::string ValidateCandidate(
        const std::shared_ptr<const KernelArtifact>& candidate,
        PublicationMode mode, CanaryPolicy canary,
        bool executable_ready) const {
        if (!candidate || candidate->key().slot_key() != slot_key ||
            candidate->compatible_abi() != required_abi ||
            !candidate->executable() || !executable_ready) {
            return "candidate does not match the slot and exact Plan ABI";
        }
        switch (mode) {
            case PublicationMode::kStable:
                if (canary.basis_points != 0) {
                    return "stable publication cannot define canary traffic";
                }
                return "";
            case PublicationMode::kCanary:
                if (canary.basis_points == 0 ||
                    canary.basis_points > 10000) {
                    return "canary basis points must be in [1, 10000]";
                }
                if (stable_heads.count(candidate->applicability()) == 0) {
                    return "canary publication requires a stable predecessor";
                }
                return "";
        }
        return "publication mode is invalid";
    }

    bool IsHead(Generation generation) const {
        for (const auto& head : stable_heads) {
            if (head.second == generation) return true;
        }
        for (const auto& head : canary_heads) {
            if (head.second == generation) return true;
        }
        return false;
    }

    bool MakeRecordSpace() {
        while (records.size() >= options.max_retained_generations) {
            auto reclaim = records.end();
            for (auto record = records.begin(); record != records.end();
                 ++record) {
                if (!IsHead(record->first)) {
                    reclaim = record;
                    break;
                }
            }
            if (reclaim == records.end()) return false;
            rollback_eligible.erase(reclaim->first);
            records.erase(reclaim);
        }
        return true;
    }

    void Emit(const AdaptiveEvent& event) const noexcept {
        if (!options.observer) return;
        try {
            options.observer(event);
        } catch (...) {
            // Slot observability never changes publication or routing state.
        }
    }

public:
    const KernelSlotKey slot_key;
    const PlanAbiFingerprint required_abi;
    const KernelSlotOptions options;
    mutable std::mutex mutex;
    Generation last_generation{0};
    std::map<Generation, std::shared_ptr<const VariantRecord>> records;
    std::map<DispatchKey, Generation> stable_heads;
    std::map<DispatchKey, Generation> canary_heads;
    std::map<DispatchKey, CanaryPolicy> canary_policies;
    std::set<Generation> rollback_eligible;
};

KernelSlot::KernelSlot(KernelSlotKey slot_key,
                       PlanAbiFingerprint required_abi,
                       KernelSlotOptions options)
    : state_(std::make_unique<State>(std::move(slot_key),
                                     std::move(required_abi),
                                     std::move(options))) {}

KernelSlot::~KernelSlot() = default;

PublishResult KernelSlot::Publish(
    std::shared_ptr<const KernelArtifact> candidate, PublicationMode mode,
    CanaryPolicy canary) {
    const bool executable_ready =
        candidate && candidate->executable() &&
        candidate->executable()->IsReady();
    return state_->Publish(std::move(candidate), mode, canary,
                           executable_ready);
}

ArtifactLease KernelSlot::Acquire(
    const DispatchKey& dispatch_key,
    const PlanAbiFingerprint& required_abi,
    RoutingContext routing) const {
    return state_->Acquire(dispatch_key, required_abi, std::move(routing));
}

SlotActionResult KernelSlot::PromoteCanary(
    const DispatchKey& dispatch_key, Generation generation,
    std::string health_evidence) {
    return state_->PromoteCanary(dispatch_key, generation,
                                 std::move(health_evidence));
}

SlotActionResult KernelSlot::WithdrawCanary(
    const DispatchKey& dispatch_key, Generation generation,
    std::string reason) {
    return state_->WithdrawCanary(dispatch_key, generation,
                                  std::move(reason));
}

SlotActionResult KernelSlot::Rollback(const DispatchKey& dispatch_key,
                                      Generation generation,
                                      std::string reason) {
    return state_->Rollback(dispatch_key, generation, std::move(reason));
}

KernelSlotSnapshot KernelSlot::Snapshot() const { return state_->Snapshot(); }

ExactPlanBinding::ExactPlanBinding(
    KernelSlotKey slot_key, DispatchKey dispatch_key,
    PlanAbiFingerprint required_abi)
    : slot_key_(std::move(slot_key)),
      dispatch_key_(std::move(dispatch_key)),
      required_abi_(std::move(required_abi)) {}

FrozenPlanVariant::FrozenPlanVariant(
    PlanVariantKey key, std::vector<ExactPlanBinding> bindings,
    std::vector<ArtifactLease> leases,
    std::shared_ptr<const PlanExecutable> executable)
    : key_(std::move(key)),
      bindings_(std::move(bindings)),
      leases_(std::move(leases)),
      executable_(std::move(executable)) {
    if (bindings_.empty() || bindings_.size() != leases_.size()) {
        throw std::invalid_argument(
            "frozen plan variant requires one lease per exact binding");
    }
    for (std::size_t i = 0; i < leases_.size(); ++i) {
        const auto& lease = leases_[i];
        const auto& binding = bindings_[i];
        if (!lease.valid() || !lease.artifact()->executable() ||
            !lease.artifact()->executable()->IsReady() ||
            lease.artifact()->key().slot_key() != binding.slot_key() ||
            lease.artifact()->applicability() != binding.dispatch_key() ||
            lease.artifact()->compatible_abi() != binding.required_abi()) {
            throw std::invalid_argument(
                "frozen plan variant lease does not match its exact binding");
        }
    }
    if (!executable_ || !executable_->IsReady()) {
        throw std::invalid_argument(
            "frozen plan variant requires a ready typed executable");
    }
}

}  // namespace kxc::api::adaptive
