/*! \file src/compiler/adaptive/kernel_slot.cc
 * \brief Experimental immutable generation publication and RCU leases.
 */

#include "../../../test/support/adaptive_v1.h"

#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace kxc::api::experimental::adaptive::v1 {

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
    VariantRecord(
        Generation generation_value, Generation predecessor_generation_value,
        std::shared_ptr<const KernelArtifact> artifact_value,
        std::shared_ptr<const ArtifactValidationRecord> validation_value,
        std::shared_ptr<const GenerationHealthRecord> health_value = nullptr)
        : generation(generation_value),
          predecessor_generation(predecessor_generation_value),
          artifact(std::move(artifact_value)),
          validation(std::move(validation_value)),
          health(std::move(health_value)) {}

    const Generation generation;
    const Generation predecessor_generation;
    const std::shared_ptr<const KernelArtifact> artifact;
    const std::shared_ptr<const ArtifactValidationRecord> validation;
    const std::shared_ptr<const GenerationHealthRecord> health;
};

std::shared_ptr<const VariantRecord> WithHealth(
    const VariantRecord& record,
    std::shared_ptr<const GenerationHealthRecord> health) {
    return std::make_shared<const VariantRecord>(
        record.generation, record.predecessor_generation, record.artifact,
        record.validation, std::move(health));
}

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
    event.artifact_bytes = artifact.byte_size();
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
          std::shared_ptr<AdaptiveValidationAuthority> authority_value,
          KernelSlotOptions options_value)
        : slot_key(std::move(slot_key_value)),
          required_abi(std::move(required_abi_value)),
          authority(std::move(authority_value)),
          options(std::move(options_value)) {
        if (!authority) {
            throw std::invalid_argument(
                "KernelSlot requires a validation authority");
        }
        if (options.max_dispatches == 0 ||
            options.max_retained_generations < 2 ||
            options.max_artifact_bytes == 0 ||
            options.max_retained_artifact_bytes == 0) {
            throw std::invalid_argument(
                "KernelSlot count and byte bounds are invalid");
        }
    }

    PublishResult Publish(
        std::shared_ptr<const KernelArtifact> candidate,
        std::shared_ptr<const ArtifactValidationRecord> validation,
        PublicationMode mode, CanaryPolicy canary) {
        PublishResult result;
        AdaptiveEvent event;
        bool should_emit = false;
        {
            std::lock_guard<std::mutex> lock(mutex);
            const std::string validation_error =
                ValidateCandidate(candidate, validation, mode, canary);
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

            std::vector<Generation> reclaim;
            if (!PlanRecordSpace(dispatch, mode, candidate->byte_size(),
                                 &reclaim)) {
                result.diagnostic =
                    "kernel slot retained generation byte budget is full";
                return result;
            }
            const auto prior = stable_heads.find(dispatch);
            const Generation predecessor =
                prior == stable_heads.end() ? 0 : prior->second;
            const Generation generation = last_generation + 1;
            const auto record = std::make_shared<const VariantRecord>(
                generation, predecessor, candidate, validation);
            if (!authority->VerifyAndConsumeArtifactValidation(
                    *candidate, *validation)) {
                result.diagnostic =
                    "artifact validation token is invalid, foreign, or replayed";
                return result;
            }

            // Allocate the new map node before removing discoverable records.
            // If allocation fails the token is burned fail-closed, but current
            // routing and existing leases remain unchanged.
            records.emplace(generation, record);
            retained_artifact_bytes += candidate->byte_size();
            last_generation = generation;
            for (const Generation reclaimed_generation : reclaim) {
                EraseRecord(reclaimed_generation);
            }
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
                                   ? "validated stable"
                                   : "validated canary";
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
            if (stable == stable_heads.end() ||
                !IsAdmissible(stable->second)) {
                return {};
            }
            Generation selected_generation = stable->second;

            const auto canary = canary_heads.find(dispatch_key);
            const auto policy = canary_policies.find(dispatch_key);
            if (routing.allow_canary && !routing.stable_request_key.empty() &&
                canary != canary_heads.end() &&
                policy != canary_policies.end() &&
                IsAdmissible(canary->second) &&
                StableRouteHash(routing.stable_request_key) % 10000ULL <
                    policy->second.basis_points) {
                selected_generation = canary->second;
            }
            selected = records.at(selected_generation);
        }

        ArtifactLease lease(selected->generation, selected->artifact);
        Emit(SlotEvent(slot_key, required_abi, *selected->artifact,
                       AdaptiveEventKind::kAcquired, selected->generation,
                       selected->predecessor_generation));
        return lease;
    }

    SlotActionResult PromoteCanary(
        std::shared_ptr<const GenerationHealthRecord> health) {
        if (!health ||
            health->disposition() != GenerationHealthDisposition::kHealthy) {
            return SlotActionResult{
                false, health ? health->generation() : 0, 0,
                "canary promotion requires a healthy generation record"};
        }

        AdaptiveEvent event;
        SlotActionResult result;
        bool should_emit = false;
        {
            std::lock_guard<std::mutex> lock(mutex);
            const DispatchKey& dispatch = health->dispatch_key();
            const Generation generation = health->generation();
            const auto canary = canary_heads.find(dispatch);
            if (health->slot_key() != slot_key ||
                health->compatible_abi() != required_abi ||
                canary == canary_heads.end() || canary->second != generation) {
                return SlotActionResult{false, generation, 0,
                                        "generation is not the active canary"};
            }
            const auto record = records.find(generation);
            if (record == records.end() ||
                record->second->artifact->key() != health->artifact_key() ||
                !IsAdmissible(generation)) {
                return SlotActionResult{
                    false, generation, 0,
                    "health record does not match an admissible canary"};
            }
            if (!authority->VerifyAndConsumeHealth(*health)) {
                return SlotActionResult{
                    false, generation, 0,
                    "health token is invalid, foreign, or replayed"};
            }

            const auto stable = stable_heads.find(dispatch);
            const Generation predecessor =
                stable == stable_heads.end() ? 0 : stable->second;
            records.insert_or_assign(generation,
                                     WithHealth(*record->second, health));
            stable_heads.insert_or_assign(dispatch, generation);
            canary_heads.erase(canary);
            canary_policies.erase(dispatch);
            rollback_eligible.insert(generation);
            result = SlotActionResult{true, generation, predecessor, ""};
            event = SlotEvent(slot_key, required_abi,
                              *record->second->artifact,
                              AdaptiveEventKind::kPromoted, generation,
                              predecessor);
            event.diagnostic = health->record_id();
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
            if (record == records.end()) {
                return SlotActionResult{false, generation, 0,
                                        "canary generation is unavailable"};
            }
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

    SlotActionResult Rollback(
        Generation target_generation,
        std::shared_ptr<const GenerationHealthRecord> regression) {
        if (!regression || regression->disposition() !=
                               GenerationHealthDisposition::kQuarantined) {
            return SlotActionResult{
                false, target_generation, 0,
                "rollback requires a quarantined runtime health record"};
        }

        AdaptiveEvent quarantine_event;
        AdaptiveEvent rollback_event;
        SlotActionResult result;
        bool should_emit = false;
        {
            std::lock_guard<std::mutex> lock(mutex);
            const DispatchKey& dispatch = regression->dispatch_key();
            const auto current_head = stable_heads.find(dispatch);
            if (regression->slot_key() != slot_key ||
                regression->compatible_abi() != required_abi ||
                current_head == stable_heads.end() ||
                current_head->second != regression->generation()) {
                return SlotActionResult{
                    false, target_generation, 0,
                    "runtime regression does not identify the stable head"};
            }
            const Generation regressed_generation = current_head->second;
            const auto regressed = records.find(regressed_generation);
            const auto target = records.find(target_generation);
            if (regressed == records.end() ||
                regressed->second->artifact->key() !=
                    regression->artifact_key()) {
                return SlotActionResult{
                    false, target_generation, regressed_generation,
                    "runtime regression does not match the stable artifact"};
            }
            if (target == records.end() || target_generation == regressed_generation ||
                rollback_eligible.count(target_generation) == 0 ||
                target->second->artifact->applicability() != dispatch ||
                target->second->artifact->key() == regression->artifact_key() ||
                !IsAdmissible(target_generation)) {
                return SlotActionResult{
                    false, target_generation, regressed_generation,
                    "rollback target is not a retained admissible stable"};
            }
            const auto quarantined_record =
                WithHealth(*regressed->second, regression);
            std::map<KernelArtifactKey,
                     std::shared_ptr<const GenerationHealthRecord>>
                pending_quarantine;
            if (quarantine_records.count(regression->artifact_key()) == 0) {
                pending_quarantine.emplace(regression->artifact_key(),
                                           regression);
            }
            if (!authority->VerifyAndConsumeHealth(*regression)) {
                return SlotActionResult{
                    false, target_generation, regressed_generation,
                    "regression token is invalid, foreign, or replayed"};
            }

            // All allocating work is complete before the one-shot token is
            // consumed. The following transition is fail-closed and does not
            // invalidate leases that linearized before quarantine.
            records.insert_or_assign(regressed_generation,
                                     quarantined_record);
            if (pending_quarantine.empty()) {
                quarantine_records.insert_or_assign(
                    regression->artifact_key(), regression);
            } else {
                quarantine_records.merge(pending_quarantine);
            }
            for (const auto& item : records) {
                if (item.second->artifact->key() ==
                    regression->artifact_key()) {
                    rollback_eligible.erase(item.first);
                }
            }
            for (auto canary = canary_heads.begin();
                 canary != canary_heads.end();) {
                const auto record = records.find(canary->second);
                if (record != records.end() &&
                    record->second->artifact->key() ==
                        regression->artifact_key()) {
                    canary_policies.erase(canary->first);
                    canary = canary_heads.erase(canary);
                } else {
                    ++canary;
                }
            }
            stable_heads.insert_or_assign(dispatch, target_generation);

            result = SlotActionResult{true, target_generation,
                                      regressed_generation, ""};
            quarantine_event = SlotEvent(
                slot_key, required_abi, *regressed->second->artifact,
                AdaptiveEventKind::kQuarantined, regressed_generation,
                target_generation);
            quarantine_event.diagnostic = regression->record_id();
            rollback_event = SlotEvent(
                slot_key, required_abi, *target->second->artifact,
                AdaptiveEventKind::kRolledBack, target_generation,
                regressed_generation);
            rollback_event.diagnostic = regression->evidence_id();
            should_emit = true;
        }
        if (should_emit) {
            Emit(quarantine_event);
            Emit(rollback_event);
        }
        return result;
    }

    KernelSlotSnapshot Snapshot() const {
        std::lock_guard<std::mutex> lock(mutex);
        KernelSlotSnapshot result(slot_key, required_abi);
        result.last_generation = last_generation;
        result.record_count = records.size();
        result.retained_artifact_bytes = retained_artifact_bytes;
        result.stable_dispatches = stable_heads.size();
        result.canary_dispatches = canary_heads.size();
        result.quarantined_artifacts = quarantine_records.size();
        return result;
    }

private:
    std::string ValidateCandidate(
        const std::shared_ptr<const KernelArtifact>& candidate,
        const std::shared_ptr<const ArtifactValidationRecord>& validation,
        PublicationMode mode, CanaryPolicy canary) const {
        if (!candidate || !validation ||
            candidate->key().slot_key() != slot_key ||
            candidate->compatible_abi() != required_abi ||
            !candidate->executable() || !candidate->executable()->IsReady()) {
            return "candidate does not match the slot and exact Plan ABI";
        }
        if (validation->artifact_key() != candidate->key() ||
            validation->dispatch_key() != candidate->applicability() ||
            validation->compatible_abi() != candidate->compatible_abi() ||
            validation->artifact_bytes() != candidate->byte_size()) {
            return "validation record does not bind the exact candidate";
        }
        if (quarantine_records.count(candidate->key()) != 0) {
            return "candidate artifact identity is quarantined";
        }
        if (candidate->byte_size() > options.max_artifact_bytes ||
            candidate->byte_size() > options.max_retained_artifact_bytes) {
            return "candidate exceeds the kernel slot artifact byte budget";
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

    bool IsHeadAfterReplacement(Generation generation,
                                const DispatchKey& dispatch,
                                PublicationMode mode) const {
        for (const auto& head : stable_heads) {
            if (head.second != generation) continue;
            if (mode == PublicationMode::kStable && head.first == dispatch) {
                return false;
            }
            return true;
        }
        for (const auto& head : canary_heads) {
            if (head.second != generation) continue;
            if (head.first == dispatch) return false;
            return true;
        }
        return false;
    }

    bool PlanRecordSpace(const DispatchKey& dispatch, PublicationMode mode,
                         std::size_t candidate_bytes,
                         std::vector<Generation>* reclaim) const {
        std::size_t projected_count = records.size() + 1;
        std::size_t projected_bytes = retained_artifact_bytes;
        auto bytes_fit = [&] {
            return candidate_bytes <=
                   options.max_retained_artifact_bytes - projected_bytes;
        };

        for (const auto& item : records) {
            if (projected_count <= options.max_retained_generations &&
                bytes_fit()) {
                break;
            }
            if (IsHeadAfterReplacement(item.first, dispatch, mode)) continue;
            reclaim->push_back(item.first);
            --projected_count;
            projected_bytes -= item.second->artifact->byte_size();
        }
        return projected_count <= options.max_retained_generations &&
               bytes_fit();
    }

    void EraseRecord(Generation generation) {
        const auto record = records.find(generation);
        if (record == records.end()) return;
        retained_artifact_bytes -= record->second->artifact->byte_size();
        rollback_eligible.erase(generation);
        records.erase(record);
    }

    bool IsAdmissible(Generation generation) const {
        const auto record = records.find(generation);
        if (record == records.end() ||
            quarantine_records.count(record->second->artifact->key()) != 0) {
            return false;
        }
        return !record->second->health ||
               record->second->health->disposition() !=
                   GenerationHealthDisposition::kQuarantined;
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
    const std::shared_ptr<AdaptiveValidationAuthority> authority;
    const KernelSlotOptions options;
    mutable std::mutex mutex;
    Generation last_generation{0};
    std::size_t retained_artifact_bytes{0};
    std::map<Generation, std::shared_ptr<const VariantRecord>> records;
    std::map<DispatchKey, Generation> stable_heads;
    std::map<DispatchKey, Generation> canary_heads;
    std::map<DispatchKey, CanaryPolicy> canary_policies;
    std::set<Generation> rollback_eligible;
    std::map<KernelArtifactKey,
             std::shared_ptr<const GenerationHealthRecord>>
        quarantine_records;
};

KernelSlot::KernelSlot(
    KernelSlotKey slot_key, PlanAbiFingerprint required_abi,
    std::shared_ptr<AdaptiveValidationAuthority> validation_authority,
    KernelSlotOptions options)
    : state_(std::make_unique<State>(
          std::move(slot_key), std::move(required_abi),
          std::move(validation_authority), std::move(options))) {}

KernelSlot::~KernelSlot() = default;

PublishResult KernelSlot::Publish(
    std::shared_ptr<const KernelArtifact> candidate,
    std::shared_ptr<const ArtifactValidationRecord> validation,
    PublicationMode mode, CanaryPolicy canary) {
    return state_->Publish(std::move(candidate), std::move(validation), mode,
                           canary);
}

ArtifactLease KernelSlot::Acquire(
    const DispatchKey& dispatch_key,
    const PlanAbiFingerprint& required_abi,
    RoutingContext routing) const {
    return state_->Acquire(dispatch_key, required_abi, std::move(routing));
}

SlotActionResult KernelSlot::PromoteCanary(
    std::shared_ptr<const GenerationHealthRecord> health) {
    return state_->PromoteCanary(std::move(health));
}

SlotActionResult KernelSlot::WithdrawCanary(
    const DispatchKey& dispatch_key, Generation generation,
    std::string reason) {
    return state_->WithdrawCanary(dispatch_key, generation,
                                  std::move(reason));
}

SlotActionResult KernelSlot::Rollback(
    Generation target_generation,
    std::shared_ptr<const GenerationHealthRecord> regression) {
    return state_->Rollback(target_generation, std::move(regression));
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

}  // namespace kxc::api::experimental::adaptive::v1
