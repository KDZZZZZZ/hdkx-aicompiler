#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kxc/compiler/capability.h"
#include "kxc/compiler/foundation_contract.h"
#include "kxc/compiler/pipeline.h"

namespace kxc::api::testing {

class FakeCapabilityVerifier final {
public:
    static std::string Key(const CapabilityRequest& request) {
        return request.graph_locator + "|" + ToString(request.boundary) + "|" +
               ToString(request.requested_mode);
    }

    void Set(const CapabilityRequest& request, CapabilityResult result) {
        results_.insert_or_assign(Key(request), std::move(result));
    }

    CapabilityResult Verify(const CapabilityRequest& request) const {
        const auto found = results_.find(Key(request));
        if (found != results_.end()) return found->second;
        CapabilityResult result;
        result.supported = false;
        result.diagnostic_locator = request.graph_locator;
        result.missing_capabilities = {"fake.explicit_result"};
        result.issues.push_back(CapabilityIssue{
            request.graph_locator, "Fake", "fake.explicit_result",
            "FakeCapabilityVerifier has no explicit result for this request"});
        return result;
    }

private:
    std::unordered_map<std::string, CapabilityResult> results_;
};

class FakePipelineResolver final {
public:
    static std::string Key(const PipelineRequest& request) {
        return ToString(request.dialect) + "|" +
               std::to_string(request.opt_level) + "|" +
               (request.target.defined() ? request.target->kind : "undefined") +
               "|" + std::string(request.named_pipeline);
    }

    void Set(const PipelineRequest& request, NormalizedPipeline pipeline) {
        if (!pipeline.defined()) {
            throw std::invalid_argument("fake normalized pipeline is undefined");
        }
        results_.insert_or_assign(Key(request), std::move(pipeline));
    }

    NormalizedPipeline Resolve(const PipelineRequest& request) const {
        const auto found = results_.find(Key(request));
        if (found == results_.end()) {
            throw std::runtime_error(
                "FakePipelineResolver requires an explicit request result");
        }
        return found->second;
    }

private:
    std::unordered_map<std::string, NormalizedPipeline> results_;
};

class FakeArtifactStore final {
public:
    ArtifactLookup Lookup(const ArtifactKey& key) {
        const auto found = ready_.find(key.canonical_bytes());
        if (found == ready_.end()) {
            events_.push_back(CacheObserverEvent{
                CacheEventKind::kMiss, key.digest(), 0,
                static_cast<uint64_t>(ready_.size()), 0, 0, ""});
            return ArtifactLookup{};
        }
        ArtifactPin pin(found->second);
        events_.push_back(CacheObserverEvent{
            CacheEventKind::kHit, key.digest(),
            found->second.record().byte_size,
            static_cast<uint64_t>(ready_.size()), 1, 0, ""});
        return ArtifactLookup{ArtifactLookupKind::kHit, std::move(pin)};
    }

    void Store(ArtifactHandle handle) {
        if (!handle.defined()) {
            throw std::invalid_argument("fake store requires an artifact handle");
        }
        const ArtifactRecord& record = handle.record();
        const auto inserted = ready_.emplace(
            record.artifact_key.canonical_bytes(), handle);
        if (!inserted.second) {
            const ArtifactRecord& existing = inserted.first->second.record();
            if (existing.signature_digest != record.signature_digest ||
                existing.validation_record != record.validation_record) {
                throw std::runtime_error(
                    "fake store detected conflicting complete artifact key");
            }
            return;
        }
        events_.push_back(CacheObserverEvent{
            CacheEventKind::kStore, record.artifact_key.digest(),
            record.byte_size, static_cast<uint64_t>(ready_.size()), 0, 0, ""});
    }

    void Evict(const ArtifactKey& key) {
        const auto found = ready_.find(key.canonical_bytes());
        if (found == ready_.end()) return;
        const uint64_t bytes = found->second.record().byte_size;
        ready_.erase(found);
        events_.push_back(CacheObserverEvent{
            CacheEventKind::kEvict, key.digest(), bytes,
            static_cast<uint64_t>(ready_.size()), 0, 0, ""});
    }

    const std::vector<CacheObserverEvent>& events() const noexcept {
        return events_;
    }

private:
    std::unordered_map<std::string, ArtifactHandle> ready_;
    std::vector<CacheObserverEvent> events_;
};

class FakeCompileCoordinator final {
public:
    using Ticket = std::shared_ptr<CompileTicket>;

    explicit FakeCompileCoordinator(uint64_t max_active)
        : max_active_(max_active) {
        if (max_active == 0) {
            throw std::invalid_argument("fake coordinator budget must be positive");
        }
    }

    Ticket Request(const CompileRequest& request, FakeArtifactStore* store) {
        const std::string key = request.CanonicalSingleflightKey();
        auto ticket = std::make_shared<CompileTicket>();
        ticket->ticket_id = "fake-ticket-" + std::to_string(next_ticket_++);
        ticket->canonical_request_key = key;
        if (request.cancellation.cancelled) {
            ticket->state = CompileRequestState::kCancelled;
            ticket->outcome = CompileOutcome{
                CompileRequestState::kCancelled, {},
                CompileFailure{CompileFailureCategory::kCancelled, 0,
                               "request was already cancelled"}};
            return ticket;
        }
        if (store != nullptr) {
            ArtifactLookup lookup = store->Lookup(request.artifact_key);
            if (lookup.kind == ArtifactLookupKind::kHit) {
                ticket->state = CompileRequestState::kReady;
                ticket->outcome = CompileOutcome{
                    CompileRequestState::kReady, lookup.pin, std::nullopt};
                return ticket;
            }
        }
        const auto failure = failures_.find(key);
        if (failure != failures_.end() && failure->second.retry_at > now_) {
            ticket->state = CompileRequestState::kFailed;
            CompileFailure value = failure->second.failure;
            value.retry_after_millis = failure->second.retry_at - now_;
            ticket->outcome = CompileOutcome{
                CompileRequestState::kFailed, {}, value};
            return ticket;
        }
        const auto active = active_.find(key);
        if (active != active_.end()) {
            ++active->second->merged_waiter_count;
            return active->second;
        }
        if (active_.size() >= max_active_) {
            ticket->state = CompileRequestState::kRejected;
            ticket->outcome = CompileOutcome{
                CompileRequestState::kRejected, {},
                CompileFailure{CompileFailureCategory::kBackpressure, 0,
                               "fake compile budget is saturated"}};
            return ticket;
        }
        ticket->state = CompileRequestState::kQueued;
        active_.emplace(key, ticket);
        return ticket;
    }

    void Begin(const Ticket& ticket) {
        RequireActive(ticket);
        ticket->state = CompileRequestState::kCompiling;
    }

    void Validate(const Ticket& ticket) {
        RequireActive(ticket);
        ticket->state = CompileRequestState::kValidating;
    }

    void Ready(const Ticket& ticket, ArtifactPin pin) {
        RequireActive(ticket);
        if (!pin.defined()) throw std::invalid_argument("ready requires a pin");
        ticket->state = CompileRequestState::kReady;
        ticket->outcome = CompileOutcome{
            CompileRequestState::kReady, std::move(pin), std::nullopt};
        active_.erase(ticket->canonical_request_key);
    }

    void Fail(const Ticket& ticket, CompileFailure failure) {
        RequireActive(ticket);
        if (failure.diagnostic.empty()) {
            throw std::invalid_argument("fake failure requires a diagnostic");
        }
        const uint64_t retry_at = now_ + failure.retry_after_millis;
        failures_.insert_or_assign(ticket->canonical_request_key,
                                   FailureState{failure, retry_at});
        ticket->state = CompileRequestState::kFailed;
        ticket->outcome = CompileOutcome{
            CompileRequestState::kFailed, {}, std::move(failure)};
        active_.erase(ticket->canonical_request_key);
    }

    CompileOutcome CancelWaiter(const Ticket&) const {
        return CompileOutcome{
            CompileRequestState::kCancelled, {},
            CompileFailure{CompileFailureCategory::kCancelled, 0,
                           "waiter cancelled without owning shared work"}};
    }

    void Advance(uint64_t millis) { now_ += millis; }

private:
    struct FailureState {
        CompileFailure failure;
        uint64_t retry_at{0};
    };

    void RequireActive(const Ticket& ticket) const {
        if (!ticket || !active_.count(ticket->canonical_request_key)) {
            throw std::logic_error("fake ticket is not active");
        }
    }

    uint64_t max_active_{0};
    uint64_t now_{0};
    uint64_t next_ticket_{1};
    std::unordered_map<std::string, Ticket> active_;
    std::unordered_map<std::string, FailureState> failures_;
};

class FakePlanAssembler final {
public:
    FrozenPlanInput Assemble(
        std::string graph_template_locator,
        std::vector<GraphValueLocator> value_routing,
        std::vector<SelectedArtifact> selected_artifacts,
        std::string value_contract,
        std::string memory_plan_version) const {
        FrozenPlanInput input{
            std::move(graph_template_locator), std::move(value_routing),
            std::move(selected_artifacts), std::move(value_contract),
            std::move(memory_plan_version)};
        input.Validate();
        return input;
    }
};

}  // namespace kxc::api::testing
