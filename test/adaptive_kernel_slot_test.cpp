/*! \file test/adaptive_kernel_slot_test.cpp
 * \brief Tests experimental v1 validated generation publication and RCU leases.
 */

#include <atomic>
#include <exception>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "kxc/compiler/adaptive.h"

namespace {

#define TEST_CHECK(condition, message)                                           \
    do {                                                                          \
        if (!(condition)) {                                                       \
            std::cerr << "[FAIL] " << __FUNCTION__ << ": " << (message) << '\n'; \
            return false;                                                        \
        }                                                                         \
    } while (0)

namespace adaptive = kxc::api::experimental::adaptive::v1;

using adaptive::AdaptiveEvent;
using adaptive::AdaptiveEventKind;
using adaptive::AdaptiveValidationAuthority;
using adaptive::ArtifactCompiler;
using adaptive::ArtifactExecutable;
using adaptive::ArtifactLease;
using adaptive::ArtifactValidationRecord;
using adaptive::ArtifactValidationToken;
using adaptive::CanaryPolicy;
using adaptive::CancellationToken;
using adaptive::CompileAttempt;
using adaptive::CompileCoordinator;
using adaptive::CompileRequest;
using adaptive::DispatchKey;
using adaptive::ExactPlanAssembler;
using adaptive::ExactPlanBinding;
using adaptive::FrozenPlanVariant;
using adaptive::Generation;
using adaptive::GenerationHealthDisposition;
using adaptive::GenerationHealthRecord;
using adaptive::GenerationHealthToken;
using adaptive::KernelArtifact;
using adaptive::KernelArtifactKey;
using adaptive::KernelSlot;
using adaptive::KernelSlotKey;
using adaptive::KernelSlotOptions;
using adaptive::PlanAbiFingerprint;
using adaptive::PlanExecutable;
using adaptive::PlanVariantKey;
using adaptive::PublicationMode;
using adaptive::RequestKind;
using adaptive::RoutingContext;

static_assert(!std::is_constructible_v<KernelSlot, KernelSlotKey,
                                       PlanAbiFingerprint>);
static_assert(!std::is_invocable_v<decltype(&KernelSlot::Publish), KernelSlot*,
                                   std::shared_ptr<const KernelArtifact>>);

constexpr const char* kSlot = "unit:add|target:fake-cpu";
constexpr const char* kDispatch = "exact:f32[4]|contiguous";
constexpr const char* kAbi =
    "abi-v1|input:f32[4]:align4|output:f32[4]:align64|workspace:0";

bool Throws(const std::function<void()>& function) {
    try {
        function();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

class FakeExecutable final : public ArtifactExecutable {
public:
    explicit FakeExecutable(std::string version) : version_(std::move(version)) {}
    bool IsReady() const noexcept override { return true; }
    std::string DebugName() const override { return version_; }

private:
    std::string version_;
};

class FakePlanExecutable final : public PlanExecutable {
public:
    explicit FakePlanExecutable(std::string name) : name_(std::move(name)) {}
    bool IsReady() const noexcept override { return true; }
    std::string DebugName() const override { return name_; }

private:
    std::string name_;
};

std::shared_ptr<const KernelArtifact> Artifact(
    std::string version, std::string slot = kSlot,
    std::string dispatch = kDispatch, std::string abi = kAbi,
    std::size_t byte_size = 4096) {
    return std::make_shared<const KernelArtifact>(
        KernelArtifactKey(KernelSlotKey(std::move(slot)), "artifact:" + version),
        DispatchKey::Exact(std::move(dispatch)), PlanAbiFingerprint(std::move(abi)),
        std::make_shared<const FakeExecutable>(version), byte_size, "fake-compiler");
}

CompileRequest Request(std::string version) {
    return CompileRequest(
        KernelArtifactKey(KernelSlotKey(kSlot), "artifact:" + version),
        DispatchKey::Exact(kDispatch), PlanAbiFingerprint(kAbi), "model@1",
        RequestKind::kDemand);
}

class FakeValidationAuthority final : public AdaptiveValidationAuthority {
public:
    // This test-only ledger is a trust seam, not a cryptographic attestation.
    std::shared_ptr<const ArtifactValidationRecord> ValidateArtifact(
        const CompileRequest& request,
        const std::shared_ptr<const KernelArtifact>& artifact) override {
        if (!artifact || artifact->key() != request.artifact_key() ||
            artifact->applicability() != request.dispatch_key() ||
            artifact->compatible_abi() != request.required_abi()) {
            return nullptr;
        }
        return ValidationFor(*artifact);
    }

    std::shared_ptr<const ArtifactValidationRecord> ValidationFor(
        const KernelArtifact& artifact) {
        std::lock_guard<std::mutex> lock(mutex_);
        const std::string token = NextTokenLocked("artifact");
        auto record = std::make_shared<const ArtifactValidationRecord>(
            "record:" + token, artifact.key(), artifact.applicability(),
            artifact.compatible_abi(), artifact.byte_size(),
            ArtifactValidationToken(token));
        artifact_records_.emplace(token, record);
        return record;
    }

    std::shared_ptr<const GenerationHealthRecord> Health(
        const KernelArtifact& artifact, Generation generation,
        GenerationHealthDisposition disposition,
        std::string evidence = "test") {
        return HealthWithAbi(artifact, generation, disposition,
                             artifact.compatible_abi(), std::move(evidence));
    }

    std::shared_ptr<const GenerationHealthRecord> HealthWithAbi(
        const KernelArtifact& artifact, Generation generation,
        GenerationHealthDisposition disposition,
        PlanAbiFingerprint compatible_abi,
        std::string evidence = "test") {
        std::lock_guard<std::mutex> lock(mutex_);
        const std::string token = NextTokenLocked("health");
        auto record = std::make_shared<const GenerationHealthRecord>(
            "record:" + token, artifact.key().slot_key(), artifact.key(),
            artifact.applicability(), std::move(compatible_abi), generation,
            disposition, std::move(evidence), GenerationHealthToken(token));
        health_records_.emplace(token, record);
        return record;
    }

    bool VerifyAndConsumeArtifactValidation(
        const KernelArtifact& artifact,
        const ArtifactValidationRecord& record) noexcept override {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = artifact_records_.find(record.token().opaque());
        if (found == artifact_records_.end() || consumed_.count(record.token().opaque()) ||
            !SameArtifactRecord(*found->second, record) ||
            record.artifact_key() != artifact.key() ||
            record.dispatch_key() != artifact.applicability() ||
            record.compatible_abi() != artifact.compatible_abi() ||
            record.artifact_bytes() != artifact.byte_size()) {
            return false;
        }
        consumed_.insert(record.token().opaque());
        return true;
    }

    bool VerifyAndConsumeHealth(
        const GenerationHealthRecord& record) noexcept override {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = health_records_.find(record.token().opaque());
        if (found == health_records_.end() || consumed_.count(record.token().opaque()) ||
            !SameHealthRecord(*found->second, record)) {
            return false;
        }
        consumed_.insert(record.token().opaque());
        return true;
    }

private:
    std::string NextTokenLocked(const char* kind) {
        return std::string(kind) + ":" + std::to_string(++next_token_);
    }

    static bool SameArtifactRecord(const ArtifactValidationRecord& lhs,
                                   const ArtifactValidationRecord& rhs) {
        return lhs.record_id() == rhs.record_id() &&
               lhs.artifact_key() == rhs.artifact_key() &&
               lhs.dispatch_key() == rhs.dispatch_key() &&
               lhs.compatible_abi() == rhs.compatible_abi() &&
               lhs.artifact_bytes() == rhs.artifact_bytes() &&
               lhs.token().opaque() == rhs.token().opaque();
    }

    static bool SameHealthRecord(const GenerationHealthRecord& lhs,
                                 const GenerationHealthRecord& rhs) {
        return lhs.record_id() == rhs.record_id() && lhs.slot_key() == rhs.slot_key() &&
               lhs.artifact_key() == rhs.artifact_key() &&
               lhs.dispatch_key() == rhs.dispatch_key() &&
               lhs.compatible_abi() == rhs.compatible_abi() &&
               lhs.generation() == rhs.generation() &&
               lhs.disposition() == rhs.disposition() &&
               lhs.evidence_id() == rhs.evidence_id() &&
               lhs.token().opaque() == rhs.token().opaque();
    }

    std::mutex mutex_;
    std::uint64_t next_token_{0};
    std::map<std::string, std::shared_ptr<const ArtifactValidationRecord>>
        artifact_records_;
    std::map<std::string, std::shared_ptr<const GenerationHealthRecord>>
        health_records_;
    std::set<std::string> consumed_;
};

class FakeCompiler final : public ArtifactCompiler {
public:
    CompileAttempt Compile(const CompileRequest& request,
                           const CancellationToken& cancellation) override {
        calls.fetch_add(1, std::memory_order_relaxed);
        if (cancellation.IsCancellationRequested()) {
            return CompileAttempt::Failed(
                adaptive::CompileFailureCategory::kCancelled, "cancelled");
        }
        const std::string canonical = request.artifact_key().canonical();
        const std::string prefix = "artifact:";
        const std::string version = canonical.rfind(prefix, 0) == 0
                                        ? canonical.substr(prefix.size())
                                        : canonical;
        return CompileAttempt::Ready(Artifact(version));
    }

    std::atomic<int> calls{0};
};

class FakePlanAssembler final : public ExactPlanAssembler {
public:
    std::shared_ptr<const FrozenPlanVariant> Assemble(
        PlanVariantKey key, std::vector<ExactPlanBinding> bindings,
        std::vector<ArtifactLease> leases) override {
        ++calls;
        auto plan = std::make_shared<const FrozenPlanVariant>(
            std::move(key), std::move(bindings), std::move(leases),
            std::make_shared<const FakePlanExecutable>("fake-static-plan"));
        last = plan;
        return plan;
    }

    int calls{0};
    std::shared_ptr<const FrozenPlanVariant> last;
};

ExactPlanBinding Binding() {
    return ExactPlanBinding(KernelSlotKey(kSlot), DispatchKey::Exact(kDispatch),
                            PlanAbiFingerprint(kAbi));
}

KernelSlotOptions SlotOptions(std::size_t generations = 64,
                              std::size_t artifact_bytes = 1024 * 1024,
                              std::size_t retained_bytes = 4 * 1024 * 1024,
                              std::size_t dispatches = 64,
                              adaptive::AdaptiveObserver observer = {}) {
    KernelSlotOptions options;
    options.max_dispatches = dispatches;
    options.max_retained_generations = generations;
    options.max_artifact_bytes = artifact_bytes;
    options.max_retained_artifact_bytes = retained_bytes;
    options.observer = std::move(observer);
    return options;
}

adaptive::PublishResult Publish(KernelSlot& slot,
                                const std::shared_ptr<FakeValidationAuthority>& authority,
                                const std::shared_ptr<const KernelArtifact>& artifact,
                                PublicationMode mode = PublicationMode::kStable,
                                CanaryPolicy canary = {}) {
    return slot.Publish(artifact, authority->ValidationFor(*artifact), mode, canary);
}

class EventLog final {
public:
    void Record(const AdaptiveEvent& event) {
        std::lock_guard<std::mutex> lock(mutex_);
        events_.push_back(event);
    }

    std::size_t Count(AdaptiveEventKind kind) const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::size_t count = 0;
        for (const auto& event : events_) {
            if (event.kind == kind) ++count;
        }
        return count;
    }

private:
    mutable std::mutex mutex_;
    std::vector<AdaptiveEvent> events_;
};

bool TestPublicationRequiresBoundOneShotValidation() {
    const auto authority = std::make_shared<FakeValidationAuthority>();
    KernelSlot slot{KernelSlotKey(kSlot), PlanAbiFingerprint(kAbi), authority};
    const auto v1 = Artifact("v1");

    TEST_CHECK(!slot.Publish(v1, nullptr).published,
               "a missing/default validation record must not publish");
    const auto wrong_artifact = Artifact("wrong");
    TEST_CHECK(!slot.Publish(v1, authority->ValidationFor(*wrong_artifact)).published,
               "a record bound to another artifact must not publish");
    const ArtifactValidationRecord forged(
        "forged", v1->key(), v1->applicability(), v1->compatible_abi(),
        v1->byte_size(), ArtifactValidationToken("forged-token"));
    TEST_CHECK(!slot.Publish(v1,
                             std::make_shared<const ArtifactValidationRecord>(forged))
                    .published,
               "a manually forged validation record must not publish");

    const auto validation = authority->ValidationFor(*v1);
    TEST_CHECK(slot.Publish(v1, validation).published &&
                   !slot.Publish(v1, validation).published,
               "a valid token is accepted once and its replay is rejected");
    return true;
}

bool TestGenerationAndExactPublicationGate() {
    const auto authority = std::make_shared<FakeValidationAuthority>();
    KernelSlot slot{KernelSlotKey(kSlot), PlanAbiFingerprint(kAbi), authority};
    const auto first = Publish(slot, authority, Artifact("v1"));
    const auto second = Publish(slot, authority, Artifact("v2"));
    TEST_CHECK(first.published && second.published && first.generation == 1 &&
                   second.generation == 2 && second.predecessor_generation == 1,
               "stable generations must increase monotonically");

    const auto current = slot.Acquire(DispatchKey::Exact(kDispatch),
                                      PlanAbiFingerprint(kAbi));
    TEST_CHECK(current.valid() && current.generation() == 2 &&
                   current.artifact()->executable()->DebugName() == "v2",
               "future acquire should see the newest stable generation");

    TEST_CHECK(!Publish(slot, authority, Artifact("bad-slot", "other-slot")).published &&
                   !Publish(slot, authority,
                            Artifact("bad-abi", kSlot, kDispatch, "abi-v2"))
                        .published &&
                   slot.Acquire(DispatchKey::Exact(kDispatch), PlanAbiFingerprint(kAbi))
                           .generation() == 2,
               "slot or Plan ABI mismatch must not replace healthy state");
    TEST_CHECK(!slot.Acquire(DispatchKey::Exact("exact:f32[8]"),
                             PlanAbiFingerprint(kAbi))
                    .valid() &&
                   !slot.Acquire(DispatchKey::Exact(kDispatch),
                                 PlanAbiFingerprint("abi-v2"))
                        .valid(),
               "acquire must use exact dispatch and exact Plan ABI");
    return true;
}

bool TestLeaseAndFrozenPlanRetainOldGeneration() {
    const auto authority = std::make_shared<FakeValidationAuthority>();
    ArtifactLease old_lease;
    std::shared_ptr<const FrozenPlanVariant> frozen;
    std::weak_ptr<const ArtifactExecutable> old_executable;
    {
        KernelSlot slot{KernelSlotKey(kSlot), PlanAbiFingerprint(kAbi), authority};
        const auto old_artifact = Artifact("v1");
        old_executable = old_artifact->executable();
        TEST_CHECK(Publish(slot, authority, old_artifact).published,
                   "first artifact should publish");
        old_lease = slot.Acquire(DispatchKey::Exact(kDispatch), PlanAbiFingerprint(kAbi));
        FakePlanAssembler assembler;
        frozen = assembler.Assemble(PlanVariantKey("plan:model@1:v1"), {Binding()},
                                    {old_lease});
        TEST_CHECK(Publish(slot, authority, Artifact("v2")).published &&
                       slot.Acquire(DispatchKey::Exact(kDispatch), PlanAbiFingerprint(kAbi))
                               .generation() == 2,
                   "new requests should move to generation two");
    }

    TEST_CHECK(old_lease.valid() && old_lease.generation() == 1 &&
                   !old_executable.expired() && frozen &&
                   frozen->leases()[0].generation() == 1 &&
                   frozen->executable()->IsReady(),
               "lease and frozen plan must outlive the slot and newer publish");
    old_lease = ArtifactLease();
    TEST_CHECK(!old_executable.expired(),
               "frozen plan must independently retain the old generation");
    frozen.reset();
    TEST_CHECK(old_executable.expired(),
               "old executable should release after the last plan/lease owner");
    return true;
}

bool TestFrozenPlanRejectsIncompatibleLease() {
    const auto authority = std::make_shared<FakeValidationAuthority>();
    KernelSlot slot{KernelSlotKey(kSlot), PlanAbiFingerprint(kAbi), authority};
    TEST_CHECK(Publish(slot, authority, Artifact("v1")).published,
               "plan fixture artifact should publish");
    const ArtifactLease lease = slot.Acquire(DispatchKey::Exact(kDispatch),
                                             PlanAbiFingerprint(kAbi));
    TEST_CHECK(Throws([&] {
                   FrozenPlanVariant invalid(
                       PlanVariantKey("plan:bad-abi"),
                       {ExactPlanBinding(KernelSlotKey(kSlot),
                                         DispatchKey::Exact(kDispatch),
                                         PlanAbiFingerprint("abi:wrong"))},
                       {lease}, std::make_shared<const FakePlanExecutable>("plan"));
               }),
               "plan assembly must reject an ABI-incompatible lease");
    return true;
}

bool TestHealthRecordsRequireExactOneShotAuthority() {
    const auto authority = std::make_shared<FakeValidationAuthority>();
    const auto g1 = Artifact("g1");
    const auto g2 = Artifact("g2");
    KernelSlot slot{KernelSlotKey(kSlot), PlanAbiFingerprint(kAbi), authority};
    const auto first = Publish(slot, authority, g1);
    const auto second = Publish(slot, authority, g2, PublicationMode::kCanary,
                                CanaryPolicy{10000});
    TEST_CHECK(first.published && second.published, "health fixture must publish");

    const GenerationHealthRecord forged(
        "forged", KernelSlotKey(kSlot), g2->key(), g2->applicability(),
        g2->compatible_abi(), second.generation,
        GenerationHealthDisposition::kHealthy, "test",
        GenerationHealthToken("forged-health-token"));
    TEST_CHECK(!slot.PromoteCanary(
                    std::make_shared<const GenerationHealthRecord>(forged))
                    .changed &&
                   !slot.PromoteCanary(authority->Health(
                        *g1, first.generation, GenerationHealthDisposition::kHealthy))
                        .changed,
               "forged or mismatched health cannot promote a canary");
    const auto wrong_abi_health = authority->HealthWithAbi(
        *g2, second.generation, GenerationHealthDisposition::kHealthy,
        PlanAbiFingerprint("abi:wrong"));
    TEST_CHECK(!slot.PromoteCanary(wrong_abi_health).changed,
               "authority-issued health record ABI must match the slot ABI");

    const auto healthy = authority->Health(*g2, second.generation,
                                           GenerationHealthDisposition::kHealthy);
    TEST_CHECK(slot.PromoteCanary(healthy).changed &&
                   !slot.PromoteCanary(healthy).changed,
               "a healthy token promotes once and replay is rejected");

    const auto wrong_regression = authority->Health(
        *g1, first.generation, GenerationHealthDisposition::kQuarantined);
    TEST_CHECK(!slot.Rollback(first.generation, wrong_regression).changed,
               "a regression record must identify the current stable generation");
    const auto wrong_abi_regression = authority->HealthWithAbi(
        *g2, second.generation, GenerationHealthDisposition::kQuarantined,
        PlanAbiFingerprint("abi:wrong"));
    TEST_CHECK(!slot.Rollback(first.generation, wrong_abi_regression).changed,
               "authority-issued regression ABI must match the slot ABI");
    const auto regression = authority->Health(
        *g2, second.generation, GenerationHealthDisposition::kQuarantined);
    TEST_CHECK(slot.Rollback(first.generation, regression).changed,
               "a quarantined current stable record rolls back to g1");

    KernelSlot replay_slot{KernelSlotKey(kSlot), PlanAbiFingerprint(kAbi), authority};
    const auto replay_g1 = Publish(replay_slot, authority, g1);
    const auto replay_g2 = Publish(replay_slot, authority, g2);
    TEST_CHECK(replay_g1.generation == first.generation &&
                   replay_g2.generation == second.generation &&
                   !replay_slot.Rollback(replay_g1.generation, regression).changed,
               "a replayed regression token is rejected even when all bindings match");
    return true;
}

bool TestQuarantinePersistsAcrossEvictionAndLeases() {
    const auto authority = std::make_shared<FakeValidationAuthority>();
    KernelSlot slot{KernelSlotKey(kSlot), PlanAbiFingerprint(kAbi), authority,
                    SlotOptions(2)};
    const auto g1 = Artifact("g1");
    const auto g2 = Artifact("g2");
    const auto g1_publish = Publish(slot, authority, g1);
    const auto g2_publish = Publish(slot, authority, g2, PublicationMode::kCanary,
                                    CanaryPolicy{10000});
    TEST_CHECK(slot.PromoteCanary(authority->Health(
                   *g2, g2_publish.generation, GenerationHealthDisposition::kHealthy))
                   .changed,
               "g2 must be promoted before its runtime regression");
    const ArtifactLease g2_lease = slot.Acquire(DispatchKey::Exact(kDispatch),
                                                PlanAbiFingerprint(kAbi));
    TEST_CHECK(g2_lease.generation() == g2_publish.generation,
               "the old g2 lease must be captured before rollback");
    TEST_CHECK(slot.Rollback(
                   g1_publish.generation,
                   authority->Health(*g2, g2_publish.generation,
                                     GenerationHealthDisposition::kQuarantined,
                                     "runtime-regression"))
                   .changed &&
                   slot.Acquire(DispatchKey::Exact(kDispatch), PlanAbiFingerprint(kAbi))
                           .generation() == g1_publish.generation &&
                   g2_lease.valid() &&
                   g2_lease.artifact()->executable()->DebugName() == "g2",
               "rollback removes g2 from routing without invalidating its old lease");

    const auto g3 = Artifact("g3");
    const auto g3_publish = Publish(slot, authority, g3);
    TEST_CHECK(g3_publish.published &&
                   !slot.Rollback(g2_publish.generation,
                                  authority->Health(
                                      *g3, g3_publish.generation,
                                      GenerationHealthDisposition::kQuarantined))
                        .changed,
               "a quarantined g2 can never be a rollback target");
    TEST_CHECK(Publish(slot, authority, Artifact("g4")).published &&
                   slot.Snapshot().record_count == 2 &&
                   slot.Snapshot().quarantined_artifacts == 1 &&
                   !slot.Publish(g2, authority->ValidationFor(*g2)).published,
               "g2 quarantine survives eviction and rejects a fresh validation");
    return true;
}

bool TestDirectPublicationRetentionBudgets() {
    const auto authority = std::make_shared<FakeValidationAuthority>();
    KernelSlot capped{KernelSlotKey(kSlot), PlanAbiFingerprint(kAbi), authority,
                      SlotOptions(3, 100, 150)};
    const auto too_large = Artifact("too-large", kSlot, kDispatch, kAbi, 101);
    TEST_CHECK(!Publish(capped, authority, too_large).published &&
                   capped.Snapshot().record_count == 0,
               "direct publish enforces the single-artifact cap");

    const auto byte_g1 = Artifact("byte-g1", kSlot, kDispatch, kAbi, 80);
    TEST_CHECK(Publish(capped, authority, byte_g1).published,
               "byte fixture g1 must publish");
    const ArtifactLease old_lease = capped.Acquire(DispatchKey::Exact(kDispatch),
                                                   PlanAbiFingerprint(kAbi));
    FakePlanAssembler assembler;
    const auto frozen = assembler.Assemble(PlanVariantKey("plan:byte-g1"),
                                           {Binding()}, {old_lease});
    TEST_CHECK(Publish(capped, authority,
                       Artifact("byte-g2", kSlot, kDispatch, kAbi, 80))
                       .published &&
                   capped.Snapshot().record_count == 1 &&
                   capped.Snapshot().retained_artifact_bytes == 80 &&
                   old_lease.valid() && frozen->leases()[0].valid(),
               "byte eviction retains only discoverable records, not old leases/plans");

    KernelSlot count_capped{KernelSlotKey(kSlot), PlanAbiFingerprint(kAbi), authority,
                            SlotOptions(2, 100, 300)};
    const auto count_g1 = Artifact("count-g1", kSlot, kDispatch, kAbi, 50);
    TEST_CHECK(Publish(count_capped, authority, count_g1).published,
               "count fixture g1 must publish");
    const ArtifactLease count_lease = count_capped.Acquire(
        DispatchKey::Exact(kDispatch), PlanAbiFingerprint(kAbi));
    const auto count_plan = assembler.Assemble(PlanVariantKey("plan:count-g1"),
                                               {Binding()}, {count_lease});
    TEST_CHECK(Publish(count_capped, authority,
                       Artifact("count-g2", kSlot, kDispatch, kAbi, 50))
                       .published &&
                   Publish(count_capped, authority,
                       Artifact("count-g3", kSlot, kDispatch, kAbi, 50))
                       .published &&
                   count_capped.Snapshot().record_count == 2 &&
                   count_lease.valid() && count_plan->leases()[0].valid(),
               "count eviction also preserves an existing lease and frozen plan");

    KernelSlot all_heads{KernelSlotKey(kSlot), PlanAbiFingerprint(kAbi), authority,
                         SlotOptions(4, 100, 200, 4)};
    const auto h1 = Artifact("head1", kSlot, "exact:h1", kAbi, 100);
    const auto h2 = Artifact("head2", kSlot, "exact:h2", kAbi, 100);
    const auto h3 = Artifact("head3", kSlot, "exact:h3", kAbi, 1);
    TEST_CHECK(Publish(all_heads, authority, h1).published &&
                   Publish(all_heads, authority, h2).published,
               "aggregate budget fixture must fill with two heads");
    const auto before = all_heads.Snapshot();
    TEST_CHECK(!Publish(all_heads, authority, h3).published &&
                   all_heads.Snapshot().last_generation == before.last_generation &&
                   all_heads.Snapshot().record_count == before.record_count &&
                   all_heads.Snapshot().retained_artifact_bytes ==
                       before.retained_artifact_bytes &&
                   all_heads.Acquire(DispatchKey::Exact("exact:h1"),
                                    PlanAbiFingerprint(kAbi))
                           .valid() &&
                   all_heads.Acquire(DispatchKey::Exact("exact:h2"),
                                    PlanAbiFingerprint(kAbi))
                           .valid(),
               "all-head aggregate budget rejection is atomic");

    KernelSlot replacement{KernelSlotKey(kSlot), PlanAbiFingerprint(kAbi), authority,
                           SlotOptions(2)};
    TEST_CHECK(Publish(replacement, authority, Artifact("stable")).published &&
                   Publish(replacement, authority, Artifact("canary"),
                           PublicationMode::kCanary, CanaryPolicy{10000})
                       .published &&
                   Publish(replacement, authority, Artifact("replacement")).published &&
                   replacement.Snapshot().record_count == 2,
               "stable replacement at max two generations works with a canary present");
    return true;
}

bool TestConcurrentPublishAcquireStress() {
    const auto authority = std::make_shared<FakeValidationAuthority>();
    KernelSlot slot{KernelSlotKey(kSlot), PlanAbiFingerprint(kAbi), authority};
    TEST_CHECK(Publish(slot, authority, Artifact("v0")).published,
               "initial stress artifact should publish");

    constexpr int kReaders = 8;
    constexpr int kPublishes = 100;
    std::atomic<bool> start{false};
    std::atomic<bool> stop{false};
    std::atomic<int> failures{0};
    std::vector<std::thread> readers;
    for (int i = 0; i < kReaders; ++i) {
        readers.emplace_back([&] {
            while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
            while (!stop.load(std::memory_order_acquire)) {
                const ArtifactLease lease = slot.Acquire(
                    DispatchKey::Exact(kDispatch), PlanAbiFingerprint(kAbi));
                if (!lease.valid() || lease.generation() == 0 ||
                    !lease.artifact()->executable()->IsReady() ||
                    lease.artifact()->executable()->DebugName().empty()) {
                    failures.fetch_add(1, std::memory_order_relaxed);
                    break;
                }
            }
        });
    }
    start.store(true, std::memory_order_release);
    for (int generation = 1; generation <= kPublishes; ++generation) {
        const auto artifact = Artifact("v" + std::to_string(generation));
        if (!slot.Publish(artifact, authority->ValidationFor(*artifact)).published) {
            failures.fetch_add(1, std::memory_order_relaxed);
            break;
        }
    }
    stop.store(true, std::memory_order_release);
    for (auto& reader : readers) reader.join();

    const auto snapshot = slot.Snapshot();
    TEST_CHECK(failures.load(std::memory_order_relaxed) == 0 &&
                   snapshot.last_generation == kPublishes + 1 &&
                   snapshot.record_count <= 64,
               "concurrent publish/acquire must use valid unique tokens safely");
    return true;
}

bool TestFakeCompilerToPlanAssemblerFlow() {
    const auto authority = std::make_shared<FakeValidationAuthority>();
    auto compiler = std::make_shared<FakeCompiler>();
    CompileCoordinator coordinator(compiler, authority);
    KernelSlot slot{KernelSlotKey(kSlot), PlanAbiFingerprint(kAbi), authority};
    FakePlanAssembler assembler;

    const auto first_compile = coordinator.Request(Request("v1")).Get();
    TEST_CHECK(first_compile.ready() && first_compile.validation(),
               "fake v1 compile should return authority validation");
    const auto first_publish = slot.Publish(first_compile.artifact(),
                                            first_compile.validation());
    const ArtifactLease first_lease = slot.Acquire(DispatchKey::Exact(kDispatch),
                                                   PlanAbiFingerprint(kAbi));
    const auto first_plan = assembler.Assemble(PlanVariantKey("plan:model@1:v1"),
                                               {Binding()}, {first_lease});

    const auto second_compile = coordinator.Request(Request("v2")).Get();
    TEST_CHECK(second_compile.ready() && second_compile.validation(),
               "fake v2 compile should return authority validation");
    const auto second_publish = slot.Publish(second_compile.artifact(),
                                             second_compile.validation());
    const ArtifactLease second_lease = slot.Acquire(DispatchKey::Exact(kDispatch),
                                                    PlanAbiFingerprint(kAbi));

    TEST_CHECK(first_publish.published && second_publish.published &&
                   first_plan->leases()[0].generation() == 1 &&
                   first_plan->leases()[0].artifact()->executable()->DebugName() == "v1" &&
                   second_lease.generation() == 2 &&
                   second_lease.artifact()->executable()->DebugName() == "v2" &&
                   compiler->calls.load(std::memory_order_relaxed) == 2 &&
                   assembler.calls == 1,
               "compile results publish through the same injected authority");
    return true;
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"publication_requires_bound_one_shot_validation",
         TestPublicationRequiresBoundOneShotValidation},
        {"generation_and_exact_publication_gate", TestGenerationAndExactPublicationGate},
        {"lease_and_frozen_plan_retain_old_generation",
         TestLeaseAndFrozenPlanRetainOldGeneration},
        {"frozen_plan_rejects_incompatible_lease", TestFrozenPlanRejectsIncompatibleLease},
        {"health_records_require_exact_one_shot_authority",
         TestHealthRecordsRequireExactOneShotAuthority},
        {"quarantine_persists_across_eviction_and_leases",
         TestQuarantinePersistsAcrossEvictionAndLeases},
        {"direct_publication_retention_budgets", TestDirectPublicationRetentionBudgets},
        {"concurrent_publish_acquire_stress", TestConcurrentPublishAcquireStress},
        {"fake_compiler_to_plan_assembler_flow", TestFakeCompilerToPlanAssemblerFlow},
    };

    int failures = 0;
    for (const auto& test : tests) {
        try {
            if (!test.second()) {
                ++failures;
                continue;
            }
            std::cout << "[PASS] " << test.first << '\n';
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << test.first << ": " << error.what() << '\n';
            ++failures;
        }
    }
    return failures == 0 ? 0 : 1;
}
