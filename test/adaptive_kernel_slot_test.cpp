/*! \file test/adaptive_kernel_slot_test.cpp
 * \brief Tests generation publication, RCU leases, canary, and fake plans.
 */

#include <atomic>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "kxc/compiler/adaptive.h"

namespace {

#define TEST_CHECK(condition, message)                                           \
    do {                                                                          \
        if (!(condition)) {                                                        \
            std::cerr << "[FAIL] " << __FUNCTION__ << ": " << (message) << '\n'; \
            return false;                                                         \
        }                                                                         \
    } while (0)

using kxc::api::adaptive::AdaptiveEvent;
using kxc::api::adaptive::AdaptiveEventKind;
using kxc::api::adaptive::ArtifactCompiler;
using kxc::api::adaptive::ArtifactExecutable;
using kxc::api::adaptive::ArtifactLease;
using kxc::api::adaptive::CanaryPolicy;
using kxc::api::adaptive::CancellationToken;
using kxc::api::adaptive::CompileAttempt;
using kxc::api::adaptive::CompileCoordinator;
using kxc::api::adaptive::CompileRequest;
using kxc::api::adaptive::DispatchKey;
using kxc::api::adaptive::ExactPlanAssembler;
using kxc::api::adaptive::ExactPlanBinding;
using kxc::api::adaptive::FrozenPlanVariant;
using kxc::api::adaptive::KernelArtifact;
using kxc::api::adaptive::KernelArtifactKey;
using kxc::api::adaptive::KernelSlot;
using kxc::api::adaptive::KernelSlotKey;
using kxc::api::adaptive::KernelSlotOptions;
using kxc::api::adaptive::PlanAbiFingerprint;
using kxc::api::adaptive::PlanExecutable;
using kxc::api::adaptive::PlanVariantKey;
using kxc::api::adaptive::PublicationMode;
using kxc::api::adaptive::RequestKind;
using kxc::api::adaptive::RoutingContext;

bool Throws(const std::function<void()>& function) {
    try {
        function();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

constexpr const char* kSlot = "unit:add|target:fake-cpu";
constexpr const char* kDispatch = "exact:f32[4]|contiguous";
constexpr const char* kAbi =
    "abi-v1|input:f32[4]:align4|output:f32[4]:align64|workspace:0";

class FakeExecutable final : public ArtifactExecutable {
public:
    explicit FakeExecutable(std::string version)
        : version_(std::move(version)) {}
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
    std::string dispatch = kDispatch, std::string abi = kAbi) {
    return std::make_shared<const KernelArtifact>(
        KernelArtifactKey(KernelSlotKey(std::move(slot)),
                          "artifact:" + version),
        DispatchKey::Exact(std::move(dispatch)),
        PlanAbiFingerprint(std::move(abi)),
        std::make_shared<const FakeExecutable>(version), 4096,
        "fake-compiler");
}

CompileRequest Request(std::string version) {
    return CompileRequest(
        KernelArtifactKey(KernelSlotKey(kSlot), "artifact:" + version),
        DispatchKey::Exact(kDispatch), PlanAbiFingerprint(kAbi), "model@1",
        RequestKind::kDemand);
}

class FakeCompiler final : public ArtifactCompiler {
public:
    CompileAttempt Compile(const CompileRequest& request,
                           const CancellationToken& cancellation) override {
        calls.fetch_add(1, std::memory_order_relaxed);
        if (cancellation.IsCancellationRequested()) {
            return CompileAttempt::Failed(
                kxc::api::adaptive::CompileFailureCategory::kCancelled,
                "cancelled");
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
    return ExactPlanBinding(KernelSlotKey(kSlot),
                            DispatchKey::Exact(kDispatch),
                            PlanAbiFingerprint(kAbi));
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

bool TestGenerationAndExactPublicationGate() {
    KernelSlot slot{KernelSlotKey(kSlot), PlanAbiFingerprint(kAbi)};
    const auto first = slot.Publish(Artifact("v1"));
    const auto second = slot.Publish(Artifact("v2"));
    TEST_CHECK(first.published && second.published && first.generation == 1 &&
                   second.generation == 2 &&
                   second.predecessor_generation == 1,
               "stable generations must increase monotonically");

    const auto current = slot.Acquire(DispatchKey::Exact(kDispatch),
                                      PlanAbiFingerprint(kAbi));
    TEST_CHECK(current.valid() && current.generation() == 2 &&
                   current.artifact()->executable()->DebugName() == "v2",
               "future acquire should see the newest stable generation");

    const auto wrong_slot = slot.Publish(Artifact("bad-slot", "other-slot"));
    const auto wrong_abi = slot.Publish(
        Artifact("bad-abi", kSlot, kDispatch, "abi-v2"));
    TEST_CHECK(!wrong_slot.published && !wrong_abi.published &&
                   slot.Acquire(DispatchKey::Exact(kDispatch),
                                PlanAbiFingerprint(kAbi))
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
    ArtifactLease old_lease;
    std::shared_ptr<const FrozenPlanVariant> frozen;
    std::weak_ptr<const ArtifactExecutable> old_executable;
    {
        KernelSlot slot{KernelSlotKey(kSlot), PlanAbiFingerprint(kAbi)};
        auto old_artifact = Artifact("v1");
        old_executable = old_artifact->executable();
        TEST_CHECK(slot.Publish(old_artifact).published,
                   "first artifact should publish");
        old_lease = slot.Acquire(DispatchKey::Exact(kDispatch),
                                 PlanAbiFingerprint(kAbi));
        FakePlanAssembler assembler;
        frozen = assembler.Assemble(PlanVariantKey("plan:model@1:v1"),
                                    {Binding()}, {old_lease});
        TEST_CHECK(slot.Publish(Artifact("v2")).published &&
                       slot.Acquire(DispatchKey::Exact(kDispatch),
                                    PlanAbiFingerprint(kAbi))
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
    KernelSlot slot{KernelSlotKey(kSlot), PlanAbiFingerprint(kAbi)};
    TEST_CHECK(slot.Publish(Artifact("v1")).published,
               "plan fixture artifact should publish");
    const ArtifactLease lease = slot.Acquire(
        DispatchKey::Exact(kDispatch), PlanAbiFingerprint(kAbi));
    TEST_CHECK(Throws([&] {
                   FrozenPlanVariant invalid(
                       PlanVariantKey("plan:bad-abi"),
                       {ExactPlanBinding(KernelSlotKey(kSlot),
                                         DispatchKey::Exact(kDispatch),
                                         PlanAbiFingerprint("abi:wrong"))},
                       {lease},
                       std::make_shared<const FakePlanExecutable>("plan"));
               }),
               "plan assembly must reject an ABI-incompatible lease");
    TEST_CHECK(Throws([&] {
                   FrozenPlanVariant invalid(
                       PlanVariantKey("plan:bad-dispatch"),
                       {ExactPlanBinding(KernelSlotKey(kSlot),
                                         DispatchKey::Exact("exact:f32[8]"),
                                         PlanAbiFingerprint(kAbi))},
                       {lease},
                       std::make_shared<const FakePlanExecutable>("plan"));
               }),
               "plan assembly must reject a dispatch-incompatible lease");
    return true;
}

bool TestCanaryWithdrawPromoteAndRollback() {
    EventLog events;
    KernelSlot slot(
        KernelSlotKey(kSlot), PlanAbiFingerprint(kAbi),
        KernelSlotOptions{64, 64, [&](const AdaptiveEvent& event) {
                              events.Record(event);
                          }});
    const auto stable = slot.Publish(Artifact("stable"));
    TEST_CHECK(stable.published, "stable predecessor should publish");

    const auto canary = slot.Publish(Artifact("canary"),
                                     PublicationMode::kCanary,
                                     CanaryPolicy{10000});
    TEST_CHECK(canary.published && canary.predecessor_generation ==
                                         stable.generation,
               "canary should retain its stable predecessor");
    const ArtifactLease normal = slot.Acquire(
        DispatchKey::Exact(kDispatch), PlanAbiFingerprint(kAbi));
    const ArtifactLease routed = slot.Acquire(
        DispatchKey::Exact(kDispatch), PlanAbiFingerprint(kAbi),
        RoutingContext{true, "request-42"});
    TEST_CHECK(normal.generation() == stable.generation &&
                   routed.generation() == canary.generation,
               "canary routing must be explicit and request-stable");

    const auto withdrawn = slot.WithdrawCanary(
        DispatchKey::Exact(kDispatch), canary.generation,
        "numeric oracle mismatch");
    TEST_CHECK(withdrawn.changed &&
                   slot.Acquire(DispatchKey::Exact(kDispatch),
                                PlanAbiFingerprint(kAbi),
                                RoutingContext{true, "request-42"})
                           .generation() == stable.generation &&
                   routed.generation() == canary.generation,
               "withdrawal affects future acquire, not an in-flight lease");
    TEST_CHECK(!slot.Rollback(DispatchKey::Exact(kDispatch),
                              canary.generation,
                              "must not restore quarantined canary")
                    .changed,
               "withdrawn canary must never become a rollback target");

    const auto next_canary = slot.Publish(
        Artifact("canary-good"), PublicationMode::kCanary,
        CanaryPolicy{10000});
    TEST_CHECK(!slot.PromoteCanary(DispatchKey::Exact(kDispatch),
                                  next_canary.generation, "")
                    .changed,
               "canary cannot promote without explicit health evidence");
    const auto promoted = slot.PromoteCanary(
        DispatchKey::Exact(kDispatch), next_canary.generation,
        "shadow numeric and launch health passed");
    TEST_CHECK(promoted.changed &&
                   slot.Acquire(DispatchKey::Exact(kDispatch),
                                PlanAbiFingerprint(kAbi))
                           .generation() == next_canary.generation,
               "healthy canary should become the future stable head");

    const auto rolled_back = slot.Rollback(
        DispatchKey::Exact(kDispatch), stable.generation,
        "runtime health regression");
    TEST_CHECK(rolled_back.changed &&
                   rolled_back.predecessor_generation ==
                       next_canary.generation &&
                   slot.Acquire(DispatchKey::Exact(kDispatch),
                                PlanAbiFingerprint(kAbi))
                           .generation() == stable.generation,
               "rollback should restore a prior validated generation");
    TEST_CHECK(events.Count(AdaptiveEventKind::kPublished) == 3 &&
                   events.Count(AdaptiveEventKind::kWithdrawn) == 1 &&
                   events.Count(AdaptiveEventKind::kPromoted) == 1 &&
                   events.Count(AdaptiveEventKind::kRolledBack) == 1,
               "publication lifecycle should be observable");
    return true;
}

bool TestCanaryRequiresHealthyPredecessor() {
    KernelSlot slot{KernelSlotKey(kSlot), PlanAbiFingerprint(kAbi)};
    const auto canary = slot.Publish(Artifact("orphan-canary"),
                                     PublicationMode::kCanary,
                                     CanaryPolicy{500});
    TEST_CHECK(!canary.published && slot.Snapshot().record_count == 0 &&
                   !slot.Acquire(DispatchKey::Exact(kDispatch),
                                 PlanAbiFingerprint(kAbi))
                    .valid(),
               "orphan canary must leave the slot unavailable");
    return true;
}

bool TestConcurrentPublishAcquireStress() {
    KernelSlot slot{KernelSlotKey(kSlot), PlanAbiFingerprint(kAbi)};
    TEST_CHECK(slot.Publish(Artifact("v0")).published,
               "initial stress artifact should publish");

    constexpr int kReaders = 16;
    constexpr int kPublishes = 500;
    std::atomic<bool> start{false};
    std::atomic<bool> stop{false};
    std::atomic<int> failures{0};
    std::vector<std::thread> readers;
    for (int i = 0; i < kReaders; ++i) {
        readers.emplace_back([&] {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
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
        if (!slot.Publish(Artifact("v" + std::to_string(generation)))
                 .published) {
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
               "concurrent publish/acquire must preserve bounded immutable generations");
    return true;
}

bool TestFakeCompilerToPlanAssemblerFlow() {
    auto compiler = std::make_shared<FakeCompiler>();
    CompileCoordinator coordinator(compiler);
    KernelSlot slot{KernelSlotKey(kSlot), PlanAbiFingerprint(kAbi)};
    FakePlanAssembler assembler;

    const auto first_compile = coordinator.Request(Request("v1")).Get();
    TEST_CHECK(first_compile.ready(), "fake v1 compile should be ready");
    const auto first_publish = slot.Publish(first_compile.artifact());
    const ArtifactLease first_lease = slot.Acquire(
        DispatchKey::Exact(kDispatch), PlanAbiFingerprint(kAbi));
    const auto first_plan = assembler.Assemble(
        PlanVariantKey("plan:model@1:v1"), {Binding()}, {first_lease});

    const auto second_compile = coordinator.Request(Request("v2")).Get();
    TEST_CHECK(second_compile.ready(), "fake v2 compile should be ready");
    const auto second_publish = slot.Publish(second_compile.artifact());
    const ArtifactLease second_lease = slot.Acquire(
        DispatchKey::Exact(kDispatch), PlanAbiFingerprint(kAbi));

    TEST_CHECK(first_publish.published && second_publish.published &&
                   first_plan->leases()[0].generation() == 1 &&
                   first_plan->leases()[0]
                           .artifact()
                           ->executable()
                           ->DebugName() == "v1" &&
                   second_lease.generation() == 2 &&
                   second_lease.artifact()->executable()->DebugName() == "v2" &&
                   compiler->calls.load(std::memory_order_relaxed) == 2 &&
                   assembler.calls == 1,
               "fake compile/publish/assemble flow must freeze selected generation");
    return true;
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"generation_and_exact_publication_gate",
         TestGenerationAndExactPublicationGate},
        {"lease_and_frozen_plan_retain_old_generation",
         TestLeaseAndFrozenPlanRetainOldGeneration},
        {"frozen_plan_rejects_incompatible_lease",
         TestFrozenPlanRejectsIncompatibleLease},
        {"canary_withdraw_promote_and_rollback",
         TestCanaryWithdrawPromoteAndRollback},
        {"canary_requires_healthy_predecessor",
         TestCanaryRequiresHealthyPredecessor},
        {"concurrent_publish_acquire_stress",
         TestConcurrentPublishAcquireStress},
        {"fake_compiler_to_plan_assembler_flow",
         TestFakeCompilerToPlanAssemblerFlow},
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
            std::cerr << "[FAIL] " << test.first << ": " << error.what()
                      << '\n';
            ++failures;
        }
    }
    return failures == 0 ? 0 : 1;
}
