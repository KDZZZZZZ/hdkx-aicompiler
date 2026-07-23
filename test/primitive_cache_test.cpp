/*! \file test/primitive_cache_test.cpp
 * \brief Verifies pinned artifacts, singleflight, failures and backpressure.
 */

#include <atomic>
#include <chrono>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "../src/compiler/internal/primitive_cache.h"
#include "kxc/compiler/foundation_contract.h"

namespace {

#define TEST_CHECK(condition, message)                                           \
    do {                                                                          \
        if (!(condition)) {                                                        \
            std::cerr << "[FAIL] " << __FUNCTION__ << ": " << (message) << "\n"; \
            return false;                                                         \
        }                                                                         \
    } while (0)

bool ThrowsWith(const std::function<void()>& fn, const std::string& expected) {
    try {
        fn();
    } catch (const std::exception& error) {
        return std::string(error.what()).find(expected) != std::string::npos;
    }
    return false;
}

class NoopLauncher final : public kxc::codegen::KernelLauncher {
public:
    bool IsReady() const noexcept override { return true; }

    kxc::AsyncOperation Launch(
        const kxc::Array<kxc::runtime::NDArray>&,
        const kxc::DeviceStream& stream,
        const kxc::ObjectRef&) const override {
        return kxc::AsyncOperation::Completed(stream);
    }
};

kxc::api::ArtifactKey MakeKey(const std::string& unit,
                              const std::string& digest = {}) {
    return kxc::api::ArtifactKey(
        kxc::api::UnitSemanticKey("unit=" + unit, digest),
        "target=cpu-test-v1", "pipeline=test-v1", 1,
        "schedule=test-v1", "backend=fake-v1", digest);
}

kxc::api::CompileRequest MakeRequest(
    const kxc::api::ArtifactKey& key, const std::string& cancellation_id) {
    kxc::api::CompileRequest request;
    request.artifact_key = key;
    request.request_origin = "primitive-cache-test";
    request.cancellation.id = cancellation_id;
    return request;
}

kxc::api::internal::CachedPrimitive MakeArtifact(
    const std::string& symbol, uint64_t bytes = 1,
    const std::shared_ptr<NoopLauncher>& launcher =
        std::make_shared<NoopLauncher>()) {
    using namespace kxc;
    using namespace kxc::codegen;
    KernelArgSpec output("output", KernelArgRole::kOutput,
                         runtime::DataTypeFromString("float32"), {1},
                         Device::CPU(), 1, true);
    KernelSignature signature(String(symbol), {output});
    KernelLaunchMetadata metadata(Device::CPU(), CodeGenBackend::kLLVM);
    CompiledKernel kernel(signature, metadata, launcher);
    return api::internal::CachedPrimitive{
        signature, metadata, kernel, bytes, "primitive-cache-test",
        "fake-launcher-validated"};
}

void Reset(kxc::api::internal::PrimitiveCacheLimits limits = {}) {
    kxc::api::internal::ClearPrimitiveCacheForTesting();
    kxc::api::internal::SetPrimitiveCacheLimitsForTesting(limits);
}

bool TestPinSurvivesEviction() {
    using namespace kxc::api::internal;
    Reset(PrimitiveCacheLimits{1, 64, 8, 8});
    const auto first_key = MakeKey("first");
    const PrimitiveCacheLease first = AcquirePrimitiveCache(first_key);
    TEST_CHECK(first.access() == PrimitiveCacheAccess::kOwner,
               "first request must own compilation");
    const PrimitiveArtifactPin first_pin =
        PublishPrimitiveCacheLease(first, MakeArtifact("first"));

    const PrimitiveCacheLease second = AcquirePrimitiveCache(MakeKey("second"));
    const PrimitiveArtifactPin second_pin =
        PublishPrimitiveCacheLease(second, MakeArtifact("second"));
    const PrimitiveCacheStats stats = GetPrimitiveCacheStats();
    TEST_CHECK(first_pin.defined() && first_pin.artifact().kernel.IsReady() &&
                   second_pin.defined() && stats.entries == 1 &&
                   stats.evictions == 1,
               "eviction must remove discoverability, not an issued pin");

    const PrimitiveCacheLease first_again = AcquirePrimitiveCache(first_key);
    TEST_CHECK(first_again.access() == PrimitiveCacheAccess::kOwner,
               "evicted key should miss while the old pin remains usable");
    FailPrimitiveCacheLease(first_again, PrimitiveFailureCategory::kCancelled,
                            "test cleanup", std::chrono::milliseconds(0));
    return true;
}

bool TestProductionAdapterPinSurvivesEviction() {
    using namespace kxc::api;
    using namespace kxc::api::internal;
    Reset(PrimitiveCacheLimits{1, 64, 8, 8});
    const ProductionArtifactCacheAdapter adapter;
    const ArtifactKey first_key = MakeKey("adapter-first");
    const ProductionCompileTransaction first =
        adapter.Acquire(MakeRequest(first_key, "adapter-first"));
    TEST_CHECK(first.owns_compile(),
               "public transaction must own the first production miss");
    const CompileOutcome first_outcome = adapter.Publish(
        first, ProductionArtifactAccess::Make(MakeArtifact("adapter_first")));
    TEST_CHECK(first_outcome.state == CompileRequestState::kReady &&
                   first_outcome.pin.defined(),
               "public owner must publish a production-backed pin");

    const ArtifactCacheStats before_lookup = adapter.stats();
    const ArtifactLookup lookup = adapter.Lookup(first_key);
    const ArtifactCacheStats after_hit = adapter.stats();
    TEST_CHECK(lookup.kind == ArtifactLookupKind::kHit && lookup.pin.defined() &&
                   lookup.pin.handle().record().artifact_key == first_key &&
                   !lookup.pin.handle().record().executable_token.empty() &&
                   !lookup.pin.handle().record().signature_digest.empty() &&
                   !lookup.pin.handle().record().launch_metadata_digest.empty() &&
                   !lookup.pin.handle().record().provenance.empty() &&
                   lookup.pin.handle().record().byte_size > 0 &&
                   !lookup.pin.handle().record().validation_record.empty() &&
                   after_hit.hits == before_lookup.hits &&
                   after_hit.misses == before_lookup.misses &&
                   after_hit.in_flight == 0,
               "adapter lookup must expose a complete pin without cache mutation");

    const ArtifactKey second_key = MakeKey("adapter-second");
    const ProductionCompileTransaction second =
        adapter.Acquire(MakeRequest(second_key, "adapter-second"));
    (void)adapter.Publish(
        second, ProductionArtifactAccess::Make(MakeArtifact("adapter_second")));
    const ArtifactCacheStats before_miss = adapter.stats();
    const ArtifactLookup evicted = adapter.Lookup(first_key);
    const ArtifactLookup missing = adapter.Lookup(MakeKey("adapter-missing"));
    const ArtifactCacheStats after_miss = adapter.stats();
    TEST_CHECK(evicted.kind == ArtifactLookupKind::kMiss &&
                   missing.kind == ArtifactLookupKind::kMiss &&
                   first_outcome.pin.handle().record().executable_token ==
                       "primitive-v1:" + first_key.canonical_bytes() &&
                   ProductionArtifactAccess::Pin(first_outcome.pin)
                       .artifact()
                       .kernel.IsReady() &&
                   after_miss.misses == before_miss.misses &&
                   after_miss.in_flight == before_miss.in_flight,
               "eviction removes discovery but cannot invalidate a public production pin");
    return true;
}

bool TestPublicProductionSameKeyMergeAndCancellation() {
    using namespace kxc::api;
    using namespace kxc::api::internal;
    Reset(PrimitiveCacheLimits{8, 1024, 8, 8});
    const ProductionArtifactCacheAdapter adapter;
    const ProductionCompileOwnership ownership = adapter.ownership();
    TEST_CHECK(ownership.exact_singleflight ==
                       CompilePolicyLayer::kPrimitiveCacheTransaction &&
                   ownership.failure_ttl ==
                       CompilePolicyLayer::kPrimitiveCacheTransaction &&
                   ownership.global_backpressure ==
                       CompilePolicyLayer::kPrimitiveCacheTransaction &&
                   ownership.asynchronous_cancellation ==
                       CompilePolicyLayer::kAdaptiveCoordinator &&
                   ownership.priority_and_budget_scheduling ==
                       CompilePolicyLayer::kAdaptiveCoordinator &&
                   ownership.dispatch_and_generation ==
                       CompilePolicyLayer::kAdaptiveCoordinator,
               "Core and adaptive policy ownership must be explicit");

    const ArtifactKey key = MakeKey("public-merge");
    const CompileRequest owner_request = MakeRequest(key, "owner");
    const CompileRequest waiter_request = MakeRequest(key, "waiter");
    const ProductionCompileTransaction owner = adapter.Acquire(owner_request);
    const ProductionCompileTransaction waiter = adapter.Acquire(waiter_request);
    TEST_CHECK(owner.owns_compile() && !waiter.owns_compile() &&
                   owner.ticket().ticket_id == waiter.ticket().ticket_id &&
                   owner.ticket().merged_waiter_count == 1,
               "same public full key must reference one production flight");

    CancellationToken cancelled = waiter_request.cancellation;
    cancelled.cancelled = true;
    const CompileOutcome waiter_cancelled = adapter.Wait(waiter, cancelled);
    TEST_CHECK(waiter_cancelled.state == CompileRequestState::kCancelled &&
                   owner.ticket().state == CompileRequestState::kCompiling,
               "snapshot waiter cancellation must not cancel shared owner work");
    const CompileOutcome ready = adapter.Publish(
        owner, ProductionArtifactAccess::Make(MakeArtifact("public_merge")));
    TEST_CHECK(ready.state == CompileRequestState::kReady &&
                   ProductionArtifactAccess::Pin(ready.pin)
                       .artifact()
                       .kernel.IsReady(),
               "owner must still publish after one waiter cancels");

    const ProductionCompileTransaction hit =
        adapter.Acquire(MakeRequest(key, "after-ready"));
    TEST_CHECK(adapter.Wait(hit).state == CompileRequestState::kReady &&
                   adapter.stats().misses == 1 &&
                   adapter.stats().merged_waiters == 1,
               "ready acquisition must reuse the one public production compile");
    return true;
}

bool TestPublicProductionFailureRetryAndBackpressure() {
    using namespace kxc::api;
    using namespace kxc::api::internal;
    Reset(PrimitiveCacheLimits{8, 1024, 1, 8});
    const ProductionArtifactCacheAdapter adapter;
    const ArtifactKey failure_key = MakeKey("public-failure");
    const ProductionCompileTransaction owner =
        adapter.Acquire(MakeRequest(failure_key, "failure-owner"));
    const ProductionCompileTransaction waiter =
        adapter.Acquire(MakeRequest(failure_key, "failure-waiter"));
    const CompileOutcome failed = adapter.Fail(
        owner, CompileFailure{CompileFailureCategory::kValidation, 60000,
                              "public ABI validation failed"});
    const CompileOutcome waiter_failed = adapter.Wait(waiter);
    const ProductionCompileTransaction negative =
        adapter.Acquire(MakeRequest(failure_key, "failure-negative"));
    const CompileOutcome negative_outcome = adapter.Wait(negative);
    TEST_CHECK(failed.state == CompileRequestState::kFailed &&
                   waiter_failed.failure &&
                   waiter_failed.failure->category ==
                       CompileFailureCategory::kValidation &&
                   waiter_failed.failure->diagnostic ==
                       "public ABI validation failed" &&
                   negative_outcome.failure &&
                   negative_outcome.failure->retry_after_millis > 0,
               "public failure TTL must reach waiters and suppress immediate retry");

    ForgetPrimitiveFailureForTesting(failure_key);
    const ProductionCompileTransaction retry =
        adapter.Acquire(MakeRequest(failure_key, "failure-retry"));
    TEST_CHECK(retry.owns_compile(),
               "public request must retry after the production failure expires");

    const ProductionCompileTransaction saturated = adapter.Acquire(
        MakeRequest(MakeKey("public-saturated"), "saturated"));
    const CompileOutcome rejected = adapter.Wait(saturated);
    TEST_CHECK(rejected.state == CompileRequestState::kRejected &&
                   rejected.failure &&
                   rejected.failure->category ==
                       CompileFailureCategory::kBackpressure,
               "production in-flight bound must map to public backpressure");

    CompileRequest adaptive =
        MakeRequest(MakeKey("public-adaptive"), "adaptive");
    adaptive.priority = CompilePriority::kUrgent;
    const CompileOutcome unsupported = adapter.Wait(adapter.Acquire(adaptive));
    CompileRequest budget =
        MakeRequest(MakeKey("public-budget"), "budget");
    budget.budget_class = CompileBudgetClass::kModel;
    const CompileOutcome unsupported_budget =
        adapter.Wait(adapter.Acquire(budget));
    CompileRequest dispatch =
        MakeRequest(MakeKey("public-dispatch"), "dispatch");
    dispatch.dispatch_key = DispatchKey(
        "family", "shape=[1];layout=contiguous;valid=[1]", "exact-v1");
    const CompileOutcome unsupported_dispatch =
        adapter.Wait(adapter.Acquire(dispatch));
    TEST_CHECK(unsupported.state == CompileRequestState::kRejected &&
                   unsupported.failure->category ==
                       CompileFailureCategory::kUnsupported &&
                   unsupported_budget.failure->category ==
                       CompileFailureCategory::kUnsupported &&
                   unsupported_dispatch.failure->category ==
                       CompileFailureCategory::kUnsupported,
               "Track03 dispatch/priority/budget policy must fail closed in Core");
    (void)adapter.Fail(
        retry, CompileFailure{CompileFailureCategory::kCancelled, 0,
                              "test cleanup"});

    const ArtifactKey cancel_key = MakeKey("public-owner-cancel");
    const ProductionCompileTransaction cancel_owner =
        adapter.Acquire(MakeRequest(cancel_key, "cancel-owner"));
    const ProductionCompileTransaction cancel_waiter =
        adapter.Acquire(MakeRequest(cancel_key, "cancel-waiter"));
    (void)adapter.Fail(
        cancel_owner,
        CompileFailure{CompileFailureCategory::kCancelled, 0,
                       "shared owner cancelled"});
    const CompileOutcome shared_cancelled = adapter.Wait(cancel_waiter);
    TEST_CHECK(shared_cancelled.state == CompileRequestState::kCancelled &&
                   shared_cancelled.failure->category ==
                       CompileFailureCategory::kCancelled,
               "shared owner cancellation must retain its public terminal state");
    return true;
}

bool TestPublicProductionConcurrentSingleflight() {
    using namespace kxc::api;
    using namespace kxc::api::internal;
    Reset(PrimitiveCacheLimits{16, 1024, 16, 16});
    constexpr int kThreads = 8;
    const ProductionArtifactCacheAdapter adapter;
    const ArtifactKey key = MakeKey("public-concurrent");
    std::atomic<int> acquired{0};
    std::atomic<int> owners{0};
    std::atomic<int> failures{0};
    std::vector<CompileOutcome> outcomes(kThreads);
    std::vector<std::string> ticket_ids(kThreads);
    std::vector<std::thread> threads;
    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([&, i] {
            bool counted = false;
            try {
                const ProductionCompileTransaction transaction = adapter.Acquire(
                    MakeRequest(key, "public-thread-" + std::to_string(i)));
                ticket_ids[i] = transaction.ticket().ticket_id;
                if (transaction.owns_compile()) ++owners;
                ++acquired;
                counted = true;
                if (transaction.owns_compile()) {
                    while (acquired.load() != kThreads) std::this_thread::yield();
                    outcomes[i] = adapter.Publish(
                        transaction, ProductionArtifactAccess::Make(
                                         MakeArtifact("public_concurrent")));
                } else {
                    outcomes[i] = adapter.Wait(transaction);
                }
            } catch (...) {
                ++failures;
                if (!counted) ++acquired;
            }
        });
    }
    for (auto& thread : threads) thread.join();
    for (int i = 0; i < kThreads; ++i) {
        TEST_CHECK(outcomes[i].state == CompileRequestState::kReady &&
                       outcomes[i].pin.defined() &&
                       ticket_ids[i] == ticket_ids[0],
                   "all public concurrent callers must share one ready ticket");
    }
    TEST_CHECK(owners == 1 && failures == 0 &&
                   adapter.stats().misses == 1 &&
                   adapter.stats().merged_waiters == kThreads - 1,
               "public adapter must linearize one owner and all merged waiters");
    return true;
}

bool TestPublicTransactionTerminalRaceAndAbandonment() {
    using namespace kxc::api;
    using namespace kxc::api::internal;
    Reset(PrimitiveCacheLimits{8, 1024, 1, 8});
    const ProductionArtifactCacheAdapter adapter;
    const ArtifactKey race_key = MakeKey("public-terminal-race");
    const ProductionCompileTransaction owner =
        adapter.Acquire(MakeRequest(race_key, "race-owner"));
    const ProductionCompileTransaction owner_copy = owner;
    const ProductionCompileTransaction waiter =
        adapter.Acquire(MakeRequest(race_key, "race-waiter"));
    CompileOutcome publish_outcome;
    CompileOutcome fail_outcome;
    std::atomic<int> terminal_exceptions{0};
    std::thread publisher([&] {
        try {
            publish_outcome = adapter.Publish(
                owner, ProductionArtifactAccess::Make(MakeArtifact("race")));
        } catch (...) {
            ++terminal_exceptions;
        }
    });
    std::thread failure([&] {
        try {
            fail_outcome = adapter.Fail(
                owner_copy,
                CompileFailure{CompileFailureCategory::kCompile, 1000,
                               "racing terminal failure"});
        } catch (...) {
            ++terminal_exceptions;
        }
    });
    publisher.join();
    failure.join();
    const CompileOutcome waiter_outcome = adapter.Wait(waiter);
    TEST_CHECK(terminal_exceptions == 0 &&
                   publish_outcome.state == fail_outcome.state &&
                   waiter_outcome.state == publish_outcome.state &&
                   owner.ticket().state == publish_outcome.state,
               "copied owner transactions must expose one linearized terminal result");

    ForgetPrimitiveFailureForTesting(race_key);
    ProductionCompileTransaction abandoned_waiter;
    {
        const ArtifactKey abandoned_key = MakeKey("public-abandoned");
        const ProductionCompileTransaction abandoned_owner =
            adapter.Acquire(MakeRequest(abandoned_key, "abandoned-owner"));
        abandoned_waiter =
            adapter.Acquire(MakeRequest(abandoned_key, "abandoned-waiter"));
        TEST_CHECK(abandoned_owner.owns_compile(),
                   "abandonment fixture requires a production owner");
    }
    const CompileOutcome abandoned = adapter.Wait(abandoned_waiter);
    const ArtifactCacheStats after_abandon = adapter.stats();
    const ProductionCompileTransaction recovered = adapter.Acquire(
        MakeRequest(MakeKey("public-after-abandon"), "after-abandon"));
    TEST_CHECK(abandoned.state == CompileRequestState::kFailed &&
                   abandoned.failure->diagnostic.find("abandoned") !=
                       std::string::npos &&
                   after_abandon.in_flight == 0 &&
                   adapter.stats().in_flight == 1 && recovered.owns_compile(),
               "owner destruction must release waiters and recover in-flight budget");
    (void)adapter.Fail(
        recovered, CompileFailure{CompileFailureCategory::kCancelled, 0,
                                  "test cleanup"});
    return true;
}

bool TestSameKeySingleflight() {
    using namespace kxc::api::internal;
    Reset(PrimitiveCacheLimits{16, 1024, 16, 16});
    constexpr int kThreads = 12;
    const auto key = MakeKey("singleflight");
    std::atomic<int> acquired{0};
    std::atomic<int> owners{0};
    std::atomic<int> failures{0};
    std::vector<const CachedPrimitive*> artifacts(kThreads, nullptr);
    std::vector<std::thread> threads;
    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([&, i] {
            try {
                const PrimitiveCacheLease lease = AcquirePrimitiveCache(key);
                if (lease.access() == PrimitiveCacheAccess::kOwner) ++owners;
                ++acquired;
                if (lease.access() == PrimitiveCacheAccess::kOwner) {
                    while (acquired.load() != kThreads) std::this_thread::yield();
                    artifacts[i] = &PublishPrimitiveCacheLease(
                                        lease, MakeArtifact("singleflight"))
                                        .artifact();
                } else {
                    artifacts[i] = &WaitPrimitiveCacheLease(lease).artifact();
                }
            } catch (...) {
                ++failures;
            }
        });
    }
    for (auto& thread : threads) thread.join();
    for (const CachedPrimitive* artifact : artifacts) {
        TEST_CHECK(artifact != nullptr && artifact == artifacts[0],
                   "all same-key callers must receive one immutable artifact");
    }
    const PrimitiveCacheStats stats = GetPrimitiveCacheStats();
    TEST_CHECK(owners == 1 && failures == 0 && stats.misses == 1 &&
                   stats.merged_waiters == kThreads - 1,
               "same full key must create exactly one compile owner");
    return true;
}

bool TestDigestCollisionDoesNotMerge() {
    using namespace kxc::api::internal;
    Reset();
    const auto first_key = MakeKey("collision-a", "same-digest");
    const auto second_key = MakeKey("collision-b", "same-digest");
    TEST_CHECK(first_key.digest() == second_key.digest() &&
                   first_key != second_key,
               "fixture must force an index digest collision");
    const PrimitiveCacheLease first = AcquirePrimitiveCache(first_key);
    const PrimitiveCacheLease second = AcquirePrimitiveCache(second_key);
    TEST_CHECK(first.access() == PrimitiveCacheAccess::kOwner &&
                   second.access() == PrimitiveCacheAccess::kOwner,
               "different canonical bytes must not share singleflight");
    (void)PublishPrimitiveCacheLease(first, MakeArtifact("collision_a"));
    (void)PublishPrimitiveCacheLease(second, MakeArtifact("collision_b"));
    TEST_CHECK(GetPrimitiveCacheStats().entries == 2,
               "different canonical bytes must occupy distinct ready records");
    return true;
}

bool TestFailureIsBoundedAndRetryable() {
    using namespace kxc::api::internal;
    Reset(PrimitiveCacheLimits{8, 1024, 8, 1});
    const auto key = MakeKey("failure");
    const PrimitiveCacheLease owner = AcquirePrimitiveCache(key);
    const PrimitiveCacheLease waiter = AcquirePrimitiveCache(key);
    FailPrimitiveCacheLease(owner, PrimitiveFailureCategory::kValidation,
                            "ABI validation failed", std::chrono::hours(1));
    TEST_CHECK(ThrowsWith([&] { (void)WaitPrimitiveCacheLease(waiter); },
                          "ABI validation failed"),
               "merged waiters must observe the structured owner failure");
    const PrimitiveCacheLease failed = AcquirePrimitiveCache(key);
    TEST_CHECK(failed.access() == PrimitiveCacheAccess::kFailed &&
                   failed.failure().category ==
                       PrimitiveFailureCategory::kValidation &&
                   failed.failure().retry_after_millis > 0,
               "negative result must suppress immediate compile storms");

    ForgetPrimitiveFailureForTesting(key);
    const PrimitiveCacheLease retry = AcquirePrimitiveCache(key);
    TEST_CHECK(retry.access() == PrimitiveCacheAccess::kOwner,
               "request becomes retryable when its failure deadline is cleared");
    FailPrimitiveCacheLease(retry, PrimitiveFailureCategory::kCancelled,
                            "caller cancelled", std::chrono::milliseconds(0));

    const auto other_key = MakeKey("other-failure");
    const PrimitiveCacheLease other = AcquirePrimitiveCache(other_key);
    FailPrimitiveCacheLease(other, PrimitiveFailureCategory::kUnsupported,
                            "unsupported", std::chrono::hours(1));
    TEST_CHECK(GetPrimitiveCacheStats().failures == 1,
               "failure records must obey their configured bound");
    return true;
}

bool TestBackpressureAndByteBudgetAreExplicit() {
    using namespace kxc::api::internal;
    Reset(PrimitiveCacheLimits{8, 1, 1, 8});
    const PrimitiveCacheLease first = AcquirePrimitiveCache(MakeKey("active"));
    const PrimitiveCacheLease rejected =
        AcquirePrimitiveCache(MakeKey("saturated"));
    TEST_CHECK(first.access() == PrimitiveCacheAccess::kOwner &&
                   rejected.access() == PrimitiveCacheAccess::kRejected &&
                   rejected.failure().category ==
                       PrimitiveFailureCategory::kBackpressure &&
                   ThrowsWith(
                       [&] { (void)WaitPrimitiveCacheLease(rejected); },
                       "backpressure"),
               "saturation must be observable and must not silently drop work");
    FailPrimitiveCacheLease(first, PrimitiveFailureCategory::kCancelled,
                            "test cleanup", std::chrono::milliseconds(0));
    ForgetPrimitiveFailureForTesting(first.key());

    const PrimitiveCacheLease oversized =
        AcquirePrimitiveCache(MakeKey("oversized"));
    const PrimitiveArtifactPin pin = PublishPrimitiveCacheLease(
        oversized, MakeArtifact("oversized", 2));
    const PrimitiveCacheStats stats = GetPrimitiveCacheStats();
    TEST_CHECK(pin.defined() && pin.artifact().kernel.IsReady() &&
                   stats.entries == 0 && stats.evictions == 1 &&
                   stats.rejections == 1,
               "byte budget may evict discoverability but not the returned pin");
    return true;
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"pin_survives_eviction", TestPinSurvivesEviction},
        {"production_adapter_eviction_pin",
         TestProductionAdapterPinSurvivesEviction},
        {"public_production_merge_cancel",
         TestPublicProductionSameKeyMergeAndCancellation},
        {"public_production_failure_retry_backpressure",
         TestPublicProductionFailureRetryAndBackpressure},
        {"public_production_concurrent_singleflight",
         TestPublicProductionConcurrentSingleflight},
        {"public_transaction_terminal_race_abandonment",
         TestPublicTransactionTerminalRaceAndAbandonment},
        {"same_key_singleflight", TestSameKeySingleflight},
        {"digest_collision_not_merged", TestDigestCollisionDoesNotMerge},
        {"failure_retry_and_bound", TestFailureIsBoundedAndRetryable},
        {"backpressure_and_bytes", TestBackpressureAndByteBudgetAreExplicit},
    };
    int failures = 0;
    for (const auto& test : tests) {
        try {
            if (!test.second()) {
                ++failures;
                continue;
            }
            std::cout << "[PASS] " << test.first << "\n";
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << test.first << ": " << error.what() << "\n";
            ++failures;
        }
    }
    kxc::api::internal::ClearPrimitiveCacheForTesting();
    return failures == 0 ? 0 : 1;
}
