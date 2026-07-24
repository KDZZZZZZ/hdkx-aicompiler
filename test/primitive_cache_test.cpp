/*! \file test/primitive_cache_test.cpp
 * \brief Verifies the real internal primitive cache's exact-key behavior.
 */

#include <atomic>
#include <chrono>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "../src/compiler/internal/primitive_cache.h"

namespace {

#define TEST_CHECK(condition, message)                                           \
    do {                                                                          \
        if (!(condition)) {                                                       \
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
    kxc::AsyncOperation Launch(const kxc::Array<kxc::runtime::NDArray>&,
                                const kxc::DeviceStream& stream,
                                const kxc::ObjectRef&) const override {
        return kxc::AsyncOperation::Completed(stream);
    }
};

kxc::api::PrimitiveArtifactKey MakeKey(const std::string& unit,
                                       const std::string& digest = {}) {
    return kxc::api::PrimitiveArtifactKey(
        kxc::api::UnitSemanticKey("unit=" + unit, digest),
        "target=cpu-test-v1", "pipeline=test-v1", 1,
        "schedule=test-v1", "backend=fake-v1", digest);
}

kxc::api::internal::CachedPrimitive MakeArtifact(
    const std::string& symbol, uint64_t bytes = 1,
    const std::shared_ptr<NoopLauncher>& launcher = std::make_shared<NoopLauncher>()) {
    using namespace kxc;
    using namespace kxc::codegen;
    KernelArgSpec output("output", KernelArgRole::kOutput,
                         runtime::DataTypeFromString("float32"), {1},
                         Device::CPU(), 1, true);
    KernelSignature signature(String(symbol), {output});
    KernelLaunchMetadata metadata(Device::CPU(), CodeGenBackend::kLLVM);
    return api::internal::CachedPrimitive{
        signature, metadata, CompiledKernel(signature, metadata, launcher), bytes,
        "primitive-cache-test", "fake-launcher-validated"};
}

void Reset(kxc::api::internal::PrimitiveCacheLimits limits = {}) {
    kxc::api::internal::ClearPrimitiveCacheForTesting();
    kxc::api::internal::SetPrimitiveCacheLimitsForTesting(limits);
}

bool TestPinSurvivesEvictionAndOversizedResult() {
    using namespace kxc::api::internal;
    Reset(PrimitiveCacheLimits{1, 64, 8, 8});
    const auto first_key = MakeKey("first");
    const PrimitiveArtifactPin first = PublishPrimitiveCacheLease(
        AcquirePrimitiveCache(first_key), MakeArtifact("first"));
    (void)PublishPrimitiveCacheLease(
        AcquirePrimitiveCache(MakeKey("second")), MakeArtifact("second"));
    TEST_CHECK(first.defined() && first.artifact().kernel.IsReady() &&
                   GetPrimitiveCacheStats().entries == 1 &&
                   GetPrimitiveCacheStats().evictions == 1,
               "eviction must not invalidate an issued pin");
    const PrimitiveCacheLease evicted = AcquirePrimitiveCache(first_key);
    TEST_CHECK(evicted.access() == PrimitiveCacheAccess::kOwner,
               "an evicted key must compile again despite its retained pin");
    FailPrimitiveCacheLease(evicted, PrimitiveFailureCategory::kCancelled,
                            "test cleanup", std::chrono::milliseconds(0));

    Reset(PrimitiveCacheLimits{8, 1, 8, 8});
    const PrimitiveArtifactPin oversized = PublishPrimitiveCacheLease(
        AcquirePrimitiveCache(MakeKey("oversized")), MakeArtifact("oversized", 2));
    const PrimitiveCacheStats stats = GetPrimitiveCacheStats();
    TEST_CHECK(oversized.defined() && oversized.artifact().kernel.IsReady() &&
                   stats.entries == 0 && stats.evictions == 1,
               "oversized results retain their returned pin but not discovery");
    return true;
}

bool TestConcurrentSameKeySingleflightAndWaiterOutcome() {
    using namespace kxc::api::internal;
    Reset(PrimitiveCacheLimits{16, 1024, 16, 16});
    constexpr int kThreads = 12;
    const auto key = MakeKey("singleflight");
    std::atomic<int> acquired{0};
    std::atomic<int> owners{0};
    std::vector<const CachedPrimitive*> artifacts(kThreads, nullptr);
    std::vector<std::thread> threads;
    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([&, i] {
            const PrimitiveCacheLease lease = AcquirePrimitiveCache(key);
            if (lease.access() == PrimitiveCacheAccess::kOwner) ++owners;
            ++acquired;
            if (lease.access() == PrimitiveCacheAccess::kOwner) {
                while (acquired.load() != kThreads) std::this_thread::yield();
                artifacts[i] = &PublishPrimitiveCacheLease(
                    lease, MakeArtifact("singleflight")).artifact();
            } else {
                const PrimitiveCacheWaitResult result =
                    WaitPrimitiveCacheLeaseResult(lease);
                artifacts[i] = result.succeeded() ? &result.pin.artifact() : nullptr;
            }
        });
    }
    for (auto& thread : threads) thread.join();
    for (const CachedPrimitive* artifact : artifacts) {
        TEST_CHECK(artifact != nullptr && artifact == artifacts.front(),
                   "all same-key waiters must receive the owner's artifact");
    }
    const PrimitiveCacheStats stats = GetPrimitiveCacheStats();
    TEST_CHECK(owners == 1 && stats.misses == 1 &&
                   stats.merged_waiters == kThreads - 1,
               "one full exact key must have exactly one concurrent owner");
    return true;
}

bool TestDeferredCrossKeyAcquisitionAvoidsCycle() {
    using namespace kxc::api::internal;
    Reset(PrimitiveCacheLimits{8, 1024, 2, 8});
    const auto first_key = MakeKey("deferred-first");
    const auto second_key = MakeKey("deferred-second");
    std::atomic<int> first_acquired{0};
    std::atomic<int> all_acquired{0};
    std::atomic<int> ready{0};
    std::vector<std::thread> threads;
    for (int i = 0; i < 2; ++i) {
        threads.emplace_back([&, i] {
            const kxc::api::PrimitiveArtifactKey& first =
                i == 0 ? first_key : second_key;
            const kxc::api::PrimitiveArtifactKey& second =
                i == 0 ? second_key : first_key;
            const PrimitiveCacheLease owner = AcquirePrimitiveCache(first);
            ++first_acquired;
            while (first_acquired.load() != 2) std::this_thread::yield();
            const PrimitiveCacheLease waiter = AcquirePrimitiveCache(second);
            ++all_acquired;
            while (all_acquired.load() != 2) std::this_thread::yield();
            if (owner.access() == PrimitiveCacheAccess::kOwner) {
                (void)PublishPrimitiveCacheLease(owner, MakeArtifact("deferred"));
            }
            if (WaitPrimitiveCacheLeaseResult(waiter).succeeded()) ++ready;
        });
    }
    for (auto& thread : threads) thread.join();
    TEST_CHECK(ready == 2 && GetPrimitiveCacheStats().in_flight == 0,
               "both owners must publish before either cross-key waiter blocks");
    return true;
}

bool TestExactCollisionFailureTtlAndBackpressure() {
    using namespace kxc::api::internal;
    Reset(PrimitiveCacheLimits{8, 64, 2, 1});
    const auto first_key = MakeKey("collision-a", "same-digest");
    const auto second_key = MakeKey("collision-b", "same-digest");
    TEST_CHECK(first_key.digest() == second_key.digest() && first_key != second_key,
               "fixture must force an index digest collision");
    const PrimitiveCacheLease first = AcquirePrimitiveCache(first_key);
    const PrimitiveCacheLease second = AcquirePrimitiveCache(second_key);
    TEST_CHECK(first.access() == PrimitiveCacheAccess::kOwner &&
                   second.access() == PrimitiveCacheAccess::kOwner,
               "digest collisions must not merge distinct canonical keys");
    (void)PublishPrimitiveCacheLease(first, MakeArtifact("collision-a"));
    (void)PublishPrimitiveCacheLease(second, MakeArtifact("collision-b"));
    TEST_CHECK(GetPrimitiveCacheStats().entries == 2,
               "distinct canonical keys must occupy distinct ready records");

    const auto failure_key = MakeKey("failure");
    const PrimitiveCacheLease owner = AcquirePrimitiveCache(failure_key);
    const PrimitiveCacheLease waiter = AcquirePrimitiveCache(failure_key);
    FailPrimitiveCacheLease(owner, PrimitiveFailureCategory::kValidation,
                            "ABI validation failed", std::chrono::hours(1));
    const PrimitiveCacheWaitResult waiter_result =
        WaitPrimitiveCacheLeaseResult(waiter);
    const PrimitiveCacheLease negative = AcquirePrimitiveCache(failure_key);
    TEST_CHECK(!waiter_result.succeeded() &&
                   waiter_result.failure.category == PrimitiveFailureCategory::kValidation &&
                   negative.access() == PrimitiveCacheAccess::kFailed &&
                   negative.failure().retry_after_millis > 0 &&
                   ThrowsWith([&] { (void)WaitPrimitiveCacheLease(waiter); },
                              "ABI validation failed"),
               "waiters and retries must observe the negative TTL outcome");

    ForgetPrimitiveFailureForTesting(failure_key);
    Reset(PrimitiveCacheLimits{8, 64, 1, 8});
    const PrimitiveCacheLease active = AcquirePrimitiveCache(MakeKey("active"));
    const PrimitiveCacheLease rejected = AcquirePrimitiveCache(MakeKey("saturated"));
    TEST_CHECK(rejected.access() == PrimitiveCacheAccess::kRejected &&
                   rejected.failure().category == PrimitiveFailureCategory::kBackpressure &&
                   ThrowsWith([&] { (void)WaitPrimitiveCacheLease(rejected); },
                              "backpressure"),
               "in-flight saturation must return an explicit failure outcome");
    FailPrimitiveCacheLease(active, PrimitiveFailureCategory::kCancelled,
                            "test cleanup", std::chrono::milliseconds(0));
    return true;
}

bool TestOwnerAbandonmentReleasesWaiters() {
    using namespace kxc::api::internal;
    Reset(PrimitiveCacheLimits{8, 1024, 1, 8});
    PrimitiveCacheLease waiter;
    {
        const PrimitiveCacheLease owner = AcquirePrimitiveCache(MakeKey("abandoned"));
        waiter = AcquirePrimitiveCache(MakeKey("abandoned"));
        TEST_CHECK(owner.access() == PrimitiveCacheAccess::kOwner &&
                       waiter.access() == PrimitiveCacheAccess::kWait,
                   "fixture requires one owner and one waiter");
    }
    const PrimitiveCacheWaitResult result = WaitPrimitiveCacheLeaseResult(waiter);
    TEST_CHECK(!result.succeeded() &&
                   result.failure.diagnostic.find("abandoned") != std::string::npos &&
                   GetPrimitiveCacheStats().in_flight == 0,
               "owner destruction must terminally release waiters and capacity");
    return true;
}

bool TestCheckedNearUint64ByteAccounting() {
    using namespace kxc::api::internal;
    Reset(PrimitiveCacheLimits{8, std::numeric_limits<uint64_t>::max(), 1, 8});
    const PrimitiveCacheLease near_limit = AcquirePrimitiveCache(MakeKey("near-limit"));
    (void)PublishPrimitiveCacheLease(
        near_limit,
        MakeArtifact("near-limit", std::numeric_limits<uint64_t>::max() - 10));
    const PrimitiveCacheLease crossing = AcquirePrimitiveCache(MakeKey("crossing"));
    const PrimitiveArtifactPin pin = PublishPrimitiveCacheLease(
        crossing, MakeArtifact("crossing", 20));
    const PrimitiveCacheStats stats = GetPrimitiveCacheStats();
    TEST_CHECK(pin.defined() && stats.entries == 1 && stats.accounted_bytes == 20 &&
                   stats.evictions == 1,
               "byte accounting must evict before addition and never wrap");
    return true;
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"pin_survives_eviction_and_oversized", TestPinSurvivesEvictionAndOversizedResult},
        {"concurrent_same_key_singleflight", TestConcurrentSameKeySingleflightAndWaiterOutcome},
        {"deferred_cross_key_acquisition", TestDeferredCrossKeyAcquisitionAvoidsCycle},
        {"exact_collision_failure_ttl_backpressure", TestExactCollisionFailureTtlAndBackpressure},
        {"owner_abandonment", TestOwnerAbandonmentReleasesWaiters},
        {"near_uint64_byte_accounting", TestCheckedNearUint64ByteAccounting},
    };
    int failures = 0;
    for (const auto& test : tests) {
        try {
            if (test.second()) std::cout << "[PASS] " << test.first << "\n";
            else ++failures;
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << test.first << ": " << error.what() << "\n";
            ++failures;
        }
    }
    kxc::api::internal::ClearPrimitiveCacheForTesting();
    return failures == 0 ? 0 : 1;
}
