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
