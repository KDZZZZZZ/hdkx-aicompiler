/*! \file src/compiler/cache/primitive_cache.cc
 * \brief Pinned ready-artifact cache with bounded same-key singleflight.
 */

#include "../internal/primitive_cache.h"

#include "../internal/execution_contract.h"
#include "support/hash.h"

#include <algorithm>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace kxc::api::internal {

struct PrimitiveArtifact final {
    PrimitiveArtifactKey key;
    CachedPrimitive entry;
};

struct PrimitiveFlight final {
    std::mutex mutex;
    std::condition_variable ready;
    uint64_t publish_stamp{0};
    uint64_t merged_waiters{0};
    bool completed{false};
    PrimitiveArtifactPin pin;
    PrimitiveFailureRecord failure;
};

namespace {

using Clock = std::chrono::steady_clock;
constexpr int kKernelABIVersion = 2;

struct CacheRecord final {
    std::shared_ptr<const PrimitiveArtifact> artifact;
    uint64_t stamp{0};
};

struct FailureEntry final {
    PrimitiveFailureRecord failure;
    Clock::time_point retry_after;
    uint64_t stamp{0};
};

struct PrimitiveCache final {
    std::mutex mutex;
    std::unordered_map<std::string, CacheRecord> entries;
    std::unordered_map<std::string, std::shared_ptr<PrimitiveFlight>> in_flight;
    std::unordered_map<std::string, FailureEntry> failures;
    PrimitiveCacheLimits limits;
    uint64_t next_stamp{1};
    uint64_t hits{0};
    uint64_t misses{0};
    uint64_t evictions{0};
    uint64_t merged_waiters{0};
    uint64_t rejections{0};
    uint64_t accounted_bytes{0};
};

PrimitiveCache& Cache() {
    static PrimitiveCache cache;
    return cache;
}

std::string KeyBytes(const PrimitiveArtifactKey& key) {
    if (!key.defined()) {
        throw std::invalid_argument(
            "primitive artifact cache requires a complete PrimitiveArtifactKey");
    }
    return key.canonical_bytes();
}

const char* FailureCategoryName(PrimitiveFailureCategory category) {
    switch (category) {
        case PrimitiveFailureCategory::kCompile: return "compile";
        case PrimitiveFailureCategory::kValidation: return "validation";
        case PrimitiveFailureCategory::kUnsupported: return "unsupported";
        case PrimitiveFailureCategory::kCancelled: return "cancelled";
        case PrimitiveFailureCategory::kBackpressure: return "backpressure";
    }
    return "unknown";
}

[[noreturn]] void ThrowFailure(const PrimitiveFailureRecord& failure) {
    throw std::runtime_error(
        std::string("primitive compile request ") +
        FailureCategoryName(failure.category) + ": " + failure.diagnostic);
}

void ValidateEntry(const CachedPrimitive& entry) {
    if (!entry.signature.defined() || !entry.launch_metadata.defined() ||
        !entry.kernel.IsReady() || entry.accounted_bytes == 0 ||
        entry.provenance.empty() || entry.validation_record.empty()) {
        throw std::invalid_argument(
            "primitive artifact requires a complete validated executable");
    }
    entry.signature.Validate();
    entry.launch_metadata.Validate();
    if (entry.kernel.signature().get() != entry.signature.get() ||
        entry.kernel.launch_metadata().get() != entry.launch_metadata.get()) {
        throw std::invalid_argument(
            "primitive artifact executable contract is inconsistent");
    }
}

void RemoveExpiredFailures(PrimitiveCache* cache, Clock::time_point now) {
    for (auto it = cache->failures.begin(); it != cache->failures.end();) {
        if (it->second.retry_after <= now) {
            it = cache->failures.erase(it);
        } else {
            ++it;
        }
    }
}

void BoundFailures(PrimitiveCache* cache) {
    while (cache->failures.size() > cache->limits.max_failures) {
        const auto oldest = std::min_element(
            cache->failures.begin(), cache->failures.end(),
            [](const auto& lhs, const auto& rhs) {
                return lhs.second.stamp < rhs.second.stamp;
            });
        if (oldest == cache->failures.end()) break;
        cache->failures.erase(oldest);
    }
}

bool EvictOldestReadyArtifact(PrimitiveCache* cache) {
    const auto oldest = std::min_element(
        cache->entries.begin(), cache->entries.end(),
        [](const auto& lhs, const auto& rhs) {
            return lhs.second.stamp < rhs.second.stamp;
        });
    if (oldest == cache->entries.end()) return false;
    cache->accounted_bytes -=
        oldest->second.artifact->entry.accounted_bytes;
    cache->entries.erase(oldest);
    ++cache->evictions;
    return true;
}

void EvictReadyArtifacts(PrimitiveCache* cache) {
    while (cache->entries.size() > cache->limits.max_entries ||
           cache->accounted_bytes > cache->limits.max_accounted_bytes) {
        if (!EvictOldestReadyArtifact(cache)) break;
    }
}

bool MakeRoomForReadyArtifact(PrimitiveCache* cache, uint64_t bytes) {
    if (bytes > cache->limits.max_accounted_bytes) return false;
    const uint64_t remaining = cache->limits.max_accounted_bytes - bytes;
    while (cache->entries.size() >= cache->limits.max_entries ||
           cache->accounted_bytes > remaining) {
        if (!EvictOldestReadyArtifact(cache)) break;
    }
    return cache->entries.size() < cache->limits.max_entries &&
           cache->accounted_bytes <= remaining;
}
void AbandonPrimitiveCacheFlight(
    const PrimitiveArtifactKey& key,
    const std::shared_ptr<PrimitiveFlight>& flight) noexcept {
    try {
        const std::string canonical = KeyBytes(key);
        PrimitiveCache& cache = Cache();
        PrimitiveFailureRecord failure{
            PrimitiveFailureCategory::kCompile,
            "primitive cache owner was abandoned", 1000};
        {
            std::lock_guard<std::mutex> lock(cache.mutex);
            const auto active = cache.in_flight.find(canonical);
            if (active == cache.in_flight.end() ||
                active->second.get() != flight.get()) {
                return;
            }
            cache.in_flight.erase(active);
            if (cache.next_stamp != std::numeric_limits<uint64_t>::max()) {
                cache.failures.insert_or_assign(
                    canonical, FailureEntry{failure,
                                            Clock::now() + std::chrono::seconds(1),
                                            cache.next_stamp++});
                BoundFailures(&cache);
            }
        }
        {
            std::lock_guard<std::mutex> lock(flight->mutex);
            if (flight->completed) return;
            flight->failure = std::move(failure);
            flight->completed = true;
        }
        flight->ready.notify_all();
    } catch (...) {
    }
}

PrimitiveFailureRecord FailureWithRemaining(
    const FailureEntry& entry, Clock::time_point now) {
    PrimitiveFailureRecord failure = entry.failure;
    const auto remaining = entry.retry_after > now
                               ? entry.retry_after - now
                               : Clock::duration::zero();
    failure.retry_after_millis = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(remaining)
            .count());
    return failure;
}

}  // namespace

std::string BuildTargetCapabilityFingerprint(const Target& target) {
    return CanonicalTargetSnapshot(target);
}

PrimitiveArtifactPin::PrimitiveArtifactPin(
    std::shared_ptr<const PrimitiveArtifact> artifact)
    : artifact_(std::move(artifact)) {}

bool PrimitiveArtifactPin::defined() const noexcept {
    return artifact_ != nullptr;
}

const PrimitiveArtifactKey& PrimitiveArtifactPin::key() const {
    if (!artifact_) throw std::logic_error("primitive artifact pin is undefined");
    return artifact_->key;
}

const CachedPrimitive& PrimitiveArtifactPin::artifact() const {
    if (!artifact_) throw std::logic_error("primitive artifact pin is undefined");
    return artifact_->entry;
}

PrimitiveCacheAccess PrimitiveCacheLease::access() const noexcept {
    return access_;
}

const PrimitiveArtifactKey& PrimitiveCacheLease::key() const {
    if (!key_.defined()) throw std::logic_error("primitive cache lease has no key");
    return key_;
}

const PrimitiveArtifactPin& PrimitiveCacheLease::pin() const {
    if (!pin_.defined()) throw std::logic_error("primitive cache lease has no pin");
    return pin_;
}

const PrimitiveFailureRecord& PrimitiveCacheLease::failure() const {
    if (access_ != PrimitiveCacheAccess::kFailed &&
        access_ != PrimitiveCacheAccess::kRejected) {
        throw std::logic_error("primitive cache lease has no failure");
    }
    return failure_;
}

uint64_t PrimitiveCacheLease::merged_waiter_count() const {
    if (!flight_) return 0;
    std::lock_guard<std::mutex> lock(flight_->mutex);
    return flight_->merged_waiters;
}

PrimitiveArtifactKey BuildPrimitiveArtifactKey(
    const UnitSemanticKey& semantic_key, const Target& target,
    const std::string& pipeline_fingerprint, const char* schedule_contract,
    const char* backend_version) {
    if (!semantic_key.defined() || pipeline_fingerprint.empty() ||
        schedule_contract == nullptr || schedule_contract[0] == '\0' ||
        backend_version == nullptr || backend_version[0] == '\0') {
        throw std::invalid_argument(
            "primitive artifact key requires semantics, pipeline, schedule, and backend");
    }
    return PrimitiveArtifactKey(
        semantic_key, CanonicalTargetSnapshot(target), pipeline_fingerprint,
        kKernelABIVersion, schedule_contract, backend_version);
}

PrimitiveArtifactPin LookupPrimitiveCache(const PrimitiveArtifactKey& key) {
    const std::string canonical = KeyBytes(key);
    PrimitiveCache& cache = Cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    const auto ready = cache.entries.find(canonical);
    return ready == cache.entries.end()
               ? PrimitiveArtifactPin{}
               : PrimitiveArtifactPin(ready->second.artifact);
}

}  // namespace kxc::api::internal

namespace kxc::api {

ArtifactPin::ArtifactPin(std::shared_ptr<const ArtifactRecord> record,
                         std::shared_ptr<const void> owner)
    : record_(std::move(record)), owner_(std::move(owner)) {
    if (!record_ || !record_->artifact_key.defined() ||
        record_->executable_token.empty() || record_->signature_digest.empty() ||
        record_->launch_metadata_digest.empty() || record_->provenance.empty() ||
        record_->byte_size == 0 || record_->validation_record.empty() || !owner_) {
        throw std::invalid_argument(
            "production ArtifactPin requires a complete record and owner");
    }
}

bool ArtifactPin::defined() const noexcept { return record_ != nullptr; }

const ArtifactRecord& ArtifactPin::record() const {
    if (!record_) throw std::logic_error("ArtifactPin is undefined");
    return *record_;
}

}  // namespace kxc::api

namespace kxc::api::internal {

ArtifactPin ArtifactPinAccess::Wrap(const PrimitiveArtifactPin& pin) {
    if (!pin.defined()) {
        throw std::invalid_argument(
            "cannot expose an undefined primitive artifact pin");
    }
    const CachedPrimitive& artifact = pin.artifact();
    ArtifactRecord record{pin.key(),
                          "primitive-v1:" + pin.key().canonical_bytes(),
                          support::HashText(artifact.signature.CanonicalBytes()),
                          support::HashText(
                              artifact.launch_metadata.CanonicalBytes()),
                          artifact.provenance,
                          artifact.accounted_bytes,
                          artifact.validation_record};
    return ArtifactPin(std::make_shared<const ArtifactRecord>(std::move(record)),
                       std::make_shared<PrimitiveArtifactPin>(pin));
}

PrimitiveCacheLease AcquirePrimitiveCache(
    const PrimitiveArtifactKey& key) {
    const std::string canonical = KeyBytes(key);
    PrimitiveCache& cache = Cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    const Clock::time_point now = Clock::now();
    RemoveExpiredFailures(&cache, now);

    PrimitiveCacheLease lease;
    lease.key_ = key;
    const auto ready = cache.entries.find(canonical);
    if (ready != cache.entries.end()) {
        if (cache.next_stamp == std::numeric_limits<uint64_t>::max()) {
            ++cache.rejections;
            lease.access_ = PrimitiveCacheAccess::kRejected;
            lease.failure_ = PrimitiveFailureRecord{
                PrimitiveFailureCategory::kBackpressure,
                "primitive cache LRU stamp space is exhausted", 0};
            return lease;
        }
        ready->second.stamp = cache.next_stamp++;
        ++cache.hits;
        lease.access_ = PrimitiveCacheAccess::kHit;
        lease.pin_ = PrimitiveArtifactPin(ready->second.artifact);
        return lease;
    }
    const auto failure = cache.failures.find(canonical);
    if (failure != cache.failures.end()) {
        lease.access_ = PrimitiveCacheAccess::kFailed;
        lease.failure_ = FailureWithRemaining(failure->second, now);
        return lease;
    }
    const auto active = cache.in_flight.find(canonical);
    if (active != cache.in_flight.end()) {
        ++cache.merged_waiters;
        {
            std::lock_guard<std::mutex> flight_lock(active->second->mutex);
            ++active->second->merged_waiters;
        }
        lease.access_ = PrimitiveCacheAccess::kWait;
        lease.flight_ = active->second;
        return lease;
    }
    if (cache.in_flight.size() >= cache.limits.max_in_flight) {
        ++cache.rejections;
        lease.access_ = PrimitiveCacheAccess::kRejected;
        lease.failure_ = PrimitiveFailureRecord{
            PrimitiveFailureCategory::kBackpressure,
            "bounded in-flight compile budget is saturated", 0};
        return lease;
    }
    if (cache.next_stamp == std::numeric_limits<uint64_t>::max()) {
        ++cache.rejections;
        lease.access_ = PrimitiveCacheAccess::kRejected;
        lease.failure_ = PrimitiveFailureRecord{
            PrimitiveFailureCategory::kBackpressure,
            "primitive cache stamp space is exhausted", 0};
        return lease;
    }

    ++cache.misses;
    lease.access_ = PrimitiveCacheAccess::kOwner;
    lease.flight_ = std::make_shared<PrimitiveFlight>();
    lease.flight_->publish_stamp = cache.next_stamp++;
    cache.in_flight.emplace(canonical, lease.flight_);
    lease.owner_guard_ = std::shared_ptr<const void>(
        new int(0), [key, flight = lease.flight_](const void* value) {
            delete static_cast<const int*>(value);
            AbandonPrimitiveCacheFlight(key, flight);
        });
    return lease;
}

PrimitiveCacheWaitResult WaitPrimitiveCacheLeaseResult(
    const PrimitiveCacheLease& lease) {
    if (lease.access_ == PrimitiveCacheAccess::kHit) {
        return PrimitiveCacheWaitResult{lease.pin(), {}};
    }
    if (lease.access_ == PrimitiveCacheAccess::kFailed ||
        lease.access_ == PrimitiveCacheAccess::kRejected) {
        return PrimitiveCacheWaitResult{{}, lease.failure_};
    }
    if (!lease.flight_) {
        throw std::logic_error("primitive cache lease has no in-flight request");
    }
    std::unique_lock<std::mutex> lock(lease.flight_->mutex);
    lease.flight_->ready.wait(lock,
                              [&] { return lease.flight_->completed; });
    return PrimitiveCacheWaitResult{lease.flight_->pin,
                                    lease.flight_->failure};
}

PrimitiveArtifactPin WaitPrimitiveCacheLease(
    const PrimitiveCacheLease& lease) {
    const PrimitiveCacheWaitResult result =
        WaitPrimitiveCacheLeaseResult(lease);
    if (result.succeeded()) return result.pin;
    ThrowFailure(result.failure);
}

PrimitiveArtifactPin PublishPrimitiveCacheLease(
    const PrimitiveCacheLease& lease, CachedPrimitive entry) {
    if (lease.access_ != PrimitiveCacheAccess::kOwner || !lease.flight_) {
        throw std::invalid_argument(
            "only the singleflight owner may publish a primitive artifact");
    }
    ValidateEntry(entry);
    auto artifact = std::make_shared<const PrimitiveArtifact>(
        PrimitiveArtifact{lease.key(), std::move(entry)});
    PrimitiveArtifactPin pin(artifact);
    const std::string canonical = KeyBytes(lease.key());

    PrimitiveCache& cache = Cache();
    {
        std::lock_guard<std::mutex> lock(cache.mutex);
        const auto active = cache.in_flight.find(canonical);
        if (active == cache.in_flight.end() ||
            active->second.get() != lease.flight_.get()) {
            throw std::logic_error(
                "primitive singleflight owner is stale or already completed");
        }
        if (MakeRoomForReadyArtifact(
                &cache, artifact->entry.accounted_bytes)) {
            cache.entries.emplace(
                canonical,
                CacheRecord{artifact, lease.flight_->publish_stamp});
            // MakeRoom proved this addition cannot overflow the byte budget.
            cache.accounted_bytes += artifact->entry.accounted_bytes;
        } else {
            // The returned pin remains valid, but this oversized artifact is
            // intentionally not discoverable in the bounded cache.
            ++cache.evictions;
        }
        cache.in_flight.erase(active);
        cache.failures.erase(canonical);
    }
    {
        std::lock_guard<std::mutex> lock(lease.flight_->mutex);
        lease.flight_->pin = pin;
        lease.flight_->completed = true;
    }
    lease.flight_->ready.notify_all();
    return pin;
}

void FailPrimitiveCacheLease(
    const PrimitiveCacheLease& lease, PrimitiveFailureCategory category,
    std::string diagnostic, std::chrono::milliseconds retry_after) {
    if (lease.access_ != PrimitiveCacheAccess::kOwner || !lease.flight_) return;
    if (diagnostic.empty() || retry_after.count() < 0) {
        throw std::invalid_argument(
            "primitive failure requires a diagnostic and non-negative retry delay");
    }
    PrimitiveFailureRecord failure{category, std::move(diagnostic),
                                   static_cast<uint64_t>(retry_after.count())};
    const std::string canonical = KeyBytes(lease.key());
    PrimitiveCache& cache = Cache();
    {
        std::lock_guard<std::mutex> lock(cache.mutex);
        const auto active = cache.in_flight.find(canonical);
        if (active == cache.in_flight.end() ||
            active->second.get() != lease.flight_.get()) {
            return;
        }
        cache.in_flight.erase(active);
        // Counter exhaustion fails closed: complete this flight but do not
        // create a wrapped/ambiguous negative-cache record.
        if (cache.next_stamp != std::numeric_limits<uint64_t>::max()) {
            cache.failures.insert_or_assign(
                canonical,
                FailureEntry{failure, Clock::now() + retry_after,
                             cache.next_stamp++});
            BoundFailures(&cache);
        }
    }
    {
        std::lock_guard<std::mutex> lock(lease.flight_->mutex);
        lease.flight_->failure = failure;
        lease.flight_->completed = true;
    }
    lease.flight_->ready.notify_all();
}

PrimitiveCacheStats GetPrimitiveCacheStats() {
    PrimitiveCache& cache = Cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    uint64_t pins = 0;
    for (const auto& item : cache.entries) {
        const long references = item.second.artifact.use_count();
        if (references > 1) {
            pins += static_cast<uint64_t>(references - 1);
        }
    }
    return PrimitiveCacheStats{
        cache.hits,
        cache.misses,
        static_cast<uint64_t>(cache.entries.size()),
        cache.accounted_bytes,
        cache.evictions,
        static_cast<uint64_t>(cache.in_flight.size()),
        cache.merged_waiters,
        static_cast<uint64_t>(cache.failures.size()),
        cache.rejections,
        pins};
}

void SetPrimitiveCacheLimitsForTesting(PrimitiveCacheLimits limits) {
    if (limits.max_entries == 0 || limits.max_accounted_bytes == 0 ||
        limits.max_in_flight == 0 || limits.max_failures == 0) {
        throw std::invalid_argument("primitive cache limits must be positive");
    }
    PrimitiveCache& cache = Cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    if (!cache.in_flight.empty()) {
        throw std::logic_error(
            "cannot change primitive cache limits with active compiles");
    }
    cache.limits = limits;
    EvictReadyArtifacts(&cache);
    BoundFailures(&cache);
}

void ForgetPrimitiveFailureForTesting(const PrimitiveArtifactKey& key) {
    PrimitiveCache& cache = Cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    cache.failures.erase(KeyBytes(key));
}

PrimitiveArtifactPin ArtifactPinAccess::Unwrap(const ArtifactPin& pin) {
    if (!pin.defined() || !pin.owner_) {
        throw std::invalid_argument("ArtifactPin is not backed by the production cache");
    }
    const auto primitive =
        std::static_pointer_cast<const PrimitiveArtifactPin>(pin.owner_);
    if (!primitive || !primitive->defined() || primitive->key() !=
                                                 pin.record().artifact_key) {
        throw std::invalid_argument("production ArtifactPin owner is inconsistent");
    }
    return *primitive;
}

void ClearPrimitiveCacheForTesting() {
    PrimitiveCache& cache = Cache();
    std::vector<std::shared_ptr<PrimitiveFlight>> active;
    {
        std::lock_guard<std::mutex> lock(cache.mutex);
        for (const auto& item : cache.in_flight) active.push_back(item.second);
        cache.entries.clear();
        cache.in_flight.clear();
        cache.failures.clear();
        cache.limits = PrimitiveCacheLimits{};
        cache.next_stamp = 1;
        cache.hits = 0;
        cache.misses = 0;
        cache.evictions = 0;
        cache.merged_waiters = 0;
        cache.rejections = 0;
        cache.accounted_bytes = 0;
    }
    for (const auto& flight : active) {
        {
            std::lock_guard<std::mutex> lock(flight->mutex);
            flight->failure = PrimitiveFailureRecord{
                PrimitiveFailureCategory::kCancelled,
                "primitive cache was cleared", 0};
            flight->completed = true;
        }
        flight->ready.notify_all();
    }
}

}  // namespace kxc::api::internal
