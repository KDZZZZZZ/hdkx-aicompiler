/*! \file src/compiler/cache/primitive_cache.cc
 * \brief Pinned ready-artifact cache with bounded same-key singleflight.
 */

#include "../internal/primitive_cache.h"

#include <algorithm>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace kxc::api::internal {

struct PrimitiveArtifact final {
    ArtifactKey key;
    CachedPrimitive entry;
};

struct PrimitiveFlight final {
    std::mutex mutex;
    std::condition_variable ready;
    bool completed{false};
    PrimitiveArtifactPin pin;
    PrimitiveFailureRecord failure;
};

namespace {

using Clock = std::chrono::steady_clock;
constexpr int kKernelABIVersion = 1;

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

std::string KeyBytes(const ArtifactKey& key) {
    if (!key.defined()) {
        throw std::invalid_argument(
            "primitive artifact cache requires a complete ArtifactKey");
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

void EvictReadyArtifacts(PrimitiveCache* cache) {
    while (cache->entries.size() > cache->limits.max_entries ||
           cache->accounted_bytes > cache->limits.max_accounted_bytes) {
        const auto oldest = std::min_element(
            cache->entries.begin(), cache->entries.end(),
            [](const auto& lhs, const auto& rhs) {
                return lhs.second.stamp < rhs.second.stamp;
            });
        if (oldest == cache->entries.end()) break;
        cache->accounted_bytes -=
            oldest->second.artifact->entry.accounted_bytes;
        cache->entries.erase(oldest);
        ++cache->evictions;
    }
}

std::string TargetFingerprint(const Target& target) {
    if (!target.defined() || !target.As<TargetNode>()) {
        throw std::invalid_argument(
            "primitive artifact identity requires a defined Target");
    }
    const TargetNode* node = target.operator->();
    std::ostringstream out;
    out << "target-v1|kind=" << node->kind
        << "|device_type=" << static_cast<int>(node->device_type)
        << "|device_id=" << node->device_id
        << "|exists=" << node->attrs.exists
        << "|device_name=" << node->attrs.device_name
        << "|arch=" << node->attrs.arch
        << "|max_clock_khz=" << node->attrs.max_clock_rate_khz
        << "|max_registers_block=" << node->attrs.max_registers_per_block
        << "|api_version=" << node->attrs.api_version
        << "|driver_version=" << node->attrs.driver_version
        << "|l2_bytes=" << node->attrs.l2_cache_size_bytes
        << "|global_bytes=" << node->attrs.total_global_memory
        << "|shared_mem_sm="
        << node->attrs.max_shared_memory_per_multiprocessor
        << "|registers_sm=" << node->attrs.max_registers_per_multiprocessor
        << "|threads_sm=" << node->attrs.max_threads_per_multiprocessor
        << "|threads_block=" << node->attrs.max_threads_per_block
        << "|warp=" << node->attrs.warp_size
        << "|compute_major=" << node->attrs.compute_version_major
        << "|compute_minor=" << node->attrs.compute_version_minor
        << "|multiprocessors=" << node->attrs.multi_processor_count;
    return out.str();
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

PrimitiveArtifactPin::PrimitiveArtifactPin(
    std::shared_ptr<const PrimitiveArtifact> artifact)
    : artifact_(std::move(artifact)) {}

bool PrimitiveArtifactPin::defined() const noexcept {
    return artifact_ != nullptr;
}

const ArtifactKey& PrimitiveArtifactPin::key() const {
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

const ArtifactKey& PrimitiveCacheLease::key() const {
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

ArtifactKey BuildPrimitiveArtifactKey(
    const UnitSemanticKey& semantic_key, const Target& target,
    const std::string& pipeline_fingerprint, const char* schedule_version,
    const char* backend_version) {
    if (!semantic_key.defined() || pipeline_fingerprint.empty() ||
        schedule_version == nullptr || schedule_version[0] == '\0' ||
        backend_version == nullptr || backend_version[0] == '\0') {
        throw std::invalid_argument(
            "primitive artifact key requires semantics, pipeline, schedule, and backend");
    }
    return ArtifactKey(semantic_key, TargetFingerprint(target),
                       pipeline_fingerprint, kKernelABIVersion,
                       schedule_version, backend_version);
}

PrimitiveCacheLease AcquirePrimitiveCache(const ArtifactKey& key) {
    const std::string canonical = KeyBytes(key);
    PrimitiveCache& cache = Cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    const Clock::time_point now = Clock::now();
    RemoveExpiredFailures(&cache, now);

    PrimitiveCacheLease lease;
    lease.key_ = key;
    const auto ready = cache.entries.find(canonical);
    if (ready != cache.entries.end()) {
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

    ++cache.misses;
    lease.access_ = PrimitiveCacheAccess::kOwner;
    lease.flight_ = std::make_shared<PrimitiveFlight>();
    cache.in_flight.emplace(canonical, lease.flight_);
    return lease;
}

PrimitiveArtifactPin WaitPrimitiveCacheLease(
    const PrimitiveCacheLease& lease) {
    if (lease.access_ == PrimitiveCacheAccess::kHit) return lease.pin();
    if (lease.access_ == PrimitiveCacheAccess::kFailed ||
        lease.access_ == PrimitiveCacheAccess::kRejected) {
        ThrowFailure(lease.failure_);
    }
    if (!lease.flight_) {
        throw std::logic_error("primitive cache lease has no in-flight request");
    }
    std::unique_lock<std::mutex> lock(lease.flight_->mutex);
    lease.flight_->ready.wait(lock,
                              [&] { return lease.flight_->completed; });
    if (lease.flight_->pin.defined()) return lease.flight_->pin;
    ThrowFailure(lease.flight_->failure);
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
        cache.in_flight.erase(active);
        cache.failures.erase(canonical);
        cache.accounted_bytes += artifact->entry.accounted_bytes;
        cache.entries.emplace(
            canonical, CacheRecord{artifact, cache.next_stamp++});
        EvictReadyArtifacts(&cache);
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
        cache.failures.insert_or_assign(
            canonical,
            FailureEntry{failure, Clock::now() + retry_after,
                         cache.next_stamp++});
        BoundFailures(&cache);
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

void ForgetPrimitiveFailureForTesting(const ArtifactKey& key) {
    PrimitiveCache& cache = Cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    cache.failures.erase(KeyBytes(key));
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
