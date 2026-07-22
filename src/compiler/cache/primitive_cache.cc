/*! \file src/compiler/cache/primitive_cache.cc
 * \brief Bounded process-local cache for immutable primitive executables.
 */

#include "../internal/primitive_cache.h"

#include <algorithm>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace kxc::api::internal {
namespace {

constexpr size_t kMaxPrimitiveCacheEntries = 256;
constexpr int kKernelABIVersion = 1;

struct CacheRecord final {
    CachedPrimitive entry;
    uint64_t stamp{0};
};

struct PrimitiveCache final {
    std::mutex mutex;
    std::unordered_map<std::string, CacheRecord> entries;
    uint64_t next_stamp{1};
    uint64_t hits{0};
    uint64_t misses{0};
};

PrimitiveCache& Cache() {
    static PrimitiveCache cache;
    return cache;
}

std::optional<CachedPrimitive> Find(const std::string& key,
                                    bool record_stats) {
    PrimitiveCache& cache = Cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    const auto found = cache.entries.find(key);
    if (found == cache.entries.end()) {
        if (record_stats) ++cache.misses;
        return std::nullopt;
    }
    if (record_stats) ++cache.hits;
    found->second.stamp = cache.next_stamp++;
    return found->second.entry;
}

}  // namespace

std::string BuildPrimitiveCacheKey(const String& structural_hash,
                                   const Target& target, int opt_level,
                                   const char* backend_version) {
    if (std::string(structural_hash).empty() || !target.defined() ||
        opt_level < 0 || opt_level > 3 || backend_version == nullptr ||
        backend_version[0] == '\0') {
        throw std::invalid_argument(
            "Primitive cache key requires hash, target, opt level, and backend version");
    }
    std::ostringstream key;
    key << "primitive-cache-v1|hash=" << std::string(structural_hash)
        << "|target=" << target.ToString()
        << "|exists=" << target->attrs.exists
        << "|device_name=" << target->attrs.device_name
        << "|max_clock_khz=" << target->attrs.max_clock_rate_khz
        << "|max_registers_block="
        << target->attrs.max_registers_per_block
        << "|api_version=" << target->attrs.api_version
        << "|driver_version=" << target->attrs.driver_version
        << "|l2_bytes=" << target->attrs.l2_cache_size_bytes
        << "|global_bytes=" << target->attrs.total_global_memory
        << "|shared_mem_sm="
        << target->attrs.max_shared_memory_per_multiprocessor
        << "|registers_sm="
        << target->attrs.max_registers_per_multiprocessor
        << "|threads_sm=" << target->attrs.max_threads_per_multiprocessor
        << "|compute_major=" << target->attrs.compute_version_major
        << "|compute_minor=" << target->attrs.compute_version_minor
        << "|multiprocessors=" << target->attrs.multi_processor_count
        << "|opt=" << opt_level
        << "|abi=" << kKernelABIVersion
        << "|backend=" << backend_version;
    return key.str();
}

std::optional<CachedPrimitive> LookupPrimitiveCache(const std::string& key) {
    return Find(key, true);
}

std::optional<CachedPrimitive> PeekPrimitiveCache(const std::string& key) {
    return Find(key, false);
}

void StorePrimitiveCache(std::string key, CachedPrimitive entry) {
    if (key.empty() || !entry.signature.defined() ||
        !entry.launch_metadata.defined() || !entry.kernel.IsReady()) {
        throw std::invalid_argument(
            "Primitive cache requires a complete executable entry");
    }
    PrimitiveCache& cache = Cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    cache.entries.insert_or_assign(
        std::move(key), CacheRecord{std::move(entry), cache.next_stamp++});
    if (cache.entries.size() <= kMaxPrimitiveCacheEntries) return;
    const auto oldest = std::min_element(
        cache.entries.begin(), cache.entries.end(),
        [](const auto& lhs, const auto& rhs) {
            return lhs.second.stamp < rhs.second.stamp;
        });
    if (oldest != cache.entries.end()) cache.entries.erase(oldest);
}

PrimitiveCacheStats GetPrimitiveCacheStats() {
    PrimitiveCache& cache = Cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    return PrimitiveCacheStats{cache.hits, cache.misses,
                               static_cast<uint64_t>(cache.entries.size())};
}

void ClearPrimitiveCacheForTesting() {
    PrimitiveCache& cache = Cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    cache.entries.clear();
    cache.next_stamp = 1;
    cache.hits = 0;
    cache.misses = 0;
}

}  // namespace kxc::api::internal
