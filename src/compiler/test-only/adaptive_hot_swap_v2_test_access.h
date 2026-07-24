#pragma once

#include "kxc/compiler/adaptive_hot_swap_v2.h"

namespace kxc::api::adaptive::hot_swap::v2::test_only {

struct AdaptiveHotSwapSnapshot final {
    size_t queued_flights{0}, active_flights{0}, discoverable_generations{0};
    uint64_t producer_reported_discoverable_bytes{0}, evictions{0}, merged_waiters{0}, retry_cached{0};
    size_t negative_cache_entries{0};
    uint64_t negative_cache_diagnostic_bytes{0}, negative_cache_evictions{0}, negative_cache_drops{0};
    bool negative_cache_compile_blocked{false};
    size_t quarantine_tombstones{0}, quarantine_compile_blocked_routes{0};
    uint64_t quarantine_saturations{0};
    size_t routes{0};
    uint64_t route_metadata_bytes{0}, quarantine_tombstone_bytes{0}, route_saturations{0}, tombstone_saturations{0};
};

class AdaptiveHotSwapTestAccess final {
public:
    static AdaptiveHotSwapSnapshot Snapshot(const AdaptiveHotSwapController& controller);
    static void ClearNegativeCache(AdaptiveHotSwapController& controller);
    static void ClearQuarantines(AdaptiveHotSwapController& controller);
};

}  // namespace kxc::api::adaptive::hot_swap::v2::test_only
