#include "runtime/kernel_cache.h"

#include <climits>

namespace kxc {
namespace runtime {

api::CompiledModule* KernelCache::GetExact(const ShapeSignature& sig) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = cache_.find(sig);
    if (it != cache_.end()) return &it->second;
    return nullptr;
}

api::CompiledModule* KernelCache::GetFuzzy(const ShapeSignature& sig) {
    std::lock_guard<std::mutex> lock(mu_);
    api::CompiledModule* best = nullptr;
    int64_t best_dist = INT64_MAX;

    for (auto& [cached_sig, module] : cache_) {
        int64_t dist = ShapeDistance(sig, cached_sig);
        if (dist >= 0 && dist < best_dist) {
            best_dist = dist;
            best = &module;
        }
    }
    return best;
}

void KernelCache::Put(const ShapeSignature& sig, api::CompiledModule module) {
    std::lock_guard<std::mutex> lock(mu_);
    cache_[sig] = std::move(module);
}

bool KernelCache::Has(const ShapeSignature& sig) const {
    std::lock_guard<std::mutex> lock(mu_);
    return cache_.count(sig) > 0;
}

size_t KernelCache::Size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return cache_.size();
}

void KernelCache::Clear() {
    std::lock_guard<std::mutex> lock(mu_);
    cache_.clear();
}

int64_t KernelCache::ShapeDistance(const ShapeSignature& query,
                                   const ShapeSignature& cached) {
    // 必须有相同数量的输入
    if (query.input_shapes.size() != cached.input_shapes.size()) return -1;

    int64_t total_dist = 0;
    for (size_t i = 0; i < query.input_shapes.size(); ++i) {
        const auto& q = query.input_shapes[i];
        const auto& c = cached.input_shapes[i];
        // 必须有相同的维度数
        if (q.size() != c.size()) return -1;

        for (size_t d = 0; d < q.size(); ++d) {
            // cached的每个维度必须 >= query（更大的kernel可兼容小输入）
            if (c[d] < q[d]) return -1;
            total_dist += (c[d] - q[d]);
        }
    }
    return total_dist;
}

}  // namespace runtime
}  // namespace kxc
