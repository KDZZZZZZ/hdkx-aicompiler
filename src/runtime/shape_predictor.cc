/*! \file src/runtime/shape_predictor.cc
 * \brief 实现 adaptive runtime、kernel cache、shape 统计和后台编译。
 */

#include "runtime/shape_predictor.h"

#include <algorithm>

namespace kxc {
namespace runtime {

ShapeSignature MakeShapeSignature(const std::vector<std::vector<int64_t>>& shapes) {
    ShapeSignature sig;
    sig.input_shapes = shapes;
    return sig;
}

void ShapePredictor::Record(const ShapeSignature& sig) {
    std::lock_guard<std::mutex> lock(mu_);
    freq_[sig]++;
    recent_.push_back(sig);
    if (recent_.size() > kMaxRecent) {
        recent_.pop_front();
    }
    total_count_++;
}

std::vector<ShapeSignature> ShapePredictor::PredictTopK(int k) const {
    std::lock_guard<std::mutex> lock(mu_);

    // 按频率排序
    std::vector<std::pair<ShapeSignature, int>> items(freq_.begin(), freq_.end());
    std::sort(items.begin(), items.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    std::vector<ShapeSignature> result;
    for (int i = 0; i < std::min(k, static_cast<int>(items.size())); ++i) {
        result.push_back(items[i].first);
    }
    return result;
}

bool ShapePredictor::ShouldCompile(const ShapeSignature& sig,
                                   int count_threshold,
                                   double ratio_threshold) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = freq_.find(sig);
    if (it == freq_.end()) return false;

    int count = it->second;
    if (count < count_threshold) return false;

    if (total_count_ > 0) {
        double ratio = static_cast<double>(count) / total_count_;
        if (ratio >= ratio_threshold) return true;
    }

    return count >= count_threshold;
}

int ShapePredictor::GetCount(const ShapeSignature& sig) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = freq_.find(sig);
    return it != freq_.end() ? it->second : 0;
}

}  // namespace runtime
}  // namespace kxc
