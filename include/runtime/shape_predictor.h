#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace kxc {
namespace runtime {

// 输入shape签名：用于kernel缓存索引
struct ShapeSignature {
    std::vector<std::vector<int64_t>> input_shapes;

    bool operator==(const ShapeSignature& other) const {
        return input_shapes == other.input_shapes;
    }

    size_t Hash() const {
        size_t seed = input_shapes.size();
        for (const auto& shape : input_shapes) {
            seed ^= shape.size() + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            for (auto dim : shape) {
                seed ^= std::hash<int64_t>{}(dim) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
        }
        return seed;
    }
};

struct ShapeSignatureHash {
    size_t operator()(const ShapeSignature& sig) const { return sig.Hash(); }
};

// 从raw指针和shape信息提取签名
ShapeSignature MakeShapeSignature(const std::vector<std::vector<int64_t>>& shapes);

// Shape预测器：统计历史输入，预测未来可能的shape
class ShapePredictor {
public:
    // 记录一次实际输入
    void Record(const ShapeSignature& sig);

    // 预测Top-K最可能的输入shape
    std::vector<ShapeSignature> PredictTopK(int k = 3) const;

    // 是否值得为此shape编译优化版本
    // 首次出现：不编译（可能是偶发）
    // 出现 >= threshold 次：编译
    // 占比 >= ratio：编译
    bool ShouldCompile(const ShapeSignature& sig,
                       int count_threshold = 2,
                       double ratio_threshold = 0.05) const;

    // 获取某shape的出现次数
    int GetCount(const ShapeSignature& sig) const;

    // 总调用次数
    int TotalCount() const { return total_count_; }

private:
    mutable std::mutex mu_;
    std::unordered_map<ShapeSignature, int, ShapeSignatureHash> freq_;
    std::deque<ShapeSignature> recent_;
    int total_count_{0};
    static constexpr int kMaxRecent = 200;
};

}  // namespace runtime
}  // namespace kxc
