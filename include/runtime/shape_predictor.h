/*! \file include/runtime/shape_predictor.h
 * \brief 定义 adaptive runtime、shape predictor、kernel cache 和后台编译器。
 */

#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace kxc {
namespace runtime {

/*! \brief 输入 shape 签名，用作 adaptive kernel cache 的索引。 */
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

/*! \brief ShapeSignature 的哈希器，供 unordered_map/unordered_set 使用。 */
struct ShapeSignatureHash {
    size_t operator()(const ShapeSignature& sig) const { return sig.Hash(); }
};

/*! \brief 从调用方传入的 shape 列表构造 ShapeSignature。 */
ShapeSignature MakeShapeSignature(const std::vector<std::vector<int64_t>>& shapes);

/*! \brief 基于历史调用频率预测常见输入 shape，驱动后台优化编译。 */
class ShapePredictor {
public:
    /*! \brief 记录一次实际输入 shape。 */
    void Record(const ShapeSignature& sig);

    /*! \brief 返回出现频率最高的 Top-K shape。 */
    std::vector<ShapeSignature> PredictTopK(int k = 3) const;

    /*! \brief 判断是否值得为该 shape 编译优化版本。 */
    bool ShouldCompile(const ShapeSignature& sig,
                       int count_threshold = 2,
                       double ratio_threshold = 0.05) const;

    /*! \brief 获取某个 shape 的历史出现次数。 */
    int GetCount(const ShapeSignature& sig) const;

    /*! \brief 返回已记录的总调用次数。 */
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
