#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>

#include "api/compiler.h"
#include "runtime/shape_predictor.h"

namespace kxc {
namespace runtime {

// Kernel缓存：精确匹配 + 模糊匹配
class KernelCache {
public:
    // 精确查找：shape完全匹配
    api::CompiledModule* GetExact(const ShapeSignature& sig);

    // 模糊查找：找最接近的已编译kernel（用于fallback）
    // 策略：找shape最接近且 >= 输入shape的kernel
    api::CompiledModule* GetFuzzy(const ShapeSignature& sig);

    // 插入新编译的kernel
    void Put(const ShapeSignature& sig, api::CompiledModule module);

    // 是否已有此shape的kernel
    bool Has(const ShapeSignature& sig) const;

    // 缓存大小
    size_t Size() const;

    // 清空缓存
    void Clear();

private:
    mutable std::mutex mu_;
    std::unordered_map<ShapeSignature, api::CompiledModule, ShapeSignatureHash> cache_;

    // 计算两个shape的"距离"（用于模糊匹配）
    // 返回：-1表示不兼容，>=0表示距离（越小越好）
    static int64_t ShapeDistance(const ShapeSignature& a, const ShapeSignature& b);
};

}  // namespace runtime
}  // namespace kxc
