/*! \file include/runtime/kernel_cache.h
 * \brief 定义 adaptive runtime、shape predictor、kernel cache 和后台编译器。
 */

#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>

#include "api/compiler.h"
#include "runtime/shape_predictor.h"

namespace kxc {
namespace runtime {

/*! \brief Adaptive kernel 缓存，支持精确 shape 匹配和向上兼容的模糊匹配。 */
class KernelCache {
public:
    /*! \brief 精确查找 shape 完全一致的 compiled module。 */
    api::CompiledModule* GetExact(const ShapeSignature& sig);

    /*! \brief 查找可兼容输入 shape 的最近 compiled module，用于 fallback 执行。 */
    api::CompiledModule* GetFuzzy(const ShapeSignature& sig);

    /*! \brief 插入新编译完成的 kernel。 */
    void Put(const ShapeSignature& sig, api::CompiledModule module);

    /*! \brief 判断缓存中是否已有指定 shape 的 kernel。 */
    bool Has(const ShapeSignature& sig) const;

    /*! \brief 返回缓存中 kernel 的数量。 */
    size_t Size() const;

    /*! \brief 清空缓存中所有 kernel。 */
    void Clear();

private:
    mutable std::mutex mu_;
    std::unordered_map<ShapeSignature, api::CompiledModule, ShapeSignatureHash> cache_;

    /*! \brief 计算两个 shape 的兼容距离，-1 表示不兼容，值越小越优。 */
    static int64_t ShapeDistance(const ShapeSignature& a, const ShapeSignature& b);
};

}  // namespace runtime
}  // namespace kxc
