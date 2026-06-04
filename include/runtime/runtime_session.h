/*! \file include/runtime/runtime_session.h
 * \brief 定义 adaptive runtime、shape predictor、kernel cache 和后台编译器。
 */

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include "api/compile_config.h"
#include "base/profiling.h"
#include "relay/relay.h"
#include "runtime/background_compiler.h"
#include "runtime/kernel_cache.h"
#include "runtime/kernel_runner.h"
#include "runtime/shape_predictor.h"

namespace kxc {
namespace runtime {

/*! \brief Adaptive runtime 会话，负责执行、shape 统计、kernel 缓存和后台优化编译。 */
class RuntimeSession {
public:
    /*! \brief 创建会话并准备初始编译配置。 */
    explicit RuntimeSession(Function relay_func, api::CompileConfig config,
                            std::shared_ptr<profiling::ProfileContext> profile_context = nullptr);
    ~RuntimeSession();

    /*! \brief 执行一次调用，并根据输入 shape 触发缓存命中或后台优化。 */
    void Run(const std::vector<void*>& args,
             const std::vector<std::vector<int64_t>>& input_shapes);
    /*! \brief 对指定输入 shape 预热，提前建立可用 kernel。 */
    void WarmUp(const std::vector<std::vector<int64_t>>& input_shapes);
    /*! \brief 等待后台编译任务完成。 */
    void WaitAll();

    /*! \brief 返回会话、缓存和后台队列的可读状态。 */
    std::string GetStatus() const;
    /*! \brief 返回当前缓存中的 kernel 数量。 */
    int CachedKernelCount() const { return cache_.Size(); }
    /*! \brief 返回后台编译器中待处理/执行中的任务数量。 */
    int PendingCompileCount() const { return bg_compiler_ ? bg_compiler_->PendingCount() : 0; }

private:
    /*! \brief 根据 shape 频率和缓存状态决定是否提交后台优化任务。 */
    void MaybeScheduleOptimization(const ShapeSignature& sig);

    Function relay_func_;
    api::CompileConfig config_;
    api::CompileConfig internal_config_;
    std::shared_ptr<profiling::ProfileContext> profile_context_;

    KernelRunner runner_;
    KernelCache cache_;
    ShapePredictor predictor_;
    std::unique_ptr<BackgroundCompiler> bg_compiler_;

    std::unordered_set<size_t> submitted_shapes_;
    mutable std::mutex schedule_mu_;
};

}  // namespace runtime
}  // namespace kxc
