#pragma once

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "api/compile_config.h"
#include "relay/relay.h"
#include "runtime/background_compiler.h"
#include "runtime/kernel_cache.h"
#include "runtime/kernel_runner.h"
#include "runtime/shape_predictor.h"

namespace kxc {
namespace runtime {

// 自适应运行时会话
// 管理kernel生命周期、shape预测、后台编译、热替换
class RuntimeSession {
public:
    explicit RuntimeSession(Function relay_func, api::CompileConfig config);
    ~RuntimeSession();

    // 主运行接口：自动选择最优kernel
    void Run(const std::vector<void*>& args,
             const std::vector<std::vector<int64_t>>& input_shapes);

    // 预热：提前为指定shape编译
    void WarmUp(const std::vector<std::vector<int64_t>>& input_shapes);

    // 等待所有后台编译完成
    void WaitAll();

    // 查询状态
    std::string GetStatus() const;
    int CachedKernelCount() const { return cache_.Size(); }
    int PendingCompileCount() const { return bg_compiler_->PendingCount(); }

private:
    // 调度优化编译
    void MaybeScheduleOptimization(const ShapeSignature& sig);

    Function relay_func_;
    api::CompileConfig config_;
    api::CompileConfig internal_config_;  // 内部编译用AOT模式

    KernelRunner runner_;
    KernelCache cache_;
    ShapePredictor predictor_;
    std::unique_ptr<BackgroundCompiler> bg_compiler_;

    // 已提交编译的shape（避免重复提交）
    std::unordered_set<size_t> submitted_shapes_;
    mutable std::mutex schedule_mu_;
};

}  // namespace runtime
}  // namespace kxc
