/*! \file include/runtime/background_compiler.h
 * \brief 定义 adaptive runtime、shape predictor、kernel cache 和后台编译器。
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "api/compile_config.h"
#include "api/compiler.h"
#include "base/profiling.h"
#include "relay/relay.h"
#include "runtime/shape_predictor.h"

namespace kxc {
namespace runtime {

/*! \brief 后台编译器，消费优先级队列中的 shape 优化任务。 */
class BackgroundCompiler {
public:
    /*! \brief 单个后台编译任务，包含 Relay 函数、目标 shape 和完成回调。 */
    struct CompileTask {
        Function relay_func;
        api::CompileConfig config;
        ShapeSignature target_shape;
        int priority{0};
        std::shared_ptr<profiling::ProfileContext> profile_context;
        std::string run_id;
        std::string parent_span_id;
        std::function<void(api::CompiledModule)> on_complete;

        bool operator<(const CompileTask& other) const { return priority < other.priority; }
    };

    /*! \brief 创建后台编译器；num_threads 为 0 时由实现选择线程数。 */
    explicit BackgroundCompiler(int num_threads = 0);
    ~BackgroundCompiler();

    /*! \brief 提交一个后台编译任务。 */
    void Submit(CompileTask task);
    /*! \brief 等待队列和当前执行中的任务全部完成。 */
    void WaitAll();

    /*! \brief 返回排队中和执行中的任务数量。 */
    int PendingCount() const;
    /*! \brief 返回已完成任务数量。 */
    int CompletedCount() const { return completed_count_.load(); }

    BackgroundCompiler(const BackgroundCompiler&) = delete;
    BackgroundCompiler& operator=(const BackgroundCompiler&) = delete;

private:
    void WorkerLoop();

    std::vector<std::thread> workers_;
    std::priority_queue<CompileTask> queue_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::condition_variable done_cv_;
    std::atomic<bool> shutdown_{false};
    std::atomic<int> active_tasks_{0};
    std::atomic<int> completed_count_{0};
};

}  // namespace runtime
}  // namespace kxc
