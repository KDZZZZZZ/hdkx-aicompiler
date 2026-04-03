#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "api/compile_config.h"
#include "api/compiler.h"
#include "relay/relay.h"
#include "runtime/shape_predictor.h"

namespace kxc {
namespace runtime {

// 后台编译器：非阻塞提交任务，后台线程池编译
class BackgroundCompiler {
public:
    struct CompileTask {
        Function relay_func;             // 要编译的Relay函数
        api::CompileConfig config;       // 编译配置
        ShapeSignature target_shape;     // 目标shape
        int priority{0};                 // 越大越优先

        // 编译完成回调（在工作线程中调用，必须线程安全）
        std::function<void(api::CompiledModule)> on_complete;

        bool operator<(const CompileTask& other) const {
            return priority < other.priority;  // min-heap → 大的先出
        }
    };

    explicit BackgroundCompiler(int num_threads = 0);
    ~BackgroundCompiler();

    // 提交编译任务（非阻塞）
    void Submit(CompileTask task);

    // 等待所有任务完成
    void WaitAll();

    // 查询状态
    int PendingCount() const;
    int CompletedCount() const { return completed_count_.load(); }

    // 禁止拷贝
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
