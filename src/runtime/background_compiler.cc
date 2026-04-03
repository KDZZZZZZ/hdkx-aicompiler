#include "runtime/background_compiler.h"

#include <iostream>

namespace kxc {
namespace runtime {

BackgroundCompiler::BackgroundCompiler(int num_threads) {
    if (num_threads <= 0) {
        num_threads = std::max(1u, std::thread::hardware_concurrency() / 2);
    }
    for (int i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this] { WorkerLoop(); });
    }
}

BackgroundCompiler::~BackgroundCompiler() {
    shutdown_.store(true);
    cv_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
}

void BackgroundCompiler::Submit(CompileTask task) {
    {
        std::lock_guard<std::mutex> lock(mu_);
        queue_.push(std::move(task));
        active_tasks_++;
    }
    cv_.notify_one();
}

void BackgroundCompiler::WaitAll() {
    std::unique_lock<std::mutex> lock(mu_);
    done_cv_.wait(lock, [this] {
        return active_tasks_.load() == 0 && queue_.empty();
    });
}

int BackgroundCompiler::PendingCount() const {
    std::lock_guard<std::mutex> lock(mu_);
    return static_cast<int>(queue_.size());
}

void BackgroundCompiler::WorkerLoop() {
    while (true) {
        CompileTask task;
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait(lock, [this] {
                return !queue_.empty() || shutdown_.load();
            });
            if (shutdown_.load() && queue_.empty()) return;
            task = std::move(const_cast<CompileTask&>(queue_.top()));
            queue_.pop();
        }

        // 执行编译
        try {
            auto module = api::Compiler::Compile(task.relay_func, task.config);
            if (task.on_complete) {
                task.on_complete(std::move(module));
            }
            completed_count_++;
        } catch (const std::exception& e) {
            std::cerr << "[BackgroundCompiler] Compilation failed: " << e.what() << std::endl;
        }

        active_tasks_--;
        done_cv_.notify_all();
    }
}

}  // namespace runtime
}  // namespace kxc
