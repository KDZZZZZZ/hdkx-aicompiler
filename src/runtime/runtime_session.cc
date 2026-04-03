#include "runtime/runtime_session.h"

#include <iostream>
#include <sstream>

namespace kxc {
namespace runtime {

RuntimeSession::RuntimeSession(Function relay_func, api::CompileConfig config)
    : relay_func_(relay_func), config_(config) {
    int num_threads = config->background_threads;
    if (num_threads == 0) {
        num_threads = std::max(1u, std::thread::hardware_concurrency() / 2);
    }
    bg_compiler_ = std::make_unique<BackgroundCompiler>(num_threads);

    // 创建内部用的AOT编译配置（RuntimeSession内部编译用AOT，外壳是Adaptive）
    internal_config_ = api::CompileConfig::AOT(config->target, config->opt_level);
}

RuntimeSession::~RuntimeSession() {
    // 等待所有后台任务完成
    if (bg_compiler_) {
        bg_compiler_->WaitAll();
    }
}

void RuntimeSession::Run(const std::vector<void*>& args,
                         const std::vector<std::vector<int64_t>>& input_shapes) {
    ShapeSignature sig = MakeShapeSignature(input_shapes);

    // 1. 记录shape统计
    predictor_.Record(sig);

    // 2. 查找精确匹配的kernel
    auto* exact_kernel = cache_.GetExact(sig);
    if (exact_kernel) {
        // 快速路径：直接执行
        exact_kernel->Run(args);
        return;
    }

    // 3. 没有精确匹配，尝试模糊匹配（fallback）
    auto* fuzzy_kernel = cache_.GetFuzzy(sig);
    if (fuzzy_kernel) {
        // 使用fallback kernel执行
        fuzzy_kernel->Run(args);
        // 同时触发后台优化编译
        MaybeScheduleOptimization(sig);
        return;
    }

    // 4. 完全没有可用kernel：同步编译（首次运行）
    std::cout << "[RuntimeSession] First run, compiling synchronously..." << std::endl;
    auto module = api::Compiler::Compile(relay_func_, internal_config_);
    cache_.Put(sig, module);

    // 设置为当前kernel并执行
    auto module_ptr = cache_.GetExact(sig);
    module_ptr->Run(args);

    // 触发后台预测编译
    MaybeScheduleOptimization(sig);
}

void RuntimeSession::WarmUp(const std::vector<std::vector<int64_t>>& input_shapes) {
    ShapeSignature sig = MakeShapeSignature(input_shapes);
    if (cache_.Has(sig)) return;

    auto module = api::Compiler::Compile(relay_func_, internal_config_);
    cache_.Put(sig, std::move(module));
}

void RuntimeSession::WaitAll() {
    if (bg_compiler_) {
        bg_compiler_->WaitAll();
    }
}

std::string RuntimeSession::GetStatus() const {
    std::ostringstream oss;
    oss << "RuntimeSession{"
        << "cached=" << cache_.Size()
        << ", pending=" << (bg_compiler_ ? bg_compiler_->PendingCount() : 0)
        << ", completed=" << (bg_compiler_ ? bg_compiler_->CompletedCount() : 0)
        << ", total_runs=" << predictor_.TotalCount()
        << "}";
    return oss.str();
}

void RuntimeSession::MaybeScheduleOptimization(const ShapeSignature& sig) {
    // 检查是否值得编译
    if (!predictor_.ShouldCompile(sig, /*count_threshold=*/2, /*ratio=*/0.05)) {
        return;
    }

    // 检查是否已提交
    size_t sig_hash = sig.Hash();
    {
        std::lock_guard<std::mutex> lock(schedule_mu_);
        if (submitted_shapes_.count(sig_hash)) return;
        submitted_shapes_.insert(sig_hash);
    }

    // 提交后台编译任务
    BackgroundCompiler::CompileTask task;
    task.relay_func = relay_func_;
    task.config = internal_config_;  // 使用AOT配置
    task.target_shape = sig;
    task.priority = predictor_.GetCount(sig);  // 频率越高优先级越高
    task.on_complete = [this, sig](api::CompiledModule module) {
        // 编译完成回调：插入缓存
        cache_.Put(sig, std::move(module));
        std::cout << "[RuntimeSession] Background compilation completed for shape" << std::endl;
    };

    bg_compiler_->Submit(std::move(task));
}

}  // namespace runtime
}  // namespace kxc
