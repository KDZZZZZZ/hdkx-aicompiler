/*! \file include/kxc/profiling/runtime_observer.h
 * \brief 把 runtime 执行观测钩子翻译成 ProfileContext 事件的适配器。
 */

// 职责简介：
// - RuntimeExecutionObserver 实现 runtime::ExecutionObserver，把执行钩子
//   写成 ProfileContext 事件：runtime_session_run span、kernel_submit
//   （phase="submit"）、kernel_exec span、alloc/copy 记账。
// - 每次运行用 NextRunId("run") 取新 run id，run span 作为本次内核与
//   分配事件的父 span；事件语义见 M1 文档，每个事件带 timing 字段。
// - 本适配器必须吞掉自身异常：观测永远不能改变执行结果。完成回调可能
//   在 run 作用域结束后、甚至别的线程触发，因此按值捕获全部关联信息。
// - 实现整体内联在本公共头中；ProfileContext 只通过公共 API 使用，
//   新增实现文件需要动共享的 CMake 源列表，超出 A 线所有权。

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kxc/profiling/profiling.h"
#include "kxc/runtime/execution_observer.h"

namespace kxc {
namespace profiling {

/*! \brief runtime 执行钩子到 ProfileContext 事件的适配器。 */
class RuntimeExecutionObserver final : public runtime::ExecutionObserver {
public:
    /*! \brief 持有 context；空 context 等价于不观测。 */
    explicit RuntimeExecutionObserver(std::shared_ptr<ProfileContext> context)
        : context_(std::move(context)) {}

    runtime::ExecutionRunCorrelation OnRunStart(
        const runtime::ExecutionRunStart& run) override {
        runtime::ExecutionRunCorrelation correlation;
        if (!context_) return correlation;
        try {
            auto state = std::make_shared<RunState>();
            state->run_id = context_->NextRunId("run");
            state->span_id = context_->NextSpanId();
            state->start_ns = context_->ElapsedMonotonicNs();
            state->device = run.device.ToString();
            {
                std::lock_guard<std::mutex> lock(mu_);
                active_runs_[state->run_id] = state;
            }
            correlation.run_id = state->run_id;
            correlation.span_id = state->span_id;
        } catch (...) {
            // 观测失败不能改变执行结果。
            correlation = runtime::ExecutionRunCorrelation{};
        }
        return correlation;
    }

    void OnRunEnd(const runtime::ExecutionRunEnd& run,
                  const runtime::ExecutionRunCorrelation& correlation) override {
        if (!context_ || correlation.run_id.empty()) return;
        try {
            std::shared_ptr<RunState> state;
            {
                std::lock_guard<std::mutex> lock(mu_);
                const auto it = active_runs_.find(correlation.run_id);
                if (it == active_runs_.end()) return;
                state = it->second;
                active_runs_.erase(it);
            }
            EventSpec spec;
            spec.component = "runtime_session";
            spec.event_type = "runtime_session_run";
            spec.device = run.device.ToString();
            spec.fields["timing"] = "host_execute";
            spec.metrics["input_count"] = static_cast<double>(run.input_count);
            spec.metrics["kernel_count"] = static_cast<double>(run.kernel_count);
            spec.metrics["submit_count"] = static_cast<double>(run.submit_count);
            if (!run.completed) {
                spec.status = "error";
                spec.message = run.error_message;
            }
            context_->RecordCompletedSpan(spec, state->run_id, state->span_id, "",
                                          state->start_ns,
                                          context_->ElapsedMonotonicNs());
        } catch (...) {
            // 观测失败不能改变执行结果。
        }
    }

    void OnKernelBegin(const runtime::KernelSubmitInfo& kernel,
                       const runtime::ExecutionRunCorrelation&) override {
        if (!context_) return;
        try {
            // 内核提交前时刻；提交路径在单线程内 begin→submitted 相邻，
            // 用线程本地栈保存起点，提交失败残留的起点由下一次 begin 覆盖。
            PendingKernelStarts().push_back(context_->ElapsedMonotonicNs());
        } catch (...) {
            // 观测失败不能改变执行结果。
        }
    }

    runtime::ExecutionCompletionCallback OnKernelSubmitted(
        const runtime::KernelSubmitInfo& kernel,
        const runtime::ExecutionRunCorrelation& correlation) override {
        if (!context_) return nullptr;
        // 主机完成一次内核提交动作的时刻；不代表执行完成。
        try {
            EventSpec spec;
            spec.component = "execution_plan";
            spec.event_type = "kernel_submit";
            spec.phase = "submit";
            spec.device = kernel.device.ToString();
            spec.kernel_symbol = kernel.kernel_symbol;
            spec.fields["timing"] = "host_submit";
            spec.fields["call_index"] = std::to_string(kernel.call_index);
            context_->RecordInstant(spec, correlation.run_id,
                                    correlation.span_id);
        } catch (...) {
            // 观测失败不能改变执行结果。
        }
        std::int64_t start_ns = 0;
        bool has_start = false;
        try {
            auto& pending = PendingKernelStarts();
            if (!pending.empty()) {
                start_ns = pending.back();
                pending.pop_back();
                has_start = true;
            }
        } catch (...) {
            has_start = false;
        }
        // 完成回调可能晚于 run 作用域在别的线程触发；全部关联信息按值捕获，
        // 不再依赖任何线程本地状态。
        if (!has_start || correlation.run_id.empty()) return nullptr;
        auto context = context_;
        return [context, start_ns, run_id = correlation.run_id,
                parent_span_id = correlation.span_id,
                kernel_symbol = kernel.kernel_symbol,
                device = kernel.device.ToString(),
                call_index = kernel.call_index](bool at_registration) {
            if (!context) return;
            try {
                EventSpec spec;
                spec.component = "execution_plan";
                spec.event_type = "kernel_exec";
                spec.device = device;
                spec.kernel_symbol = kernel_symbol;
                // 起点=主机提交前时刻，终点=主机观测到完成的时刻；同步后端
                // 该区间是主机执行区间，异步后端是提交到观测完成的等待区间。
                spec.fields["timing"] =
                    at_registration ? "host_execute" : "host_observed_complete";
                spec.fields["call_index"] = std::to_string(call_index);
                context->RecordCompletedSpan(spec, run_id, context->NextSpanId(),
                                             parent_span_id, start_ns,
                                             context->ElapsedMonotonicNs());
            } catch (...) {
                // 观测失败不能改变执行结果。
            }
        };
    }

    void OnAllocation(const runtime::AllocationInfo& allocation,
                      const runtime::ExecutionRunCorrelation& correlation) override {
        if (!context_) return;
        try {
            EventSpec spec;
            spec.component = "device_api";
            spec.event_type = "alloc";
            spec.device = allocation.device.ToString();
            spec.metrics["bytes"] = static_cast<double>(allocation.bytes);
            spec.fields["timing"] = "host_execute";
            spec.fields["alignment"] = std::to_string(allocation.alignment);
            spec.fields["reused"] =
                allocation.kind == runtime::AllocationKind::kFresh ? "false" : "true";
            switch (allocation.kind) {
                case runtime::AllocationKind::kFresh:
                    spec.fields["alloc_kind"] = "fresh";
                    break;
                case runtime::AllocationKind::kReuse:
                    spec.fields["alloc_kind"] = "reuse";
                    break;
                case runtime::AllocationKind::kAlias:
                    spec.fields["alloc_kind"] = "alias";
                    break;
            }
            if (!allocation.error_message.empty()) {
                spec.status = "error";
                spec.message = allocation.error_message;
            }
            const std::int64_t end_ns = context_->ElapsedMonotonicNs();
            std::int64_t start_ns = end_ns - allocation.duration_ns;
            if (start_ns < 0) start_ns = 0;
            context_->RecordCompletedSpan(spec, correlation.run_id,
                                          context_->NextSpanId(),
                                          correlation.span_id, start_ns, end_ns);
        } catch (...) {
            // 观测失败不能改变执行结果。
        }
    }

    void OnCopy(const runtime::CopyInfo& copy,
                const runtime::ExecutionRunCorrelation& correlation) override {
        if (!context_) return;
        try {
            EventSpec spec;
            spec.component = "device_api";
            spec.event_type = "copy";
            spec.device = copy.to_device.ToString();
            spec.metrics["bytes"] = static_cast<double>(copy.bytes);
            spec.fields["timing"] = "host_execute";
            spec.fields["from_device"] = copy.from_device.ToString();
            spec.fields["to_device"] = copy.to_device.ToString();
            spec.fields["submitted_async"] = copy.submitted_async ? "true" : "false";
            if (!copy.error_message.empty()) {
                spec.status = "error";
                spec.message = copy.error_message;
            }
            const std::int64_t end_ns = context_->ElapsedMonotonicNs();
            std::int64_t start_ns = end_ns - copy.duration_ns;
            if (start_ns < 0) start_ns = 0;
            context_->RecordCompletedSpan(spec, correlation.run_id,
                                          context_->NextSpanId(),
                                          correlation.span_id, start_ns, end_ns);
        } catch (...) {
            // 观测失败不能改变执行结果。
        }
    }

    runtime::ExecutionCompletionCallback OnCopySubmitted(
        const runtime::CopyInfo& copy,
        const runtime::ExecutionRunCorrelation& correlation) override {
        if (!context_) return nullptr;
        const std::int64_t submit_ns = context_->ElapsedMonotonicNs();
        try {
            EventSpec spec;
            spec.component = "device_api";
            spec.event_type = "copy";
            spec.phase = "submit";
            spec.device = copy.to_device.ToString();
            spec.metrics["bytes"] = static_cast<double>(copy.bytes);
            spec.fields["timing"] = "host_submit";
            spec.fields["from_device"] = copy.from_device.ToString();
            spec.fields["to_device"] = copy.to_device.ToString();
            context_->RecordInstant(spec, correlation.run_id,
                                    correlation.span_id);
        } catch (...) {
            // 观测失败不能改变执行结果。
        }
        if (correlation.run_id.empty()) return nullptr;
        auto context = context_;
        return [context, submit_ns, bytes = copy.bytes,
                from_device = copy.from_device.ToString(),
                to_device = copy.to_device.ToString(),
                run_id = correlation.run_id,
                parent_span_id = correlation.span_id](bool at_registration) {
            if (!context) return;
            try {
                EventSpec spec;
                spec.component = "device_api";
                spec.event_type = "copy";
                spec.device = to_device;
                spec.metrics["bytes"] = static_cast<double>(bytes);
                spec.fields["timing"] = at_registration ? "host_execute"
                                                        : "host_observed_complete";
                spec.fields["from_device"] = from_device;
                spec.fields["to_device"] = to_device;
                spec.fields["submitted_async"] = "true";
                context->RecordCompletedSpan(spec, run_id, context->NextSpanId(),
                                             parent_span_id, submit_ns,
                                             context->ElapsedMonotonicNs());
            } catch (...) {
                // 观测失败不能改变执行结果。
            }
        };
    }

private:
    /*! \brief 一次运行的在途状态；OnRunEnd 依据 run id 找回并关闭 span。 */
    struct RunState {
        std::string run_id;
        std::string span_id;
        std::string device;
        std::int64_t start_ns{0};
    };

    /*! \brief 本线程待提交内核的执行区间起点；begin→submitted 相邻消费。 */
    static std::vector<std::int64_t>& PendingKernelStarts() {
        thread_local std::vector<std::int64_t> starts;
        return starts;
    }

    std::shared_ptr<ProfileContext> context_;
    std::mutex mu_;
    std::unordered_map<std::string, std::shared_ptr<RunState>> active_runs_;
};

/*! \brief profiling 存在且启用时返回执行观测适配器，否则返回空。
 *  RuntimeSession 从模块继承该观测器；模块没有观测器时运行行为与
 *  未装配时完全一致。 */
inline std::shared_ptr<runtime::ExecutionObserver> MakeRuntimeExecutionObserver(
    std::shared_ptr<ProfileContext> context) {
    if (!context || !context->options().enabled) return nullptr;
    return std::make_shared<RuntimeExecutionObserver>(std::move(context));
}

}  // namespace profiling
}  // namespace kxc
