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
// - 同步作用域只覆盖实际提交线程；异步完成只记录捕获的关联，不迁移 TLS。

#pragma once

#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
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

    runtime::ExecutionScopeExit OnScopeEnter(
        const runtime::ExecutionRunCorrelation& correlation,
        const runtime::KernelSubmitInfo* kernel) override {
        return EnterScope(correlation, kernel, nullptr);
    }

    runtime::ExecutionScopeExit OnCopyScopeEnter(
        const runtime::ExecutionRunCorrelation& correlation,
        const runtime::CopyInfo& copy) override {
        return EnterScope(correlation, nullptr, &copy);
    }

private:
    runtime::ExecutionScopeExit EnterScope(
        const runtime::ExecutionRunCorrelation& correlation,
        const runtime::KernelSubmitInfo* kernel, const runtime::CopyInfo* copy) {
        if (!context_) return {};
        try {
            const bool own_context = CurrentContext() == context_;
            const runtime::ExecutionRunCorrelation effective{
                correlation.run_id.empty() && own_context ? CurrentRunId() : correlation.run_id,
                correlation.span_id.empty() && own_context ? CurrentSpanId() : correlation.span_id};
            auto scope = std::make_shared<ActiveScope>(context_, effective);
            if (kernel) {
                EventSpec spec;
                spec.component = "execution_plan";
                spec.event_type = "kernel_launch";
                spec.device = kernel->device.ToString();
                spec.kernel_symbol = kernel->kernel_symbol;
                spec.fields["timing"] = "host_submit";
                spec.fields["call_index"] = std::to_string(kernel->call_index);
                AddRunMetadata(&spec, LookupMetadata(effective.run_id));
                scope->launch.emplace(context_, std::move(spec), effective.run_id, effective.span_id);
            }
            if (copy) {
                EventSpec spec;
                spec.component = "device_api";
                spec.event_type = "copy_launch";
                spec.device = copy->to_device.ToString();
                spec.fields["timing"] = "host_submit";
                spec.fields["from_device"] = copy->from_device.ToString();
                spec.fields["to_device"] = copy->to_device.ToString();
                spec.fields["submitted_async"] = copy->submitted_async ? "true" : "false";
                spec.metrics["bytes"] = static_cast<double>(copy->bytes);
                AddRunMetadata(&spec, LookupMetadata(effective.run_id));
                scope->launch.emplace(context_, std::move(spec), effective.run_id, effective.span_id);
                PendingCopyIds().push_back({context_.get(), scope->launch->span_id(), effective});
            }
            return [scope = std::move(scope)]() mutable { scope.reset(); };
        } catch (...) {
            return {};
        }
    }

public:
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
            state->metadata = run.metadata;
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
            AddRunMetadata(&spec, state->metadata);
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
            // 用线程本地栈保存起点，提交失败由 lexical scope 清理残留起点。
            PendingKernelStarts().push_back(context_->ElapsedMonotonicNs());
        } catch (...) {
            // 观测失败不能改变执行结果。
        }
    }

    runtime::ExecutionCompletionCallback OnKernelSubmitted(
        const runtime::KernelSubmitInfo& kernel,
        const runtime::ExecutionRunCorrelation& correlation) override {
        if (!context_) return nullptr;
        const runtime::ExecutionMetadata metadata = LookupMetadata(correlation.run_id);
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
            AddRunMetadata(&spec, metadata);
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
                call_index = kernel.call_index,
                metadata](bool at_registration) {
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
                AddRunMetadata(&spec, metadata);
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
        const runtime::ExecutionMetadata metadata = LookupMetadata(correlation.run_id);
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
            AddRunMetadata(&spec, metadata);
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
            const auto effective = CopyCorrelation(correlation);
            const runtime::ExecutionMetadata metadata = LookupMetadata(effective.run_id);
            EventSpec spec;
            spec.component = "device_api";
            spec.event_type = "copy";
            spec.device = copy.to_device.ToString();
            spec.metrics["bytes"] = static_cast<double>(copy.bytes);
            spec.fields["timing"] = "host_execute";
            spec.fields["from_device"] = copy.from_device.ToString();
            spec.fields["to_device"] = copy.to_device.ToString();
            spec.fields["submitted_async"] = copy.submitted_async ? "true" : "false";
            const std::string copy_id = CurrentCopyId();
            spec.fields["copy_id"] = copy_id;
            AddRunMetadata(&spec, metadata);
            if (!copy.error_message.empty()) {
                spec.status = "error";
                spec.message = copy.error_message;
            }
            const std::int64_t end_ns = context_->ElapsedMonotonicNs();
            std::int64_t start_ns = end_ns - copy.duration_ns;
            if (start_ns < 0) start_ns = 0;
            context_->RecordCompletedSpan(spec, effective.run_id,
                                          context_->NextSpanId(),
                                          effective.span_id, start_ns, end_ns);
        } catch (...) {
            // 观测失败不能改变执行结果。
        }
    }

    runtime::ExecutionCompletionCallback OnCopySubmitted(
        const runtime::CopyInfo& copy,
        const runtime::ExecutionRunCorrelation& correlation) override {
        if (!context_) return nullptr;
        const auto effective = CopyCorrelation(correlation);
        const runtime::ExecutionMetadata metadata = LookupMetadata(effective.run_id);
        const std::int64_t submit_ns = context_->ElapsedMonotonicNs();
        const std::string copy_id = CurrentCopyId();
        try {
            EventSpec spec;
            spec.component = "device_api";
            spec.event_type = "copy";
            spec.phase = "submit";
            spec.device = copy.to_device.ToString();
            spec.metrics["bytes"] = static_cast<double>(copy.bytes);
            spec.fields["timing"] = "host_submit";
            spec.fields["copy_id"] = copy_id;
            spec.fields["from_device"] = copy.from_device.ToString();
            spec.fields["to_device"] = copy.to_device.ToString();
            AddRunMetadata(&spec, metadata);
            // Keep empty correlation explicit when another profile is active.
            const ActivationScope scope(context_,effective.run_id);
            context_->RecordInstant(spec, effective.run_id,effective.span_id);
        } catch (...) {
            // 观测失败不能改变执行结果。
        }
        auto context = context_;
        return [context, submit_ns, copy_id, duration_ns = copy.duration_ns, bytes = copy.bytes,
                from_device = copy.from_device.ToString(),
                to_device = copy.to_device.ToString(),
                run_id = effective.run_id,
                parent_span_id = effective.span_id,
                metadata](bool at_registration) {
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
                spec.fields["copy_id"] = copy_id;
                AddRunMetadata(&spec, metadata);
                // CPU CopyDataAsync already executed before this callback was
                // registered. Report that measured work, not observer overhead.
                const std::int64_t end_ns = at_registration
                    ? submit_ns : context->ElapsedMonotonicNs();
                const std::int64_t start_ns = at_registration
                    ? (duration_ns < submit_ns ? submit_ns - duration_ns : 0) : submit_ns;
                context->RecordCompletedSpan(spec, run_id, context->NextSpanId(),
                                             parent_span_id, start_ns, end_ns);
            } catch (...) {
                // 观测失败不能改变执行结果。
            }
        };
    }

private:
    struct ActiveScope {
        ActivationScope activation;
        std::optional<ScopedSpan> launch;
        std::size_t kernel_depth{PendingKernelStarts().size()};
        std::size_t copy_depth{PendingCopyIds().size()};
        int exceptions{std::uncaught_exceptions()};
        ActiveScope(const std::shared_ptr<ProfileContext>& context,
                    const runtime::ExecutionRunCorrelation& correlation)
            : activation(context, correlation.run_id, correlation.span_id) {}
        ~ActiveScope() {
            if (PendingKernelStarts().size() > kernel_depth)
                PendingKernelStarts().resize(kernel_depth);
            if (PendingCopyIds().size() > copy_depth)
                PendingCopyIds().resize(copy_depth);
            if (launch && std::uncaught_exceptions() > exceptions) {
                try { launch->SetStatus("error"); } catch (...) {}
            }
        }
    };

    struct ActiveCopy {
        const ProfileContext* context;
        std::string id;
        runtime::ExecutionRunCorrelation correlation;
    };

    static std::vector<ActiveCopy>& PendingCopyIds() {
        thread_local std::vector<ActiveCopy> ids;
        return ids;
    }

    const ActiveCopy* CurrentCopy() const {
        const auto& ids = PendingCopyIds();
        if (!ids.empty() && ids.back().context == context_.get() &&
            ids.back().id == CurrentSpanId()) return &ids.back();
        return nullptr;
    }

    std::string CurrentCopyId() {
        if (const auto* copy = CurrentCopy()) return copy->id;
        return context_->NextSpanId();
    }

    runtime::ExecutionRunCorrelation CopyCorrelation(
        const runtime::ExecutionRunCorrelation& correlation) const {
        // The copy launch is a child of the caller. Preserve the caller for
        // submit/completion even when the runtime supplied empty correlation.
        if (const auto* copy = CurrentCopy()) return copy->correlation;
        const bool own_context = CurrentContext() == context_;
        return {correlation.run_id.empty() && own_context ? CurrentRunId() : correlation.run_id,
                correlation.span_id.empty() && own_context ? CurrentSpanId() : correlation.span_id};
    }

    /*! \brief 将调用方字段附加到 runtime 事件，保留运行时保留字段。 */
    static void AddRunMetadata(EventSpec* spec,
                               const runtime::ExecutionMetadata& metadata) {
        for (const auto& item : metadata) {
            spec->fields.emplace(item.first, item.second);
        }
    }

    runtime::ExecutionMetadata LookupMetadata(const std::string& run_id) {
        if (run_id.empty()) return {};
        std::lock_guard<std::mutex> lock(mu_);
        const auto it = active_runs_.find(run_id);
        return it == active_runs_.end() ? runtime::ExecutionMetadata{} : it->second->metadata;
    }

    /*! \brief 一次运行的在途状态；OnRunEnd 依据 run id 找回并关闭 span。 */
    struct RunState {
        std::string run_id;
        std::string span_id;
        std::string device;
        std::int64_t start_ns{0};
        runtime::ExecutionMetadata metadata;
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
