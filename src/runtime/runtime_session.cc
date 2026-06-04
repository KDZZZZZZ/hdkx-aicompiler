/*! \file src/runtime/runtime_session.cc
 * \brief 实现 adaptive runtime、kernel cache、shape 统计和后台编译。
 */

#include "runtime/runtime_session.h"

#include <algorithm>
#include <sstream>

#include "base/profiling.h"

namespace kxc {
namespace runtime {

namespace {

profiling::EventSpec MakeRuntimeSpec(const std::string& event_type) {
    profiling::EventSpec spec;
    spec.component = "runtime_session";
    spec.event_type = event_type;
    return spec;
}

}  // namespace

RuntimeSession::RuntimeSession(Function relay_func, api::CompileConfig config,
                               std::shared_ptr<profiling::ProfileContext> profile_context)
    : relay_func_(relay_func), config_(config), profile_context_(std::move(profile_context)) {
    int num_threads = config->background_threads;
    if (num_threads == 0) {
        num_threads = std::max(1u, std::thread::hardware_concurrency() / 2);
    }
    bg_compiler_ = std::make_unique<BackgroundCompiler>(num_threads);
    internal_config_ = api::CompileConfig::AOT(config->target, config->opt_level);
    internal_config_->profile_options = config->profile_options;
}

RuntimeSession::~RuntimeSession() {
    if (bg_compiler_) {
        bg_compiler_->WaitAll();
    }
}

void RuntimeSession::Run(const std::vector<void*>& args,
                         const std::vector<std::vector<int64_t>>& input_shapes) {
    profiling::EventSpec session_spec = MakeRuntimeSpec("runtime_session_run");
    session_spec.shape_signature = profiling::ShapeSignatureToString(input_shapes);
    profiling::ScopedSpan span(profile_context_, std::move(session_spec));

    ShapeSignature sig = MakeShapeSignature(input_shapes);
    predictor_.Record(sig);
    span.AddField("shape_hash", std::to_string(sig.Hash()));
    span.AddMetric("shape_seen_count", static_cast<double>(predictor_.GetCount(sig)));

    auto* exact_kernel = cache_.GetExact(sig);
    if (exact_kernel) {
        profiling::ScopedSpan hit_span(profile_context_, MakeRuntimeSpec("cache_exact_hit"));
        hit_span.AddField("shape_hash", std::to_string(sig.Hash()));
        exact_kernel->Run(args);
        return;
    }

    auto* fuzzy_kernel = cache_.GetFuzzy(sig);
    if (fuzzy_kernel) {
        profiling::ScopedSpan hit_span(profile_context_, MakeRuntimeSpec("cache_fuzzy_hit"));
        hit_span.AddField("shape_hash", std::to_string(sig.Hash()));
        fuzzy_kernel->Run(args);
        MaybeScheduleOptimization(sig);
        return;
    }

    profiling::ScopedSpan miss_span(profile_context_, MakeRuntimeSpec("cache_miss_sync_compile"));
    miss_span.AddField("shape_hash", std::to_string(sig.Hash()));
    auto module = api::Compiler::Compile(relay_func_, internal_config_);
    cache_.Put(sig, module);

    auto module_ptr = cache_.GetExact(sig);
    module_ptr->Run(args);
    MaybeScheduleOptimization(sig);
}

void RuntimeSession::WarmUp(const std::vector<std::vector<int64_t>>& input_shapes) {
    profiling::EventSpec spec = MakeRuntimeSpec("runtime_session_warmup");
    spec.shape_signature = profiling::ShapeSignatureToString(input_shapes);
    profiling::ScopedSpan span(profile_context_, std::move(spec));

    ShapeSignature sig = MakeShapeSignature(input_shapes);
    span.AddField("shape_hash", std::to_string(sig.Hash()));
    if (cache_.Has(sig)) {
        span.AddField("warmup_cache_status", "already_cached");
        return;
    }

    auto module = api::Compiler::Compile(relay_func_, internal_config_);
    cache_.Put(sig, std::move(module));
}

void RuntimeSession::WaitAll() {
    profiling::ScopedSpan span(profile_context_, MakeRuntimeSpec("runtime_session_wait_all"));
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
        << ", total_runs=" << predictor_.TotalCount() << "}";
    return oss.str();
}

void RuntimeSession::MaybeScheduleOptimization(const ShapeSignature& sig) {
    const int observed_count = predictor_.GetCount(sig);
    const bool should_compile =
        predictor_.ShouldCompile(sig, /*count_threshold=*/2, /*ratio=*/0.05);
    if (!should_compile) {
        return;
    }

    size_t sig_hash = sig.Hash();
    {
        std::lock_guard<std::mutex> lock(schedule_mu_);
        if (submitted_shapes_.count(sig_hash)) {
            return;
        }
        submitted_shapes_.insert(sig_hash);
    }

    BackgroundCompiler::CompileTask task;
    task.relay_func = relay_func_;
    task.config = internal_config_;
    task.target_shape = sig;
    task.priority = observed_count;
    task.profile_context = profile_context_;
    task.run_id = profiling::CurrentRunId();
    task.parent_span_id = profiling::CurrentSpanId();
    task.on_complete = [this, sig, run_id = task.run_id,
                        parent_span_id = task.parent_span_id](api::CompiledModule module) {
        cache_.Put(sig, std::move(module));
        if (profile_context_) {
            profiling::EventSpec spec = MakeRuntimeSpec("background_compile_completed");
            spec.shape_signature = profiling::ShapeSignatureToString(sig.input_shapes);
            spec.fields["shape_hash"] = std::to_string(sig.Hash());
            profile_context_->RecordInstant(std::move(spec), run_id, parent_span_id);
        }
    };

    if (profile_context_) {
        profiling::EventSpec submit_spec = MakeRuntimeSpec("background_compile_submitted");
        submit_spec.shape_signature = profiling::ShapeSignatureToString(sig.input_shapes);
        submit_spec.fields["shape_hash"] = std::to_string(sig.Hash());
        submit_spec.metrics["priority"] = static_cast<double>(task.priority);
        submit_spec.metrics["seen_count"] = static_cast<double>(observed_count);
        profile_context_->RecordInstant(std::move(submit_spec), task.run_id, task.parent_span_id);
    }

    bg_compiler_->Submit(std::move(task));
}

}  // namespace runtime
}  // namespace kxc
