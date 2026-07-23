/*! \file test/adaptive_coordinator_test.cpp
 * \brief Concurrent tests for exact singleflight, backpressure, and retry.
 */

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "kxc/compiler/adaptive.h"

namespace {

#define TEST_CHECK(condition, message)                                           \
    do {                                                                          \
        if (!(condition)) {                                                        \
            std::cerr << "[FAIL] " << __FUNCTION__ << ": " << (message) << '\n'; \
            return false;                                                         \
        }                                                                         \
    } while (0)

using namespace std::chrono_literals;
using kxc::api::adaptive::AdaptiveEvent;
using kxc::api::adaptive::AdaptiveEventKind;
using kxc::api::adaptive::ArtifactCompiler;
using kxc::api::adaptive::ArtifactExecutable;
using kxc::api::adaptive::CancellationToken;
using kxc::api::adaptive::CancelResult;
using kxc::api::adaptive::CompileAttempt;
using kxc::api::adaptive::CompileCoordinator;
using kxc::api::adaptive::CompileFailureCategory;
using kxc::api::adaptive::CompileRequest;
using kxc::api::adaptive::CompileResult;
using kxc::api::adaptive::CompileStatus;
using kxc::api::adaptive::CompileTicket;
using kxc::api::adaptive::CoordinatorOptions;
using kxc::api::adaptive::DispatchKey;
using kxc::api::adaptive::KernelArtifact;
using kxc::api::adaptive::KernelArtifactKey;
using kxc::api::adaptive::KernelSlotKey;
using kxc::api::adaptive::PlanAbiFingerprint;
using kxc::api::adaptive::RequestKind;

class FakeExecutable final : public ArtifactExecutable {
public:
    explicit FakeExecutable(std::string name) : name_(std::move(name)) {}
    bool IsReady() const noexcept override { return true; }
    std::string DebugName() const override { return name_; }

private:
    std::string name_;
};

CompileRequest Request(std::string name,
                       RequestKind kind = RequestKind::kDemand,
                       int priority = 0, std::string dispatch = "exact:f32[4]",
                       std::string abi = "abi:input-f32[4],output-f32[4]") {
    return CompileRequest(
        KernelArtifactKey(KernelSlotKey("slot:" + name),
                          "artifact:" + name),
        DispatchKey::Exact(std::move(dispatch)),
        PlanAbiFingerprint(std::move(abi)), "model@1", kind, priority);
}

std::shared_ptr<const KernelArtifact> ArtifactFor(
    const CompileRequest& request, std::size_t bytes = 4096) {
    return std::make_shared<const KernelArtifact>(
        request.artifact_key(), request.dispatch_key(), request.required_abi(),
        std::make_shared<const FakeExecutable>(
            request.artifact_key().canonical()),
        bytes, "fake-compiler");
}

class ScriptedCompiler final : public ArtifactCompiler {
public:
    CompileAttempt Compile(const CompileRequest& request,
                           const CancellationToken& cancellation) override {
        const std::string name = request.artifact_key().canonical();
        int key_attempt = 0;
        std::function<void()> callback;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            ++calls_;
            key_attempt = ++attempts_[name];
            order_.push_back(name);
            if (calls_ == 1) callback = on_first_call_;
            condition_.notify_all();
            while (blocked_ && !released_ &&
                   !cancellation.IsCancellationRequested()) {
                condition_.wait_for(lock, 1ms);
            }
        }
        if (callback) callback();
        if (cancellation.IsCancellationRequested()) {
            cancellations_.fetch_add(1, std::memory_order_relaxed);
            return CompileAttempt::Failed(CompileFailureCategory::kCancelled,
                                          "fake observed cancellation");
        }
        if (name.find("unsupported") != std::string::npos) {
            return CompileAttempt::Failed(CompileFailureCategory::kUnsupported,
                                          "fake unsupported request");
        }
        if (name.find("transient") != std::string::npos && key_attempt == 1) {
            return CompileAttempt::Failed(CompileFailureCategory::kTransient,
                                          "fake transient failure");
        }
        if (name.find("deterministic") != std::string::npos) {
            return CompileAttempt::Failed(
                CompileFailureCategory::kDeterministic,
                "fake deterministic failure");
        }
        if (name.find("invalid") != std::string::npos) {
            const CompileRequest wrong = Request("wrong");
            return CompileAttempt::Ready(ArtifactFor(wrong));
        }
        if (name.find("oversized") != std::string::npos) {
            return CompileAttempt::Ready(ArtifactFor(request, 8192));
        }
        return CompileAttempt::Ready(ArtifactFor(request));
    }

    void Block() {
        std::lock_guard<std::mutex> lock(mutex_);
        blocked_ = true;
        released_ = false;
    }

    void Release() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            released_ = true;
        }
        condition_.notify_all();
    }

    bool WaitForCalls(int expected) {
        std::unique_lock<std::mutex> lock(mutex_);
        return condition_.wait_for(lock, 2s,
                                   [&] { return calls_ >= expected; });
    }

    int calls() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return calls_;
    }

    std::vector<std::string> order() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return order_;
    }

    int cancellations() const noexcept {
        return cancellations_.load(std::memory_order_relaxed);
    }

    void SetOnFirstCall(std::function<void()> callback) {
        std::lock_guard<std::mutex> lock(mutex_);
        on_first_call_ = std::move(callback);
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    bool blocked_{false};
    bool released_{false};
    int calls_{0};
    std::map<std::string, int> attempts_;
    std::vector<std::string> order_;
    std::function<void()> on_first_call_;
    std::atomic<int> cancellations_{0};
};

class EventLog final {
public:
    void Record(const AdaptiveEvent& event) {
        std::lock_guard<std::mutex> lock(mutex_);
        events_.push_back(event);
    }

    std::size_t Count(AdaptiveEventKind kind) const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::size_t result = 0;
        for (const auto& event : events_) {
            if (event.kind == kind) ++result;
        }
        return result;
    }

    bool LifecycleEventsAreCorrelated() const {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& event : events_) {
            if ((event.kind == AdaptiveEventKind::kCompileStarted ||
                 event.kind == AdaptiveEventKind::kValidating ||
                 event.kind == AdaptiveEventKind::kReady ||
                 event.kind == AdaptiveEventKind::kFailed) &&
                (event.request_id == 0 || event.artifact_key.empty() ||
                 event.dispatch_key.empty() ||
                 event.abi_fingerprint.empty() || event.attempt == 0)) {
                return false;
            }
        }
        return true;
    }

private:
    mutable std::mutex mutex_;
    std::vector<AdaptiveEvent> events_;
};

bool TestConcurrentSameKeySingleflight() {
    auto compiler = std::make_shared<ScriptedCompiler>();
    compiler->Block();
    EventLog events;
    CoordinatorOptions options;
    options.worker_count = 2;
    options.observer = [&](const AdaptiveEvent& event) { events.Record(event); };
    CompileCoordinator coordinator(compiler, options);

    constexpr int kWaiters = 64;
    std::vector<CompileTicket> tickets(kWaiters);
    std::vector<std::thread> callers;
    std::atomic<bool> start{false};
    for (int i = 0; i < kWaiters; ++i) {
        callers.emplace_back([&, i] {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            tickets[i] = coordinator.Request(Request("singleflight"));
        });
    }
    start.store(true, std::memory_order_release);
    for (auto& caller : callers) caller.join();
    TEST_CHECK(compiler->WaitForCalls(1), "fake compiler did not start");
    TEST_CHECK(compiler->calls() == 1,
               "same full key must invoke the compiler exactly once");

    compiler->Release();
    std::shared_ptr<const KernelArtifact> first;
    for (const auto& ticket : tickets) {
        const CompileResult result = ticket.Get();
        TEST_CHECK(result.ready(), "every merged waiter should become ready");
        if (!first) first = result.artifact();
        TEST_CHECK(result.artifact() == first,
                   "merged waiters should share one immutable artifact");
    }
    const auto snapshot = coordinator.Snapshot();
    TEST_CHECK(snapshot.compile_attempts == 1 &&
                   snapshot.merged == kWaiters - 1 && snapshot.ready == 1,
               "singleflight metrics are inconsistent");
    TEST_CHECK(events.Count(AdaptiveEventKind::kRequestMerged) ==
                   kWaiters - 1 && events.LifecycleEventsAreCorrelated(),
               "merge and lifecycle events should remain request-correlated");
    return true;
}

bool TestFullKeySeparationAndReadyReuse() {
    auto compiler = std::make_shared<ScriptedCompiler>();
    CoordinatorOptions options;
    options.max_cached_artifact_bytes = 4096;
    CompileCoordinator coordinator(compiler, options);
    const CompileRequest first = Request("separate", RequestKind::kDemand, 0,
                                         "exact:f32[4]", "abi:v1");
    const CompileRequest different_dispatch =
        Request("separate", RequestKind::kDemand, 0, "exact:f32[8]",
                "abi:v1");
    const CompileRequest different_abi =
        Request("separate", RequestKind::kDemand, 0, "exact:f32[4]",
                "abi:v2");

    const auto first_result = coordinator.Request(first).Get();
    const auto cached_result = coordinator.Request(first).Get();
    const auto dispatch_result = coordinator.Request(different_dispatch).Get();
    const auto abi_result = coordinator.Request(different_abi).Get();
    TEST_CHECK(first_result.ready() && cached_result.ready() &&
                   dispatch_result.ready() && abi_result.ready(),
               "all exact requests should compile successfully");
    TEST_CHECK(compiler->calls() == 3 &&
                   first_result.artifact() == cached_result.artifact(),
               "ready reuse must use full artifact/dispatch/ABI equality");
    TEST_CHECK(coordinator.Snapshot().terminal_records == 1 &&
                   coordinator.Snapshot().cached_artifact_bytes == 4096,
               "terminal artifact retention must honor aggregate byte budget");
    return true;
}

bool TestBoundedQueueAndPriority() {
    auto compiler = std::make_shared<ScriptedCompiler>();
    compiler->Block();
    CoordinatorOptions options;
    options.worker_count = 1;
    options.max_queue_size = 2;
    CompileCoordinator coordinator(compiler, options);

    CompileTicket active = coordinator.Request(Request("active"));
    TEST_CHECK(compiler->WaitForCalls(1), "active compile did not start");
    CompileTicket prewarm = coordinator.Request(
        Request("prewarm", RequestKind::kPrewarm, -100));
    CompileTicket demand = coordinator.Request(
        Request("demand", RequestKind::kDemand, 1));
    CompileTicket displaced_demand = coordinator.Request(
        Request("overflow", RequestKind::kDemand, 100));
    const CompileResult rejected = coordinator.Request(
        Request("late-prewarm", RequestKind::kPrewarm, 100)).Get();
    TEST_CHECK(prewarm.Get().status() == CompileStatus::kRejected &&
                   rejected.status() == CompileStatus::kRejected &&
                   rejected.failure_category() ==
                       CompileFailureCategory::kBackpressure,
               "demand should displace prewarm, then full queue rejects explicitly");

    compiler->Release();
    TEST_CHECK(active.Get().ready() && demand.Get().ready() &&
                   displaced_demand.Get().ready(),
               "admitted demand work should complete");
    const auto order = compiler->order();
    TEST_CHECK(order.size() == 3 && order[0] == "artifact:active" &&
                   order[1] == "artifact:overflow" &&
                   order[2] == "artifact:demand",
               "higher-priority demand must run before queued demand");
    TEST_CHECK(coordinator.Snapshot().rejected == 2,
               "prewarm displacement and rejection should be observable");
    return true;
}

bool TestWaiterAndQueuedCancellation() {
    auto compiler = std::make_shared<ScriptedCompiler>();
    compiler->Block();
    CoordinatorOptions options;
    options.worker_count = 1;
    CompileCoordinator coordinator(compiler, options);

    CompileTicket first = coordinator.Request(Request("cancel-joined"));
    TEST_CHECK(compiler->WaitForCalls(1), "joined compile did not start");
    CompileTicket survivor = coordinator.Request(Request("cancel-joined"));
    CompileTicket queued = coordinator.Request(Request("cancel-queued"));
    TEST_CHECK(coordinator.Cancel(first.request_id()) == CancelResult::kCancelled,
               "one merged waiter should cancel");
    TEST_CHECK(coordinator.Cancel(queued.request_id()) == CancelResult::kCancelled,
               "last queued waiter should cancel its work");
    TEST_CHECK(first.Get().status() == CompileStatus::kCancelled &&
                   queued.Get().status() == CompileStatus::kCancelled,
               "cancelled tickets should complete deterministically");

    compiler->Release();
    TEST_CHECK(survivor.Get().ready(),
               "cancelling one waiter must not cancel the shared compile");
    TEST_CHECK(compiler->calls() == 1,
               "cancelled queued work must not reach the compiler");
    TEST_CHECK(coordinator.Cancel(first.request_id()) ==
                   CancelResult::kAlreadyCompleted,
               "completed cancellation should be reported deterministically");
    return true;
}

bool TestFreshRequestAfterLastActiveCancellation() {
    auto compiler = std::make_shared<ScriptedCompiler>();
    compiler->Block();
    CoordinatorOptions options;
    options.worker_count = 2;
    CompileCoordinator coordinator(compiler, options);

    CompileTicket abandoned =
        coordinator.Request(Request("cancel-replacement"));
    TEST_CHECK(compiler->WaitForCalls(1), "abandoned compile did not start");
    TEST_CHECK(coordinator.Cancel(abandoned.request_id()) ==
                   CancelResult::kCancelled,
               "last active waiter should cancel");
    CompileTicket replacement =
        coordinator.Request(Request("cancel-replacement"));
    TEST_CHECK(compiler->WaitForCalls(2),
               "fresh request must not merge into abandoned active flight");
    compiler->Release();
    TEST_CHECK(abandoned.Get().status() == CompileStatus::kCancelled &&
                   replacement.Get().ready() && compiler->calls() == 2,
               "replacement request should complete independently");
    return true;
}

bool TestSingleflightWaiterBudget() {
    auto compiler = std::make_shared<ScriptedCompiler>();
    compiler->Block();
    CoordinatorOptions options;
    options.max_waiters_per_flight = 2;
    CompileCoordinator coordinator(compiler, options);

    CompileTicket first = coordinator.Request(Request("waiter-budget"));
    TEST_CHECK(compiler->WaitForCalls(1), "budget compile did not start");
    CompileTicket second = coordinator.Request(Request("waiter-budget"));
    const CompileResult rejected =
        coordinator.Request(Request("waiter-budget")).Get();
    TEST_CHECK(rejected.status() == CompileStatus::kRejected &&
                   rejected.failure_category() ==
                       CompileFailureCategory::kBackpressure,
               "singleflight waiter growth must be bounded");
    compiler->Release();
    TEST_CHECK(first.Get().ready() && second.Get().ready() &&
                   coordinator.Snapshot().waiters == 0,
               "admitted waiters should finish and release accounting");
    return true;
}

bool TestNegativeCacheAndTransientRetry() {
    auto compiler = std::make_shared<ScriptedCompiler>();
    std::atomic<std::int64_t> now_ms{0};
    CoordinatorOptions options;
    options.retry_policy.max_transient_attempts = 2;
    options.retry_policy.initial_backoff = 10ms;
    options.retry_policy.max_backoff = 10ms;
    options.now = [&] {
        return std::chrono::steady_clock::time_point(
            std::chrono::milliseconds(now_ms.load(std::memory_order_relaxed)));
    };
    CompileCoordinator coordinator(compiler, options);

    const CompileRequest transient = Request("transient");
    const CompileResult first = coordinator.Request(transient).Get();
    const CompileResult negative_hit = coordinator.Request(transient).Get();
    TEST_CHECK(first.status() == CompileStatus::kFailed && first.retryable() &&
                   negative_hit.status() == CompileStatus::kFailed &&
                   compiler->calls() == 1,
               "transient failure should be negatively cached until retry_after");
    now_ms.store(10, std::memory_order_relaxed);
    const CompileResult retried = coordinator.Request(transient).Get();
    TEST_CHECK(retried.ready() && retried.attempt() == 2 &&
                   compiler->calls() == 2,
               "transient failure should retry once only after backoff");

    const CompileRequest unsupported = Request("unsupported");
    const CompileResult unsupported_first =
        coordinator.Request(unsupported).Get();
    now_ms.store(100000, std::memory_order_relaxed);
    const CompileResult unsupported_cached =
        coordinator.Request(unsupported).Get();
    TEST_CHECK(unsupported_first.failure_category() ==
                       CompileFailureCategory::kUnsupported &&
                   !unsupported_first.retryable() &&
                   unsupported_cached.status() == CompileStatus::kFailed &&
                   compiler->calls() == 3,
               "unsupported failure must not cause a retry storm");
    return true;
}

bool TestValidationFailureAndArtifactBudget() {
    auto compiler = std::make_shared<ScriptedCompiler>();
    CoordinatorOptions options;
    options.max_artifact_bytes = 4096;
    CompileCoordinator coordinator(compiler, options);

    const CompileResult invalid = coordinator.Request(Request("invalid")).Get();
    const CompileResult oversized =
        coordinator.Request(Request("oversized")).Get();
    TEST_CHECK(invalid.status() == CompileStatus::kFailed &&
                   invalid.failure_category() ==
                       CompileFailureCategory::kValidation &&
                   oversized.status() == CompileStatus::kFailed &&
                   oversized.failure_category() ==
                       CompileFailureCategory::kValidation,
               "mismatched or oversized artifacts must fail validation");
    TEST_CHECK(coordinator.Request(Request("invalid")).Get().status() ==
                       CompileStatus::kFailed &&
                   compiler->calls() == 2,
               "validation failures should be negatively cached");
    return true;
}

bool TestWorkerAndObserverShutdownReentry() {
    auto compiler = std::make_shared<ScriptedCompiler>();
    EventLog events;
    CompileCoordinator* coordinator_ptr = nullptr;
    std::atomic<bool> observer_reentered{false};
    CoordinatorOptions options;
    options.observer = [&](const AdaptiveEvent& event) {
        events.Record(event);
        if (event.kind == AdaptiveEventKind::kShutdownStarted &&
            !observer_reentered.exchange(true, std::memory_order_relaxed)) {
            coordinator_ptr->Shutdown();
        }
    };
    CompileCoordinator coordinator(compiler, options);
    coordinator_ptr = &coordinator;
    compiler->SetOnFirstCall([&] { coordinator.Shutdown(); });

    const CompileResult stopped =
        coordinator.Request(Request("worker-stop")).Get();
    TEST_CHECK(stopped.status() == CompileStatus::kCancelled,
               "worker-requested stop should cancel its ticket");
    coordinator.Shutdown();
    TEST_CHECK(observer_reentered.load(std::memory_order_relaxed) &&
                   events.Count(AdaptiveEventKind::kShutdownStarted) == 1 &&
                   events.Count(AdaptiveEventKind::kShutdownCompleted) == 1,
               "worker and observer shutdown reentry must not self-join or deadlock");
    return true;
}

bool TestDeterministicShutdownAndRejection() {
    auto compiler = std::make_shared<ScriptedCompiler>();
    compiler->Block();
    EventLog events;
    CoordinatorOptions options;
    options.worker_count = 1;
    options.observer = [&](const AdaptiveEvent& event) { events.Record(event); };
    CompileCoordinator coordinator(compiler, options);

    CompileTicket active = coordinator.Request(Request("shutdown-active"));
    TEST_CHECK(compiler->WaitForCalls(1), "shutdown compile did not start");
    CompileTicket queued = coordinator.Request(Request("shutdown-queued"));
    coordinator.Shutdown();
    TEST_CHECK(active.Get().status() == CompileStatus::kCancelled &&
                   queued.Get().status() == CompileStatus::kCancelled &&
                   compiler->cancellations() == 1,
               "shutdown must cancel queued and cooperative active work");
    const CompileResult rejected =
        coordinator.Request(Request("after-shutdown")).Get();
    TEST_CHECK(rejected.status() == CompileStatus::kRejected &&
                   rejected.failure_category() ==
                       CompileFailureCategory::kShutdown,
               "shutdown coordinator must reject new work");
    TEST_CHECK(!coordinator.Snapshot().accepting &&
                   events.Count(AdaptiveEventKind::kShutdownStarted) == 1 &&
                   events.Count(AdaptiveEventKind::kShutdownCompleted) == 1,
               "shutdown state and events should be deterministic");
    return true;
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"concurrent_same_key_singleflight",
         TestConcurrentSameKeySingleflight},
        {"full_key_separation_and_ready_reuse",
         TestFullKeySeparationAndReadyReuse},
        {"bounded_queue_and_priority", TestBoundedQueueAndPriority},
        {"waiter_and_queued_cancellation",
         TestWaiterAndQueuedCancellation},
        {"fresh_request_after_last_active_cancellation",
         TestFreshRequestAfterLastActiveCancellation},
        {"singleflight_waiter_budget", TestSingleflightWaiterBudget},
        {"negative_cache_and_transient_retry",
         TestNegativeCacheAndTransientRetry},
        {"validation_failure_and_artifact_budget",
         TestValidationFailureAndArtifactBudget},
        {"worker_and_observer_shutdown_reentry",
         TestWorkerAndObserverShutdownReentry},
        {"deterministic_shutdown_and_rejection",
         TestDeterministicShutdownAndRejection},
    };

    int failures = 0;
    for (const auto& test : tests) {
        try {
            if (!test.second()) {
                ++failures;
                continue;
            }
            std::cout << "[PASS] " << test.first << '\n';
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << test.first << ": " << error.what()
                      << '\n';
            ++failures;
        }
    }
    return failures == 0 ? 0 : 1;
}
