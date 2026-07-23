/*! \file src/compiler/adaptive/compile_coordinator.cc
 * \brief Bounded singleflight coordinator for static-exact compilation.
 */

#include "kxc/compiler/adaptive.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace kxc::api::adaptive {

struct CancellationToken::State final {
    std::atomic<bool> requested{false};
};

CancellationToken::CancellationToken(std::shared_ptr<State> state)
    : state_(std::move(state)) {}

bool CancellationToken::IsCancellationRequested() const noexcept {
    return state_ && state_->requested.load(std::memory_order_acquire);
}

CompileTicket::CompileTicket(std::uint64_t request_id,
                             std::shared_future<CompileResult> future)
    : request_id_(request_id), future_(std::move(future)) {}

CompileResult CompileTicket::Get() const {
    if (!future_.valid()) {
        throw std::logic_error("compile ticket is not valid");
    }
    return future_.get();
}

std::future_status CompileTicket::WaitFor(
    std::chrono::milliseconds timeout) const {
    if (!future_.valid()) {
        throw std::logic_error("compile ticket is not valid");
    }
    return future_.wait_for(timeout);
}

namespace {

struct RequestIdentity final {
    KernelArtifactKey artifact_key;
    DispatchKey dispatch_key;
    PlanAbiFingerprint required_abi;

    explicit RequestIdentity(const CompileRequest& request)
        : artifact_key(request.artifact_key()),
          dispatch_key(request.dispatch_key()),
          required_abi(request.required_abi()) {}

    friend bool operator<(const RequestIdentity& lhs,
                          const RequestIdentity& rhs) noexcept {
        if (lhs.artifact_key != rhs.artifact_key) {
            return lhs.artifact_key < rhs.artifact_key;
        }
        if (lhs.dispatch_key != rhs.dispatch_key) {
            return lhs.dispatch_key < rhs.dispatch_key;
        }
        return lhs.required_abi < rhs.required_abi;
    }
};

enum class FlightState : std::uint8_t {
    kQueued,
    kCompiling,
    kValidating,
    kDone,
};

struct Waiter final {
    explicit Waiter(std::shared_ptr<std::promise<CompileResult>> promise_value)
        : promise(std::move(promise_value)) {}
    std::shared_ptr<std::promise<CompileResult>> promise;
};

struct Flight final {
    Flight(CompileRequest request_value, std::uint32_t attempt_value,
           std::uint64_t sequence_value)
        : request(std::move(request_value)),
          identity(request),
          attempt(attempt_value),
          sequence(sequence_value),
          cancellation(std::make_shared<CancellationToken::State>()) {}

    CompileRequest request;
    RequestIdentity identity;
    std::uint32_t attempt{1};
    std::uint64_t sequence{0};
    FlightState state{FlightState::kQueued};
    bool discard{false};
    std::shared_ptr<CancellationToken::State> cancellation;
    std::map<std::uint64_t, Waiter> waiters;
};

struct TerminalRecord final {
    TerminalRecord(CompileResult result_value, std::uint64_t stamp_value)
        : result(std::move(result_value)), stamp(stamp_value) {}
    CompileResult result;
    std::uint64_t stamp{0};
};

int KindOrder(RequestKind kind) {
    switch (kind) {
        case RequestKind::kDemand:
            return 0;
        case RequestKind::kCanary:
            return 1;
        case RequestKind::kPrewarm:
            return 2;
    }
    return 3;
}

AdaptiveEvent EventFor(const Flight& flight, AdaptiveEventKind kind) {
    AdaptiveEvent event;
    event.kind = kind;
    event.slot_key = flight.request.artifact_key().slot_key().canonical();
    event.artifact_key = flight.request.artifact_key().canonical();
    event.dispatch_key = flight.request.dispatch_key().canonical();
    event.abi_fingerprint = flight.request.required_abi().canonical();
    event.model_revision = flight.request.model_revision();
    event.waiter_count = flight.waiters.size();
    return event;
}

}  // namespace

class CompileCoordinator::State final {
public:
    State(std::shared_ptr<ArtifactCompiler> compiler_value,
          CoordinatorOptions options_value)
        : compiler_(std::move(compiler_value)),
          options_(std::move(options_value)) {
        if (!compiler_) {
            throw std::invalid_argument(
                "CompileCoordinator requires an ArtifactCompiler");
        }
        if (options_.worker_count == 0 || options_.max_queue_size == 0 ||
            options_.max_terminal_records == 0 ||
            options_.max_artifact_bytes == 0) {
            throw std::invalid_argument(
                "CompileCoordinator bounds must be non-zero");
        }
        if (options_.retry_policy.max_transient_attempts == 0 ||
            options_.retry_policy.initial_backoff.count() < 0 ||
            options_.retry_policy.max_backoff <
                options_.retry_policy.initial_backoff) {
            throw std::invalid_argument(
                "CompileCoordinator retry policy is invalid");
        }
        if (!options_.now) {
            options_.now = [] { return std::chrono::steady_clock::now(); };
        }

        try {
            for (std::size_t i = 0; i < options_.worker_count; ++i) {
                workers_.emplace_back([this] { WorkerLoop(); });
            }
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stopping_ = true;
            }
            condition_.notify_all();
            for (auto& worker : workers_) {
                if (worker.joinable()) worker.join();
            }
            throw;
        }
    }

    CompileTicket Request(CompileRequest request) {
        const RequestIdentity identity(request);
        std::shared_ptr<std::promise<CompileResult>> promise =
            std::make_shared<std::promise<CompileResult>>();
        std::shared_future<CompileResult> future = promise->get_future().share();
        AdaptiveEvent event;
        bool emit_event = false;
        std::uint64_t request_id = 0;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            request_id = next_request_id_++;
            ++metrics_.requests;
            if (stopping_) {
                CompileResult result = CompileResult::Failure(
                    CompileStatus::kRejected,
                    CompileFailureCategory::kShutdown,
                    "compile coordinator is shutting down", 0, false);
                promise->set_value(result);
                MarkCompletedLocked(request_id);
                ++metrics_.rejected;
                event = EventForRequest(request, AdaptiveEventKind::kBudgetRejected);
                event.request_id = request_id;
                event.diagnostic = result.diagnostic();
                emit_event = true;
            } else {
                const auto terminal = terminal_.find(identity);
                if (terminal != terminal_.end() &&
                    !RetryExpiredLocked(terminal->second.result)) {
                    terminal->second.stamp = next_stamp_++;
                    promise->set_value(terminal->second.result);
                    MarkCompletedLocked(request_id);
                    return CompileTicket(request_id, std::move(future));
                }

                std::uint32_t attempt = 1;
                if (terminal != terminal_.end()) {
                    attempt = terminal->second.result.attempt() + 1;
                    terminal_.erase(terminal);
                }

                const auto active = in_flight_.find(identity);
                if (active != in_flight_.end()) {
                    active->second->waiters.emplace(request_id,
                                                    Waiter(promise));
                    waiter_flights_[request_id] = active->second;
                    ++metrics_.merged;
                    event = EventFor(*active->second,
                                     AdaptiveEventKind::kRequestMerged);
                    event.request_id = request_id;
                    event.queue_depth = queue_.size();
                    emit_event = true;
                } else if (queue_.size() >= options_.max_queue_size) {
                    CompileResult result = CompileResult::Failure(
                        CompileStatus::kRejected,
                        CompileFailureCategory::kBackpressure,
                        "compile queue is full", attempt, true, Now());
                    promise->set_value(result);
                    MarkCompletedLocked(request_id);
                    ++metrics_.rejected;
                    event = EventForRequest(
                        request, AdaptiveEventKind::kBudgetRejected);
                    event.request_id = request_id;
                    event.queue_depth = queue_.size();
                    event.diagnostic = result.diagnostic();
                    emit_event = true;
                } else {
                    auto flight = std::make_shared<Flight>(
                        std::move(request), attempt, next_sequence_++);
                    flight->waiters.emplace(request_id, Waiter(promise));
                    in_flight_.emplace(flight->identity, flight);
                    waiter_flights_[request_id] = flight;
                    queue_.push_back(flight);
                    event = EventFor(*flight, AdaptiveEventKind::kQueued);
                    event.request_id = request_id;
                    event.queue_depth = queue_.size();
                    emit_event = true;
                    condition_.notify_one();
                }
            }
        }
        if (emit_event) Emit(event);
        return CompileTicket(request_id, std::move(future));
    }

    CancelResult Cancel(std::uint64_t request_id) {
        AdaptiveEvent event;
        bool emit_event = false;
        CancelResult result = CancelResult::kNotFound;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (completed_ids_.count(request_id) != 0) {
                return CancelResult::kAlreadyCompleted;
            }
            const auto waiter_flight = waiter_flights_.find(request_id);
            if (waiter_flight == waiter_flights_.end()) {
                return CancelResult::kNotFound;
            }
            const auto flight = waiter_flight->second.lock();
            if (!flight) {
                waiter_flights_.erase(waiter_flight);
                return CancelResult::kNotFound;
            }
            const auto waiter = flight->waiters.find(request_id);
            if (waiter == flight->waiters.end()) {
                waiter_flights_.erase(waiter_flight);
                return CancelResult::kNotFound;
            }

            waiter->second.promise->set_value(CompileResult::Failure(
                CompileStatus::kCancelled,
                CompileFailureCategory::kCancelled,
                "compile waiter was cancelled", flight->attempt, false));
            flight->waiters.erase(waiter);
            waiter_flights_.erase(waiter_flight);
            MarkCompletedLocked(request_id);
            ++metrics_.cancelled;
            result = CancelResult::kCancelled;

            event = EventFor(*flight, AdaptiveEventKind::kCancelled);
            event.request_id = request_id;
            event.queue_depth = queue_.size();
            event.diagnostic = "compile waiter was cancelled";
            emit_event = true;

            if (flight->waiters.empty()) {
                flight->cancellation->requested.store(
                    true, std::memory_order_release);
                flight->discard = true;
                if (flight->state == FlightState::kQueued) {
                    queue_.erase(std::remove(queue_.begin(), queue_.end(), flight),
                                 queue_.end());
                    in_flight_.erase(flight->identity);
                }
            }
        }
        if (emit_event) Emit(event);
        return result;
    }

    CoordinatorSnapshot Snapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        CoordinatorSnapshot result = metrics_;
        result.accepting = !stopping_;
        result.queued = queue_.size();
        result.active = active_count_;
        result.in_flight_keys = in_flight_.size();
        result.terminal_records = terminal_.size();
        return result;
    }

    void Shutdown() {
        std::lock_guard<std::mutex> shutdown_lock(shutdown_mutex_);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (joined_) return;
            if (!stopping_) {
                stopping_ = true;
                AdaptiveEvent started;
                started.kind = AdaptiveEventKind::kShutdownStarted;
                shutdown_events_.push_back(std::move(started));

                for (const auto& item : in_flight_) {
                    const auto& flight = item.second;
                    flight->discard = true;
                    flight->cancellation->requested.store(
                        true, std::memory_order_release);
                    for (const auto& waiter : flight->waiters) {
                        waiter.second.promise->set_value(CompileResult::Failure(
                            CompileStatus::kCancelled,
                            CompileFailureCategory::kShutdown,
                            "compile coordinator shutdown", flight->attempt,
                            false));
                        waiter_flights_.erase(waiter.first);
                        MarkCompletedLocked(waiter.first);
                        ++metrics_.cancelled;
                        AdaptiveEvent cancelled = EventFor(
                            *flight, AdaptiveEventKind::kCancelled);
                        cancelled.request_id = waiter.first;
                        cancelled.diagnostic =
                            "compile coordinator shutdown";
                        shutdown_events_.push_back(std::move(cancelled));
                    }
                    flight->waiters.clear();
                }
                queue_.clear();
                in_flight_.clear();
            }
        }
        condition_.notify_all();
        EmitShutdownEvents();
        for (auto& worker : workers_) {
            if (worker.joinable()) worker.join();
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            joined_ = true;
        }
        AdaptiveEvent completed;
        completed.kind = AdaptiveEventKind::kShutdownCompleted;
        Emit(completed);
    }

private:
    std::chrono::steady_clock::time_point Now() const {
        return options_.now();
    }

    AdaptiveEvent EventForRequest(const CompileRequest& request,
                                  AdaptiveEventKind kind) const {
        AdaptiveEvent event;
        event.kind = kind;
        event.slot_key = request.artifact_key().slot_key().canonical();
        event.artifact_key = request.artifact_key().canonical();
        event.dispatch_key = request.dispatch_key().canonical();
        event.abi_fingerprint = request.required_abi().canonical();
        event.model_revision = request.model_revision();
        return event;
    }

    bool RetryExpiredLocked(const CompileResult& result) const {
        return result.status() == CompileStatus::kFailed &&
               result.retryable() && Now() >= result.retry_after();
    }

    void Emit(const AdaptiveEvent& event) const noexcept {
        if (!options_.observer) return;
        try {
            options_.observer(event);
        } catch (...) {
            // Observability cannot alter compilation state or shutdown.
        }
    }

    void EmitShutdownEvents() {
        std::vector<AdaptiveEvent> events;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            events.swap(shutdown_events_);
        }
        for (const auto& event : events) Emit(event);
    }

    void MarkCompletedLocked(std::uint64_t request_id) {
        completed_ids_.insert(request_id);
        completed_order_.push_back(request_id);
        const std::size_t limit =
            std::max<std::size_t>(1024, options_.max_terminal_records * 4);
        while (completed_order_.size() > limit) {
            completed_ids_.erase(completed_order_.front());
            completed_order_.pop_front();
        }
    }

    void StoreTerminalLocked(const RequestIdentity& identity,
                             CompileResult result) {
        terminal_.erase(identity);
        terminal_.emplace(identity,
                          TerminalRecord(std::move(result), next_stamp_++));
        while (terminal_.size() > options_.max_terminal_records) {
            const auto oldest = std::min_element(
                terminal_.begin(), terminal_.end(),
                [](const auto& lhs, const auto& rhs) {
                    return lhs.second.stamp < rhs.second.stamp;
                });
            terminal_.erase(oldest);
        }
    }

    std::shared_ptr<Flight> PopNextLocked() {
        const auto best = std::min_element(
            queue_.begin(), queue_.end(),
            [](const auto& lhs, const auto& rhs) {
                const int lhs_kind = KindOrder(lhs->request.kind());
                const int rhs_kind = KindOrder(rhs->request.kind());
                if (lhs_kind != rhs_kind) return lhs_kind < rhs_kind;
                if (lhs->request.priority() != rhs->request.priority()) {
                    return lhs->request.priority() > rhs->request.priority();
                }
                return lhs->sequence < rhs->sequence;
            });
        auto flight = *best;
        queue_.erase(best);
        return flight;
    }

    std::chrono::milliseconds RetryDelay(std::uint32_t attempt) const {
        std::chrono::milliseconds delay =
            options_.retry_policy.initial_backoff;
        for (std::uint32_t i = 1; i < attempt; ++i) {
            if (delay >= options_.retry_policy.max_backoff / 2) {
                return options_.retry_policy.max_backoff;
            }
            delay *= 2;
        }
        return std::min(delay, options_.retry_policy.max_backoff);
    }

    CompileResult FailureResult(CompileFailureCategory category,
                                std::string diagnostic,
                                std::uint32_t attempt) const {
        if (category == CompileFailureCategory::kCancelled ||
            category == CompileFailureCategory::kShutdown) {
            return CompileResult::Failure(
                CompileStatus::kCancelled, category, std::move(diagnostic),
                attempt, false);
        }
        const bool retryable =
            category == CompileFailureCategory::kTransient &&
            attempt < options_.retry_policy.max_transient_attempts;
        const auto retry_after = retryable ? Now() + RetryDelay(attempt)
                                           : std::chrono::steady_clock::time_point{};
        return CompileResult::Failure(CompileStatus::kFailed, category,
                                      std::move(diagnostic), attempt,
                                      retryable, retry_after);
    }

    CompileResult ValidateAttempt(const Flight& flight,
                                  CompileAttempt attempt) const {
        if (!attempt.ready()) {
            return FailureResult(attempt.failure_category(),
                                 attempt.diagnostic(), flight.attempt);
        }
        const auto& artifact = attempt.artifact();
        if (!artifact || artifact->key() != flight.request.artifact_key() ||
            artifact->applicability() != flight.request.dispatch_key() ||
            artifact->compatible_abi() != flight.request.required_abi() ||
            !artifact->executable() || !artifact->executable()->IsReady()) {
            return FailureResult(
                CompileFailureCategory::kValidation,
                "compiled artifact does not match the full exact request",
                flight.attempt);
        }
        if (artifact->byte_size() > options_.max_artifact_bytes) {
            return FailureResult(
                CompileFailureCategory::kValidation,
                "compiled artifact exceeds the configured byte budget",
                flight.attempt);
        }
        return CompileResult::Ready(artifact, flight.attempt);
    }

    void WorkerLoop() {
        while (true) {
            std::shared_ptr<Flight> flight;
            AdaptiveEvent started;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                condition_.wait(lock,
                                [this] { return stopping_ || !queue_.empty(); });
                if (stopping_ && queue_.empty()) return;
                flight = PopNextLocked();
                if (flight->discard) continue;
                flight->state = FlightState::kCompiling;
                ++active_count_;
                ++metrics_.compile_attempts;
                started = EventFor(*flight, AdaptiveEventKind::kCompileStarted);
                started.queue_depth = queue_.size();
            }
            Emit(started);

            CompileAttempt attempt = CompileAttempt::Failed(
                CompileFailureCategory::kDeterministic,
                "artifact compiler returned no result");
            try {
                attempt = compiler_->Compile(
                    flight->request, CancellationToken(flight->cancellation));
            } catch (const std::exception& error) {
                attempt = CompileAttempt::Failed(
                    CompileFailureCategory::kDeterministic,
                    std::string("artifact compiler threw: ") + error.what());
            } catch (...) {
                attempt = CompileAttempt::Failed(
                    CompileFailureCategory::kDeterministic,
                    "artifact compiler threw an unknown exception");
            }

            if (flight->cancellation->requested.load(
                    std::memory_order_acquire)) {
                FinishFlight(
                    flight,
                    FailureResult(CompileFailureCategory::kCancelled,
                                  "compile was cancelled", flight->attempt));
                continue;
            }

            AdaptiveEvent validating;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                flight->state = FlightState::kValidating;
                validating =
                    EventFor(*flight, AdaptiveEventKind::kValidating);
            }
            Emit(validating);
            FinishFlight(flight, ValidateAttempt(*flight, std::move(attempt)));
        }
    }

    void FinishFlight(const std::shared_ptr<Flight>& flight,
                      CompileResult result) {
        AdaptiveEvent event;
        bool emit_event = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (active_count_ != 0) --active_count_;
            flight->state = FlightState::kDone;
            const auto in_flight = in_flight_.find(flight->identity);
            if (in_flight != in_flight_.end() &&
                in_flight->second == flight) {
                in_flight_.erase(in_flight);
            }

            if (!flight->discard) {
                if (result.status() == CompileStatus::kReady) {
                    ++metrics_.ready;
                    StoreTerminalLocked(flight->identity, result);
                    event = EventFor(*flight, AdaptiveEventKind::kReady);
                } else if (result.status() == CompileStatus::kFailed) {
                    ++metrics_.failed;
                    StoreTerminalLocked(flight->identity, result);
                    event = EventFor(*flight, AdaptiveEventKind::kFailed);
                    event.diagnostic = result.diagnostic();
                } else {
                    event = EventFor(*flight, AdaptiveEventKind::kCancelled);
                    event.diagnostic = result.diagnostic();
                }
                event.queue_depth = queue_.size();
                emit_event = true;
            }

            for (const auto& waiter : flight->waiters) {
                waiter.second.promise->set_value(result);
                waiter_flights_.erase(waiter.first);
                MarkCompletedLocked(waiter.first);
            }
            flight->waiters.clear();
        }
        condition_.notify_all();
        if (emit_event) Emit(event);
    }

    std::shared_ptr<ArtifactCompiler> compiler_;
    CoordinatorOptions options_;

    mutable std::mutex mutex_;
    std::mutex shutdown_mutex_;
    std::condition_variable condition_;
    bool stopping_{false};
    bool joined_{false};
    std::vector<std::thread> workers_;
    std::vector<std::shared_ptr<Flight>> queue_;
    std::map<RequestIdentity, std::shared_ptr<Flight>> in_flight_;
    std::unordered_map<std::uint64_t, std::weak_ptr<Flight>> waiter_flights_;
    std::map<RequestIdentity, TerminalRecord> terminal_;
    std::unordered_set<std::uint64_t> completed_ids_;
    std::deque<std::uint64_t> completed_order_;
    std::vector<AdaptiveEvent> shutdown_events_;
    std::uint64_t next_request_id_{1};
    std::uint64_t next_sequence_{1};
    std::uint64_t next_stamp_{1};
    std::size_t active_count_{0};
    CoordinatorSnapshot metrics_;
};

CompileCoordinator::CompileCoordinator(
    std::shared_ptr<ArtifactCompiler> compiler, CoordinatorOptions options)
    : state_(std::make_unique<State>(std::move(compiler),
                                     std::move(options))) {}

CompileCoordinator::~CompileCoordinator() { Shutdown(); }

CompileTicket CompileCoordinator::Request(CompileRequest request) {
    return state_->Request(std::move(request));
}

CancelResult CompileCoordinator::Cancel(std::uint64_t request_id) {
    return state_->Cancel(request_id);
}

CoordinatorSnapshot CompileCoordinator::Snapshot() const {
    return state_->Snapshot();
}

void CompileCoordinator::Shutdown() {
    if (state_) state_->Shutdown();
}

}  // namespace kxc::api::adaptive
