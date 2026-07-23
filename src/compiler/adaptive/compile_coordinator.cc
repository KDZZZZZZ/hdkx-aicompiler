/*! \file src/compiler/adaptive/compile_coordinator.cc
 * \brief Bounded singleflight coordinator for static-exact compilation.
 */

#include "kxc/compiler/adaptive.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace kxc::api::experimental::adaptive::v1 {

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
    if (!future_.valid()) throw std::logic_error("compile ticket is not valid");
    return future_.get();
}

std::future_status CompileTicket::WaitFor(
    std::chrono::milliseconds timeout) const {
    if (!future_.valid()) throw std::logic_error("compile ticket is not valid");
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

enum class FlightState : std::uint8_t { kQueued, kCompiling, kValidating, kDone };

struct Waiter final {
    Waiter(CompileRequest request_value,
           std::shared_ptr<std::promise<CompileResult>> promise_value)
        : request(std::move(request_value)), promise(std::move(promise_value)) {}
    CompileRequest request;
    std::shared_ptr<std::promise<CompileResult>> promise;
};

struct Flight final {
    Flight(CompileRequest request_value, std::uint32_t attempt_value,
           std::uint64_t sequence_value, std::uint64_t root_request_id_value,
           std::chrono::steady_clock::time_point queued_at_value)
        : request(std::move(request_value)),
          identity(request),
          attempt(attempt_value),
          sequence(sequence_value),
          root_request_id(root_request_id_value),
          queued_at(queued_at_value),
          cancellation(std::make_shared<CancellationToken::State>()) {}

    // This request is replaced only while queued with the selected effective
    // waiter request; a compiler therefore always observes an immutable value.
    CompileRequest request;
    RequestIdentity identity;
    std::uint32_t attempt{1};
    std::uint64_t sequence{0};
    std::uint64_t root_request_id{0};
    std::chrono::steady_clock::time_point queued_at;
    std::chrono::steady_clock::time_point compile_started_at{};
    std::chrono::steady_clock::time_point validating_at{};
    FlightState state{FlightState::kQueued};
    bool discard{false};
    std::shared_ptr<CancellationToken::State> cancellation;
    std::map<std::uint64_t, Waiter> waiters;
};

struct TerminalRecord final {
    TerminalRecord(CompileResult result_value, std::uint64_t stamp_value)
        : result(std::move(result_value)), stamp(stamp_value) {
        if (result.artifact()) bytes = result.artifact()->byte_size();
    }
    CompileResult result;
    std::uint64_t stamp{0};
    std::size_t bytes{0};
};

int KindOrder(RequestKind kind) {
    switch (kind) {
        case RequestKind::kDemand: return 0;
        case RequestKind::kCanary: return 1;
        case RequestKind::kPrewarm: return 2;
    }
    return 3;
}

AdaptiveEvent EventForRequest(const CompileRequest& request,
                              AdaptiveEventKind kind) {
    AdaptiveEvent event;
    event.kind = kind;
    event.priority = request.priority();
    event.slot_key = request.artifact_key().slot_key().canonical();
    event.artifact_key = request.artifact_key().canonical();
    event.dispatch_key = request.dispatch_key().canonical();
    event.abi_fingerprint = request.required_abi().canonical();
    event.model_revision = request.model_revision();
    return event;
}

AdaptiveEvent EventFor(const Flight& flight, AdaptiveEventKind kind) {
    AdaptiveEvent event = EventForRequest(flight.request, kind);
    event.request_id = flight.root_request_id;
    event.attempt = flight.attempt;
    event.waiter_count = flight.waiters.size();
    return event;
}

}  // namespace

class CompileCoordinator::State final {
public:
    State(std::shared_ptr<ArtifactCompiler> compiler_value,
          std::shared_ptr<AdaptiveValidationAuthority> authority_value,
          CoordinatorOptions options_value)
        : compiler_(std::move(compiler_value)),
          authority_(std::move(authority_value)),
          options_(std::move(options_value)) {
        if (!compiler_ || !authority_) {
            throw std::invalid_argument(
                "CompileCoordinator requires a compiler and validation authority");
        }
        if (options_.worker_count == 0 || options_.max_queue_size == 0 ||
            options_.max_waiters_per_flight == 0 ||
            options_.max_terminal_records == 0 || options_.max_artifact_bytes == 0 ||
            options_.max_cached_artifact_bytes == 0) {
            throw std::invalid_argument("CompileCoordinator bounds must be non-zero");
        }
        if (options_.retry_policy.max_transient_attempts == 0 ||
            options_.retry_policy.initial_backoff.count() < 0 ||
            options_.retry_policy.max_backoff < options_.retry_policy.initial_backoff) {
            throw std::invalid_argument("CompileCoordinator retry policy is invalid");
        }
        if (!options_.now) {
            options_.now = [] { return std::chrono::steady_clock::now(); };
        }
    }

    std::size_t worker_count() const noexcept { return options_.worker_count; }

    CompileTicket Request(CompileRequest request) {
        const RequestIdentity identity(request);
        const auto now = Now();
        auto promise = std::make_shared<std::promise<CompileResult>>();
        auto future = promise->get_future().share();
        std::vector<AdaptiveEvent> events;
        std::uint64_t request_id = 0;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            request_id = next_request_id_++;
            ++metrics_.requests;
            ExpireQueuedLocked(now, &events);

            if (request.expired(now)) {
                const CompileResult expired = ExpiredResult();
                promise->set_value(expired);
                MarkCompletedLocked(request_id);
                ++metrics_.expired;
                AdaptiveEvent event = EventForRequest(request, AdaptiveEventKind::kExpired);
                event.request_id = request_id;
                event.diagnostic = expired.diagnostic();
                events.push_back(std::move(event));
            } else if (stopping_) {
                const CompileResult result = CompileResult::Failure(
                    CompileStatus::kRejected, CompileFailureCategory::kShutdown,
                    "compile coordinator is shutting down", 0, false);
                promise->set_value(result);
                MarkCompletedLocked(request_id);
                ++metrics_.rejected;
                AdaptiveEvent event = EventForRequest(request, AdaptiveEventKind::kBudgetRejected);
                event.request_id = request_id;
                event.diagnostic = result.diagnostic();
                events.push_back(std::move(event));
            } else {
                auto terminal = terminal_.find(identity);
                if (terminal != terminal_.end() &&
                    !RetryExpiredLocked(terminal->second.result, now)) {
                    terminal->second.stamp = next_stamp_++;
                    promise->set_value(terminal->second.result);
                    MarkCompletedLocked(request_id);
                } else {
                    std::uint32_t attempt = 1;
                    if (terminal != terminal_.end()) {
                        attempt = terminal->second.result.attempt() + 1;
                        cached_artifact_bytes_ -= terminal->second.bytes;
                        terminal_.erase(terminal);
                    }

                    const auto active = in_flight_.find(identity);
                    if (active != in_flight_.end() &&
                        active->second->waiters.size() >= options_.max_waiters_per_flight) {
                        const CompileResult result = CompileResult::Failure(
                            CompileStatus::kRejected, CompileFailureCategory::kBackpressure,
                            "singleflight waiter budget is full", attempt, true, now);
                        promise->set_value(result);
                        MarkCompletedLocked(request_id);
                        ++metrics_.rejected;
                        AdaptiveEvent event = EventForRequest(request, AdaptiveEventKind::kBudgetRejected);
                        event.request_id = request_id;
                        event.waiter_count = active->second->waiters.size();
                        event.queue_depth = queue_.size();
                        event.diagnostic = result.diagnostic();
                        events.push_back(std::move(event));
                    } else if (active != in_flight_.end()) {
                        const auto& flight = active->second;
                        flight->waiters.emplace(request_id, Waiter(std::move(request), promise));
                        waiter_flights_[request_id] = flight;
                        if (flight->state == FlightState::kQueued) RecomputeScheduleLocked(*flight);
                        ++metrics_.merged;
                        AdaptiveEvent event = EventFor(*flight, AdaptiveEventKind::kRequestMerged);
                        event.request_id = request_id;
                        event.queue_depth = queue_.size();
                        events.push_back(std::move(event));
                    } else {
                        if (queue_.size() >= options_.max_queue_size &&
                            request.kind() != RequestKind::kPrewarm) {
                            EvictOnePrewarmLocked(now, &events);
                        }
                        if (queue_.size() >= options_.max_queue_size) {
                            const CompileResult result = CompileResult::Failure(
                                CompileStatus::kRejected, CompileFailureCategory::kBackpressure,
                                "compile queue is full", attempt, true, now);
                            promise->set_value(result);
                            MarkCompletedLocked(request_id);
                            ++metrics_.rejected;
                            AdaptiveEvent event = EventForRequest(request, AdaptiveEventKind::kBudgetRejected);
                            event.request_id = request_id;
                            event.queue_depth = queue_.size();
                            event.diagnostic = result.diagnostic();
                            events.push_back(std::move(event));
                        } else {
                            auto flight = std::make_shared<Flight>(
                                std::move(request), attempt, next_sequence_++, request_id, now);
                            flight->waiters.emplace(request_id, Waiter(flight->request, promise));
                            in_flight_.emplace(flight->identity, flight);
                            waiter_flights_[request_id] = flight;
                            queue_.push_back(flight);
                            AdaptiveEvent event = EventFor(*flight, AdaptiveEventKind::kQueued);
                            event.request_id = request_id;
                            event.queue_depth = queue_.size();
                            events.push_back(std::move(event));
                            condition_.notify_one();
                        }
                    }
                }
            }
        }
        deadline_condition_.notify_one();
        Emit(events);
        return CompileTicket(request_id, std::move(future));
    }

    CancelResult Cancel(std::uint64_t request_id) {
        std::vector<AdaptiveEvent> events;
        CancelResult result = CancelResult::kNotFound;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (completed_ids_.count(request_id) != 0) return CancelResult::kAlreadyCompleted;
            const auto found = waiter_flights_.find(request_id);
            if (found == waiter_flights_.end()) return CancelResult::kNotFound;
            const auto flight = found->second.lock();
            if (!flight) {
                waiter_flights_.erase(found);
                return CancelResult::kNotFound;
            }
            const auto waiter = flight->waiters.find(request_id);
            if (waiter == flight->waiters.end()) return CancelResult::kNotFound;

            waiter->second.promise->set_value(CompileResult::Failure(
                CompileStatus::kCancelled, CompileFailureCategory::kCancelled,
                "compile waiter was cancelled", flight->attempt, false));
            flight->waiters.erase(waiter);
            waiter_flights_.erase(found);
            MarkCompletedLocked(request_id);
            ++metrics_.cancelled;
            AdaptiveEvent event = EventFor(*flight, AdaptiveEventKind::kCancelled);
            event.request_id = request_id;
            event.queue_depth = queue_.size();
            event.diagnostic = "compile waiter was cancelled";
            events.push_back(std::move(event));
            result = CancelResult::kCancelled;

            if (flight->waiters.empty()) {
                DiscardFlightLocked(flight);
            } else if (flight->state == FlightState::kQueued) {
                RecomputeScheduleLocked(*flight);
            }
        }
        condition_.notify_all();
        deadline_condition_.notify_one();
        Emit(events);
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
        result.waiters = waiter_flights_.size();
        result.cached_artifact_bytes = cached_artifact_bytes_;
        return result;
    }

    // Returns events to be emitted after dropping coordinator locks.
    std::vector<AdaptiveEvent> RequestStop() {
        std::vector<AdaptiveEvent> events;
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_) return events;
        stopping_ = true;
        events.push_back(AdaptiveEvent{});
        events.back().kind = AdaptiveEventKind::kShutdownStarted;
        for (const auto& item : in_flight_) {
            const auto& flight = item.second;
            flight->discard = true;
            flight->cancellation->requested.store(true, std::memory_order_release);
            for (const auto& waiter : flight->waiters) {
                waiter.second.promise->set_value(CompileResult::Failure(
                    CompileStatus::kCancelled, CompileFailureCategory::kShutdown,
                    "compile coordinator shutdown", flight->attempt, false));
                waiter_flights_.erase(waiter.first);
                MarkCompletedLocked(waiter.first);
                ++metrics_.cancelled;
                AdaptiveEvent event = EventFor(*flight, AdaptiveEventKind::kCancelled);
                event.request_id = waiter.first;
                event.diagnostic = "compile coordinator shutdown";
                events.push_back(std::move(event));
            }
            flight->waiters.clear();
        }
        queue_.clear();
        in_flight_.clear();
        condition_.notify_all();
        deadline_condition_.notify_all();
        return events;
    }

    void CompleteShutdown() {
        {
            std::lock_guard<std::mutex> lock(completion_mutex_);
            if (shutdown_completed_) return;
            shutdown_completed_ = true;
        }
        AdaptiveEvent event;
        event.kind = AdaptiveEventKind::kShutdownCompleted;
        Emit(event);
    }

    void Emit(const std::vector<AdaptiveEvent>& events) const noexcept {
        for (const auto& event : events) Emit(event);
    }

    void Emit(const AdaptiveEvent& event) const noexcept {
        if (!options_.observer) return;
        try {
            options_.observer(event);
        } catch (...) {
        }
    }

    void DeadlineLoop() {
        while (true) {
            std::vector<AdaptiveEvent> events;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                deadline_condition_.wait(lock, [this] {
                    return stopping_ || !queue_.empty();
                });
                while (!stopping_ && !queue_.empty()) {
                    deadline_condition_.wait_for(lock,
                                                 std::chrono::milliseconds(1));
                    if (stopping_) break;
                    ExpireQueuedLocked(Now(), &events);
                    if (!events.empty()) break;
                }
                if (stopping_) return;
            }
            Emit(events);
        }
    }

    void WorkerLoop() {
        while (true) {
            std::shared_ptr<Flight> flight;
            AdaptiveEvent started;
            std::vector<AdaptiveEvent> expired_events;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                condition_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
                if (stopping_ && queue_.empty()) return;
                flight = PopNextLocked(Now(), &expired_events);
                if (flight) {
                    flight->state = FlightState::kCompiling;
                    flight->compile_started_at = Now();
                    ++active_count_;
                    ++metrics_.compile_attempts;
                    started = EventFor(*flight,
                                       AdaptiveEventKind::kCompileStarted);
                    started.queue_depth = queue_.size();
                    started.queue_wait = flight->compile_started_at -
                                         flight->queued_at;
                }
            }
            Emit(expired_events);
            if (!flight) continue;
            Emit(started);

            CompileAttempt attempt = CompileAttempt::Failed(
                CompileFailureCategory::kDeterministic,
                "artifact compiler returned no result");
            try {
                attempt = compiler_->Compile(
                    flight->request, CancellationToken(flight->cancellation));
            } catch (const std::exception& error) {
                attempt = CompileAttempt::Failed(CompileFailureCategory::kDeterministic,
                                                  std::string("artifact compiler threw: ") + error.what());
            } catch (...) {
                attempt = CompileAttempt::Failed(CompileFailureCategory::kDeterministic,
                                                  "artifact compiler threw an unknown exception");
            }

            AdaptiveEvent validating;
            bool cancelled = false;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                cancelled = flight->discard ||
                            flight->cancellation->requested.load(
                                std::memory_order_acquire);
                if (!cancelled) {
                    flight->state = FlightState::kValidating;
                    flight->validating_at = Now();
                    validating =
                        EventFor(*flight, AdaptiveEventKind::kValidating);
                    validating.queue_wait = flight->compile_started_at -
                                            flight->queued_at;
                    validating.compile_time = flight->validating_at -
                                              flight->compile_started_at;
                }
            }
            if (cancelled) {
                FinishFlight(
                    flight,
                    FailureResult(CompileFailureCategory::kCancelled,
                                  "compile was cancelled", flight->attempt));
                continue;
            }
            Emit(validating);
            FinishFlight(flight, ValidateAttempt(*flight, std::move(attempt)));
        }
    }

private:
    std::chrono::steady_clock::time_point Now() const { return options_.now(); }

    static CompileResult ExpiredResult() {
        return CompileResult::Failure(CompileStatus::kExpired,
                                      CompileFailureCategory::kDeadlineExceeded,
                                      "compile queue deadline exceeded", 0, false);
    }

    static bool RetryExpiredLocked(const CompileResult& result,
                                   std::chrono::steady_clock::time_point now) {
        return result.status() == CompileStatus::kFailed && result.retryable() &&
               now >= result.retry_after();
    }

    void MarkCompletedLocked(std::uint64_t request_id) {
        completed_ids_.insert(request_id);
        completed_order_.push_back(request_id);
        const std::size_t scaled_limit =
            options_.max_terminal_records >
                    std::numeric_limits<std::size_t>::max() / 4
                ? std::numeric_limits<std::size_t>::max()
                : options_.max_terminal_records * 4;
        const std::size_t limit =
            std::max<std::size_t>(1024, scaled_limit);
        while (completed_order_.size() > limit) {
            completed_ids_.erase(completed_order_.front());
            completed_order_.pop_front();
        }
    }

    void RecomputeScheduleLocked(Flight& flight) {
        if (flight.state != FlightState::kQueued || flight.waiters.empty()) return;
        const auto selected = std::min_element(
            flight.waiters.begin(), flight.waiters.end(),
            [](const auto& lhs, const auto& rhs) {
                const CompileRequest& left = lhs.second.request;
                const CompileRequest& right = rhs.second.request;
                const int left_kind = KindOrder(left.kind());
                const int right_kind = KindOrder(right.kind());
                if (left_kind != right_kind) return left_kind < right_kind;
                if (left.priority() != right.priority()) {
                    return left.priority() > right.priority();
                }
                if (left.deadline() != right.deadline()) {
                    return left.deadline() < right.deadline();
                }
                return lhs.first < rhs.first;
            });
        flight.request = selected->second.request;
        flight.root_request_id = selected->first;
    }

    void RemoveFlightLocked(const std::shared_ptr<Flight>& flight) {
        const auto found = in_flight_.find(flight->identity);
        if (found != in_flight_.end() && found->second == flight) {
            in_flight_.erase(found);
        }
    }

    void DiscardFlightLocked(const std::shared_ptr<Flight>& flight) {
        flight->discard = true;
        flight->cancellation->requested.store(true, std::memory_order_release);
        if (flight->state == FlightState::kQueued) {
            queue_.erase(std::remove(queue_.begin(), queue_.end(), flight), queue_.end());
        }
        RemoveFlightLocked(flight);
    }

    void ExpireWaiterLocked(const std::shared_ptr<Flight>& flight,
                            std::map<std::uint64_t, Waiter>::iterator waiter,
                            std::vector<AdaptiveEvent>* events) {
        const std::uint64_t request_id = waiter->first;
        const CompileRequest request = waiter->second.request;
        waiter->second.promise->set_value(ExpiredResult());
        flight->waiters.erase(waiter);
        waiter_flights_.erase(request_id);
        MarkCompletedLocked(request_id);
        ++metrics_.expired;
        AdaptiveEvent event = EventForRequest(request, AdaptiveEventKind::kExpired);
        event.request_id = request_id;
        event.queue_depth = queue_.size();
        event.diagnostic = "compile queue deadline exceeded";
        events->push_back(std::move(event));
    }

    void ExpireQueuedLocked(std::chrono::steady_clock::time_point now,
                            std::vector<AdaptiveEvent>* events) {
        for (auto queued = queue_.begin(); queued != queue_.end();) {
            const auto flight = *queued;
            for (auto waiter = flight->waiters.begin(); waiter != flight->waiters.end();) {
                if (waiter->second.request.expired(now)) {
                    auto expired = waiter++;
                    ExpireWaiterLocked(flight, expired, events);
                } else {
                    ++waiter;
                }
            }
            if (flight->waiters.empty()) {
                flight->discard = true;
                flight->cancellation->requested.store(true, std::memory_order_release);
                RemoveFlightLocked(flight);
                queued = queue_.erase(queued);
            } else {
                RecomputeScheduleLocked(*flight);
                ++queued;
            }
        }
    }

    void EvictOnePrewarmLocked(std::chrono::steady_clock::time_point now,
                               std::vector<AdaptiveEvent>* events) {
        auto victim = queue_.end();
        for (auto queued = queue_.begin(); queued != queue_.end(); ++queued) {
            const auto& candidate = *queued;
            if (candidate->request.kind() != RequestKind::kPrewarm) continue;
            if (victim == queue_.end() ||
                candidate->request.priority() < (*victim)->request.priority() ||
                (candidate->request.priority() == (*victim)->request.priority() &&
                 (candidate->request.deadline() > (*victim)->request.deadline() ||
                  (candidate->request.deadline() == (*victim)->request.deadline() &&
                   candidate->sequence > (*victim)->sequence)))) {
                victim = queued;
            }
        }
        if (victim == queue_.end()) return;
        const auto flight = *victim;
        const CompileResult rejected = CompileResult::Failure(
            CompileStatus::kRejected, CompileFailureCategory::kBackpressure,
            "queued prewarm displaced by demand", flight->attempt, true, now);
        for (const auto& waiter : flight->waiters) {
            waiter.second.promise->set_value(rejected);
            waiter_flights_.erase(waiter.first);
            MarkCompletedLocked(waiter.first);
            ++metrics_.rejected;
            AdaptiveEvent event = EventForRequest(waiter.second.request,
                                                   AdaptiveEventKind::kBudgetRejected);
            event.request_id = waiter.first;
            event.queue_depth = queue_.size() - 1;
            event.diagnostic = rejected.diagnostic();
            events->push_back(std::move(event));
        }
        flight->waiters.clear();
        flight->discard = true;
        flight->cancellation->requested.store(true, std::memory_order_release);
        RemoveFlightLocked(flight);
        queue_.erase(victim);
    }

    std::shared_ptr<Flight> PopNextLocked(
        std::chrono::steady_clock::time_point now,
        std::vector<AdaptiveEvent>* expired_events) {
        ExpireQueuedLocked(now, expired_events);
        if (queue_.empty()) return nullptr;
        const auto best = std::min_element(
            queue_.begin(), queue_.end(),
            [](const auto& lhs, const auto& rhs) {
                const int left_kind = KindOrder(lhs->request.kind());
                const int right_kind = KindOrder(rhs->request.kind());
                if (left_kind != right_kind) return left_kind < right_kind;
                if (lhs->request.priority() != rhs->request.priority()) {
                    return lhs->request.priority() > rhs->request.priority();
                }
                if (lhs->request.deadline() != rhs->request.deadline()) {
                    return lhs->request.deadline() < rhs->request.deadline();
                }
                return lhs->sequence < rhs->sequence;
            });
        auto flight = *best;
        queue_.erase(best);
        return flight;
    }

    void StoreTerminalLocked(const RequestIdentity& identity,
                             CompileResult result) {
        const auto existing = terminal_.find(identity);
        if (existing != terminal_.end()) {
            cached_artifact_bytes_ -= existing->second.bytes;
            terminal_.erase(existing);
        }

        const std::size_t bytes =
            result.artifact() ? result.artifact()->byte_size() : 0;
        if (bytes > options_.max_cached_artifact_bytes) return;
        while (!terminal_.empty() &&
               (terminal_.size() >= options_.max_terminal_records ||
                bytes > options_.max_cached_artifact_bytes -
                            cached_artifact_bytes_)) {
            const auto oldest = std::min_element(
                terminal_.begin(), terminal_.end(),
                [](const auto& lhs, const auto& rhs) {
                    return lhs.second.stamp < rhs.second.stamp;
                });
            cached_artifact_bytes_ -= oldest->second.bytes;
            terminal_.erase(oldest);
        }
        if (terminal_.size() >= options_.max_terminal_records ||
            bytes > options_.max_cached_artifact_bytes -
                        cached_artifact_bytes_) {
            return;
        }
        const auto inserted = terminal_.emplace(
            identity, TerminalRecord(std::move(result), next_stamp_++));
        cached_artifact_bytes_ += inserted.first->second.bytes;
    }

    std::chrono::milliseconds RetryDelay(std::uint32_t attempt) const {
        auto delay = options_.retry_policy.initial_backoff;
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
            return CompileResult::Failure(CompileStatus::kCancelled, category,
                                          std::move(diagnostic), attempt, false);
        }
        const bool retryable = category == CompileFailureCategory::kTransient &&
                               attempt < options_.retry_policy.max_transient_attempts;
        const auto retry_after = retryable ? Now() + RetryDelay(attempt)
                                           : std::chrono::steady_clock::time_point{};
        return CompileResult::Failure(CompileStatus::kFailed, category,
                                      std::move(diagnostic), attempt, retryable,
                                      retry_after);
    }

    CompileResult ValidateAttempt(const Flight& flight, CompileAttempt attempt) const {
        if (!attempt.ready()) {
            return FailureResult(attempt.failure_category(), attempt.diagnostic(),
                                 flight.attempt);
        }
        const auto& artifact = attempt.artifact();
        if (!artifact || artifact->key() != flight.request.artifact_key() ||
            artifact->applicability() != flight.request.dispatch_key() ||
            artifact->compatible_abi() != flight.request.required_abi() ||
            !artifact->executable() || !artifact->executable()->IsReady()) {
            return FailureResult(CompileFailureCategory::kValidation,
                                 "compiled artifact does not match the full exact request",
                                 flight.attempt);
        }
        if (artifact->byte_size() > options_.max_artifact_bytes) {
            return FailureResult(CompileFailureCategory::kValidation,
                                 "compiled artifact exceeds the configured byte budget",
                                 flight.attempt);
        }
        try {
            const auto validation = authority_->ValidateArtifact(flight.request, artifact);
            if (!validation ||
                validation->artifact_key() != flight.request.artifact_key() ||
                validation->dispatch_key() != flight.request.dispatch_key() ||
                validation->compatible_abi() != flight.request.required_abi() ||
                validation->artifact_bytes() != artifact->byte_size()) {
                return FailureResult(CompileFailureCategory::kValidation,
                                     "validation authority returned an unbound record",
                                     flight.attempt);
            }
            return CompileResult::Ready(artifact, validation, flight.attempt);
        } catch (const std::exception& error) {
            return FailureResult(CompileFailureCategory::kValidation,
                                 std::string("validation authority threw: ") + error.what(),
                                 flight.attempt);
        } catch (...) {
            return FailureResult(CompileFailureCategory::kValidation,
                                 "validation authority threw an unknown exception",
                                 flight.attempt);
        }
    }

    void FinishFlight(const std::shared_ptr<Flight>& flight, CompileResult result) {
        AdaptiveEvent event;
        bool emit_event = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (active_count_ != 0) --active_count_;
            flight->state = FlightState::kDone;
            RemoveFlightLocked(flight);
            if (!flight->discard) {
                if (result.status() == CompileStatus::kReady) {
                    ++metrics_.ready;
                    StoreTerminalLocked(flight->identity, result);
                    event = EventFor(*flight, AdaptiveEventKind::kReady);
                    event.artifact_bytes = result.artifact()->byte_size();
                } else if (result.status() == CompileStatus::kFailed) {
                    ++metrics_.failed;
                    StoreTerminalLocked(flight->identity, result);
                    event = EventFor(*flight, AdaptiveEventKind::kFailed);
                    event.failure_category = result.failure_category();
                    event.diagnostic = result.diagnostic();
                } else {
                    event = EventFor(*flight, AdaptiveEventKind::kCancelled);
                    event.failure_category = result.failure_category();
                    event.diagnostic = result.diagnostic();
                }
                event.queue_depth = queue_.size();
                event.queue_wait = flight->compile_started_at - flight->queued_at;
                event.compile_time = flight->validating_at - flight->compile_started_at;
                event.validation_time = Now() - flight->validating_at;
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
        deadline_condition_.notify_one();
        if (emit_event) Emit(event);
    }

    std::shared_ptr<ArtifactCompiler> compiler_;
    std::shared_ptr<AdaptiveValidationAuthority> authority_;
    CoordinatorOptions options_;

    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::condition_variable deadline_condition_;
    mutable std::mutex completion_mutex_;
    bool stopping_{false};
    bool shutdown_completed_{false};
    std::vector<std::shared_ptr<Flight>> queue_;
    std::map<RequestIdentity, std::shared_ptr<Flight>> in_flight_;
    std::unordered_map<std::uint64_t, std::weak_ptr<Flight>> waiter_flights_;
    std::map<RequestIdentity, TerminalRecord> terminal_;
    std::unordered_set<std::uint64_t> completed_ids_;
    std::deque<std::uint64_t> completed_order_;
    std::uint64_t next_request_id_{1};
    std::uint64_t next_sequence_{1};
    std::uint64_t next_stamp_{1};
    std::size_t active_count_{0};
    std::size_t cached_artifact_bytes_{0};
    CoordinatorSnapshot metrics_;
};

class CompileCoordinator::ThreadGroup final {
private:
    class JoinService;

public:
    ThreadGroup(std::shared_ptr<State> state, std::size_t worker_count) {
        try {
            for (std::size_t i = 0; i < worker_count; ++i) {
                workers_.emplace_back([state] { state->WorkerLoop(); });
                worker_ids_.push_back(workers_.back().get_id());
            }
            workers_.emplace_back([state] { state->DeadlineLoop(); });
            worker_ids_.push_back(workers_.back().get_id());
        } catch (...) {
            state->RequestStop();
            for (auto& worker : workers_) {
                if (worker.joinable()) worker.join();
            }
            throw;
        }
    }

    bool IsWorkerThread() const noexcept {
        const auto current = std::this_thread::get_id();
        return std::find(worker_ids_.begin(), worker_ids_.end(), current) !=
               worker_ids_.end();
    }

    void Join() {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (joined_) return;
            if (joining_) {
                if (joining_thread_ == std::this_thread::get_id()) return;
                joined_condition_.wait(lock, [this] { return joined_; });
                return;
            }
            joining_ = true;
            joining_thread_ = std::this_thread::get_id();
        }
        for (auto& worker : workers_) {
            if (worker.joinable()) worker.join();
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            // Publish joined before State emits shutdown completion observers.
            joined_ = true;
            joining_ = false;
            joining_thread_ = std::thread::id();
        }
        joined_condition_.notify_all();
    }

    static void EnsureDeferredJoinService() {
        (void)JoinService::Instance();
    }

    static void DeferJoin(std::shared_ptr<ThreadGroup> group,
                          std::shared_ptr<State> state) noexcept {
        // Instance is created by CompileCoordinator before any worker starts.
        // Enqueue is intrusive and allocation-free, so a worker-side
        // destructor cannot lose ownership of its joinable thread handles.
        JoinService::Instance().Enqueue(std::move(group), std::move(state));
    }

private:
    class JoinService final {
    public:
        static JoinService& Instance() {
            static JoinService service;
            return service;
        }

        JoinService(const JoinService&) = delete;
        JoinService& operator=(const JoinService&) = delete;

        void Enqueue(std::shared_ptr<ThreadGroup> group,
                     std::shared_ptr<State> state) noexcept {
            ThreadGroup* const node = group.get();
            node->deferred_self_ = std::move(group);
            node->deferred_state_ = std::move(state);
            node->deferred_next_ = nullptr;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (tail_) {
                    tail_->deferred_next_ = node;
                } else {
                    head_ = node;
                }
                tail_ = node;
            }
            condition_.notify_one();
        }

    private:
        JoinService() : worker_([this] { Run(); }) {}
        ~JoinService() {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stopping_ = true;
            }
            condition_.notify_all();
            if (worker_.joinable()) worker_.join();
        }

        void Run() {
            while (true) {
                std::shared_ptr<ThreadGroup> group;
                std::shared_ptr<State> state;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    condition_.wait(lock, [this] {
                        return stopping_ || head_ != nullptr;
                    });
                    if (stopping_ && head_ == nullptr) return;
                    ThreadGroup* const node = head_;
                    head_ = node->deferred_next_;
                    if (!head_) tail_ = nullptr;
                    node->deferred_next_ = nullptr;
                    group = std::move(node->deferred_self_);
                    state = std::move(node->deferred_state_);
                }
                group->Join();
                state->CompleteShutdown();
            }
        }

        std::mutex mutex_;
        std::condition_variable condition_;
        ThreadGroup* head_{nullptr};
        ThreadGroup* tail_{nullptr};
        bool stopping_{false};
        std::thread worker_;
    };

    mutable std::mutex mutex_;
    std::condition_variable joined_condition_;
    std::vector<std::thread> workers_;
    std::vector<std::thread::id> worker_ids_;
    ThreadGroup* deferred_next_{nullptr};
    std::shared_ptr<ThreadGroup> deferred_self_;
    std::shared_ptr<State> deferred_state_;
    bool joining_{false};
    bool joined_{false};
    std::thread::id joining_thread_;
};

CompileCoordinator::CompileCoordinator(
    std::shared_ptr<ArtifactCompiler> compiler,
    std::shared_ptr<AdaptiveValidationAuthority> validation_authority,
    CoordinatorOptions options)
    : state_(std::make_shared<State>(std::move(compiler),
                                     std::move(validation_authority),
                                     std::move(options))) {
    ThreadGroup::EnsureDeferredJoinService();
    threads_ = std::make_shared<ThreadGroup>(state_, state_->worker_count());
}

CompileCoordinator::~CompileCoordinator() {
    const auto state = state_;
    const auto threads = threads_;
    if (!state || !threads) return;
    const auto events = state->RequestStop();
    state->Emit(events);
    if (threads->IsWorkerThread()) {
        ThreadGroup::DeferJoin(threads, state);
    } else {
        threads->Join();
        state->CompleteShutdown();
    }
}

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
    const auto state = state_;
    const auto threads = threads_;
    if (!state || !threads) return;
    const auto events = state->RequestStop();
    state->Emit(events);
    if (threads->IsWorkerThread()) return;
    threads->Join();
    state->CompleteShutdown();
}

}  // namespace kxc::api::experimental::adaptive::v1
