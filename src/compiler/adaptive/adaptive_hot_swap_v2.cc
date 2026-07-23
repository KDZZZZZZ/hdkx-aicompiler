/*! \file adaptive_hot_swap_v2.cc
 * \brief Bounded v2 whole-plan scheduling and routing authority.
 */

#include "kxc/compiler/adaptive_hot_swap_v2.h"

#if KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <limits>
#include <list>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace kxc::api::adaptive::hot_swap::v2 {
namespace {

std::string KeyPart(const std::string& value) {
    return std::to_string(value.size()) + ":" + value + ";";
}

std::string FlightKey(const ProductionCompileRequest& request) {
    return KeyPart(request.artifact_key().canonical_bytes()) +
           KeyPart(request.dispatch_key().canonical_bytes()) +
           KeyPart(request.plan_abi().canonical_bytes());
}

std::string RouteKey(const DispatchKey& dispatch, const PlanAbiFingerprint& abi) {
    return KeyPart(dispatch.canonical_bytes()) + KeyPart(abi.canonical_bytes());
}

Failure MakeFailure(FailureCategory category, std::string diagnostic,
                    std::chrono::milliseconds retry_after = std::chrono::milliseconds{0},
                    bool retryable = true) {
    return Failure{category, std::move(diagnostic), retry_after, retryable};
}

CompileResult Failed(Failure failure) {
    return CompileResult{nullptr, std::move(failure)};
}

Failure FailureFromException(const std::exception& error,
                             const Options& options) {
    if (const auto* typed = dynamic_cast<const CompileError*>(&error)) {
        const bool permanent = typed->category() == FailureCategory::kPermanent ||
                               typed->category() == FailureCategory::kUnsupported;
        return MakeFailure(typed->category(), typed->what(),
                           permanent ? std::chrono::milliseconds::max()
                           : typed->category() == FailureCategory::kTimeout
                               ? options.timeout_backoff
                               : options.transient_backoff,
                           !permanent);
    }
    if (dynamic_cast<const std::overflow_error*>(&error)) {
        return MakeFailure(FailureCategory::kPermanent, error.what(),
                           std::chrono::milliseconds::max(), false);
    }
    if (dynamic_cast<const std::invalid_argument*>(&error)) {
        return MakeFailure(FailureCategory::kPermanent, error.what(),
                           std::chrono::milliseconds::max(), false);
    }
    return MakeFailure(FailureCategory::kTransient, error.what(),
                       options.transient_backoff, true);
}

uint64_t ProducerBytes(const FrozenPlanVariant& variant) {
    uint64_t total = 0;
    for (const auto& pin : variant.compiled_graph().artifact_pins) {
        const uint64_t bytes = pin.handle().record().byte_size;
        if (bytes > std::numeric_limits<uint64_t>::max() - total) {
            throw std::overflow_error("adaptive v2 producer byte sum overflow");
        }
        total += bytes;
    }
    return total;
}

class CallbackScope final {
public:
    explicit CallbackScope(std::atomic<size_t>& count) noexcept : count_(count) {
        count_.fetch_add(1, std::memory_order_acq_rel);
    }
    ~CallbackScope() { count_.fetch_sub(1, std::memory_order_release); }
    CallbackScope(const CallbackScope&) = delete;
    CallbackScope& operator=(const CallbackScope&) = delete;
private:
    std::atomic<size_t>& count_;
};

struct RunRetention final {
    std::shared_ptr<const GenerationLease> lease;
};

}  // namespace

struct CancellationToken::State final {
    std::atomic<bool> cancelled{false};
};

CancellationToken::CancellationToken(std::shared_ptr<State> state)
    : state_(std::move(state)) {}

bool CancellationToken::cancelled() const noexcept {
    return state_ && state_->cancelled.load(std::memory_order_acquire);
}

CancellationSource::CancellationSource()
    : state_(std::make_shared<CancellationToken::State>()) {}

CancellationToken CancellationSource::token() const noexcept {
    return CancellationToken(state_);
}

void CancellationSource::Cancel() noexcept {
    if (state_) state_->cancelled.store(true, std::memory_order_release);
}

CompileError::CompileError(FailureCategory category, std::string diagnostic)
    : std::runtime_error(std::move(diagnostic)), category_(category) {}

FailureCategory CompileError::category() const noexcept { return category_; }

GenerationLease::GenerationLease(Generation generation,
                                 std::shared_ptr<const FrozenPlanVariant> variant,
                                 uint64_t producer_reported_bytes)
    : generation_(generation), variant_(std::move(variant)),
      producer_reported_bytes_(producer_reported_bytes) {
    if (generation_ == 0 || !variant_) {
        throw std::invalid_argument("adaptive v2 lease requires a generation and variant");
    }
}

Generation GenerationLease::generation() const noexcept { return generation_; }
const std::shared_ptr<const FrozenPlanVariant>& GenerationLease::variant() const noexcept {
    return variant_;
}
const DispatchKey& GenerationLease::dispatch_key() const noexcept {
    return variant_->dispatch_key();
}
const PlanAbiFingerprint& GenerationLease::plan_abi() const noexcept {
    return variant_->plan_abi();
}
uint64_t GenerationLease::producer_reported_bytes() const noexcept {
    return producer_reported_bytes_;
}

CompileTicket::CompileTicket(std::shared_future<CompileResult> result,
                             std::chrono::steady_clock::time_point deadline,
                             CancellationToken cancellation)
    : result_(std::move(result)), deadline_(deadline),
      cancellation_(std::move(cancellation)) {}

bool CompileTicket::valid() const noexcept { return result_.valid(); }

CompileResult CompileTicket::Wait() const {
    if (!result_.valid()) {
        return Failed(MakeFailure(FailureCategory::kPermanent, "invalid compile ticket", {}, false));
    }
    constexpr auto kCancellationPoll = std::chrono::milliseconds(1);
    for (;;) {
        if (cancellation_.cancelled()) {
            return Failed(MakeFailure(FailureCategory::kCancelled, "waiter cancelled", {}, false));
        }
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline_) {
            return Failed(MakeFailure(FailureCategory::kTimeout, "waiter deadline expired", {}, true));
        }
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline_ - now);
        if (remaining <= std::chrono::milliseconds::zero()) continue;
        if (result_.wait_for(std::min(kCancellationPoll, remaining)) ==
            std::future_status::ready) {
            return cancellation_.cancelled()
                ? Failed(MakeFailure(FailureCategory::kCancelled, "waiter cancelled", {}, false))
                : result_.get();
        }
    }
}

std::future_status CompileTicket::WaitFor(std::chrono::milliseconds timeout) const {
    if (!result_.valid()) return std::future_status::timeout;
    constexpr auto kCancellationPoll = std::chrono::milliseconds(1);
    const auto finish = std::min(deadline_, std::chrono::steady_clock::now() + timeout);
    for (;;) {
        if (cancellation_.cancelled() || std::chrono::steady_clock::now() >= finish) {
            return std::future_status::timeout;
        }
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
            finish - std::chrono::steady_clock::now());
        if (remaining <= std::chrono::milliseconds::zero()) continue;
        if (result_.wait_for(std::min(kCancellationPoll, remaining)) ==
            std::future_status::ready) {
            return std::future_status::ready;
        }
    }
}

class AdaptiveHotSwapController::State final
    : public std::enable_shared_from_this<AdaptiveHotSwapController::State> {
public:
    struct Flight final {
        explicit Flight(ProductionCompileRequest value, std::string key_value,
                        std::string route_value)
            : request(std::move(value)), key(std::move(key_value)),
              route_key(std::move(route_value)), future(promise.get_future().share()) {}
        ProductionCompileRequest request;
        std::string key;
        std::string route_key;
        std::promise<CompileResult> promise;
        std::shared_future<CompileResult> future;
        size_t waiters{1};
    };

    struct Route final {
        std::shared_ptr<const GenerationLease> current;
        std::deque<std::shared_ptr<const GenerationLease>> history;
        std::unordered_set<std::string> quarantined_artifacts;
        bool compile_blocked{false};
    };

    struct Negative final {
        Failure failure;
        std::chrono::steady_clock::time_point expires;
        std::list<std::string>::iterator order;
    };

    State(std::shared_ptr<ProductionPathCompilerAdapter> compiler_value,
          Options options_value)
        : options(std::move(options_value)) {
        if (options.initial_generation == 0 || options.worker_count == 0 ||
            options.max_queued_flights == 0 || options.max_in_flight == 0 ||
            options.max_waiters_per_flight == 0 ||
            options.max_discoverable_generations == 0 ||
            options.max_producer_reported_bytes == 0 ||
            options.max_negative_cache_entries == 0 ||
            options.max_negative_diagnostic_bytes == 0 ||
            options.max_quarantine_tombstones_per_route == 0 ||
            options.max_queued_flights >
                std::numeric_limits<size_t>::max() - options.max_in_flight) {
            throw std::invalid_argument("adaptive v2 bounds are invalid");
        }
        legacy::AdaptiveControllerOptions legacy_options;
        legacy_options.max_in_flight_compiles = options.max_in_flight;
        legacy_options.max_slots = options.max_queued_flights + options.max_in_flight;
        legacy_options.max_discoverable_generations = std::max<size_t>(
            2, options.max_discoverable_generations);
        compiler = std::make_unique<legacy::AdaptiveController>(
            std::move(compiler_value), std::move(legacy_options));
        next_generation = options.initial_generation;
    }

    void StartWorkers() {
        workers.reserve(options.worker_count);
        for (size_t index = 0; index < options.worker_count; ++index) {
            workers.emplace_back([this] { Worker(); });
        }
    }

    ~State() { Stop(); }

    void Stop() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        wake.notify_all();
        const std::thread::id self = std::this_thread::get_id();
        for (auto& worker : workers) {
            if (!worker.joinable()) continue;
            if (worker.get_id() == self) {
                worker.detach();
            } else {
                worker.join();
            }
        }
    }

    void RejectReentry() const {
        if (callbacks.load(std::memory_order_acquire) != 0) {
            throw std::logic_error("adaptive v2 API called while callback is active");
        }
    }

    void Emit(Event event) const noexcept {
        if (!options.observer) return;
        try {
            CallbackScope scope(callbacks);
            options.observer(event);
        } catch (...) {
            // Observability is never routing authority.
        }
    }

    void Worker() {
        // An observer may release the final controller reference on this thread.
        // Keep State alive until the worker has observed shutdown and returned.
        const std::shared_ptr<State> keep_alive = shared_from_this();
        for (;;) {
            std::shared_ptr<Flight> flight;
            {
                std::unique_lock<std::mutex> lock(mutex);
                wake.wait(lock, [this] { return stopping || !queue.empty(); });
                if (stopping && queue.empty()) return;
                flight = queue.front();
                queue.pop_front();
                ++active;
            }
            Compile(flight);
            {
                std::lock_guard<std::mutex> lock(mutex);
                --active;
            }
            wake.notify_all();
        }
    }

    static bool IsPermanentNegative(const Failure& failure) noexcept {
        return failure.category == FailureCategory::kPermanent ||
               failure.category == FailureCategory::kUnsupported;
    }

    void EraseNegativeLocked(const std::string& key) {
        const auto found = negative.find(key);
        if (found == negative.end()) return;
        negative_diagnostic_bytes -= found->second.failure.diagnostic.size();
        negative_order.erase(found->second.order);
        negative.erase(found);
    }

    bool EvictOldestRetryableNegativeLocked(std::vector<Event>* events) {
        for (auto it = negative_order.begin(); it != negative_order.end(); ++it) {
            const auto found = negative.find(*it);
            if (found == negative.end()) continue;
            if (IsPermanentNegative(found->second.failure)) continue;
            Event event;
            event.kind = EventKind::kNegativeEvicted;
            event.diagnostic = "bounded negative-cache eviction";
            events->push_back(std::move(event));
            EraseNegativeLocked(*it);
            ++negative_cache_evictions;
            return true;
        }
        return false;
    }

    void RemoveExpiredNegativesLocked(
        const std::chrono::steady_clock::time_point now, std::vector<Event>* events) {
        for (auto it = negative_order.begin(); it != negative_order.end();) {
            const std::string key = *it++;
            const auto found = negative.find(key);
            if (found != negative.end() && found->second.expires <= now) {
                Event event;
                event.kind = EventKind::kNegativeEvicted;
                event.diagnostic = "expired negative-cache entry";
                events->push_back(std::move(event));
                EraseNegativeLocked(key);
                ++negative_cache_evictions;
            }
        }
    }

    void CacheFailureLocked(const std::string& key, const Failure& failure,
                            std::vector<Event>* events) {
        if (failure.category == FailureCategory::kCancelled ||
            failure.category == FailureCategory::kBackpressure ||
            (failure.retry_after != std::chrono::milliseconds::max() &&
             failure.retry_after <= std::chrono::milliseconds::zero())) {
            return;
        }
        RemoveExpiredNegativesLocked(std::chrono::steady_clock::now(), events);
        const bool permanent = IsPermanentNegative(failure);
        while (negative.size() >= options.max_negative_cache_entries &&
               EvictOldestRetryableNegativeLocked(events)) {
        }
        if (negative.size() >= options.max_negative_cache_entries) {
            ++negative_cache_drops;
            if (permanent) {
                negative_cache_compile_blocked = true;
                Event event;
                event.kind = EventKind::kNegativeCacheSaturated;
                event.diagnostic = "permanent negative-cache capacity exhausted; compilation blocked";
                events->push_back(std::move(event));
            }
            return;
        }
        Failure cached = failure;
        const size_t available = options.max_negative_diagnostic_bytes -
            negative_diagnostic_bytes;
        if (cached.diagnostic.size() > available) cached.diagnostic.resize(available);
        const auto expires = failure.retry_after == std::chrono::milliseconds::max()
            ? std::chrono::steady_clock::time_point::max()
            : std::chrono::steady_clock::now() + failure.retry_after;
        negative_order.push_back(key);
        auto order = std::prev(negative_order.end());
        negative.emplace(key, Negative{std::move(cached), expires, order});
        negative_diagnostic_bytes += negative.find(key)->second.failure.diagnostic.size();
    }

    void Finish(const std::shared_ptr<Flight>& flight, CompileResult result,
                bool cache_failure) {
        Event event;
        event.kind = result.ready() ? EventKind::kPublished : EventKind::kRejected;
        event.dispatch_key_digest = flight->request.dispatch_key().digest();
        event.plan_abi_digest = flight->request.plan_abi().digest();
        if (result.ready()) {
            event.generation = result.lease->generation();
            event.producer_reported_bytes = result.lease->producer_reported_bytes();
        } else {
            event.diagnostic = result.failure.diagnostic;
            event.retry_after = result.failure.retry_after;
        }
        std::vector<Event> cache_events;
        {
            std::lock_guard<std::mutex> lock(mutex);
            flights.erase(flight->key);
            if (cache_failure) CacheFailureLocked(flight->key, result.failure, &cache_events);
        }
        flight->promise.set_value(std::move(result));
        Emit(std::move(event));
        for (auto& cache_event : cache_events) {
            cache_event.dispatch_key_digest = flight->request.dispatch_key().digest();
            cache_event.plan_abi_digest = flight->request.plan_abi().digest();
            Emit(std::move(cache_event));
        }
    }

    void Compile(const std::shared_ptr<Flight>& flight) {
        try {
            const auto candidate = compiler->CompileAndPublish(flight->request);
            if (!candidate || candidate->dispatch_key() != flight->request.dispatch_key() ||
                candidate->plan_abi() != flight->request.plan_abi()) {
                throw CompileError(FailureCategory::kPermanent,
                    "candidate exact dispatch or PlanAbi compatibility mismatch");
            }
            const uint64_t bytes = ProducerBytes(*candidate);
            if (bytes > options.max_producer_reported_bytes) {
                throw CompileError(FailureCategory::kPermanent,
                    "candidate exceeds adaptive v2 producer byte budget");
            }

            std::shared_ptr<const GenerationLease> lease;
            Event published;
            std::vector<Event> evictions;
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (next_generation == std::numeric_limits<Generation>::max()) {
                    throw std::overflow_error("adaptive v2 generation space is exhausted");
                }
                if (negative_cache_compile_blocked) {
                    throw CompileError(FailureCategory::kPermanent,
                        "adaptive v2 permanent negative-cache capacity is fail-closed");
                }
                // Stage every allocation and eviction before replacing routing authority.
                auto staged_routes = routes;
                auto staged_discoverable = discoverable;
                uint64_t staged_bytes = discoverable_bytes;
                auto route = staged_routes.find(flight->route_key);
                if (route == staged_routes.end()) {
                    route = staged_routes.emplace(flight->route_key, Route{}).first;
                }
                if (route->second.compile_blocked) {
                    throw CompileError(FailureCategory::kPermanent,
                        "route compilation is fail-closed after quarantine tombstone saturation");
                }
                if (route->second.quarantined_artifacts.count(
                        flight->request.artifact_key().canonical_bytes()) != 0) {
                    throw CompileError(FailureCategory::kPermanent,
                        "candidate whole-plan identity is quarantined");
                }
                if (bytes > std::numeric_limits<uint64_t>::max() - staged_bytes) {
                    throw std::overflow_error(
                        "adaptive v2 discoverable byte accounting overflow");
                }
                lease = std::shared_ptr<const GenerationLease>(
                    new GenerationLease(next_generation, candidate, bytes));
                const Generation predecessor =
                    route->second.current ? route->second.current->generation() : 0;
                route->second.current = lease;
                route->second.history.push_back(lease);
                staged_discoverable.push_back(lease);
                staged_bytes += bytes;
                EvictStaged(&staged_routes, &staged_discoverable, &staged_bytes, &evictions);

                published.kind = EventKind::kPublished;
                published.generation = lease->generation();
                published.predecessor_generation = predecessor;
                published.producer_reported_bytes = bytes;
                published.dispatch_key_digest = lease->dispatch_key().digest();
                published.plan_abi_digest = lease->plan_abi().digest();

                routes.swap(staged_routes);
                discoverable.swap(staged_discoverable);
                discoverable_bytes = staged_bytes;
                next_generation = lease->generation() + 1;
                this->evictions += evictions.size();
                flights.erase(flight->key);
            }
            flight->promise.set_value(CompileResult{lease, {}});
            Emit(std::move(published));
            for (auto& event : evictions) Emit(std::move(event));
        } catch (const std::exception& error) {
            Finish(flight, Failed(FailureFromException(error, options)), true);
        } catch (...) {
            Finish(flight, Failed(MakeFailure(FailureCategory::kTransient,
                "non-standard compilation failure", options.transient_backoff, true)), true);
        }
    }

    void EvictStaged(std::unordered_map<std::string, Route>* staged_routes,
                     std::deque<std::shared_ptr<const GenerationLease>>* staged_discoverable,
                     uint64_t* staged_bytes, std::vector<Event>* events) const {
        while (!staged_discoverable->empty() &&
               (staged_discoverable->size() > options.max_discoverable_generations ||
                *staged_bytes > options.max_producer_reported_bytes)) {
            const auto lease = staged_discoverable->front();
            staged_discoverable->pop_front();
            *staged_bytes -= lease->producer_reported_bytes();
            const std::string key = RouteKey(lease->dispatch_key(), lease->plan_abi());
            const auto route = staged_routes->find(key);
            if (route != staged_routes->end()) {
                auto& history = route->second.history;
                history.erase(std::remove(history.begin(), history.end(), lease), history.end());
                if (route->second.current == lease) route->second.current.reset();
                if (!route->second.current && history.empty() &&
                    route->second.quarantined_artifacts.empty()) {
                    staged_routes->erase(route);
                }
            }
            Event event;
            event.kind = EventKind::kEvicted;
            event.generation = lease->generation();
            event.producer_reported_bytes = lease->producer_reported_bytes();
            event.dispatch_key_digest = lease->dispatch_key().digest();
            event.plan_abi_digest = lease->plan_abi().digest();
            events->push_back(std::move(event));
        }
    }

    std::unique_ptr<legacy::AdaptiveController> compiler;
    const Options options;
    mutable std::mutex mutex;
    std::condition_variable wake;
    std::deque<std::shared_ptr<Flight>> queue;
    std::unordered_map<std::string, std::shared_ptr<Flight>> flights;
    std::unordered_map<std::string, Route> routes;
    std::unordered_map<std::string, Negative> negative;
    std::list<std::string> negative_order;
    std::deque<std::shared_ptr<const GenerationLease>> discoverable;
    std::vector<std::thread> workers;
    mutable std::atomic<size_t> callbacks{0};
    Generation next_generation{1};
    size_t active{0};
    uint64_t discoverable_bytes{0};
    uint64_t evictions{0};
    uint64_t merged_waiters{0};
    uint64_t retry_cached{0};
    size_t negative_diagnostic_bytes{0};
    uint64_t negative_cache_evictions{0};
    uint64_t negative_cache_drops{0};
    bool negative_cache_compile_blocked{false};
    uint64_t quarantine_saturations{0};
    bool stopping{false};
};

AdaptiveHotSwapController::AdaptiveHotSwapController(
    std::shared_ptr<ProductionPathCompilerAdapter> compiler, Options options)
    : state_(std::make_shared<State>(std::move(compiler), std::move(options))) {
    state_->StartWorkers();
}

AdaptiveHotSwapController::~AdaptiveHotSwapController() {
    if (state_) state_->Stop();
}

CompileTicket AdaptiveHotSwapController::Submit(CompileRequest request) {
    state_->RejectReentry();
    request.production.Validate();
    const std::string key = FlightKey(request.production);
    const std::string route = RouteKey(request.production.dispatch_key(), request.production.plan_abi());
    std::shared_future<CompileResult> result;
    Event event;
    bool emit = false;
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        const auto now = std::chrono::steady_clock::now();
        if (request.cancellation.cancelled()) {
            std::promise<CompileResult> promise;
            result = promise.get_future().share();
            promise.set_value(Failed(MakeFailure(FailureCategory::kCancelled,
                "waiter cancelled", {}, false)));
            event.kind = EventKind::kCancelled;
            emit = true;
        } else if (now >= request.deadline) {
            std::promise<CompileResult> promise;
            result = promise.get_future().share();
            promise.set_value(Failed(MakeFailure(FailureCategory::kTimeout,
                "waiter deadline expired", {}, true)));
            event.kind = EventKind::kCancelled;
            emit = true;
        } else if (const auto cached = state_->negative.find(key);
                   cached != state_->negative.end() && cached->second.expires > now) {
            std::promise<CompileResult> promise;
            result = promise.get_future().share();
            Failure failure = cached->second.failure;
            if (cached->second.expires != std::chrono::steady_clock::time_point::max()) {
                failure.retry_after = std::chrono::duration_cast<std::chrono::milliseconds>(
                    cached->second.expires - now);
            }
            promise.set_value(Failed(std::move(failure)));
            ++state_->retry_cached;
            event.kind = EventKind::kRetryCached;
            event.retry_after = cached->second.failure.retry_after;
            emit = true;
        } else {
            state_->EraseNegativeLocked(key);
            if (state_->negative_cache_compile_blocked) {
                std::promise<CompileResult> promise;
                result = promise.get_future().share();
                promise.set_value(Failed(MakeFailure(FailureCategory::kPermanent,
                    "adaptive v2 permanent negative-cache capacity is fail-closed", {}, false)));
                event.kind = EventKind::kRejected;
                emit = true;
            } else if (const auto existing = state_->flights.find(key);
                       existing != state_->flights.end()) {
                if (existing->second->waiters >= state_->options.max_waiters_per_flight) {
                    std::promise<CompileResult> promise;
                    result = promise.get_future().share();
                    promise.set_value(Failed(MakeFailure(FailureCategory::kBackpressure,
                        "adaptive v2 waiter budget is full", {}, true)));
                    event.kind = EventKind::kRejected;
                } else {
                    ++existing->second->waiters;
                    ++state_->merged_waiters;
                    result = existing->second->future;
                    event.kind = EventKind::kMerged;
                }
                emit = true;
            } else if (state_->queue.size() >= state_->options.max_queued_flights ||
                       state_->flights.size() >= state_->options.max_in_flight) {
                std::promise<CompileResult> promise;
                result = promise.get_future().share();
                promise.set_value(Failed(MakeFailure(FailureCategory::kBackpressure,
                    "adaptive v2 queue or in-flight budget is full", {}, true)));
                event.kind = EventKind::kRejected;
                emit = true;
            } else {
                auto flight = std::make_shared<State::Flight>(
                    std::move(request.production), key, route);
                result = flight->future;
                state_->flights.emplace(key, flight);
                state_->queue.push_back(std::move(flight));
                event.kind = EventKind::kQueued;
                emit = true;
                state_->wake.notify_one();
            }
        }
    }
    event.dispatch_key_digest = request.production.dispatch_key().digest();
    event.plan_abi_digest = request.production.plan_abi().digest();
    if (emit) state_->Emit(std::move(event));
    return CompileTicket(std::move(result), request.deadline, std::move(request.cancellation));
}

std::shared_ptr<const GenerationLease>
AdaptiveHotSwapController::CompileAndPublish(CompileRequest request) {
    const CompileResult result = Submit(std::move(request)).Wait();
    if (result.ready()) return result.lease;
    throw CompileError(result.failure.category, result.failure.diagnostic);
}

std::shared_ptr<const GenerationLease>
AdaptiveHotSwapController::Acquire(const ProductionExecutionRequest& request) const {
    state_->RejectReentry();
    request.Validate();
    std::shared_ptr<const GenerationLease> result;
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        const auto route = state_->routes.find(RouteKey(request.dispatch_key(), request.plan_abi()));
        if (route != state_->routes.end() && route->second.current &&
            route->second.current->plan_abi() == request.plan_abi() &&
            route->second.quarantined_artifacts.count(route->second.current->variant()->artifact_lease()
                .artifact_key().canonical_bytes()) == 0) {
            result = route->second.current;
        }
    }
    if (!result) throw std::out_of_range("no exact published adaptive v2 generation");
    return result;
}

RunAsyncResult AdaptiveHotSwapController::RunAsync(
    const ProductionExecutionRequest& request,
    const Array<runtime::NDArray>& inputs, const DeviceStream& stream) const {
    const auto lease = Acquire(request);
    runtime::RunAsyncResult result = lease->variant()->session()->RunAsync(inputs, stream);
    result.completion.RetainDependencies({}, std::make_shared<RunRetention>(RunRetention{lease}));
    return RunAsyncResult{std::move(result.outputs), std::move(result.completion), lease};
}

bool AdaptiveHotSwapController::EvaluateHealth(
    const std::shared_ptr<const GenerationLease>& lease) {
    state_->RejectReentry();
    if (!lease || !state_->options.health_authority) return false;
    HealthDecision decision;
    bool verified = false;
    try {
        CallbackScope scope(state_->callbacks);
        decision = state_->options.health_authority->Evaluate(*lease);
        verified = state_->options.health_authority->VerifyAndConsume(decision, *lease);
    } catch (...) {
        return false;
    }
    Event health;
    health.kind = EventKind::kHealthDecision;
    health.generation = lease->generation();
    health.diagnostic = verified ? decision.evidence_id : "health decision rejected";
    state_->Emit(std::move(health));
    if (!verified || decision.generation != lease->generation() ||
        decision.disposition != HealthDisposition::kQuarantine) {
        return verified;
    }

    std::shared_ptr<const GenerationLease> predecessor;
    bool tombstone_saturated = false;
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        const auto found = state_->routes.find(RouteKey(lease->dispatch_key(), lease->plan_abi()));
        if (found == state_->routes.end() || found->second.current != lease) return false;
        for (auto it = found->second.history.rbegin(); it != found->second.history.rend(); ++it) {
            if ((*it)->generation() < lease->generation() &&
                found->second.quarantined_artifacts.count((*it)->variant()->artifact_lease()
                    .artifact_key().canonical_bytes()) == 0) {
                predecessor = *it;
                break;
            }
        }
        const std::string artifact =
            lease->variant()->artifact_lease().artifact_key().canonical_bytes();
        if (found->second.quarantined_artifacts.count(artifact) == 0 &&
            found->second.quarantined_artifacts.size() >=
                state_->options.max_quarantine_tombstones_per_route) {
            // Never evict a tombstone: blocking the route preserves fail-closed
            // publication while a healthy predecessor remains acquirable.
            found->second.compile_blocked = true;
            ++state_->quarantine_saturations;
            tombstone_saturated = true;
        } else {
            found->second.quarantined_artifacts.insert(artifact);
        }
        // Quarantine is durable even when no rollback target exists: never leave
        // a verified-bad generation routable merely because it was the first one.
        found->second.current = predecessor;
    }
    Event quarantined;
    quarantined.kind = tombstone_saturated ? EventKind::kQuarantineSaturated
                                           : EventKind::kQuarantined;
    quarantined.generation = lease->generation();
    quarantined.predecessor_generation = predecessor ? predecessor->generation() : 0;
    quarantined.diagnostic = tombstone_saturated
        ? "quarantine tombstone capacity exhausted; route publication blocked: " +
            decision.evidence_id
        : decision.evidence_id;
    state_->Emit(std::move(quarantined));
    if (predecessor) {
        Event rollback;
        rollback.kind = EventKind::kRolledBack;
        rollback.generation = predecessor->generation();
        rollback.predecessor_generation = lease->generation();
        rollback.diagnostic = decision.evidence_id;
        state_->Emit(std::move(rollback));
    }
    return true;
}

Snapshot AdaptiveHotSwapController::SnapshotForTesting() const {
    state_->RejectReentry();
    std::lock_guard<std::mutex> lock(state_->mutex);
    size_t tombstones = 0;
    size_t blocked_routes = 0;
    for (const auto& [key, route] : state_->routes) {
        static_cast<void>(key);
        tombstones += route.quarantined_artifacts.size();
        if (route.compile_blocked) ++blocked_routes;
    }
    return Snapshot{state_->next_generation, state_->queue.size(), state_->active,
                    state_->discoverable.size(), state_->discoverable_bytes,
                    state_->evictions, state_->merged_waiters, state_->retry_cached,
                    state_->negative.size(), state_->negative_diagnostic_bytes,
                    state_->negative_cache_evictions, state_->negative_cache_drops,
                    state_->negative_cache_compile_blocked, tombstones, blocked_routes,
                    state_->quarantine_saturations, false, false};
}

void AdaptiveHotSwapController::ClearNegativeCacheForTesting() {
    state_->RejectReentry();
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->negative.clear();
    state_->negative_order.clear();
    state_->negative_diagnostic_bytes = 0;
    state_->negative_cache_compile_blocked = false;
}

void AdaptiveHotSwapController::ClearQuarantinesForTesting() {
    state_->RejectReentry();
    std::lock_guard<std::mutex> lock(state_->mutex);
    for (auto it = state_->routes.begin(); it != state_->routes.end();) {
        it->second.quarantined_artifacts.clear();
        it->second.compile_blocked = false;
        if (!it->second.current && it->second.history.empty()) {
            it = state_->routes.erase(it);
        } else {
            ++it;
        }
    }
}

}  // namespace kxc::api::adaptive::hot_swap::v2
#endif  // KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2
