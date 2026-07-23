/*! \file adaptive_hot_swap_v2.h
 * \brief Default-OFF v2 whole-plan adaptive lifecycle control plane.
 *
 * This is a new API.  It does not change the v1 experimental claims.  It is
 * process-local CPU control-plane evidence only: it has no remote
 * authentication/attestation, CUDA pending-event proof, or device-resident
 * byte accounting.
 */
#pragma once

#ifndef KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2
#define KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2 0
#endif

#if KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2

/* v2 deliberately reuses the existing production compiler/primitive-cache
 * transaction adapter.  v1 remains separately named and source compatible. */
#ifndef KXC_ENABLE_EXPERIMENTAL_ADAPTIVE_PRODUCTION
#define KXC_ENABLE_EXPERIMENTAL_ADAPTIVE_PRODUCTION 1
#elif !KXC_ENABLE_EXPERIMENTAL_ADAPTIVE_PRODUCTION
#error "KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2 requires the v1 production adapter"
#endif

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>

#include "kxc/compiler/adaptive_production_experimental.h"

namespace kxc::api::adaptive::hot_swap::v2 {

namespace legacy = experimental::production_path;
using ProductionCompileRequest = legacy::ProductionCompileRequest;
using ProductionExecutionRequest = legacy::ProductionExecutionRequest;
using ProductionPathCompilerAdapter = legacy::ProductionPathCompilerAdapter;
using FrozenPlanVariant = legacy::FrozenPlanVariant;

inline constexpr uint32_t kAdaptiveHotSwapContractVersion = 2;
using Generation = uint64_t;

/*! \brief Per-waiter cooperative cancellation; cancelling one waiter never cancels a flight. */
class CancellationToken final {
public:
    CancellationToken() = default;
    bool cancelled() const noexcept;

private:
    struct State;
    explicit CancellationToken(std::shared_ptr<State> state);
    std::shared_ptr<State> state_;
    friend class CancellationSource;
};

class CancellationSource final {
public:
    CancellationSource();
    CancellationToken token() const noexcept;
    void Cancel() noexcept;

private:
    std::shared_ptr<CancellationToken::State> state_;
};

enum class FailureCategory : uint8_t {
    kPermanent,
    kTransient,
    kCancelled,
    kTimeout,
    kUnsupported,
    kBackpressure,
};

struct Failure final {
    FailureCategory category{FailureCategory::kTransient};
    std::string diagnostic;
    std::chrono::milliseconds retry_after{0};
    bool retryable{true};
};

/*! \brief Optional typed adapter failure; unsupported is permanently negative-cached. */
class CompileError final : public std::runtime_error {
public:
    CompileError(FailureCategory category, std::string diagnostic);
    FailureCategory category() const noexcept;
private:
    FailureCategory category_;
};

struct CompileRequest final {
    ProductionCompileRequest production;
    std::chrono::steady_clock::time_point deadline{
        std::chrono::steady_clock::time_point::max()};
    CancellationToken cancellation;
};

/*! \brief Immutable authority generation plus the frozen data-plane-only variant. */
class GenerationLease final {
public:
    Generation generation() const noexcept;
    const std::shared_ptr<const FrozenPlanVariant>& variant() const noexcept;
    const DispatchKey& dispatch_key() const noexcept;
    const PlanAbiFingerprint& plan_abi() const noexcept;
    uint64_t producer_reported_bytes() const noexcept;

private:
    GenerationLease(Generation generation, std::shared_ptr<const FrozenPlanVariant> variant,
                    uint64_t producer_reported_bytes);
    Generation generation_{0};
    std::shared_ptr<const FrozenPlanVariant> variant_;
    uint64_t producer_reported_bytes_{0};
    friend class AdaptiveHotSwapController;
};

struct CompileResult final {
    std::shared_ptr<const GenerationLease> lease;
    Failure failure;
    bool ready() const noexcept { return lease != nullptr; }
};

class CompileTicket final {
public:
    CompileTicket() = default;
    bool valid() const noexcept;
    CompileResult Wait() const;
    std::future_status WaitFor(std::chrono::milliseconds timeout) const;

private:
    CompileTicket(std::shared_future<CompileResult> result,
                  std::chrono::steady_clock::time_point deadline,
                  CancellationToken cancellation);
    std::shared_future<CompileResult> result_;
    std::chrono::steady_clock::time_point deadline_;
    CancellationToken cancellation_;
    friend class AdaptiveHotSwapController;
};

enum class HealthDisposition : uint8_t { kHealthy, kQuarantine };

/*! \brief Authority-issued decision; the token must be one-shot verified by its authority. */
struct HealthDecision final {
    Generation generation{0};
    HealthDisposition disposition{HealthDisposition::kHealthy};
    std::string evidence_id;
    std::string replay_token;
};

/*! \brief Process-local injection seam. No authentication is claimed unless supplied here. */
class HealthAuthority {
public:
    virtual ~HealthAuthority() = default;
    virtual HealthDecision Evaluate(const GenerationLease& lease) = 0;
    virtual bool VerifyAndConsume(const HealthDecision& decision,
                                  const GenerationLease& lease) noexcept = 0;
};

enum class EventKind : uint8_t {
    kQueued, kMerged, kPublished, kCancelled, kRetryCached, kEvicted,
    kNegativeEvicted, kNegativeCacheSaturated,
    kHealthDecision, kQuarantined, kQuarantineSaturated, kRolledBack, kRejected,
};

struct Event final {
    EventKind kind{EventKind::kQueued};
    Generation generation{0};
    Generation predecessor_generation{0};
    uint64_t producer_reported_bytes{0};
    std::chrono::milliseconds retry_after{0};
    std::string dispatch_key_digest;
    std::string plan_abi_digest;
    std::string diagnostic;
};
using Observer = std::function<void(const Event&)>;

struct Options final {
    /*! Test seam; production uses the default first generation. */
    Generation initial_generation{1};
    size_t worker_count{2};
    size_t max_queued_flights{64};
    size_t max_in_flight{64};
    size_t max_waiters_per_flight{1024};
    size_t max_discoverable_generations{8};
    uint64_t max_producer_reported_bytes{64ULL * 1024ULL * 1024ULL};
    size_t max_negative_cache_entries{64};
    size_t max_negative_diagnostic_bytes{64ULL * 1024ULL};
    size_t max_quarantine_tombstones_per_route{64};
    std::chrono::milliseconds transient_backoff{100};
    std::chrono::milliseconds timeout_backoff{100};
    Observer observer;
    std::shared_ptr<HealthAuthority> health_authority;
};

struct Snapshot final {
    Generation next_generation{1};
    size_t queued_flights{0};
    size_t active_flights{0};
    size_t discoverable_generations{0};
    uint64_t producer_reported_discoverable_bytes{0};
    uint64_t evictions{0};
    uint64_t merged_waiters{0};
    uint64_t retry_cached{0};
    size_t negative_cache_entries{0};
    size_t negative_cache_diagnostic_bytes{0};
    uint64_t negative_cache_evictions{0};
    uint64_t negative_cache_drops{0};
    bool negative_cache_compile_blocked{false};
    size_t quarantine_tombstones{0};
    size_t quarantine_compile_blocked_routes{0};
    uint64_t quarantine_saturations{0};
    // Not measured: process/device resident bytes or external lease ownership.
    bool process_resident_bytes_known{false};
    bool device_resident_bytes_known{false};
};

struct RunAsyncResult final {
    Array<runtime::NDArray> outputs;
    AsyncOperation completion;
    std::shared_ptr<const GenerationLease> lease;
};

/*! \brief v2 owns whole-plan scheduling/routing; RuntimeSession sees only FrozenPlanVariant. */
class AdaptiveHotSwapController final {
public:
    explicit AdaptiveHotSwapController(
        std::shared_ptr<ProductionPathCompilerAdapter> compiler = nullptr,
        Options options = {});
    ~AdaptiveHotSwapController();
    AdaptiveHotSwapController(const AdaptiveHotSwapController&) = delete;
    AdaptiveHotSwapController& operator=(const AdaptiveHotSwapController&) = delete;

    CompileTicket Submit(CompileRequest request);
    std::shared_ptr<const GenerationLease> CompileAndPublish(CompileRequest request);
    std::shared_ptr<const GenerationLease> Acquire(const ProductionExecutionRequest& request) const;
    RunAsyncResult RunAsync(const ProductionExecutionRequest& request,
                            const Array<runtime::NDArray>& inputs,
                            const DeviceStream& stream) const;
    /*! \brief Automatic policy evaluation is unavailable unless an authority was injected. */
    bool EvaluateHealth(const std::shared_ptr<const GenerationLease>& lease);
    /*! \brief Test-only cache administration; it never revokes a completed lease. */
    void ClearNegativeCacheForTesting();
    /*! \brief Test-only quarantine administration; it never revokes a completed lease. */
    void ClearQuarantinesForTesting();
    Snapshot SnapshotForTesting() const;

private:
    class State;
    std::shared_ptr<State> state_;
};

}  // namespace kxc::api::adaptive::hot_swap::v2
#endif  // KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2
