/*! \file adaptive_hot_swap_v2.h
 * \brief Default-OFF v2 whole-plan adaptive lifecycle control plane.
 *
 * This API is process-local experimental control-plane evidence only.  The
 * default authority below is test authority, not authentication or attestation.
 */
#pragma once

#ifndef KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2
#define KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2 0
#endif

#if KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2
#ifndef KXC_ENABLE_EXPERIMENTAL_ADAPTIVE_PRODUCTION
#define KXC_ENABLE_EXPERIMENTAL_ADAPTIVE_PRODUCTION 1
#elif !KXC_ENABLE_EXPERIMENTAL_ADAPTIVE_PRODUCTION
#error "KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2 requires the production adapter"
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
using PreparedCandidate = legacy::PreparedCandidate;
inline constexpr uint32_t kAdaptiveHotSwapContractVersion = 3;
using Generation = uint64_t;

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

enum class FailureCategory : uint8_t { kPermanent, kTransient, kCancelled, kTimeout, kUnsupported, kBackpressure };
struct Failure final {
    FailureCategory category{FailureCategory::kTransient};
    std::string diagnostic;
    std::chrono::milliseconds retry_after{0};
    bool retryable{true};
};
class CompileError final : public std::runtime_error {
public:
    CompileError(FailureCategory category, std::string diagnostic);
    FailureCategory category() const noexcept;
private:
    FailureCategory category_;
};
struct CompileRequest final {
    ProductionCompileRequest production;
    std::chrono::steady_clock::time_point deadline{std::chrono::steady_clock::time_point::max()};
    CancellationToken cancellation;
};

/*! \brief Non-forgeable validation evidence supplied by an injected authority. */
class ValidationReceipt final {
public:
    const std::string& value() const noexcept;
private:
    explicit ValidationReceipt(std::string value);
    std::string value_;
    friend class CandidateValidationAuthority;
};
/*! \brief Must perform policy validation after structural typed preparation.
 * Implementations must be thread-safe: different flights may call concurrently. */
class CandidateValidationAuthority {
public:
    virtual ~CandidateValidationAuthority() = default;
    virtual ValidationReceipt Validate(const ProductionCompileRequest& request,
                                       const CompiledGraph& graph) = 0;
protected:
    static ValidationReceipt IssueReceipt(std::string value);
};

class GenerationLease;
struct GenerationAuthorityRequest final {
    const DispatchKey& route;
    const ArtifactKey& selection_artifact;
    const PlanAbiFingerprint& plan_abi;
    const ValidationReceipt& validation_receipt;
    std::shared_ptr<const PreparedCandidate> candidate;
    uint64_t producer_reported_bytes{0};
};
/*! \brief Issues opaque monotonic leases. Controller code cannot mint lease internals.
 * Authorities must be thread-safe; v2 serializes issuance but may be called by
 * other control planes. The built-in authority is process-local test evidence,
 * not an authentication/attestation service. */
class GenerationAuthority {
public:
    virtual ~GenerationAuthority() = default;
    virtual std::shared_ptr<const GenerationLease> Issue(
        const GenerationAuthorityRequest& request) = 0;
    virtual Generation NextGenerationForTesting() const noexcept = 0;
protected:
    static std::shared_ptr<const GenerationLease> MakeLease(
        Generation generation, const GenerationAuthorityRequest& request);
};

/*! \brief Opaque authority lease retaining immutable candidate/pins and receipt. */
class GenerationLease final {
public:
    Generation generation() const noexcept;
    const std::shared_ptr<const FrozenPlanVariant>& variant() const noexcept;
    const DispatchKey& dispatch_key() const noexcept;
    const PlanAbiFingerprint& plan_abi() const noexcept;
    const ArtifactKey& selection_artifact_key() const noexcept;
    const std::string& validation_receipt() const noexcept;
    uint64_t producer_reported_bytes() const noexcept;
private:
    GenerationLease(Generation generation, std::shared_ptr<const FrozenPlanVariant> variant,
                    DispatchKey route, ArtifactKey selection_artifact,
                    PlanAbiFingerprint plan_abi, std::string validation_receipt,
                    uint64_t producer_reported_bytes);
    Generation generation_{0};
    std::shared_ptr<const FrozenPlanVariant> variant_;
    DispatchKey route_;
    ArtifactKey selection_artifact_;
    PlanAbiFingerprint plan_abi_;
    std::string validation_receipt_;
    uint64_t producer_reported_bytes_{0};
    friend class GenerationAuthority;
};

struct CompileResult final { std::shared_ptr<const GenerationLease> lease; Failure failure; bool ready() const noexcept { return lease != nullptr; } };
class CompileTicket final {
public:
    CompileTicket() = default;
    bool valid() const noexcept;
    CompileResult Wait() const;
    std::future_status WaitFor(std::chrono::milliseconds timeout) const;
private:
    CompileTicket(std::shared_future<CompileResult> result, std::chrono::steady_clock::time_point deadline,
                  CancellationToken cancellation);
    std::shared_future<CompileResult> result_;
    std::chrono::steady_clock::time_point deadline_;
    CancellationToken cancellation_;
    friend class AdaptiveHotSwapController;
};

enum class HealthDisposition : uint8_t { kHealthy, kQuarantine };
struct HealthDecision final { Generation generation{0}; HealthDisposition disposition{HealthDisposition::kHealthy}; std::string evidence_id; std::string replay_token; };
/*! \brief Evaluate may run concurrently; VerifyAndConsume is serialized by v2 and
 * invoked only while the referenced lease is still the route head. */
class HealthAuthority {
public:
    virtual ~HealthAuthority() = default;
    virtual HealthDecision Evaluate(const GenerationLease& lease) = 0;
    virtual bool VerifyAndConsume(const HealthDecision& decision, const GenerationLease& lease) noexcept = 0;
};

enum class PublicationStage : uint8_t { kByteBudget, kRoute, kQuarantine, kAuthority, kMapAllocation, kFinalCommit };
enum class EventKind : uint8_t { kQueued, kMerged, kPublished, kCancelled, kRetryCached, kEvicted, kNegativeEvicted, kNegativeCacheSaturated, kHealthDecision, kQuarantined, kQuarantineSaturated, kRolledBack, kRejected, kRouteSaturated, kTombstoneSaturated };
struct Event final { EventKind kind{EventKind::kQueued}; Generation generation{0}; Generation predecessor_generation{0}; uint64_t producer_reported_bytes{0}; std::chrono::milliseconds retry_after{0}; std::string dispatch_key_digest; std::string plan_abi_digest; std::string diagnostic; };
using Observer = std::function<void(const Event&)>;

struct Options final {
    Generation initial_generation{1};
    size_t worker_count{2}, max_queued_flights{64}, max_in_flight{64}, max_waiters_per_flight{1024};
    size_t max_discoverable_generations{8};
    uint64_t max_producer_reported_bytes{64ULL * 1024ULL * 1024ULL};
    size_t max_negative_cache_entries{64}; uint64_t max_negative_diagnostic_bytes{64ULL * 1024ULL * 1024ULL};
    size_t max_quarantine_tombstones_per_route{64};
    size_t max_routes{64}, max_quarantine_tombstones{256};
    uint64_t max_route_metadata_bytes{256ULL * 1024ULL}, max_quarantine_tombstone_bytes{256ULL * 1024ULL};
    std::chrono::milliseconds transient_backoff{100}, timeout_backoff{100};
    Observer observer;
    std::shared_ptr<HealthAuthority> health_authority;
    std::shared_ptr<CandidateValidationAuthority> validation_authority;
    std::shared_ptr<GenerationAuthority> generation_authority;
    /*! Test seam. Throwing/failing stages are checked before any route mutation. */
    std::function<bool(PublicationStage)> fail_publication_stage;
};
struct Snapshot final {
    Generation next_generation{1}; size_t queued_flights{0}, active_flights{0}, discoverable_generations{0};
    uint64_t producer_reported_discoverable_bytes{0}, evictions{0}, merged_waiters{0}, retry_cached{0};
    size_t negative_cache_entries{0}; uint64_t negative_cache_diagnostic_bytes{0}, negative_cache_evictions{0}, negative_cache_drops{0}; bool negative_cache_compile_blocked{false};
    size_t quarantine_tombstones{0}, quarantine_compile_blocked_routes{0}; uint64_t quarantine_saturations{0};
    size_t routes{0}; uint64_t route_metadata_bytes{0}, quarantine_tombstone_bytes{0}, route_saturations{0}, tombstone_saturations{0};
    bool process_resident_bytes_known{false}, device_resident_bytes_known{false};
};
struct RunAsyncResult final { Array<runtime::NDArray> outputs; AsyncOperation completion; std::shared_ptr<const GenerationLease> lease; };

class AdaptiveHotSwapController final {
public:
    explicit AdaptiveHotSwapController(std::shared_ptr<ProductionPathCompilerAdapter> compiler = nullptr, Options options = {});
    ~AdaptiveHotSwapController();
    AdaptiveHotSwapController(const AdaptiveHotSwapController&) = delete;
    AdaptiveHotSwapController& operator=(const AdaptiveHotSwapController&) = delete;
    CompileTicket Submit(CompileRequest request);
    std::shared_ptr<const GenerationLease> CompileAndPublish(CompileRequest request);
    std::shared_ptr<const GenerationLease> Acquire(const ProductionExecutionRequest& request) const;
    RunAsyncResult RunAsync(const ProductionExecutionRequest& request, const Array<runtime::NDArray>& inputs, const DeviceStream& stream) const;
    bool EvaluateHealth(const std::shared_ptr<const GenerationLease>& lease);
    void ClearNegativeCacheForTesting();
    void ClearQuarantinesForTesting();
    Snapshot SnapshotForTesting() const;
private:
    class State; std::shared_ptr<State> state_;
};
}  // namespace kxc::api::adaptive::hot_swap::v2
#endif
