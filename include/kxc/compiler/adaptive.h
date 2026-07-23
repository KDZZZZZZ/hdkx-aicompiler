/*! \file include/kxc/compiler/adaptive.h
 * \brief Static-exact adaptive compilation control-plane contracts.
 */

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace kxc::api::adaptive {

/*! \brief A validated canonical key with value equality, never hash-only identity. */
template <typename Tag>
class CanonicalKey final {
public:
    explicit CanonicalKey(std::string canonical)
        : canonical_(std::move(canonical)) {
        if (canonical_.empty()) {
            throw std::invalid_argument("canonical key must not be empty");
        }
    }

    const std::string& canonical() const noexcept { return canonical_; }

    friend bool operator==(const CanonicalKey& lhs,
                           const CanonicalKey& rhs) noexcept {
        return lhs.canonical_ == rhs.canonical_;
    }
    friend bool operator!=(const CanonicalKey& lhs,
                           const CanonicalKey& rhs) noexcept {
        return !(lhs == rhs);
    }
    friend bool operator<(const CanonicalKey& lhs,
                          const CanonicalKey& rhs) noexcept {
        return lhs.canonical_ < rhs.canonical_;
    }

private:
    std::string canonical_;
};

struct KernelSlotKeyTag final {};
struct PlanAbiFingerprintTag final {};
struct PlanVariantKeyTag final {};

/*! \brief Stable unit/target publication identity supplied by Core Track. */
using KernelSlotKey = CanonicalKey<KernelSlotKeyTag>;
/*! \brief Complete static-exact Plan ABI canonical fingerprint. */
using PlanAbiFingerprint = CanonicalKey<PlanAbiFingerprintTag>;
/*! \brief Frozen plan identity, separate from artifact and graph value identity. */
using PlanVariantKey = CanonicalKey<PlanVariantKeyTag>;

/*! \brief Exact-only dispatch identity; Shape Track may add proven kinds later. */
class DispatchKey final {
public:
    static DispatchKey Exact(std::string canonical);

    const std::string& canonical() const noexcept { return canonical_; }

    friend bool operator==(const DispatchKey& lhs,
                           const DispatchKey& rhs) noexcept {
        return lhs.canonical_ == rhs.canonical_;
    }
    friend bool operator!=(const DispatchKey& lhs,
                           const DispatchKey& rhs) noexcept {
        return !(lhs == rhs);
    }
    friend bool operator<(const DispatchKey& lhs,
                          const DispatchKey& rhs) noexcept {
        return lhs.canonical_ < rhs.canonical_;
    }

private:
    explicit DispatchKey(std::string canonical);
    std::string canonical_;
};

/*! \brief Artifact identity includes its slot identity and compiler fingerprint. */
class KernelArtifactKey final {
public:
    KernelArtifactKey(KernelSlotKey slot_key, std::string canonical);

    const KernelSlotKey& slot_key() const noexcept { return slot_key_; }
    const std::string& canonical() const noexcept { return canonical_; }

    friend bool operator==(const KernelArtifactKey& lhs,
                           const KernelArtifactKey& rhs) noexcept {
        return lhs.slot_key_ == rhs.slot_key_ &&
               lhs.canonical_ == rhs.canonical_;
    }
    friend bool operator!=(const KernelArtifactKey& lhs,
                           const KernelArtifactKey& rhs) noexcept {
        return !(lhs == rhs);
    }
    friend bool operator<(const KernelArtifactKey& lhs,
                          const KernelArtifactKey& rhs) noexcept {
        if (lhs.slot_key_ != rhs.slot_key_) {
            return lhs.slot_key_ < rhs.slot_key_;
        }
        return lhs.canonical_ < rhs.canonical_;
    }

private:
    KernelSlotKey slot_key_;
    std::string canonical_;
};

/*! \brief Scheduling intent; it never participates in artifact equality. */
enum class RequestKind : std::uint8_t {
    kDemand = 0,
    kCanary = 1,
    kPrewarm = 2,
};

/*! \brief Strong, complete compile request for one exact artifact. */
class CompileRequest final {
public:
    CompileRequest(KernelArtifactKey artifact_key, DispatchKey dispatch_key,
                   PlanAbiFingerprint required_abi,
                   std::string model_revision, RequestKind kind,
                   int priority = 0);

    const KernelArtifactKey& artifact_key() const noexcept {
        return artifact_key_;
    }
    const DispatchKey& dispatch_key() const noexcept { return dispatch_key_; }
    const PlanAbiFingerprint& required_abi() const noexcept {
        return required_abi_;
    }
    const std::string& model_revision() const noexcept {
        return model_revision_;
    }
    RequestKind kind() const noexcept { return kind_; }
    int priority() const noexcept { return priority_; }

private:
    KernelArtifactKey artifact_key_;
    DispatchKey dispatch_key_;
    PlanAbiFingerprint required_abi_;
    std::string model_revision_;
    RequestKind kind_{RequestKind::kDemand};
    int priority_{0};
};

/*! \brief Typed immutable executable payload; no raw void/function pointer ABI. */
class ArtifactExecutable {
public:
    virtual ~ArtifactExecutable() = default;
    virtual bool IsReady() const noexcept = 0;
    virtual std::string DebugName() const = 0;
};

/*! \brief Immutable validated result of compiling one static-exact request. */
class KernelArtifact final {
public:
    KernelArtifact(KernelArtifactKey key, DispatchKey applicability,
                   PlanAbiFingerprint compatible_abi,
                   std::shared_ptr<const ArtifactExecutable> executable,
                   std::size_t byte_size, std::string provenance);

    const KernelArtifactKey& key() const noexcept { return key_; }
    const DispatchKey& applicability() const noexcept {
        return applicability_;
    }
    const PlanAbiFingerprint& compatible_abi() const noexcept {
        return compatible_abi_;
    }
    const std::shared_ptr<const ArtifactExecutable>& executable() const noexcept {
        return executable_;
    }
    std::size_t byte_size() const noexcept { return byte_size_; }
    const std::string& provenance() const noexcept { return provenance_; }

private:
    KernelArtifactKey key_;
    DispatchKey applicability_;
    PlanAbiFingerprint compatible_abi_;
    std::shared_ptr<const ArtifactExecutable> executable_;
    std::size_t byte_size_{0};
    std::string provenance_;
};

/*! \brief Cooperative cancellation visible to an injected compiler. */
class CancellationToken final {
public:
    struct State;

    CancellationToken() = default;
    explicit CancellationToken(std::shared_ptr<State> state);
    bool IsCancellationRequested() const noexcept;

private:
    std::shared_ptr<State> state_;
};

/*! \brief Structured compiler failure before coordinator retry policy. */
enum class CompileFailureCategory : std::uint8_t {
    kUnsupported = 0,
    kDeterministic = 1,
    kTransient = 2,
    kValidation = 3,
    kBackpressure = 4,
    kCancelled = 5,
    kShutdown = 6,
};

/*! \brief Exactly one compiler attempt result. */
class CompileAttempt final {
public:
    static CompileAttempt Ready(
        std::shared_ptr<const KernelArtifact> artifact);
    static CompileAttempt Failed(CompileFailureCategory category,
                                 std::string diagnostic);

    bool ready() const noexcept { return artifact_ != nullptr; }
    const std::shared_ptr<const KernelArtifact>& artifact() const noexcept {
        return artifact_;
    }
    CompileFailureCategory failure_category() const noexcept {
        return failure_category_;
    }
    const std::string& diagnostic() const noexcept { return diagnostic_; }

private:
    CompileAttempt(std::shared_ptr<const KernelArtifact> artifact,
                   CompileFailureCategory category, std::string diagnostic);
    std::shared_ptr<const KernelArtifact> artifact_;
    CompileFailureCategory failure_category_{
        CompileFailureCategory::kDeterministic};
    std::string diagnostic_;
};

/*! \brief Testable compiler seam; implementations must observe cancellation. */
class ArtifactCompiler {
public:
    virtual ~ArtifactCompiler() = default;
    virtual CompileAttempt Compile(const CompileRequest& request,
                                   const CancellationToken& cancellation) = 0;
};

/*! \brief Terminal ticket state, separate from in-flight request state. */
enum class CompileStatus : std::uint8_t {
    kReady = 0,
    kFailed = 1,
    kCancelled = 2,
    kRejected = 3,
};

/*! \brief Immutable terminal result shared with request waiters. */
class CompileResult final {
public:
    static CompileResult Ready(
        std::shared_ptr<const KernelArtifact> artifact,
        std::uint32_t attempt);
    static CompileResult Failure(
        CompileStatus status, CompileFailureCategory category,
        std::string diagnostic, std::uint32_t attempt, bool retryable,
        std::chrono::steady_clock::time_point retry_after = {});

    CompileStatus status() const noexcept { return status_; }
    bool ready() const noexcept { return status_ == CompileStatus::kReady; }
    const std::shared_ptr<const KernelArtifact>& artifact() const noexcept {
        return artifact_;
    }
    CompileFailureCategory failure_category() const noexcept {
        return failure_category_;
    }
    const std::string& diagnostic() const noexcept { return diagnostic_; }
    std::uint32_t attempt() const noexcept { return attempt_; }
    bool retryable() const noexcept { return retryable_; }
    std::chrono::steady_clock::time_point retry_after() const noexcept {
        return retry_after_;
    }

private:
    CompileResult(CompileStatus status,
                  std::shared_ptr<const KernelArtifact> artifact,
                  CompileFailureCategory failure_category,
                  std::string diagnostic, std::uint32_t attempt,
                  bool retryable,
                  std::chrono::steady_clock::time_point retry_after);

    CompileStatus status_{CompileStatus::kFailed};
    std::shared_ptr<const KernelArtifact> artifact_;
    CompileFailureCategory failure_category_{
        CompileFailureCategory::kDeterministic};
    std::string diagnostic_;
    std::uint32_t attempt_{0};
    bool retryable_{false};
    std::chrono::steady_clock::time_point retry_after_{};
};

/*! \brief Per-caller handle for one shared singleflight compilation. */
class CompileTicket final {
public:
    CompileTicket() = default;
    std::uint64_t request_id() const noexcept { return request_id_; }
    bool valid() const noexcept { return future_.valid(); }
    CompileResult Get() const;
    std::future_status WaitFor(std::chrono::milliseconds timeout) const;

private:
    CompileTicket(std::uint64_t request_id,
                  std::shared_future<CompileResult> future);
    std::uint64_t request_id_{0};
    std::shared_future<CompileResult> future_;
    friend class CompileCoordinator;
};

enum class CancelResult : std::uint8_t {
    kCancelled = 0,
    kAlreadyCompleted = 1,
    kNotFound = 2,
};

enum class AdaptiveEventKind : std::uint8_t {
    kQueued = 0,
    kRequestMerged = 1,
    kCompileStarted = 2,
    kValidating = 3,
    kReady = 4,
    kFailed = 5,
    kCancelled = 6,
    kBudgetRejected = 7,
    kShutdownStarted = 8,
    kShutdownCompleted = 9,
    kPublished = 10,
    kAcquired = 11,
    kPromoted = 12,
    kWithdrawn = 13,
    kRolledBack = 14,
};

/*! \brief Stable event fields for coordinator and slot observability. */
struct AdaptiveEvent final {
    AdaptiveEventKind kind{AdaptiveEventKind::kQueued};
    std::uint64_t request_id{0};
    std::uint64_t generation{0};
    std::uint64_t predecessor_generation{0};
    std::size_t waiter_count{0};
    std::size_t queue_depth{0};
    std::string slot_key;
    std::string artifact_key;
    std::string dispatch_key;
    std::string abi_fingerprint;
    std::string model_revision;
    std::string diagnostic;
};

using AdaptiveObserver = std::function<void(const AdaptiveEvent&)>;

struct RetryPolicy final {
    std::uint32_t max_transient_attempts{3};
    std::chrono::milliseconds initial_backoff{10};
    std::chrono::milliseconds max_backoff{1000};
};

struct CoordinatorOptions final {
    std::size_t worker_count{1};
    std::size_t max_queue_size{64};
    std::size_t max_terminal_records{256};
    std::size_t max_artifact_bytes{256U * 1024U * 1024U};
    RetryPolicy retry_policy;
    std::function<std::chrono::steady_clock::time_point()> now;
    AdaptiveObserver observer;
};

struct CoordinatorSnapshot final {
    bool accepting{false};
    std::size_t queued{0};
    std::size_t active{0};
    std::size_t in_flight_keys{0};
    std::size_t terminal_records{0};
    std::uint64_t requests{0};
    std::uint64_t merged{0};
    std::uint64_t compile_attempts{0};
    std::uint64_t ready{0};
    std::uint64_t failed{0};
    std::uint64_t cancelled{0};
    std::uint64_t rejected{0};
};

/*! \brief Bounded asynchronous exact compiler coordinator. */
class CompileCoordinator final {
public:
    CompileCoordinator(std::shared_ptr<ArtifactCompiler> compiler,
                       CoordinatorOptions options = {});
    ~CompileCoordinator();

    CompileCoordinator(const CompileCoordinator&) = delete;
    CompileCoordinator& operator=(const CompileCoordinator&) = delete;

    CompileTicket Request(CompileRequest request);
    CancelResult Cancel(std::uint64_t request_id);
    CoordinatorSnapshot Snapshot() const;
    void Shutdown();

private:
    class State;
    std::unique_ptr<State> state_;
};

}  // namespace kxc::api::adaptive
