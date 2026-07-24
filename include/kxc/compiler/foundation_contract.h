/*! \file include/kxc/compiler/foundation_contract.h
 * \brief Versioned DTO contract shared by compiler-foundation control tracks.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "kxc/compiler/identity.h"

namespace kxc::api {

inline constexpr int kCompilerFoundationContractVersion = 2;

enum class ArtifactLookupKind { kHit, kMiss };
enum class CompilePriority { kPrewarm, kNormal, kUrgent };
enum class CompileBudgetClass { kGlobal, kModel, kTargetBackend };
enum class CompileRequestState {
    kAbsent,
    kQueued,
    kCompiling,
    kValidating,
    kReady,
    kFailed,
    kCancelled,
    kRejected,
};
enum class CompileFailureCategory {
    kCompile,
    kValidation,
    kUnsupported,
    kCancelled,
    kBackpressure,
};
enum class CacheEventKind { kHit, kMiss, kStore, kEvict, kPin, kFailure };

/*! \brief Immutable, validated ready-artifact description for fakes and adapters. */
struct ArtifactRecord final {
    PrimitiveArtifactKey artifact_key;
    std::string executable_token;
    std::string signature_digest;
    std::string launch_metadata_digest;
    std::string provenance;
    uint64_t byte_size{0};
    std::string validation_record;
};

class ArtifactPin;
class ProductionArtifactCandidate;
namespace internal {
class PrimitiveArtifactPin;
struct ProductionArtifactAccess;
ArtifactPin ToArtifactPin(const PrimitiveArtifactPin& pin);
}  // namespace internal

class ArtifactHandle final {
public:
    ArtifactHandle() = default;
    explicit ArtifactHandle(ArtifactRecord record);

    bool defined() const noexcept;
    const ArtifactRecord& record() const;

private:
    std::shared_ptr<const ArtifactRecord> record_;
};

/*! \brief Strong handle issued by lookup/result; eviction cannot invalidate it. */
class ArtifactPin final {
public:
    ArtifactPin() = default;
    explicit ArtifactPin(ArtifactHandle handle);

    bool defined() const noexcept;
    const ArtifactHandle& handle() const;

private:
    friend ArtifactPin internal::ToArtifactPin(
        const internal::PrimitiveArtifactPin& pin);
    friend struct internal::ProductionArtifactAccess;

    ArtifactPin(ArtifactHandle handle, std::shared_ptr<const void> owner);

    ArtifactHandle handle_;
    // Production pins keep the internal cache pin alive without exposing it.
    std::shared_ptr<const void> owner_;
};

struct ArtifactLookup final {
    ArtifactLookupKind kind{ArtifactLookupKind::kMiss};
    ArtifactPin pin;
};

struct ArtifactCacheStats final {
    uint64_t hits{0};
    uint64_t misses{0};
    uint64_t entries{0};
    uint64_t accounted_bytes{0};
    uint64_t evictions{0};
    uint64_t in_flight{0};
    uint64_t merged_waiters{0};
    uint64_t failures{0};
    uint64_t rejections{0};
    /*! \brief External refs to discoverable entries; excludes evicted pins. */
    uint64_t active_pins{0};
};

enum class CompilePolicyLayer {
    kPrimitiveCacheTransaction,
    kAdaptiveCoordinator,
};

/*! \brief Explicit ownership boundary between Core static-exact cache work and Track03. */
struct ProductionCompileOwnership final {
    CompilePolicyLayer exact_singleflight{
        CompilePolicyLayer::kPrimitiveCacheTransaction};
    CompilePolicyLayer failure_ttl{
        CompilePolicyLayer::kPrimitiveCacheTransaction};
    CompilePolicyLayer global_backpressure{
        CompilePolicyLayer::kPrimitiveCacheTransaction};
    CompilePolicyLayer asynchronous_cancellation{
        CompilePolicyLayer::kAdaptiveCoordinator};
    CompilePolicyLayer priority_and_budget_scheduling{
        CompilePolicyLayer::kAdaptiveCoordinator};
    CompilePolicyLayer dispatch_and_generation{
        CompilePolicyLayer::kAdaptiveCoordinator};
};

struct CancellationToken final {
    std::string id;
    bool cancelled{false};
};

struct CompileRequest final {
    PrimitiveArtifactKey artifact_key;
    std::optional<DispatchKey> dispatch_key;
    CompilePriority priority{CompilePriority::kNormal};
    CompileBudgetClass budget_class{CompileBudgetClass::kGlobal};
    std::string request_origin;
    CancellationToken cancellation;

    std::string CanonicalSingleflightKey() const;
};

struct CompileFailure final {
    CompileFailureCategory category{CompileFailureCategory::kCompile};
    uint64_t retry_after_millis{0};
    std::string diagnostic;
};

struct CompileOutcome final {
    CompileRequestState state{CompileRequestState::kAbsent};
    ArtifactPin pin;
    std::optional<CompileFailure> failure;
};

struct CompileTicket final {
    std::string ticket_id;
    std::string canonical_request_key;
    CompileRequestState state{CompileRequestState::kAbsent};
    uint64_t merged_waiter_count{0};
    std::optional<CompileOutcome> outcome;
};

/*! \brief Opaque validated executable candidate; only production internals create it. */
class ProductionArtifactCandidate final {
public:
    ProductionArtifactCandidate() = default;
    bool defined() const noexcept;

private:
    friend struct internal::ProductionArtifactAccess;
    explicit ProductionArtifactCandidate(std::shared_ptr<const void> owner);

    std::shared_ptr<const void> owner_;
};

/*! \brief One public waiter/owner view over an opaque production cache lease. */
class ProductionCompileTransaction final {
public:
    ProductionCompileTransaction() = default;

    bool defined() const noexcept;
    bool owns_compile() const noexcept;
    CompileTicket ticket() const;

private:
    struct Impl;
    explicit ProductionCompileTransaction(std::shared_ptr<Impl> impl);

    friend class ProductionArtifactCacheAdapter;
    std::shared_ptr<Impl> impl_;
};

/*! \brief Static-exact adapter over the process-local production primitive cache. */
class ProductionArtifactCacheAdapter final {
public:
    ProductionCompileOwnership ownership() const noexcept;
    ArtifactLookup Lookup(const PrimitiveArtifactKey& key) const;
    ProductionCompileTransaction Acquire(const CompileRequest& request) const;
    CompileOutcome Wait(const ProductionCompileTransaction& transaction) const;
    CompileOutcome Wait(const ProductionCompileTransaction& transaction,
                        const CancellationToken& cancellation) const;
    CompileOutcome Publish(const ProductionCompileTransaction& transaction,
                           ProductionArtifactCandidate candidate) const;
    CompileOutcome Fail(const ProductionCompileTransaction& transaction,
                        CompileFailure failure) const;
    ArtifactCacheStats stats() const;
};

struct CacheObserverEvent final {
    CacheEventKind kind{CacheEventKind::kMiss};
    std::string artifact_key_digest;
    uint64_t bytes{0};
    uint64_t entry_count{0};
    uint64_t pin_count{0};
    uint64_t age_millis{0};
    std::string diagnostic;
};

struct SelectedArtifact final {
    ArtifactPin artifact_pin;
    uint64_t generation{0};
    std::string applicability_proof;
    std::string signature_digest;
};

struct FrozenPlanInput final {
    std::string graph_template_locator;
    std::vector<GraphValueLocator> value_routing;
    std::vector<SelectedArtifact> selected_artifacts;
    std::string logical_physical_valid_extent_contract;
    std::string memory_plan_version;

    void Validate() const;
};

}  // namespace kxc::api
