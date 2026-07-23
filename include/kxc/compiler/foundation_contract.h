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

inline constexpr int kCompilerFoundationContractVersion = 1;

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

/*! \brief Immutable, validated ready-artifact description for cross-track fakes. */
struct ArtifactRecord final {
    ArtifactKey artifact_key;
    std::string executable_token;
    std::string signature_digest;
    std::string launch_metadata_digest;
    std::string provenance;
    uint64_t byte_size{0};
    std::string validation_record;
};

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
    ArtifactHandle handle_;
};

struct ArtifactLookup final {
    ArtifactLookupKind kind{ArtifactLookupKind::kMiss};
    ArtifactPin pin;
};

struct CancellationToken final {
    std::string id;
    bool cancelled{false};
};

struct CompileRequest final {
    ArtifactKey artifact_key;
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
