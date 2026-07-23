#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

#include "../../codegen/internal/compiled_kernel.h"
#include "kxc/compiler/identity.h"
#include "kxc/target/target.h"

namespace kxc::api::internal {

/*! \brief Complete ready artifact accepted by the process-local store. */
struct CachedPrimitive final {
    codegen::KernelSignature signature;
    codegen::KernelLaunchMetadata launch_metadata;
    codegen::CompiledKernel kernel;
    uint64_t accounted_bytes{1};
    std::string provenance{"compiler"};
    std::string validation_record{"validated"};
};

struct PrimitiveArtifact;
struct PrimitiveFlight;
class PrimitiveCacheLease;

/*! \brief Strong immutable handle; cache eviction only removes discoverability. */
class PrimitiveArtifactPin final {
public:
    PrimitiveArtifactPin() = default;

    bool defined() const noexcept;
    const ArtifactKey& key() const;
    const CachedPrimitive& artifact() const;

private:
    friend class PrimitiveCacheLease;
    friend PrimitiveCacheLease AcquirePrimitiveCache(const ArtifactKey&);
    friend PrimitiveArtifactPin PublishPrimitiveCacheLease(
        const PrimitiveCacheLease&, CachedPrimitive);
    explicit PrimitiveArtifactPin(
        std::shared_ptr<const PrimitiveArtifact> artifact);

    std::shared_ptr<const PrimitiveArtifact> artifact_;
};

enum class PrimitiveCacheAccess {
    kHit,
    kOwner,
    kWait,
    kFailed,
    kRejected,
};

enum class PrimitiveFailureCategory {
    kCompile,
    kValidation,
    kUnsupported,
    kCancelled,
    kBackpressure,
};

struct PrimitiveFailureRecord final {
    PrimitiveFailureCategory category{PrimitiveFailureCategory::kCompile};
    std::string diagnostic;
    uint64_t retry_after_millis{0};
};

/*! \brief One lookup/singleflight result; owner is unique per full key. */
class PrimitiveCacheLease final {
public:
    PrimitiveCacheLease() = default;

    PrimitiveCacheAccess access() const noexcept;
    const ArtifactKey& key() const;
    const PrimitiveArtifactPin& pin() const;
    const PrimitiveFailureRecord& failure() const;

private:
    friend PrimitiveCacheLease AcquirePrimitiveCache(const ArtifactKey&);
    friend PrimitiveArtifactPin WaitPrimitiveCacheLease(
        const PrimitiveCacheLease&);
    friend PrimitiveArtifactPin PublishPrimitiveCacheLease(
        const PrimitiveCacheLease&, CachedPrimitive);
    friend void FailPrimitiveCacheLease(
        const PrimitiveCacheLease&, PrimitiveFailureCategory,
        std::string, std::chrono::milliseconds);

    PrimitiveCacheAccess access_{PrimitiveCacheAccess::kRejected};
    ArtifactKey key_;
    PrimitiveArtifactPin pin_;
    PrimitiveFailureRecord failure_;
    std::shared_ptr<PrimitiveFlight> flight_;
};

struct PrimitiveCacheLimits final {
    uint64_t max_entries{256};
    uint64_t max_accounted_bytes{64ULL * 1024ULL * 1024ULL};
    uint64_t max_in_flight{1024};
    uint64_t max_failures{256};
};

struct PrimitiveCacheStats final {
    uint64_t hits{0};
    uint64_t misses{0};
    uint64_t entries{0};
    uint64_t accounted_bytes{0};
    uint64_t evictions{0};
    uint64_t in_flight{0};
    uint64_t merged_waiters{0};
    uint64_t failures{0};
    uint64_t rejections{0};
    uint64_t active_pins{0};
};

ArtifactKey BuildPrimitiveArtifactKey(
    const UnitSemanticKey& semantic_key, const Target& target,
    const std::string& pipeline_fingerprint,
    const char* schedule_version, const char* backend_version);

PrimitiveCacheLease AcquirePrimitiveCache(const ArtifactKey& key);
PrimitiveArtifactPin WaitPrimitiveCacheLease(
    const PrimitiveCacheLease& lease);
PrimitiveArtifactPin PublishPrimitiveCacheLease(
    const PrimitiveCacheLease& lease, CachedPrimitive entry);
void FailPrimitiveCacheLease(
    const PrimitiveCacheLease& lease, PrimitiveFailureCategory category,
    std::string diagnostic,
    std::chrono::milliseconds retry_after = std::chrono::seconds(1));

PrimitiveCacheStats GetPrimitiveCacheStats();
void SetPrimitiveCacheLimitsForTesting(PrimitiveCacheLimits limits);
void ForgetPrimitiveFailureForTesting(const ArtifactKey& key);
void ClearPrimitiveCacheForTesting();

}  // namespace kxc::api::internal
