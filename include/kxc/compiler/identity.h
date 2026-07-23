/*! \file include/kxc/compiler/identity.h
 * \brief Canonical, collision-safe identities shared by compiler control tracks.
 */

#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace kxc::api {

/*! \brief Plan-local graph value locator; never a semantic cache key. */
struct GraphValueLocator final {
    std::string graph_revision;
    int64_t value_id{-1};
    int64_t output_index{0};

    std::string CanonicalBytes() const;
};

/*! \brief Unit semantics independent of graph numbering, symbol, and storage. */
class UnitSemanticKey final {
public:
    UnitSemanticKey() = default;
    explicit UnitSemanticKey(std::string canonical_bytes,
                             std::string index_digest = {});

    bool defined() const noexcept;
    const std::string& canonical_bytes() const noexcept;
    const std::string& digest() const noexcept;

    bool operator==(const UnitSemanticKey& other) const noexcept;
    bool operator!=(const UnitSemanticKey& other) const noexcept;

private:
    std::string canonical_bytes_;
    std::string digest_;
};

/*! \brief Ready artifact identity, including target, pipeline, ABI and backend. */
class ArtifactKey final {
public:
    ArtifactKey() = default;
    ArtifactKey(UnitSemanticKey unit_semantic_key,
                std::string target_capability_fingerprint,
                std::string pipeline_fingerprint, int abi_version,
                std::string schedule_version, std::string backend_version,
                std::string index_digest = {});

    bool defined() const noexcept;
    const UnitSemanticKey& unit_semantic_key() const noexcept;
    const std::string& canonical_bytes() const noexcept;
    const std::string& digest() const noexcept;

    bool operator==(const ArtifactKey& other) const noexcept;
    bool operator!=(const ArtifactKey& other) const noexcept;

private:
    UnitSemanticKey unit_semantic_key_;
    std::string canonical_bytes_;
    std::string digest_;
};

/*! \brief Shape/layout applicability identity, separate from primitive semantics. */
class DispatchKey final {
public:
    DispatchKey() = default;
    DispatchKey(std::string artifact_family,
                std::string shape_layout_valid_extent,
                std::string variant_policy_version,
                std::string index_digest = {});

    bool defined() const noexcept;
    const std::string& canonical_bytes() const noexcept;
    const std::string& digest() const noexcept;
    bool operator==(const DispatchKey& other) const noexcept;

private:
    std::string canonical_bytes_;
    std::string digest_;
};

/*! \brief Frozen graph/profile/artifact-generation and memory-plan identity. */
class PlanVariantKey final {
public:
    PlanVariantKey() = default;
    PlanVariantKey(
        std::string graph_template_revision,
        std::vector<std::pair<std::string, uint64_t>> artifact_generations,
        std::string concrete_profile, std::string memory_plan_version,
        std::string index_digest = {});

    bool defined() const noexcept;
    const std::string& canonical_bytes() const noexcept;
    const std::string& digest() const noexcept;
    bool operator==(const PlanVariantKey& other) const noexcept;

private:
    std::string canonical_bytes_;
    std::string digest_;
};

/*! \brief Backend module lookup identity, explicitly not cache equivalence. */
struct LinkSymbol final {
    std::string value;

    std::string CanonicalBytes() const;
};

/*! \brief Physical allocation identity scoped to one frozen plan. */
struct StorageId final {
    int64_t value{-1};

    std::string CanonicalBytes() const;
};

}  // namespace kxc::api
