/*! \file include/kxc/compiler/identity.h
 * \brief Stable canonical identities required by installed compiler APIs.
 */

#pragma once

#include <string>

namespace kxc::api {
namespace internal {
class IdentityAccess;
}  // namespace internal

/*! \brief Whole-graph Relay semantics, independent of target and compiler policy. */
class GraphSemanticKey final {
public:
    GraphSemanticKey() = default;

    bool defined() const noexcept;
    const std::string& canonical_bytes() const noexcept;
    const std::string& digest() const noexcept;

    bool operator==(const GraphSemanticKey& other) const noexcept;
    bool operator!=(const GraphSemanticKey& other) const noexcept;
    bool operator<(const GraphSemanticKey& other) const noexcept;

private:
    explicit GraphSemanticKey(std::string canonical_bytes);

    std::string canonical_bytes_;
    std::string digest_;

    friend class internal::IdentityAccess;
};

/*! \brief Unit semantics independent of graph numbering, symbol, and storage. */
class UnitSemanticKey final {
public:
    UnitSemanticKey() = default;
    explicit UnitSemanticKey(std::string canonical_bytes);

    bool defined() const noexcept;
    const std::string& canonical_bytes() const noexcept;
    const std::string& digest() const noexcept;

    bool operator==(const UnitSemanticKey& other) const noexcept;
    bool operator!=(const UnitSemanticKey& other) const noexcept;
    bool operator<(const UnitSemanticKey& other) const noexcept;

private:
    UnitSemanticKey(std::string canonical_bytes, std::string digest);

    std::string canonical_bytes_;
    std::string digest_;

    friend class internal::IdentityAccess;
};

/*! \brief Ready primitive artifact identity, including target, pipeline, ABI and backend. */
class PrimitiveArtifactKey final {
public:
    PrimitiveArtifactKey() = default;
    PrimitiveArtifactKey(UnitSemanticKey unit_semantic_key,
                         std::string target_capability_fingerprint,
                         std::string pipeline_fingerprint, int abi_version,
                         std::string schedule_contract,
                         std::string backend_version);

    bool defined() const noexcept;
    const UnitSemanticKey& unit_semantic_key() const noexcept;
    const std::string& target_capability_fingerprint() const noexcept;
    const std::string& canonical_bytes() const noexcept;
    const std::string& digest() const noexcept;

    bool operator==(const PrimitiveArtifactKey& other) const noexcept;
    bool operator!=(const PrimitiveArtifactKey& other) const noexcept;
    bool operator<(const PrimitiveArtifactKey& other) const noexcept;

private:
    PrimitiveArtifactKey(UnitSemanticKey unit_semantic_key,
                         std::string target_capability_fingerprint,
                         std::string pipeline_fingerprint, int abi_version,
                         std::string schedule_contract,
                         std::string backend_version,
                         std::string digest);

    UnitSemanticKey unit_semantic_key_;
    std::string target_capability_fingerprint_;
    std::string canonical_bytes_;
    std::string digest_;

    friend class internal::IdentityAccess;
};

}  // namespace kxc::api
