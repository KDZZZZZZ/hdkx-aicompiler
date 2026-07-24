/*! \file include/kxc/compiler/artifact.h
 * \brief Read-only public views of compiler-produced primitive artifacts.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "kxc/compiler/identity.h"

namespace kxc::api {

/*! \brief Immutable description of a validated compiler-produced artifact. */
struct ArtifactRecord final {
    PrimitiveArtifactKey artifact_key;
    std::string executable_token;
    std::string signature_digest;
    std::string launch_metadata_digest;
    std::string provenance;
    uint64_t byte_size{0};
    std::string validation_record;
};

namespace internal {
struct ArtifactPinAccess;
}  // namespace internal

/*! \brief Opaque retention pin minted only by the production primitive cache. */
class ArtifactPin final {
public:
    ArtifactPin() = default;

    bool defined() const noexcept;
    const ArtifactRecord& record() const;

private:
    friend struct internal::ArtifactPinAccess;

    ArtifactPin(std::shared_ptr<const ArtifactRecord> record,
                std::shared_ptr<const void> owner);

    std::shared_ptr<const ArtifactRecord> record_;
    std::shared_ptr<const void> owner_;
};

}  // namespace kxc::api
