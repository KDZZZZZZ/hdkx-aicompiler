#pragma once

#include <string>
#include <utility>

#include "kxc/compiler/identity.h"

namespace kxc {
class Function;
}

namespace kxc::api::internal {

/*! Compiler-private mint and source-private collision fixture access. */
class IdentityAccess final {
public:
    static GraphSemanticKey Graph(std::string canonical_bytes);

    static UnitSemanticKey UnitWithDigest(std::string canonical_bytes,
                                          std::string digest) {
        return UnitSemanticKey(std::move(canonical_bytes), std::move(digest));
    }

    static PrimitiveArtifactKey ArtifactWithDigest(
        UnitSemanticKey unit_semantic_key,
        std::string target_capability_fingerprint,
        std::string pipeline_fingerprint, int abi_version,
        std::string schedule_version, std::string backend_version,
        std::string digest) {
        return PrimitiveArtifactKey(
            std::move(unit_semantic_key),
            std::move(target_capability_fingerprint),
            std::move(pipeline_fingerprint), abi_version,
            std::move(schedule_version), std::move(backend_version),
            std::move(digest));
    }
};

GraphSemanticKey BuildGraphSemanticKey(const Function& function);

}  // namespace kxc::api::internal
