#pragma once

#include <vector>

#include "kxc/compiler/compiler.h"

namespace kxc::api::internal {

/*! \brief Compiler-private minting seam for fully validated compiled graphs. */
struct CompiledGraphAccess final {
    static CompiledGraph Create(CompiledModule module,
                                runtime::ExecutablePlan plan,
                                std::vector<ArtifactPin> artifact_pins,
                                GraphSemanticKey graph_semantic_key);
};

}  // namespace kxc::api::internal
