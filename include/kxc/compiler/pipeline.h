/*! \file include/kxc/compiler/pipeline.h
 * \brief Deterministic production pipeline request and normalized result.
 */

#pragma once

#include <string>
#include <vector>

#include "kxc/pass/pass.h"
#include "kxc/target/target.h"

namespace kxc::api {

struct PipelineRequest final {
    IRDialect dialect{IRDialect::kUnknown};
    PassScope requested_scope{PassScope::kUnknown};
    Target target;
    int opt_level{2};
    String named_pipeline{String("compiler")};
    Array<String> enabled;
    Array<String> disabled;
    Array<String> initial_invariants;
    Array<String> initial_analyses;
};

struct PipelineInvariantTransition final {
    String pass_name;
    String phase;
    Array<String> required;
    Array<String> produced;
    Array<String> preserved_analyses;
    Array<String> invalidated_analyses;
    Array<String> invariants_before;
    Array<String> invariants_after;
    Array<String> analyses_before;
    Array<String> analyses_after;
};

struct NormalizedPipeline final {
    IRDialect dialect{IRDialect::kUnknown};
    PassScope scope{PassScope::kUnknown};
    Array<String> ordered_passes;
    std::vector<PipelineInvariantTransition> invariant_transitions;
    Array<String> target_requirements;
    Array<String> contract_versions;
    String canonical_bytes;
    String fingerprint;

    bool defined() const noexcept;
};

class PipelineResolver final {
public:
    static NormalizedPipeline Resolve(const PipelineRequest& request);
};

}  // namespace kxc::api
