/*! \file include/kxc/compiler/pipeline.h
 * \brief Deterministic production pipeline request and normalized result.
 */

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "kxc/pass/pass.h"
#include "kxc/relay/relay.h"
#include "kxc/target/target.h"
#include "kxc/tir/stmt.h"

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
    Array<String> required_control_capabilities;
};

struct PipelineExecutionStep final {
    IRDialect dialect{IRDialect::kUnknown};
    PassScope scope{PassScope::kUnknown};
    size_t occurrence{0};
    String pass_name;
    String phase;
    int schema_version{0};
    String implementation_key;
};

struct PipelineInvariantTransition final {
    Array<String> required;
    Array<String> produced;
    Array<String> declarative_only;
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
    int opt_level{0};
    String named_pipeline;
    Array<String> initial_invariants;
    Array<String> initial_analyses;
    Array<String> required_control_capabilities;
    std::vector<PipelineExecutionStep> execution_steps;
    std::vector<PipelineInvariantTransition> invariant_transitions;
    Array<String> target_requirements;
    String canonical_bytes;
    String fingerprint;

    bool defined() const noexcept;
};

class PipelineResolver final {
public:
    static NormalizedPipeline Resolve(const PipelineRequest& request);
};

/*! \brief Executable proof registry for production pipeline invariants. */
class PipelineInvariantValidator final {
public:
    static bool IsExecutable(IRDialect dialect, const String& invariant);
    static void ValidateProductionContract(const PassSpec& spec);
    static void ValidateRelay(const Array<String>& invariants,
                              const Function& function);
    static void ValidateTIR(const Array<String>& invariants,
                            const tir::PrimFunc& function);
};

/*! Executes exactly the audited pass steps in a NormalizedPipeline. */
class PipelineExecutor final {
public:
    static void Validate(const NormalizedPipeline& pipeline,
                         const Target& target);
    static Function ExecuteRelay(const NormalizedPipeline& pipeline,
                                 const Function& function,
                                 const Target& target);
    static tir::PrimFunc ExecuteTIR(const NormalizedPipeline& pipeline,
                                    const tir::PrimFunc& function,
                                    const Target& target);
};

}  // namespace kxc::api
