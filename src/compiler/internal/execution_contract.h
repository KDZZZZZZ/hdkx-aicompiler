#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "compile_state.h"
#include "prepared_static_graph.h"
#include "kxc/compiler/compiler.h"
#include "kxc/compiler/pipeline.h"
#include "kxc/relay/relay.h"

namespace kxc::profiling {
class ProfileContext;
}

namespace kxc::api::internal {

/*! \brief One immutable description of the production plan actually executed. */
struct PreparedCompilerGraph final {
    GraphSemanticKey graph_semantic_key;
    CompileResult optimized;
    PreparedStaticGraph graph;
    Target target;
    std::string execution_contract_canonical;
    std::shared_ptr<profiling::ProfileContext> profile_context;
    std::string profile_run_id;
    size_t relay_graph_pipelines{0};
    size_t capability_boundary_checks{0};
    size_t value_graph_builds{0};
    size_t partitions{0};
};

struct CompilerExecutionContract final {
    NormalizedPipeline relay_pipeline;
    NormalizedPipeline tir_pipeline;
    std::string canonical_bytes;
    std::string fingerprint;
    std::string schedule_version;
    std::string backend_version;
};

CompilerExecutionContract ResolveCompilerExecutionContract(
    const CompileConfig& config);
Target CloneTargetSnapshot(const Target& target);
std::string CanonicalTargetSnapshot(const Target& target);

/*! \brief Executes the real compiler path; success is executable proof. */
void ProbeCompilerExecution(Function function, CompileConfig config,
                            const CompilerExecutionContract& contract);
PreparedCompilerGraph PrepareCompilerGraph(
    Function function, CompileConfig config,
    const CompilerExecutionContract& contract);
CompiledGraph FinishCompilerGraph(const PreparedCompilerGraph& prepared,
                                  CompileConfig config,
                                  const CompilerExecutionContract& contract);

}  // namespace kxc::api::internal
