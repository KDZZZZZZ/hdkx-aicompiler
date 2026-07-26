#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "prepared_static_graph.h"
#include "primitive_cache.h"
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
    std::string schedule_policy;
    std::string backend_version;
};

CompilerExecutionContract ResolveCompilerExecutionContract(
    const CompileConfig& config);
CompilerExecutionContract ResolveCompilerExecutionContract(
    const CompileConfig& config,
    const Array<String>& required_relay_control_capabilities);
std::string CanonicalTargetSnapshot(const Target& target);

PreparedCompilerGraph PrepareCompilerGraph(
    Function function, CompileConfig config,
    const CompilerExecutionContract& contract);
CompiledGraph AssembleCompiledGraph(
    const PreparedCompilerGraph& prepared,
    const std::vector<PrimitiveArtifactPin>& ordered_pins,
    const Map<String, runtime::NDArray>& constants);

}  // namespace kxc::api::internal
