#pragma once

#include <string>

#include "kxc/compiler/compile_config.h"
#include "kxc/compiler/pipeline.h"
#include "kxc/relay/relay.h"

namespace kxc::api::internal {

/*! \brief One immutable description of the production plan actually executed. */
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

/*! \brief Executes the real compiler path; success is executable proof. */
void ProbeCompilerExecution(Function function, CompileConfig config,
                            const CompilerExecutionContract& contract);

}  // namespace kxc::api::internal
