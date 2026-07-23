/*! \file include/kxc/compiler/compiler.h
 * \brief 定义面向调用方的编译配置、Compiler 和 CompiledModule 公共 API。
 */

#pragma once

#include <cstddef>
#include <vector>

#include "kxc/compiler/compile_config.h"
#include "kxc/compiler/foundation_contract.h"
#include "kxc/runtime/compiled_module.h"
#include "kxc/runtime/executable_plan.h"
#include "kxc/runtime/task_plan.h"
#include "kxc/support/container.h"
#include "kxc/relay/relay.h"

namespace kxc {
namespace api {

/*! \brief Ordered production artifact ownership for one immutable plan call. */
struct ArtifactPlanBinding final {
    size_t call_index{0};
    LinkSymbol link_symbol;
    ArtifactPin artifact_pin;
    std::string signature_digest;
    std::string launch_metadata_digest;
};

struct CompiledGraph final {
    CompiledModule module;
    runtime::ExecutablePlan plan;
    // Compiler::Compile populates this with production-backed cache pins.
    std::vector<ArtifactPin> artifact_pins;
    // Compiler declares pin identities and retains those pins in the lease.
    // Runtime observes the declaration but cannot authenticate its provenance.
    runtime::PlanVariant variant;
    // Ordered call-to-artifact pins; no per-unit relinking is exposed.
    std::vector<ArtifactPlanBinding> artifact_plan_bindings;
    // Canonical whole-graph + target + normalized compiler contract identity.
    ArtifactKey graph_artifact_key;
};

/*!
 * \brief Relay Function 到可执行模块的统一编译入口。
 *
 * Compiler 负责串起 Relay pass、Relay->TIR lowering、TIR pass 和真实 codegen。
 * backend 完全由 CompileConfig.target 决定，编译入口不再创建运行时会话。
 */
class Compiler {
public:
    /*!
     * \brief 从 Relay Function 编译出 CompiledModule。
     * \param func 输入 Relay 函数。
     * \param config 编译配置。
     * \return 可运行的编译模块。
     */
    static CompiledGraph Compile(Function func, CompileConfig config);

    /*! \brief Builds the canonical whole-graph compile identity used by adapters. */
    static ArtifactKey BuildGraphArtifactKey(const Function& func,
                                             const CompileConfig& config);

    /*! \brief Compatibility view of Relay optimization passes only.
     *
     * Production pre/post InferType steps are present in PipelineResolver's
     * NormalizedPipeline and intentionally omitted from this legacy view.
     */
    static Array<String> RelayPassPolicy(int opt_level);
    /*! \brief Compatibility view of TIR optimization passes only.
     *
     * Production target scheduling remains visible only in the normalized
     * execution plan consumed by PipelineExecutor.
     */
    static Array<String> TIRPassPolicy(int opt_level, const Target& target);
};

}  // namespace api
}  // namespace kxc
