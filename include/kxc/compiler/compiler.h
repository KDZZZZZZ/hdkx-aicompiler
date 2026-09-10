/*! \file include/kxc/compiler/compiler.h
 * \brief 定义面向调用方的编译配置、Compiler 和 CompiledModule 公共 API。
 */

#pragma once

#include <cstddef>
#include <memory>
#include <type_traits>
#include <vector>

#include "kxc/compiler/compile_config.h"
#include "kxc/compiler/artifact.h"
#include "kxc/runtime/compiled_module.h"
#include "kxc/runtime/executable_plan.h"
#include "kxc/support/container.h"
#include "kxc/relay/relay.h"

namespace kxc::api::experimental::restricted_symbolic_shape::v1 {
class BoundedCompileRequest;
}

namespace kxc {
namespace api {

namespace internal {
struct CompiledGraphAccess;
}  // namespace internal

/*! \brief Immutable validated compiler output. Plan order is pin order. */
class CompiledGraph final {
public:
    CompiledGraph() = default;

    bool defined() const noexcept;
    const CompiledModule& module() const;
    const runtime::ExecutablePlan& plan() const;
    const std::vector<ArtifactPin>& artifact_pins() const;
    const GraphSemanticKey& graph_semantic_key() const;

    /*! \brief Derive a state plan through the existing validated bindings,
     *  retaining this module, artifact pins and graph semantics. */
    CompiledGraph BindStateOutputs(std::vector<runtime::StateOutputBinding> bindings,
                                   double state_fill = 0) const;
    CompiledGraph BindBoundedStateOutputs(std::vector<runtime::StateOutputBinding> bindings,
        std::vector<Array<int64_t>> physical_shapes, double state_fill = 0) const;
    /*! \brief Declare the existing bounded plan's independent request rows. */
    CompiledGraph BindRequestBatching(int64_t max_batch_size) const;

private:
    friend struct internal::CompiledGraphAccess;

    struct State;
    explicit CompiledGraph(std::shared_ptr<const State> state);
    std::shared_ptr<const State> state_;
};

static_assert(std::is_copy_constructible_v<CompiledGraph>);
static_assert(std::is_copy_assignable_v<CompiledGraph>);

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
     *
     * 当输入图残留结构化控制拓扑且 `KXC_ENABLE_CONTROL_RUNTIME` 打开时，
     * 该入口发布携带 `structured_schedule` 的普通 `CompiledGraph`，由普通
     * `RuntimeSession` 执行；否则走静态数据流路径并在准备阶段拒绝控制流。
     * \param func 输入 Relay 函数。
     * \param config 编译配置。
     * \return 可运行的编译模块。
     */
    static CompiledGraph Compile(Function func, CompileConfig config);

    /*! \brief Compiles one adapter-minted bounded LLVM or CUDA graph artifact. */
    static CompiledGraph CompileBounded(
        const experimental::restricted_symbolic_shape::v1::BoundedCompileRequest&
            request);

    /*! \brief Builds whole-graph Relay semantics without target/compiler policy. */
    static GraphSemanticKey BuildGraphSemanticKey(const Function& func);

};

}  // namespace api
}  // namespace kxc
