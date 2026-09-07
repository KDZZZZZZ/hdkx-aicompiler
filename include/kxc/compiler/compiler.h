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
#include "kxc/runtime/control_execution_plan.h"
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

private:
    friend struct internal::CompiledGraphAccess;

    struct State;
    explicit CompiledGraph(std::shared_ptr<const State> state);
    std::shared_ptr<const State> state_;
};

static_assert(std::is_copy_constructible_v<CompiledGraph>);
static_assert(std::is_copy_assignable_v<CompiledGraph>);

/*! \brief Immutable compiler result for one resolved control execution plan.
 *
 * Bound kernels privately retain the same process-local pin owner. The result
 * contains no Relay, TE, cache lookup, or compiler callback.
 */
class CompiledControlFlowGraph final {
public:
    bool defined() const noexcept;
    const runtime::ControlExecutionPlan& plan() const;

private:
    struct State;
    explicit CompiledControlFlowGraph(std::shared_ptr<const State> state);
    friend class Compiler;
    std::shared_ptr<const State> state_;
};

static_assert(std::is_copy_constructible_v<CompiledControlFlowGraph>);
static_assert(std::is_copy_assignable_v<CompiledControlFlowGraph>);

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

    /*! \brief Compiles one adapter-minted bounded CPU/LLVM graph artifact. */
    static CompiledGraph CompileBounded(
        const experimental::restricted_symbolic_shape::v1::BoundedCompileRequest&
            request);

    /*! \brief Explicit default-OFF production path for static CPU Relay control.
     *
     * Supports static If and bounded condition-before-body While on CPU:0/default
     * stream with real available backend artifacts. Bound kernels privately
     * retain one process-local pin owner; it is not authentication or external
     * provenance. Compiler::Compile remains the static-dataflow
     * API and continues to reject Relay control flow.
     */
    static CompiledControlFlowGraph CompileControlFlowExact(
        Function func, CompileConfig config);

    /*! \brief Builds whole-graph Relay semantics without target/compiler policy. */
    static GraphSemanticKey BuildGraphSemanticKey(const Function& func);

};

}  // namespace api
}  // namespace kxc
