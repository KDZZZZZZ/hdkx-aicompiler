/*! \file include/kxc/compiler/control_flow.h
 * \brief Static-exact Relay preparation for ControlPlan v2.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kxc/relay/relay.h"
#include "kxc/runtime/compiled_module.h"
#include "kxc/runtime/control_execution_plan.h"
#include "kxc/runtime/control_plan.h"

namespace kxc::api {

/*! \brief Lowers checked static Relay, including Relay If, into a ControlPlan v2.
 *
 * This is compiler preparation/reference semantics only.  It neither creates a
 * runtime session nor performs eager execution, tracing, or backend execution.
 */
runtime::ControlPlan LowerRelayToControlPlan(Function function);

/*! \brief Immutable process-local production selection retained by the runtime.
 *
 * This is deliberately not authentication or provenance: it is a typed lifetime
 * binding minted only by Compiler::CompileControlFlowExact in this process.
 */
class ControlFlowArtifactLease final {
public:
    std::uint64_t generation() const noexcept;
    bool Covers(runtime::TaskId task_id, const String& entry_symbol,
                const std::string& signature_digest,
                const std::string& launch_metadata_digest) const;

private:
    struct Entry;
    struct State;
    explicit ControlFlowArtifactLease(std::shared_ptr<const State> state);
    friend class Compiler;
    std::shared_ptr<const State> state_;
};

/*! \brief One ControlPlan kernel binding: fixture revision or compiler-minted lease. */
struct ControlKernelBinding final {
    runtime::TaskId task_id{-1};
    CompiledModule module;
    String entry_symbol;
    /*! \brief Legacy fixture label; set only for fixture bindings. */
    std::uint64_t binding_revision{0};
    /*! \brief Exact physical input-then-constant ABI ids, first-occurrence unique. */
    std::vector<runtime::ValueId> abi_non_output_value_ids;
    /*! \brief Compiler-minted process-local production lease; never caller labels. */
    std::shared_ptr<const ControlFlowArtifactLease> production_lease;
};

/*! \brief Binds a verified ControlPlan v2 into runtime-only execution schema v1.
 *
 * Bindings are selected solely by task id and supplied module entry.  The
 * unresolved kernel reference is validated as provenance, never interpreted
 * or used for module-entry selection/dispatch.  This fixture adapter provides
 * no authority lease, staleness check, or hot-swap proof.
 */
runtime::ControlExecutionPlan BindControlPlanForRuntime(
    const runtime::ControlPlan& plan,
    const std::vector<ControlKernelBinding>& bindings);


}  // namespace kxc::api
