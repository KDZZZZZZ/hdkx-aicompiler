/*! \file include/kxc/compiler/control_flow.h
 * \brief Static-exact Relay preparation for ControlPlan v2.
 */
#pragma once

#include <cstdint>
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

/*! \brief Experimental fixture module-entry binding for one ControlPlan kernel. */
struct ControlKernelBinding final {
    runtime::TaskId task_id{-1};
    CompiledModule module;
    String entry_symbol;
    /*! \brief Caller label only; it proves no freshness, lease, or hot-swap safety. */
    std::uint64_t binding_revision{0};
    /*! \brief Exact input-then-constant ABI ids, preserving source order per role. */
    std::vector<runtime::ValueId> abi_non_output_value_ids;
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
