/*! \file src/compiler/control_flow/internal_lowering.h
 * \brief Compiler-private frozen Relay payloads for control-plan resolution.
 */
#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "kxc/relay/relay.h"
#include "kxc/runtime/compiled_module.h"
#include "kxc/runtime/control_execution_plan.h"
#include "control_plan.h"

namespace kxc::api::internal {

/* This sidecar is deliberately compiler-private.  kernel_ref is provenance
 * text only; branch compilation uses the frozen Call/attrs/constants here. */
struct ControlPlanLowering final {
    runtime::ControlPlan plan;
    std::unordered_map<runtime::TaskId, Function> kernel_functions;
};

ControlPlanLowering LowerRelayToControlPlanWithSidecar(Function function);

struct ControlKernelBinding final {
    runtime::TaskId task_id{-1};
    CompiledModule module;
    String entry_symbol;
    std::vector<runtime::ValueId> abi_non_output_value_ids;
    std::shared_ptr<const void> retention_owner;
};

runtime::ControlExecutionPlan BindControlPlanForRuntime(
    const runtime::ControlPlan& plan,
    const std::vector<ControlKernelBinding>& bindings);

}  // namespace kxc::api::internal
