/*! \file src/compiler/control_flow/internal_lowering.h
 * \brief Compiler-private frozen Relay payloads for control-plan resolution.
 */
#pragma once

#include <memory>
#include <vector>

#include "kxc/relay/relay.h"
#include "kxc/runtime/compiled_module.h"
#include "kxc/runtime/control_execution_plan.h"
#include "../internal/relay_program.h"
#include "control_plan.h"

namespace kxc::api::internal {

/*! \brief Compiler-internal topology plus its shared primitive units. */
struct ControlPlanLowering final {
    runtime::ControlPlan plan;
    std::vector<PrimitiveUnit> primitive_units;
};

ControlPlanLowering LowerRelayToControlPlanWithSidecar(Function function);
ControlPlanLowering LowerPreparedRelayToControlPlanWithSidecar(
    const PreparedRelayProgram& program);

struct ControlKernelBinding final {
    PrimitiveUnitId primitive_unit_id{-1};
    CompiledModule module;
    String entry_symbol;
    std::vector<runtime::ValueId> abi_non_output_value_ids;
    std::shared_ptr<const void> retention_owner;
};

runtime::ControlExecutionPlan BindControlPlanForRuntime(
    const runtime::ControlPlan& plan,
    const std::vector<ControlKernelBinding>& bindings);

}  // namespace kxc::api::internal
