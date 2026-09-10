/*! \file src/compiler/control_flow/internal_lowering.h
 * \brief Compiler-private frozen Relay payloads for control-plan resolution.
 */
#pragma once

#include <memory>
#include <vector>

#include "kxc/relay/relay.h"
#include "kxc/runtime/compiled_module.h"
#include "kxc/runtime/executable_plan.h"
#include "../internal/relay_program.h"
#include "control_plan.h"

namespace kxc::api::internal {

/*! \brief Compiler-internal topology plus its shared primitive units. */
struct ControlPlanLowering final {
    ControlPlan plan;
    std::vector<PrimitiveUnit> primitive_units;
};

ControlPlanLowering LowerRelayToControlPlanWithSidecar(Function function);
ControlPlanLowering LowerPreparedRelayToControlPlanWithSidecar(
    const PreparedRelayProgram& program);

/*! \brief Convert ControlPlan v2 into a normal ExecutablePlan with an optional
 *  structured schedule.
 *
 *  Kernel tasks reference the plan's own `calls()` by PrimitiveUnitId, so the
 *  runtime executes them through the ordinary module/ValueTable machinery. The
 *  returned plan uses ExecutablePlanMode::kStatic. */
runtime::ExecutablePlan BuildStructuredExecutablePlan(
    const ControlPlan& plan, const std::vector<PrimitiveUnit>& units);

/*! \brief Compile a Relay function with residual control topology into an
 *  ordinary CompiledGraph carrying an optional structured schedule.
 *
 *  Reuses the production primitive compiler and module assembly; only the
 *  published plan differs from the linear static path. LLVM CPU:0 only. */
CompiledGraph CompileStructuredPipeline(Function function, CompileConfig config);

}  // namespace kxc::api::internal
