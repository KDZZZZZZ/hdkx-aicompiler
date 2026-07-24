#pragma once

#include "control_execution_plan_spec.h"

namespace kxc::runtime::internal {

/*! Source-private minting seam for compiler and focused fixture tests. */
struct ControlExecutionPlanAccess final {
    static BoundControlKernel BindKernel(
        api::CompiledModule module, String entry_symbol,
        std::shared_ptr<const void> retention_owner);
    static ControlExecutionPlan Create(ControlExecutionPlanSpec spec);
    static ControlExecutionPlanSpec CopySpec(const ControlExecutionPlan& plan);
};

}  // namespace kxc::runtime::internal
