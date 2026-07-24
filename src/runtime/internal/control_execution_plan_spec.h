#pragma once

#include "kxc/runtime/control_execution_plan.h"

namespace kxc::runtime::internal {

/*! Compiler-private authoring form consumed once by ControlExecutionPlanAccess. */
struct ControlExecutionPlanSpec final {
    static constexpr std::int64_t kSchemaVersion = 1;
    std::int64_t schema_version{kSchemaVersion};
    std::int64_t source_control_plan_version{2};
    ControlExecutionEffectModel effect_model{
        ControlExecutionEffectModel::kPureFreshKernelOutputsV1};
    std::vector<ControlExecutionValueSpec> values;
    ControlExecutionRegionId entry_region{-1};
    std::vector<ControlExecutionRegionId> region_order;
    std::vector<ControlExecutionRegion> regions;
    std::vector<ControlExecutionValueId> graph_inputs;
    std::vector<ControlExecutionValueId> constant_values;
    std::vector<ControlExecutionValueId> graph_outputs;
};

}  // namespace kxc::runtime::internal
