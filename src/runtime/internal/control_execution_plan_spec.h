#pragma once

#include "kxc/runtime/control_execution_plan.h"

namespace kxc::runtime::internal {

struct ControlExecutionValueMetadata final {
    ControlExecutionValueId value_id{-1};
    std::string source_locator;
};

/*! Compiler-private authoring form consumed once by ControlExecutionPlanAccess. */
struct ControlExecutionPlanSpec final {
    static constexpr std::int64_t kSchemaVersion = 1;
    std::int64_t schema_version{kSchemaVersion};
    ControlExecutionEffectModel effect_model{
        ControlExecutionEffectModel::kPureFreshKernelOutputsV1};
    Array<ValueSpec> values;
    std::vector<ControlExecutionValueMetadata> value_metadata;
    ControlExecutionRegionId entry_region{-1};
    std::vector<ControlExecutionRegionId> region_order;
    std::vector<ControlExecutionRegion> regions;
    std::vector<ControlExecutionValueId> graph_inputs;
    std::vector<ControlExecutionValueId> constant_values;
    std::vector<ControlExecutionValueId> graph_outputs;
};

}  // namespace kxc::runtime::internal
