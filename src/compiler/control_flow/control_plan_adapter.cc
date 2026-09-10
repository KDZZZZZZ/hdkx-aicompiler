/*! \file src/compiler/control_flow/control_plan_adapter.cc
 * \brief Compiler-private conversion from ControlPlan v2 to a normal
 *        ExecutablePlan carrying an optional structured schedule.
 */

#include "internal_lowering.h"

#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace kxc::api::internal {
namespace {

[[noreturn]] void Fail(const std::string& detail) {
    throw std::invalid_argument("BuildStructuredExecutablePlan: " + detail);
}

}  // namespace

runtime::ExecutablePlan BuildStructuredExecutablePlan(
    const ControlPlan& plan, const std::vector<PrimitiveUnit>& units) {
    plan.ValidateStaticExact();
    std::unordered_set<ValueId> constant_values(
        plan.constant_values.begin(), plan.constant_values.end());
    const std::unordered_set<ValueId> graph_inputs(
        plan.graph_inputs.begin(), plan.graph_inputs.end());
    const std::unordered_set<ValueId> graph_outputs(
        plan.graph_outputs.begin(), plan.graph_outputs.end());

    // One KernelCall per dense PrimitiveUnit. The ABI non-output order is the
    // same deduped constants-last order the runtime expects.
    std::unordered_map<PrimitiveUnitId, std::size_t> call_index_by_unit;
    Array<runtime::KernelCall> calls;
    for (std::size_t index = 0; index < units.size(); ++index) {
        const PrimitiveUnit& unit = units[index];
        if (unit.id != static_cast<PrimitiveUnitId>(index)) {
            Fail("structured plan requires dense ordered PrimitiveUnits");
        }
        call_index_by_unit.emplace(unit.id, calls.size());
        calls.push_back(runtime::KernelCall(unit.symbol, unit.boundary_input_value_ids,
                                           unit.output_value_ids));
    }

    Array<runtime::ValueSpec> value_specs;
    for (const auto& value : plan.values) {
        const TensorTypeNode& tensor =
            RequireLogicalTensorType(value, "structured runtime binding");
        value_specs.push_back(runtime::ValueSpec(
            value.id, value.id, tensor.shape,
            runtime::DataTypeFromString(tensor.dtype), value.device,
            graph_inputs.count(value.id) != 0,
            constant_values.count(value.id) != 0,
            graph_outputs.count(value.id) != 0));
    }

    runtime::StructuredSchedule schedule;
    schedule.schema_version = runtime::StructuredSchedule::kSchemaVersion;
    schedule.entry_region = plan.entry_region;
    for (const auto& region : plan.regions) {
        runtime::StructuredRegion target;
        target.id = region.id;
        target.live_ins = region.live_ins;
        target.live_outs = region.live_outs;
        for (const auto& task : region.tasks) {
            runtime::StructuredTask bound;
            bound.id = task.id;
            bound.inputs = task.inputs;
            bound.outputs = task.outputs;
            switch (task.kind) {
                case ControlTaskKind::kKernel: {
                    const auto found = call_index_by_unit.find(task.primitive_unit_id);
                    if (found == call_index_by_unit.end()) {
                        Fail("structured kernel task references an unknown PrimitiveUnit");
                    }
                    bound.kind = runtime::StructuredTaskKind::kKernel;
                    bound.call_index = static_cast<int64_t>(found->second);
                    break;
                }
                case ControlTaskKind::kBranch:
                    bound.kind = runtime::StructuredTaskKind::kBranch;
                    bound.branch.predicate = task.branch.predicate;
                    bound.branch.then_region = task.branch.then_region;
                    bound.branch.else_region = task.branch.else_region;
                    for (const auto& phi : task.branch.phis) {
                        bound.branch.phis.push_back(
                            {phi.result, phi.then_value, phi.else_value});
                    }
                    break;
                case ControlTaskKind::kLoop:
                    bound.kind = runtime::StructuredTaskKind::kLoop;
                    bound.loop.condition_region = task.loop.condition_region;
                    bound.loop.body_region = task.loop.body_region;
                    bound.loop.condition_value = task.loop.condition_value;
                    bound.loop.max_trip_count = task.loop.max_trip_count;
                    for (const auto& carried : task.loop.carried) {
                        bound.loop.carried.push_back({carried.result, carried.initial,
                                                      carried.body_argument,
                                                      carried.backedge});
                    }
                    break;
            }
            target.tasks.push_back(std::move(bound));
        }
        schedule.regions.push_back(std::move(target));
    }
    schedule.region_order = plan.region_order;

    Array<int64_t> inputs(plan.graph_inputs);
    Array<int64_t> constants(plan.constant_values);
    Array<int64_t> outputs(plan.graph_outputs);
    runtime::ExecutablePlan executable(
        std::move(value_specs), std::move(calls), std::move(inputs),
        std::move(constants), std::move(outputs), {}, runtime::ExecutablePlanMode::kStatic,
        {}, {}, -1, {}, std::nullopt, std::move(schedule));
    executable.Validate();
    return executable;
}

}  // namespace kxc::api::internal
