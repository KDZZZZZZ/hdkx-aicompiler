/*! \file src/compiler/control_flow/control_plan_adapter.cc
 * \brief Compiler-private binding from ControlPlan v2 to runtime schema v1.
 */

#include "internal_lowering.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "runtime/internal/control_execution_plan_access.h"

namespace kxc::api::internal {
namespace {

[[noreturn]] void Fail(const std::string& detail) {
    throw std::invalid_argument("BindControlPlanForRuntime: " + detail);
}

std::vector<ValueId> AbiOrderedArguments(
    const std::vector<ValueId>& logical_arguments,
    const std::unordered_set<ValueId>& constants) {
    std::vector<ValueId> result;
    std::unordered_set<ValueId> seen;
    result.reserve(logical_arguments.size());
    for (const auto value : logical_arguments) {
        if (seen.insert(value).second) result.push_back(value);
    }
    std::stable_partition(result.begin(), result.end(), [&constants](const auto value) {
        return !constants.count(value);
    });
    return result;
}

runtime::ControlExecutionBranchSpec Convert(const BranchSpec& source) {
    runtime::ControlExecutionBranchSpec result;
    result.predicate = source.predicate;
    result.then_region = source.then_region;
    result.else_region = source.else_region;
    for (const auto& phi : source.phis) {
        result.phis.push_back({phi.result, phi.then_value, phi.else_value});
    }
    return result;
}

runtime::ControlExecutionLoopSpec Convert(const LoopSpec& source) {
    runtime::ControlExecutionLoopSpec result;
    result.condition_region = source.condition_region;
    result.body_region = source.body_region;
    result.condition_value = source.condition_value;
    result.max_trip_count = source.max_trip_count;
    for (const auto& carried : source.carried) {
        result.carried.push_back({carried.result, carried.initial,
                                  carried.body_argument, carried.backedge});
    }
    return result;
}

}  // namespace

runtime::ControlExecutionPlan BindControlPlanForRuntime(
    const ControlPlan& plan,
    const std::vector<ControlKernelBinding>& bindings) {
    // Validate every preparation invariant before dropping its effect/alias
    // fields. Artifact selection uses PrimitiveUnitId, never diagnostic text.
    plan.ValidateStaticExact();
    std::unordered_set<ValueId> constant_values(
        plan.constant_values.begin(), plan.constant_values.end());
    std::unordered_map<PrimitiveUnitId, const ControlKernelBinding*> binding_by_unit;
    for (const auto& binding : bindings) {
        if (binding.primitive_unit_id < 0 || !binding.module.defined() ||
            binding.entry_symbol == "" ||
            !binding.module.HasFunction(binding.entry_symbol) ||
            !binding.retention_owner) {
            Fail("each binding requires task id, ready module entry, and retention owner");
        }
        if (!binding_by_unit.emplace(binding.primitive_unit_id, &binding).second) {
            Fail("duplicate PrimitiveUnit binding id");
        }
    }

    runtime::internal::ControlExecutionPlanSpec resolved;
    resolved.schema_version =
        runtime::internal::ControlExecutionPlanSpec::kSchemaVersion;
    resolved.effect_model =
        runtime::ControlExecutionEffectModel::kPureFreshKernelOutputsV1;
    resolved.entry_region = plan.entry_region;
    resolved.region_order = plan.region_order;
    resolved.graph_inputs = plan.graph_inputs;
    resolved.constant_values = plan.constant_values;
    resolved.graph_outputs = plan.graph_outputs;
    const std::unordered_set<ValueId> graph_inputs(
        plan.graph_inputs.begin(), plan.graph_inputs.end());
    const std::unordered_set<ValueId> graph_outputs(
        plan.graph_outputs.begin(), plan.graph_outputs.end());
    for (const auto& value : plan.values) {
        const TensorTypeNode& tensor =
            RequireLogicalTensorType(value, "ControlPlan runtime binding");
        resolved.values.push_back(runtime::ValueSpec(
            value.id, value.id, tensor.shape,
            runtime::DataTypeFromString(tensor.dtype), value.device,
            graph_inputs.count(value.id) != 0,
            constant_values.count(value.id) != 0,
            graph_outputs.count(value.id) != 0));
        resolved.value_metadata.push_back(
            {value.id, value.source_locator});
    }

    std::unordered_set<PrimitiveUnitId> kernel_units;
    for (const auto& region : plan.regions) {
        runtime::ControlExecutionRegion target;
        target.id = region.id;
        target.live_ins = region.live_ins;
        target.live_outs = region.live_outs;
        target.source_locator = region.source_locator;
        for (const auto& task : region.tasks) {
            runtime::ControlExecutionTask bound;
            bound.id = task.id;
            bound.inputs = task.inputs;
            bound.outputs = task.outputs;
            bound.dependencies = task.dependencies;
            bound.source_locator = task.source_locator;
            bound.device = task.device;
            bound.stream = task.stream;
            switch (task.kind) {
                case ControlTaskKind::kKernel: {
                    kernel_units.insert(task.primitive_unit_id);
                    const auto found =
                        binding_by_unit.find(task.primitive_unit_id);
                    if (found == binding_by_unit.end()) {
                        Fail("missing compiled PrimitiveUnit binding for unit " +
                             std::to_string(task.primitive_unit_id));
                    }
                    const ControlKernelBinding& supplied = *found->second;
                    runtime::BoundControlKernel kernel =
                        runtime::internal::ControlExecutionPlanAccess::BindKernel(
                            supplied.module, supplied.entry_symbol,
                            supplied.retention_owner);
                    const Array<codegen::KernelArgSpec> signature = kernel.signature().arguments();
                    std::size_t expected_non_outputs = 0;
                    for (const auto& argument : signature) {
                        if (argument->role != codegen::KernelArgRole::kOutput) {
                            ++expected_non_outputs;
                        }
                    }
                    if (supplied.abi_non_output_value_ids.size() != expected_non_outputs ||
                        supplied.abi_non_output_value_ids !=
                            AbiOrderedArguments(task.argument_values,
                                                constant_values)) {
                        Fail("binding ABI non-output values do not exactly match unresolved task arguments");
                    }
                    std::size_t non_output = 0;
                    std::size_t output = 0;
                    for (const auto& argument : signature) {
                        if (argument->role == codegen::KernelArgRole::kOutput) {
                            if (output >= task.outputs.size()) {
                                Fail("binding signature has more outputs than its task");
                            }
                            bound.argument_values.push_back(task.outputs[output++]);
                        } else {
                            bound.argument_values.push_back(
                                supplied.abi_non_output_value_ids[non_output++]);
                        }
                    }
                    if (output != task.outputs.size()) {
                        Fail("binding signature has fewer outputs than its task");
                    }
                    bound.kind = runtime::ControlExecutionTaskKind::kKernel;
                    bound.kernel = std::move(kernel);
                    break;
                }
                case ControlTaskKind::kBranch:
                    bound.kind = runtime::ControlExecutionTaskKind::kBranch;
                    bound.branch = Convert(task.branch);
                    break;
                case ControlTaskKind::kLoop:
                    bound.kind = runtime::ControlExecutionTaskKind::kLoop;
                    bound.loop = Convert(task.loop);
                    break;
            }
            target.tasks.push_back(std::move(bound));
        }
        resolved.regions.push_back(std::move(target));
    }
    if (binding_by_unit.size() != kernel_units.size()) {
        Fail("extra binding for a non-kernel or absent task");
    }
    for (const auto& binding : binding_by_unit) {
        if (!kernel_units.count(binding.first)) {
            Fail("binding targets a PrimitiveUnit absent from the control plan");
        }
    }
    return runtime::internal::ControlExecutionPlanAccess::Create(
        std::move(resolved));
}

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
