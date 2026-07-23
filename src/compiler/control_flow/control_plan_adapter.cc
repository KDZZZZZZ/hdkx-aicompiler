/*! \file src/compiler/control_flow/control_plan_adapter.cc
 * \brief Experimental fixture binding from ControlPlan v2 to runtime schema v1.
 */

#include "kxc/compiler/control_flow.h"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace kxc::api {
namespace {

[[noreturn]] void Fail(const std::string& detail) {
    throw std::invalid_argument("BindControlPlanForRuntime: " + detail);
}

std::vector<runtime::ValueId> AbiOrderedArguments(
    const std::vector<runtime::ValueId>& logical_arguments,
    const std::unordered_set<runtime::ValueId>& constants) {
    std::vector<runtime::ValueId> result;
    result.reserve(logical_arguments.size());
    for (const auto value : logical_arguments) {
        if (!constants.count(value)) result.push_back(value);
    }
    for (const auto value : logical_arguments) {
        if (constants.count(value)) result.push_back(value);
    }
    return result;
}

runtime::ControlExecutionBranchSpec Convert(const runtime::BranchSpec& source) {
    runtime::ControlExecutionBranchSpec result;
    result.predicate = source.predicate;
    result.then_region = source.then_region;
    result.else_region = source.else_region;
    for (const auto& phi : source.phis) {
        result.phis.push_back({phi.result, phi.then_value, phi.else_value});
    }
    return result;
}

runtime::ControlExecutionLoopSpec Convert(const runtime::LoopSpec& source) {
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
    const runtime::ControlPlan& plan,
    const std::vector<ControlKernelBinding>& bindings) {
    // Validate every preparation invariant before dropping its effect/alias
    // fields.  Fixture selection still never reads unresolved kernel_ref text.
    plan.ValidateStaticExact();
    std::unordered_set<runtime::ValueId> constant_values(
        plan.constant_values.begin(), plan.constant_values.end());
    std::unordered_map<runtime::TaskId, const ControlKernelBinding*> binding_by_task;
    for (const auto& binding : bindings) {
        if (binding.task_id < 0 || !binding.module.defined() ||
            binding.entry_symbol == "" || binding.binding_revision == 0 ||
            !binding.module.HasFunction(binding.entry_symbol)) {
            Fail("each fixture binding requires task id, ready module entry, and binding_revision > 0");
        }
        if (!binding_by_task.emplace(binding.task_id, &binding).second) {
            Fail("duplicate binding task id");
        }
    }

    runtime::ControlExecutionPlanSpec resolved;
    resolved.schema_version = runtime::ControlExecutionPlanSpec::kSchemaVersion;
    resolved.source_control_plan_version = runtime::ControlPlan::kSchemaVersion;
    resolved.effect_model =
        runtime::ControlExecutionEffectModel::kPureFreshKernelOutputsV1;
    resolved.entry_region = plan.entry_region;
    resolved.region_order = plan.region_order;
    resolved.graph_inputs = plan.graph_inputs;
    resolved.constant_values = plan.constant_values;
    resolved.graph_outputs = plan.graph_outputs;
    for (const auto& value : plan.values) {
        resolved.values.push_back({value.id, value.dtype, value.shape, value.device,
                                   value.source_locator});
    }

    std::unordered_set<runtime::TaskId> kernel_tasks;
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
                case runtime::ControlTaskKind::kKernel: {
                    kernel_tasks.insert(task.id);
                    const auto found = binding_by_task.find(task.id);
                    if (found == binding_by_task.end()) {
                        Fail("missing kernel task binding for task " + std::to_string(task.id));
                    }
                    const ControlKernelBinding& supplied = *found->second;
                    runtime::BoundControlKernel kernel(
                        supplied.module, supplied.entry_symbol,
                        supplied.binding_revision);
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
                case runtime::ControlTaskKind::kBranch:
                    bound.kind = runtime::ControlExecutionTaskKind::kBranch;
                    bound.branch = Convert(task.branch);
                    break;
                case runtime::ControlTaskKind::kLoop:
                    bound.kind = runtime::ControlExecutionTaskKind::kLoop;
                    bound.loop = Convert(task.loop);
                    break;
            }
            target.tasks.push_back(std::move(bound));
        }
        resolved.regions.push_back(std::move(target));
    }
    if (binding_by_task.size() != kernel_tasks.size()) {
        Fail("extra binding for a non-kernel or absent task");
    }
    for (const auto& binding : binding_by_task) {
        if (!kernel_tasks.count(binding.first)) Fail("binding targets a non-kernel task");
    }
    return runtime::ControlExecutionPlan(std::move(resolved));
}

}  // namespace kxc::api
