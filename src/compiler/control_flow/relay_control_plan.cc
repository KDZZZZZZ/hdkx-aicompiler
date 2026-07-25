/*! \file src/compiler/control_flow/relay_control_plan.cc
 * \brief Static-exact Relay-to-ControlPlan preparation lowering.
 */

#include "internal_lowering.h"

#include "../internal/executable_capability.h"
#include "../internal/logical_value.h"
#include "../internal/resolved_relay_call.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "kxc/relay/transforms/infer_type.h"
#include "kxc/relay/transforms/normalize_to_anf.h"

namespace kxc::api {
namespace {

using Leaves = std::vector<runtime::ValueId>;

[[noreturn]] void Fail(const std::string& path, const std::string& detail) {
    throw std::invalid_argument("LowerRelayToControlPlan: path=" + path + "; " + detail);
}

struct RegionState {
    std::unordered_set<runtime::ValueId> local_values;
    std::unordered_map<runtime::ValueId, runtime::TaskId> local_producers;
    std::unordered_set<runtime::ValueId> live_in_set;
};

class ControlPlanBuilder final {
public:
    explicit ControlPlanBuilder(Function function, bool prepared = false)
        : function_(std::move(function)), prepared_(prepared) {}

    internal::ControlPlanLowering BuildWithSidecar() {
        if (!function_.defined()) {
            throw std::invalid_argument("LowerRelayToControlPlan requires a defined Function");
        }
        if (!prepared_) {
            function_ = relay::InferTypePass(function_);
            function_ = relay::NormalizeToANF(function_);
            relay::VerifyANF(function_);
            internal::ExecutableCapabilityOptions options;
            options.version = internal::ExecutableCapabilityOptions::kVersion;
            options.allow_if = true;
            options.allow_while = true;
            options.allow_tuple_parameters = true;
            options.allow_nested_tuple_call_outputs = true;
            options.allow_device_regions = true;
            internal::VerifyExecutableCapability(function_, options);
        } else {
            relay::VerifyANF(function_);
        }

        const runtime::RegionId root = NewRegion("function");
        plan_.entry_region = root;
        Env environment;
        for (std::size_t i = 0; i < function_->params.size(); ++i) {
            const Var& parameter = function_->params[i];
            const std::string path = "function.params[" + std::to_string(i) + "]";
            const Leaves ids = AddLeaves(Expr(ObjectRef(parameter)),
                                         parameter.checked_type(),
                                         internal::LogicalValueOrigin::kParameter,
                                         path);
            environment.emplace(parameter.get(), ids);
            plan_.graph_inputs.insert(plan_.graph_inputs.end(), ids.begin(), ids.end());
        }

        const Leaves outputs = LowerTerminal(function_->body, root, &environment,
                                             "function.body");
        ValidateAliasPlacement(Expr(ObjectRef(function_)), outputs, "function");
        if (outputs.empty()) Fail("function.body", "requires at least one tensor graph output");
        std::unordered_set<runtime::ValueId> output_set;
        for (const runtime::ValueId output : outputs) {
            if (!output_set.insert(output).second) {
                Fail("function.body", "duplicate graph output value is not representable in ControlPlan v2");
            }
        }
        plan_.graph_outputs = outputs;
        runtime::ControlRegion& entry = Region(root);
        entry.live_ins = plan_.graph_inputs;
        entry.live_ins.insert(entry.live_ins.end(), plan_.constant_values.begin(),
                              plan_.constant_values.end());
        entry.live_outs = outputs;
        FinalizeRegions();
        plan_.ValidateStaticExact();
        return internal::ControlPlanLowering{std::move(plan_),
                                                   std::move(kernel_functions_)};
    }

private:
    using Env = std::unordered_map<const Object*, Leaves>;

    Function function_;
    bool prepared_{false};
    runtime::ControlPlan plan_;
    runtime::ValueId next_value_{0};
    runtime::RegionId next_region_{0};
    runtime::TaskId next_task_{0};
    std::unordered_map<runtime::RegionId, std::size_t> region_index_;
    std::unordered_map<runtime::RegionId, RegionState> region_state_;
    std::unordered_map<const Object*, Leaves> constants_;
    std::unordered_map<runtime::TaskId, Function> kernel_functions_;

    runtime::ControlRegion& Region(runtime::RegionId id) {
        return plan_.regions.at(region_index_.at(id));
    }

    RegionState& State(runtime::RegionId id) { return region_state_.at(id); }

    runtime::RegionId NewRegion(const std::string& path) {
        const runtime::RegionId id = next_region_++;
        region_index_.emplace(id, plan_.regions.size());
        plan_.regions.push_back(runtime::ControlRegion{});
        runtime::ControlRegion& region = plan_.regions.back();
        region.id = id;
        region.source_locator = path;
        plan_.region_order.push_back(id);
        region_state_.emplace(id, RegionState{});
        return id;
    }

    const runtime::ControlValueSpec& Value(runtime::ValueId id) const {
        if (id < 0 || static_cast<std::size_t>(id) >= plan_.values.size() ||
            plan_.values[static_cast<std::size_t>(id)].id != id) {
            throw std::logic_error("ControlPlanBuilder lost its dense value ids");
        }
        return plan_.values[static_cast<std::size_t>(id)];
    }

    void ValidateAliasPlacement(const Expr& alias, const Leaves& leaves,
                                const std::string& path) const {
        const auto* relay_node = dynamic_cast<const RelayNode*>(alias.get());
        if (!relay_node || !relay_node->virtual_device_.defined()) return;
        const Device expected =
            internal::ResolveLogicalValueDevice(alias, Device::CPU(), path);
        for (runtime::ValueId leaf : leaves) {
            if (Value(leaf).device != expected) {
                Fail(path,
                     "structural alias placement requires an explicit copy task");
            }
        }
    }

    Leaves AddLeaves(const Expr& source, const Type& type,
                     internal::LogicalValueOrigin origin,
                     const std::string& path) {
        std::vector<internal::LogicalValueContract> values =
            internal::MakeLogicalValueLeaves(
                source, type, origin, next_value_, Device::CPU(), path);
        Leaves ids;
        ids.reserve(values.size());
        for (auto& value : values) {
            ++next_value_;
            plan_.values.push_back(std::move(value));
            ids.push_back(plan_.values.back().id);
        }
        return ids;
    }

    void MarkRead(runtime::RegionId region, runtime::ValueId value) {
        RegionState& state = State(region);
        if (state.local_values.count(value) || !state.live_in_set.insert(value).second) return;
        Region(region).live_ins.push_back(value);
    }

    Leaves ResolveAtomic(const Expr& expr, runtime::RegionId region,
                         const Env& environment, const std::string& path) {
        Leaves ids;
        if (const auto* var = expr.As<VarNode>()) {
            const auto it = environment.find(expr.get());
            if (it == environment.end()) {
                Fail(path, "encountered free or unbound Var '" + var->vid->name_hint + "'");
            }
            ids = it->second;
        } else if (expr.As<ConstantNode>()) {
            const auto found = constants_.find(expr.get());
            if (found != constants_.end()) {
                ids = found->second;
            } else {
                ids = AddLeaves(expr, expr.checked_type(),
                                internal::LogicalValueOrigin::kConstant, path);
                constants_.emplace(expr.get(), ids);
                plan_.constant_values.insert(plan_.constant_values.end(), ids.begin(), ids.end());
            }
        } else {
            Fail(path, "ANF lowering expected an atomic Var or Constant");
        }
        ValidateAliasPlacement(expr, ids, path);
        for (const runtime::ValueId id : ids) MarkRead(region, id);
        return ids;
    }

    static Leaves UniqueBoundaryInputs(const Leaves& arguments) {
        Leaves inputs;
        std::unordered_set<runtime::ValueId> seen;
        for (const runtime::ValueId argument : arguments) {
            if (seen.insert(argument).second) inputs.push_back(argument);
        }
        return inputs;
    }

    std::vector<runtime::TaskId> Dependencies(runtime::RegionId region,
                                               const Leaves& inputs) {
        std::vector<runtime::TaskId> dependencies;
        std::unordered_set<runtime::TaskId> seen;
        const RegionState& state = State(region);
        for (const runtime::ValueId input : inputs) {
            const auto producer = state.local_producers.find(input);
            if (producer != state.local_producers.end() && seen.insert(producer->second).second) {
                dependencies.push_back(producer->second);
            }
        }
        return dependencies;
    }

    runtime::TaskId AddTask(runtime::RegionId region, runtime::ControlTask task) {
        RegionState& state = State(region);
        task.id = next_task_++;
        task.dependencies = Dependencies(region, task.inputs);
        task.effect.reads = task.inputs;
        Region(region).tasks.push_back(std::move(task));
        const runtime::ControlTask& stored = Region(region).tasks.back();
        for (const runtime::ValueId output : stored.outputs) {
            state.local_values.insert(output);
            state.local_producers.emplace(output, stored.id);
        }
        return stored.id;
    }

    Leaves LowerCall(const Expr& expr, const CallNode* call, runtime::RegionId region,
                     const Env& environment, const std::string& path) {
        internal::ResolvedRelayCall resolved = [&] {
            try {
                return internal::ResolveRelayCall(
                    expr,
                    internal::OperatorCapabilityPolicy::StructuredControl(),
                    path);
            } catch (const internal::RelayCallResolutionError& error) {
                Fail(error.issue().path,
                     "capability=" + error.issue().capability + "; " +
                         error.issue().detail);
            }
        }();

        Leaves arguments;
        for (std::size_t i = 0; i < call->args.size(); ++i) {
            const Leaves leaves = ResolveAtomic(call->args[i], region, environment,
                                                path + ".args[" + std::to_string(i) + "]");
            arguments.insert(arguments.end(), leaves.begin(), leaves.end());
        }
        const Leaves outputs = AddLeaves(
            expr, expr.checked_type(),
            internal::LogicalValueOrigin::kPrimitiveOutput, path);
        if (outputs.size() != resolved.output_leaf_types.size()) {
            Fail(path,
                 "resolved output leaves do not match control value leaves");
        }
        runtime::ControlTask task;
        task.kind = runtime::ControlTaskKind::kKernel;
        task.binding_state =
            runtime::KernelBindingState::kUnresolvedRelayKernel;
        task.inputs = UniqueBoundaryInputs(arguments);
        task.argument_values = arguments;
        task.outputs = outputs;
        task.device = Value(outputs.front()).device;
        for (const runtime::ValueId input : task.inputs) {
            if (Value(input).device != task.device) {
                Fail(path, "kernel inputs and outputs require one explicit device");
            }
        }
        for (const runtime::ValueId output : task.outputs) {
            if (Value(output).device != task.device) {
                Fail(path, "kernel outputs require one explicit device");
            }
        }
        task.kernel_ref =
            "relay.kernel.v2;" +
            relay::SerializeOperatorSpec(resolved.spec) +
            ";attrs=" + relay::SerializeAttrs(resolved.attrs);
        task.source_locator = path;
        const runtime::TaskId task_id = AddTask(region, std::move(task));
        Array<Var> parameters;
        Array<Expr> substituted_arguments;
        std::unordered_map<const Object*, Expr> substitutions;
        for (std::size_t index = 0; index < call->args.size(); ++index) {
            const Expr& argument = call->args[index];
            if (argument.As<ConstantNode>()) {
                substituted_arguments.push_back(argument);
                continue;
            }
            const auto* variable = argument.As<VarNode>();
            if (!variable || !argument.checked_type().defined()) {
                Fail(path, "frozen kernel sidecar requires typed ANF atomic arguments");
            }
            auto found = substitutions.find(argument.get());
            if (found == substitutions.end()) {
                Var fresh("control_task_" + std::to_string(task_id) + "_arg_" +
                              std::to_string(parameters.size()),
                          argument.checked_type());
                found = substitutions.emplace(argument.get(), Expr(fresh)).first;
                parameters.push_back(std::move(fresh));
            }
            substituted_arguments.push_back(found->second);
        }
        // Branch-local and ANF lexical Vars are replaced with fresh typed function
        // parameters.  The task's ValueId ABI is retained separately above.
        kernel_functions_.emplace(
            task_id, Function(std::move(parameters),
                              Call(call->op, std::move(substituted_arguments),
                                   call->attrs)));
        return outputs;
    }

    Leaves LowerTupleGetItem(const Expr& expr,
                             const TupleGetItemNode* get_item,
                             runtime::RegionId region, const Env& environment,
                             const std::string& path) {
        const Leaves tuple = ResolveAtomic(get_item->tuple, region, environment,
                                           path + ".tuple");
        const auto* tuple_type = get_item->tuple.checked_type().As<TupleTypeNode>();
        if (!tuple_type || get_item->index < 0 ||
            static_cast<std::size_t>(get_item->index) >= tuple_type->fields.size()) {
            Fail(path, "TupleGetItem is outside its checked TupleType");
        }
        std::size_t begin = 0;
        for (int i = 0; i < get_item->index; ++i) {
            begin += internal::FlattenLogicalTensorTypes(
                         tuple_type->fields[static_cast<std::size_t>(i)],
                         path + ".tuple.checked_type")
                         .size();
        }
        const std::vector<Type> selected_types =
            internal::FlattenLogicalTensorTypes(
                tuple_type->fields[static_cast<std::size_t>(get_item->index)],
                path + ".checked_type");
        if (begin + selected_types.size() > tuple.size()) {
            Fail(path, "TupleGetItem flattening is inconsistent with its checked TupleType");
        }
        Leaves selected(
            tuple.begin() + static_cast<std::ptrdiff_t>(begin),
            tuple.begin() +
                static_cast<std::ptrdiff_t>(begin + selected_types.size()));
        ValidateAliasPlacement(expr, selected, path);
        return selected;
    }

    Leaves LowerWhile(const Expr& expr, const WhileNode* while_node,
                      runtime::RegionId parent, const Env& environment,
                      const std::string& path) {
        const Leaves initial = ResolveAtomic(while_node->initial_state, parent,
                                             environment, path + ".initial_state");
        const Leaves results = AddLeaves(
            expr, expr.checked_type(),
            internal::LogicalValueOrigin::kLoopCarried, path);
        if (initial.empty() || initial.size() != results.size()) {
            Fail(path, "While state flattening does not match its result type");
        }
        runtime::ControlTask task;
        task.kind = runtime::ControlTaskKind::kLoop;
        task.outputs = results;
        task.source_locator = path;
        AddTask(parent, std::move(task));
        const std::size_t parent_task_index = Region(parent).tasks.size() - 1;

        const Leaves arguments = AddLeaves(Expr(ObjectRef(while_node->loop_var)),
                                           while_node->loop_var.checked_type(),
                                           internal::LogicalValueOrigin::kLoopCarried,
                                           path + ".loop_var");
        const runtime::RegionId condition_region = NewRegion(path + ".condition");
        Env condition_environment = environment;
        condition_environment[while_node->loop_var.get()] = arguments;
        const Leaves condition = LowerTerminal(while_node->condition, condition_region,
                                               &condition_environment, path + ".condition");
        if (condition.size() != 1 ||
            !internal::IsCpuScalarBool(Value(condition.front()))) {
            Fail(path + ".condition", "requires a CPU scalar bool condition");
        }
        Region(condition_region).live_outs = condition;
        for (const runtime::ValueId argument : arguments) MarkRead(condition_region, argument);

        const runtime::RegionId body_region = NewRegion(path + ".body");
        Env body_environment = environment;
        body_environment[while_node->loop_var.get()] = arguments;
        const Leaves backedges = LowerTerminal(while_node->body, body_region,
                                               &body_environment, path + ".body");
        if (backedges.size() != results.size()) {
            Fail(path + ".body", "While body flattening does not match state type");
        }
        Region(body_region).live_outs = backedges;
        for (const runtime::ValueId argument : arguments) MarkRead(body_region, argument);

        runtime::ControlTask& loop = Region(parent).tasks.at(parent_task_index);
        loop.loop.condition_region = condition_region;
        loop.loop.body_region = body_region;
        loop.loop.condition_value = condition.front();
        loop.loop.max_trip_count = while_node->max_trip_count;
        for (std::size_t i = 0; i < results.size(); ++i) {
            loop.loop.carried.push_back({results[i], initial[i], arguments[i], backedges[i]});
        }
        const auto append_captures = [this, parent, &loop, &arguments](
                                         const runtime::ControlRegion& child) {
            for (const runtime::ValueId capture : child.live_ins) {
                if (std::find(arguments.begin(), arguments.end(), capture) != arguments.end()) continue;
                MarkRead(parent, capture);
                if (std::find(loop.inputs.begin(), loop.inputs.end(), capture) == loop.inputs.end()) {
                    loop.inputs.push_back(capture);
                }
            }
        };
        for (const runtime::ValueId value : initial) loop.inputs.push_back(value);
        append_captures(Region(condition_region));
        append_captures(Region(body_region));
        loop.effect.reads = loop.inputs;
        loop.dependencies = Dependencies(parent, loop.inputs);
        return results;
    }

    Leaves LowerIf(const Expr& expr, const IfNode* if_node, runtime::RegionId parent,
                   const Env& environment, const std::string& path) {
        const Leaves predicate = ResolveAtomic(if_node->cond, parent, environment,
                                               path + ".cond");
        if (predicate.size() != 1) Fail(path + ".cond", "requires one scalar bool predicate leaf");
        const Leaves results = AddLeaves(
            expr, expr.checked_type(), internal::LogicalValueOrigin::kPhi,
            path);
        if (results.empty()) Fail(path, "If must produce at least one tensor leaf");

        runtime::ControlTask task;
        task.kind = runtime::ControlTaskKind::kBranch;
        task.outputs = results;
        task.source_locator = path;
        AddTask(parent, std::move(task));
        const std::size_t parent_task_index = Region(parent).tasks.size() - 1;

        const runtime::RegionId then_region = NewRegion(path + ".true_branch");
        Env then_environment = environment;
        const Leaves then_values = LowerTerminal(if_node->true_branch, then_region,
                                                 &then_environment, path + ".true_branch");
        Region(then_region).live_outs = then_values;
        const runtime::RegionId else_region = NewRegion(path + ".false_branch");
        Env else_environment = environment;
        const Leaves else_values = LowerTerminal(if_node->false_branch, else_region,
                                                 &else_environment, path + ".false_branch");
        Region(else_region).live_outs = else_values;
        if (then_values.size() != results.size() || else_values.size() != results.size()) {
            Fail(path, "If branch flattening does not match the result type");
        }

        runtime::ControlTask& branch_task = Region(parent).tasks.at(parent_task_index);
        branch_task.branch.predicate = predicate.front();
        branch_task.branch.then_region = then_region;
        branch_task.branch.else_region = else_region;
        for (std::size_t i = 0; i < results.size(); ++i) {
            branch_task.branch.phis.push_back({results[i], then_values[i], else_values[i]});
        }
        branch_task.inputs.push_back(predicate.front());
        const auto append_captures = [this, parent, &branch_task](const runtime::ControlRegion& child) {
            for (const runtime::ValueId capture : child.live_ins) {
                MarkRead(parent, capture);
                if (std::find(branch_task.inputs.begin(), branch_task.inputs.end(), capture) ==
                    branch_task.inputs.end()) {
                    branch_task.inputs.push_back(capture);
                }
            }
        };
        append_captures(Region(then_region));
        append_captures(Region(else_region));
        branch_task.effect.reads = branch_task.inputs;
        branch_task.dependencies = Dependencies(parent, branch_task.inputs);
        return results;
    }

    Leaves LowerValue(const Expr& expr, runtime::RegionId region, const Env& environment,
                      const std::string& path) {
        if (expr.As<VarNode>() || expr.As<ConstantNode>()) {
            return ResolveAtomic(expr, region, environment, path);
        }
        if (const auto* call = expr.As<CallNode>()) return LowerCall(expr, call, region, environment, path);
        if (const auto* if_node = expr.As<IfNode>()) return LowerIf(expr, if_node, region, environment, path);
        if (const auto* while_node = expr.As<WhileNode>()) return LowerWhile(expr, while_node, region, environment, path);
        if (const auto* tuple = expr.As<TupleNode>()) {
            Leaves values;
            for (std::size_t i = 0; i < tuple->fields.size(); ++i) {
                const Leaves field = ResolveAtomic(tuple->fields[i], region, environment,
                                                   path + ".fields[" + std::to_string(i) + "]");
                values.insert(values.end(), field.begin(), field.end());
            }
            ValidateAliasPlacement(expr, values, path);
            return values;
        }
        if (const auto* get_item = expr.As<TupleGetItemNode>()) {
            return LowerTupleGetItem(expr, get_item, region, environment, path);
        }
        Fail(path, "encountered unsupported non-ANF Relay value");
    }

    Leaves LowerTerminal(const Expr& expr, runtime::RegionId region, Env* environment,
                         const std::string& path) {
        if (const auto* let = expr.As<LetNode>()) {
            const Leaves value = LowerValue(let->value, region, *environment, path + ".value");
            ValidateAliasPlacement(Expr(ObjectRef(let->var)), value,
                                   path + ".var");
            const auto existing = environment->find(let->var.get());
            const bool had_existing = existing != environment->end();
            const Leaves saved = had_existing ? existing->second : Leaves{};
            (*environment)[let->var.get()] = value;
            const Leaves result = LowerTerminal(let->body, region, environment, path + ".body");
            ValidateAliasPlacement(expr, result, path);
            if (had_existing) {
                (*environment)[let->var.get()] = saved;
            } else {
                environment->erase(let->var.get());
            }
            return result;
        }
        return LowerValue(expr, region, *environment, path);
    }

    void FinalizeRegions() {
        for (runtime::ControlRegion& region : plan_.regions) {
            region.effect.reads = region.live_ins;
        }
    }
};

}  // namespace

namespace internal {
ControlPlanLowering LowerRelayToControlPlanWithSidecar(Function function) {
    return ControlPlanBuilder(std::move(function)).BuildWithSidecar();
}

ControlPlanLowering LowerPreparedRelayToControlPlanWithSidecar(
    const PreparedRelayProgram& program) {
    if (!program.residual_profile().requires_control_topology()) {
        throw std::invalid_argument(
            "LowerPreparedRelayToControlPlanWithSidecar requires residual "
            "structured control");
    }
    return ControlPlanBuilder(program.typed_anf(), true).BuildWithSidecar();
}
}  // namespace internal

}  // namespace kxc::api
