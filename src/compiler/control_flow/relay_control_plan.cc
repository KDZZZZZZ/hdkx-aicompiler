/*! \file src/compiler/control_flow/relay_control_plan.cc
 * \brief Static-exact Relay-to-ControlPlan preparation lowering.
 */

#include "internal_lowering.h"

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

using Leaves = std::vector<internal::ValueId>;

[[noreturn]] void Fail(const std::string& path, const std::string& detail) {
    throw std::invalid_argument("LowerRelayToControlPlan: path=" + path + "; " + detail);
}

struct RegionState {
    std::unordered_set<internal::ValueId> local_values;
    std::unordered_map<internal::ValueId, internal::TaskId> local_producers;
    std::unordered_set<internal::ValueId> live_in_set;
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
        } else {
            relay::VerifyANF(function_);
        }

        if (!function_->body.defined()) {
            Fail("function", "capability=defined_typed_relay; Function has no body");
        }
        (void)RequireCheckedType(Expr(ObjectRef(function_)), "function");
        const internal::RegionId root = NewRegion("function");
        plan_.entry_region = root;
        Env environment;
        for (std::size_t i = 0; i < function_->params.size(); ++i) {
            const Var& parameter = function_->params[i];
            const std::string path = "function.params[" + std::to_string(i) + "]";
            if (!parameter.defined() || !parameter->type_annotation.defined()) {
                Fail(path, "capability=typed_parameter; parameter annotation is missing");
            }
            const Type parameter_type = RequireCheckedType(
                Expr(ObjectRef(parameter)), path);
            if (!TypeEqual(parameter->type_annotation, parameter_type)) {
                Fail(path, "capability=typed_parameter; annotation and checked_type differ");
            }
            const Leaves ids = AddLeaves(Expr(ObjectRef(parameter)),
                                         parameter_type,
                                         internal::LogicalValueOrigin::kParameter,
                                         path);
            environment.emplace(parameter.get(), ids);
            plan_.graph_inputs.insert(plan_.graph_inputs.end(), ids.begin(), ids.end());
        }

        const Leaves outputs = LowerTerminal(function_->body, root, &environment,
                                             "function.body");
        if (!TypeEqual(RequireCheckedType(Expr(ObjectRef(function_)), "function"),
                       RequireCheckedType(function_->body, "function.body"))) {
            Fail("function", "capability=typed_function_result; Function and body checked_type differ");
        }
        ValidateAliasPlacement(Expr(ObjectRef(function_)), outputs, "function");
        if (outputs.empty()) Fail("function.body", "requires at least one tensor graph output");
        std::unordered_set<internal::ValueId> output_set;
        for (const internal::ValueId output : outputs) {
            if (!output_set.insert(output).second) {
                Fail("function.body", "duplicate graph output value is not representable in ControlPlan v2");
            }
        }
        plan_.graph_outputs = outputs;
        internal::ControlRegion& entry = Region(root);
        entry.live_ins = plan_.graph_inputs;
        entry.live_ins.insert(entry.live_ins.end(), plan_.constant_values.begin(),
                              plan_.constant_values.end());
        entry.live_outs = outputs;
        FinalizeRegions();
        plan_.ValidateStaticExact();
        return internal::ControlPlanLowering{std::move(plan_),
                                              std::move(primitive_units_)};
    }

private:
    using Env = std::unordered_map<const Object*, Leaves>;

    Function function_;
    bool prepared_{false};
    internal::ControlPlan plan_;
    internal::ValueId next_value_{0};
    internal::RegionId next_region_{0};
    internal::TaskId next_task_{0};
    std::unordered_map<internal::RegionId, std::size_t> region_index_;
    std::unordered_map<internal::RegionId, RegionState> region_state_;
    std::unordered_map<const Object*, Leaves> constants_;
    std::vector<internal::PrimitiveUnit> primitive_units_;

    internal::ControlRegion& Region(internal::RegionId id) {
        return plan_.regions.at(region_index_.at(id));
    }

    RegionState& State(internal::RegionId id) { return region_state_.at(id); }

    internal::RegionId NewRegion(const std::string& path) {
        const internal::RegionId id = next_region_++;
        region_index_.emplace(id, plan_.regions.size());
        plan_.regions.push_back(internal::ControlRegion{});
        internal::ControlRegion& region = plan_.regions.back();
        region.id = id;
        region.source_locator = path;
        plan_.region_order.push_back(id);
        region_state_.emplace(id, RegionState{});
        return id;
    }

    const internal::ControlValueSpec& Value(internal::ValueId id) const {
        if (id < 0 || static_cast<std::size_t>(id) >= plan_.values.size() ||
            plan_.values[static_cast<std::size_t>(id)].id != id) {
            throw std::logic_error("ControlPlanBuilder lost its dense value ids");
        }
        return plan_.values[static_cast<std::size_t>(id)];
    }

    Type RequireCheckedType(const Expr& expr, const std::string& path) const {
        const Type type = expr.checked_type();
        if (!type.defined()) {
            Fail(path, "capability=defined_typed_relay; checked_type is missing");
        }
        return type;
    }

    Device Placement(const Expr& expr, const std::string& path) const {
        return internal::ResolveLogicalValueDevice(expr, Device::CPU(), path);
    }

    void ValidateAliasPlacement(const Expr& alias, const Leaves& leaves,
                                const std::string& path) const {
        const auto* relay_node = dynamic_cast<const RelayNode*>(alias.get());
        if (!relay_node || !relay_node->virtual_device_.defined()) return;
        const Device expected =
            internal::ResolveLogicalValueDevice(alias, Device::CPU(), path);
        for (internal::ValueId leaf : leaves) {
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

    void MarkRead(internal::RegionId region, internal::ValueId value) {
        RegionState& state = State(region);
        if (state.local_values.count(value) || !state.live_in_set.insert(value).second) return;
        Region(region).live_ins.push_back(value);
    }

    Leaves ResolveAtomic(const Expr& expr, internal::RegionId region,
                         const Env& environment, const std::string& path) {
        Leaves ids;
        if (const auto* var = expr.As<VarNode>()) {
            const auto it = environment.find(expr.get());
            if (it == environment.end()) {
                Fail(path, "capability=lexically_bound_var; encountered free or unbound Var '" + var->vid->name_hint + "'");
            }
            (void)RequireCheckedType(expr, path);
            ids = it->second;
        } else if (const auto* constant = expr.As<ConstantNode>()) {
            if (!constant->data.defined()) {
                Fail(path, "capability=constant_payload; constant payload is undefined");
            }
            const auto found = constants_.find(expr.get());
            if (found != constants_.end()) {
                ids = found->second;
            } else {
                ids = AddLeaves(expr, RequireCheckedType(expr, path),
                                internal::LogicalValueOrigin::kConstant, path);
                constants_.emplace(expr.get(), ids);
                plan_.constant_values.insert(plan_.constant_values.end(), ids.begin(), ids.end());
            }
        } else {
            Fail(path, "ANF lowering expected an atomic Var or Constant");
        }
        ValidateAliasPlacement(expr, ids, path);
        for (const internal::ValueId id : ids) MarkRead(region, id);
        return ids;
    }

    std::vector<internal::TaskId> Dependencies(internal::RegionId region,
                                               const Leaves& inputs) {
        std::vector<internal::TaskId> dependencies;
        std::unordered_set<internal::TaskId> seen;
        const RegionState& state = State(region);
        for (const internal::ValueId input : inputs) {
            const auto producer = state.local_producers.find(input);
            if (producer != state.local_producers.end() && seen.insert(producer->second).second) {
                dependencies.push_back(producer->second);
            }
        }
        return dependencies;
    }

    internal::TaskId AddTask(internal::RegionId region, internal::ControlTask task) {
        RegionState& state = State(region);
        task.id = next_task_++;
        task.dependencies = Dependencies(region, task.inputs);
        task.effect.reads = task.inputs;
        Region(region).tasks.push_back(std::move(task));
        const internal::ControlTask& stored = Region(region).tasks.back();
        for (const internal::ValueId output : stored.outputs) {
            state.local_values.insert(output);
            state.local_producers.emplace(output, stored.id);
        }
        return stored.id;
    }

    Leaves LowerCall(const Expr& expr, const CallNode* call, internal::RegionId region,
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
            expr, RequireCheckedType(expr, path),
            internal::LogicalValueOrigin::kPrimitiveOutput, path);
        if (outputs.size() != resolved.output_leaf_types.size()) {
            Fail(path,
                 "resolved output leaves do not match control value leaves");
        }
        internal::ControlTask task;
        task.kind = internal::ControlTaskKind::kKernel;
        internal::PrimitiveUnit unit = internal::BuildPrimitiveUnit(
            static_cast<internal::PrimitiveUnitId>(primitive_units_.size()),
            std::move(resolved), arguments, outputs, plan_.values);
        task.primitive_unit_id = unit.id;
        task.inputs.assign(unit.boundary_input_value_ids.begin(),
                           unit.boundary_input_value_ids.end());
        task.argument_values = arguments;
        task.outputs = outputs;
        task.device = unit.device;
        for (const internal::ValueId input : task.inputs) {
            if (Value(input).device != task.device) {
                Fail(path, "kernel inputs and outputs require one explicit device");
            }
        }
        for (const internal::ValueId output : task.outputs) {
            if (Value(output).device != task.device) {
                Fail(path, "kernel outputs require one explicit device");
            }
        }
        task.source_locator = path;
        AddTask(region, std::move(task));
        primitive_units_.push_back(std::move(unit));
        return outputs;
    }

    Leaves LowerTupleGetItem(const Expr& expr,
                             const TupleGetItemNode* get_item,
                             internal::RegionId region, const Env& environment,
                             const std::string& path) {
        const Leaves tuple = ResolveAtomic(get_item->tuple, region, environment,
                                           path + ".tuple");
        const Type result_type = RequireCheckedType(expr, path);
        const auto* tuple_type =
            RequireCheckedType(get_item->tuple, path + ".tuple").As<TupleTypeNode>();
        if (!tuple_type || get_item->index < 0 ||
            static_cast<std::size_t>(get_item->index) >= tuple_type->fields.size()) {
            Fail(path, "capability=well_typed_tuple_get_item; TupleGetItem is outside its checked TupleType");
        }
        if (!TypeEqual(result_type,
                       tuple_type->fields[static_cast<std::size_t>(get_item->index)])) {
            Fail(path, "capability=well_typed_tuple_get_item; TupleGetItem checked type differs from its selected field");
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
            Fail(path, "capability=well_typed_tuple_get_item; TupleGetItem flattening is inconsistent with its checked TupleType");
        }
        Leaves selected(
            tuple.begin() + static_cast<std::ptrdiff_t>(begin),
            tuple.begin() +
                static_cast<std::ptrdiff_t>(begin + selected_types.size()));
        ValidateAliasPlacement(expr, selected, path);
        return selected;
    }

    Leaves LowerWhile(const Expr& expr, const WhileNode* while_node,
                      internal::RegionId parent, const Env& environment,
                      const std::string& path) {
        if (!while_node->loop_var.defined() || while_node->max_trip_count < 0) {
            Fail(path, "capability=bounded_loop; While requires a defined binder and non-negative max_trip_count");
        }
        const Type initial_type = RequireCheckedType(while_node->initial_state,
                                                     path + ".initial_state");
        const Type loop_type = RequireCheckedType(
            Expr(ObjectRef(while_node->loop_var)), path + ".loop_var");
        const Type result_type = RequireCheckedType(expr, path);
        if (!TypeEqual(initial_type, loop_type)) {
            Fail(path + ".loop_var", "capability=typed_loop_binding; loop binder must exactly match initial state");
        }
        if (!TypeEqual(result_type, initial_type)) {
            Fail(path, "capability=exact_loop_state_type; While result must exactly match initial state");
        }
        const Device state_device = Placement(while_node->initial_state,
                                               path + ".initial_state");
        if (Placement(Expr(ObjectRef(while_node->loop_var)), path + ".loop_var") != state_device ||
            Placement(while_node->body, path + ".body") != state_device ||
            Placement(expr, path) != state_device) {
            Fail(path, "capability=exact_loop_state_placement; While state must retain one exact device placement");
        }
        if (Placement(while_node->condition, path + ".condition") != Device::CPU()) {
            Fail(path + ".condition", "capability=cpu_loop_condition_placement; While condition must be placed on CPU:0");
        }
        const Leaves initial = ResolveAtomic(while_node->initial_state, parent,
                                             environment, path + ".initial_state");
        const Leaves results = AddLeaves(
            expr, result_type, internal::LogicalValueOrigin::kLoopCarried, path);
        if (initial.empty() || initial.size() != results.size()) {
            Fail(path, "While state flattening does not match its result type");
        }
        internal::ControlTask task;
        task.kind = internal::ControlTaskKind::kLoop;
        task.outputs = results;
        task.source_locator = path;
        AddTask(parent, std::move(task));
        const std::size_t parent_task_index = Region(parent).tasks.size() - 1;

        const Leaves arguments = AddLeaves(Expr(ObjectRef(while_node->loop_var)),
                                           loop_type,
                                           internal::LogicalValueOrigin::kLoopCarried,
                                           path + ".loop_var");
        const internal::RegionId condition_region = NewRegion(path + ".condition");
        Env condition_environment = environment;
        condition_environment[while_node->loop_var.get()] = arguments;
        const Leaves condition = LowerTerminal(while_node->condition, condition_region,
                                               &condition_environment, path + ".condition");
        if (condition.size() != 1 ||
            !internal::IsCpuScalarBool(Value(condition.front()))) {
            Fail(path + ".condition", "requires a CPU scalar bool condition");
        }
        Region(condition_region).live_outs = condition;
        for (const internal::ValueId argument : arguments) MarkRead(condition_region, argument);

        const internal::RegionId body_region = NewRegion(path + ".body");
        Env body_environment = environment;
        body_environment[while_node->loop_var.get()] = arguments;
        const Leaves backedges = LowerTerminal(while_node->body, body_region,
                                               &body_environment, path + ".body");
        if (backedges.size() != results.size()) {
            Fail(path + ".body", "While body flattening does not match state type");
        }
        Region(body_region).live_outs = backedges;
        for (const internal::ValueId argument : arguments) MarkRead(body_region, argument);

        internal::ControlTask& loop = Region(parent).tasks.at(parent_task_index);
        loop.loop.condition_region = condition_region;
        loop.loop.body_region = body_region;
        loop.loop.condition_value = condition.front();
        loop.loop.max_trip_count = while_node->max_trip_count;
        for (std::size_t i = 0; i < results.size(); ++i) {
            loop.loop.carried.push_back({results[i], initial[i], arguments[i], backedges[i]});
        }
        const auto append_captures = [this, parent, &loop, &arguments](
                                         const internal::ControlRegion& child) {
            for (const internal::ValueId capture : child.live_ins) {
                if (std::find(arguments.begin(), arguments.end(), capture) != arguments.end()) continue;
                MarkRead(parent, capture);
                if (std::find(loop.inputs.begin(), loop.inputs.end(), capture) == loop.inputs.end()) {
                    loop.inputs.push_back(capture);
                }
            }
        };
        for (const internal::ValueId value : initial) loop.inputs.push_back(value);
        append_captures(Region(condition_region));
        append_captures(Region(body_region));
        loop.effect.reads = loop.inputs;
        loop.dependencies = Dependencies(parent, loop.inputs);
        return results;
    }

    Leaves LowerIf(const Expr& expr, const IfNode* if_node, internal::RegionId parent,
                   const Env& environment, const std::string& path) {
        const Leaves predicate = ResolveAtomic(if_node->cond, parent, environment,
                                               path + ".cond");
        if (predicate.size() != 1 || !internal::IsCpuScalarBool(Value(predicate.front()))) {
            Fail(path + ".cond", "capability=scalar_bool_if_predicate; branch predicate must be a CPU scalar bool");
        }
        const Type result_type = RequireCheckedType(expr, path);
        const Type then_type = RequireCheckedType(if_node->true_branch,
                                                  path + ".true_branch");
        const Type else_type = RequireCheckedType(if_node->false_branch,
                                                  path + ".false_branch");
        if (!TypeEqual(result_type, then_type) || !TypeEqual(then_type, else_type)) {
            Fail(path, "capability=exact_if_branch_type; If branches and result must exactly match");
        }
        const Leaves results = AddLeaves(
            expr, result_type, internal::LogicalValueOrigin::kPhi, path);
        if (results.empty()) Fail(path, "If must produce at least one tensor leaf");

        internal::ControlTask task;
        task.kind = internal::ControlTaskKind::kBranch;
        task.outputs = results;
        task.source_locator = path;
        AddTask(parent, std::move(task));
        const std::size_t parent_task_index = Region(parent).tasks.size() - 1;

        const internal::RegionId then_region = NewRegion(path + ".true_branch");
        Env then_environment = environment;
        const Leaves then_values = LowerTerminal(if_node->true_branch, then_region,
                                                 &then_environment, path + ".true_branch");
        Region(then_region).live_outs = then_values;
        const internal::RegionId else_region = NewRegion(path + ".false_branch");
        Env else_environment = environment;
        const Leaves else_values = LowerTerminal(if_node->false_branch, else_region,
                                                 &else_environment, path + ".false_branch");
        Region(else_region).live_outs = else_values;
        if (then_values.size() != results.size() || else_values.size() != results.size()) {
            Fail(path, "If branch flattening does not match the result type");
        }

        internal::ControlTask& branch_task = Region(parent).tasks.at(parent_task_index);
        branch_task.branch.predicate = predicate.front();
        branch_task.branch.then_region = then_region;
        branch_task.branch.else_region = else_region;
        for (std::size_t i = 0; i < results.size(); ++i) {
            branch_task.branch.phis.push_back({results[i], then_values[i], else_values[i]});
        }
        branch_task.inputs.push_back(predicate.front());
        const auto append_captures = [this, parent, &branch_task](const internal::ControlRegion& child) {
            for (const internal::ValueId capture : child.live_ins) {
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

    Leaves LowerValue(const Expr& expr, internal::RegionId region, const Env& environment,
                      const std::string& path) {
        if (expr.As<VarNode>() || expr.As<ConstantNode>()) {
            return ResolveAtomic(expr, region, environment, path);
        }
        if (const auto* call = expr.As<CallNode>()) return LowerCall(expr, call, region, environment, path);
        if (const auto* if_node = expr.As<IfNode>()) return LowerIf(expr, if_node, region, environment, path);
        if (const auto* while_node = expr.As<WhileNode>()) return LowerWhile(expr, while_node, region, environment, path);
        if (const auto* tuple = expr.As<TupleNode>()) {
            (void)RequireCheckedType(expr, path);
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
        if (expr.As<FunctionNode>()) {
            Fail(path, "capability=first_order_relay; function values are unsupported");
        }
        Fail(path, "capability=supported_static_relay_node; encountered unsupported non-ANF Relay value");
    }

    Leaves LowerTerminal(const Expr& expr, internal::RegionId region, Env* environment,
                         const std::string& path) {
        if (const auto* let = expr.As<LetNode>()) {
            if (!let->var.defined()) {
                Fail(path + ".var", "capability=lexical_let_binding; Let binder is undefined");
            }
            const Type binder_type = RequireCheckedType(
                Expr(ObjectRef(let->var)), path + ".var");
            const Type value_type = RequireCheckedType(let->value, path + ".value");
            if (!TypeEqual(binder_type, value_type)) {
                Fail(path + ".var", "capability=typed_let_binding; Let binder and value differ");
            }
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
        for (internal::ControlRegion& region : plan_.regions) {
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
