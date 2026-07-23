/*! \file src/compiler/control_flow/relay_control_plan.cc
 * \brief Static-exact Relay-to-ControlPlan preparation lowering.
 */

#include "kxc/compiler/control_flow.h"

#include "../internal/executable_capability.h"

#include <algorithm>
#include <any>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "kxc/relay/op.h"
#include "kxc/relay/op_attr_types.h"
#include "kxc/relay/transforms/infer_type.h"
#include "kxc/relay/transforms/normalize_to_anf.h"

namespace kxc::api {
namespace {

using Leaves = std::vector<runtime::ValueId>;

[[noreturn]] void Fail(const std::string& path, const std::string& detail) {
    throw std::invalid_argument("LowerRelayToControlPlan: path=" + path + "; " + detail);
}

void FlattenTensorTypes(const Type& type, std::vector<Type>* leaves,
                        const std::string& path) {
    if (type.As<TensorTypeNode>()) {
        leaves->push_back(type);
        return;
    }
    if (const auto* tuple = type.As<TupleTypeNode>()) {
        for (std::size_t i = 0; i < tuple->fields.size(); ++i) {
            FlattenTensorTypes(tuple->fields[i], leaves,
                               path + ".fields[" + std::to_string(i) + "]");
        }
        return;
    }
    Fail(path, "requires a static TensorType or TupleType");
}

Device DeviceFor(const Expr& expr, const std::string& path) {
    const VirtualDevice device = Relay(expr).virtual_device();
    if (!device.defined()) return Device::CPU();
    if (!device->device.defined()) {
        Fail(path, "explicit Relay VirtualDevice must define a CPU or CUDA device");
    }
    switch (device->device.device_type()) {
    case kCPU:
    case kCUDA:
        return device->device;
    default:
        Fail(path, "Relay VirtualDevice must be CPU or CUDA");
    }
}

void ValidateInputArity(const relay::OperatorSpec& spec, std::size_t actual,
                        const std::string& path) {
    if (spec.input_arity.num_inputs >= 0) {
        if (actual != static_cast<std::size_t>(spec.input_arity.num_inputs)) {
            Fail(path, "OperatorSpec input arity does not match Call");
        }
        return;
    }
    if (actual < static_cast<std::size_t>(spec.input_arity.min_inputs) ||
        actual > static_cast<std::size_t>(spec.input_arity.max_inputs)) {
        Fail(path, "OperatorSpec variable input arity does not match Call");
    }
}

void ValidateCallAttrs(const relay::OperatorSpec& spec, const CallNode* call,
                       const std::string& path) {
    if (!call->attrs.defined()) return;
    if (spec.attrs_type_key.empty()) {
        Fail(path, "Call attrs are outside the OperatorSpec schema");
    }
    const std::string actual(call->attrs.get()->GetTypeKey());
    if (actual != spec.attrs_type_key && actual != spec.attrs_type_key + "Node") {
        Fail(path, "Call attrs do not match the OperatorSpec schema");
    }
}

bool IsKernelLowering(relay::OperatorLoweringKind kind) {
    return kind == relay::OperatorLoweringKind::kSingleTE ||
           kind == relay::OperatorLoweringKind::kMultiTE;
}

void ValidateOperatorBindings(const relay::OpNode* op, const CallNode* call,
                              const Expr& call_expr, const std::string& path) {
    const auto relation = op->attrs.find(op->spec.type_relation_key);
    const auto lowering = op->attrs.find(op->spec.lowering_key);
    if (relation == op->attrs.end() || lowering == op->attrs.end()) {
        Fail(path, "OperatorSpec implementation binding is missing");
    }
    const auto* infer = std::any_cast<relay::FInferType>(&relation->second);
    if (!infer) Fail(path, "OperatorSpec type relation binding has the wrong type");
    if (op->spec.lowering_kind == relay::OperatorLoweringKind::kSingleTE &&
        !std::any_cast<relay::FRelayToTE>(&lowering->second)) {
        Fail(path, "OperatorSpec single-output lowering binding has the wrong type");
    }
    if (op->spec.lowering_kind == relay::OperatorLoweringKind::kMultiTE &&
        !std::any_cast<relay::FRelayToTEMulti>(&lowering->second)) {
        Fail(path, "OperatorSpec multi-output lowering binding has the wrong type");
    }
    Array<Type> input_types;
    for (const Expr& argument : call->args) input_types.push_back(argument.checked_type());
    const relay::Attrs attrs =
        call->attrs.defined() ? relay::Attrs(call->attrs) : relay::Attrs();
    if (!TypeEqual((*infer)(attrs, input_types), call_expr.checked_type())) {
        Fail(path, "Call checked_type is stale for its OperatorSpec type relation");
    }
}

struct RegionState {
    std::unordered_set<runtime::ValueId> local_values;
    std::unordered_map<runtime::ValueId, runtime::TaskId> local_producers;
    std::unordered_set<runtime::ValueId> live_in_set;
};

class ControlPlanBuilder final {
public:
    explicit ControlPlanBuilder(Function function) : function_(std::move(function)) {}

    runtime::ControlPlan Build() {
        if (!function_.defined()) {
            throw std::invalid_argument("LowerRelayToControlPlan requires a defined Function");
        }
        function_ = relay::InferTypePass(function_);
        function_ = relay::NormalizeToANF(function_);
        relay::VerifyANF(function_);
        internal::ExecutableCapabilityOptions options;
        options.version = internal::ExecutableCapabilityOptions::kVersion;
        options.allow_if = true;
        options.allow_tuple_parameters = true;
        options.allow_nested_tuple_call_outputs = true;
        options.allow_device_regions = true;
        internal::VerifyExecutableCapability(function_, options);

        const runtime::RegionId root = NewRegion("function");
        plan_.entry_region = root;
        Env environment;
        for (std::size_t i = 0; i < function_->params.size(); ++i) {
            const Var& parameter = function_->params[i];
            const std::string path = "function.params[" + std::to_string(i) + "]";
            const Leaves ids = AddLeaves(Expr(ObjectRef(parameter)),
                                         parameter.checked_type(), path);
            environment.emplace(parameter.get(), ids);
            plan_.graph_inputs.insert(plan_.graph_inputs.end(), ids.begin(), ids.end());
        }

        const Leaves outputs = LowerTerminal(function_->body, root, &environment,
                                             "function.body");
        if (outputs.empty()) Fail("function.body", "requires at least one tensor graph output");
        std::unordered_set<runtime::ValueId> output_set;
        for (const runtime::ValueId output : outputs) {
            if (!output_set.insert(output).second) {
                Fail("function.body", "duplicate graph output value is not representable in ControlPlan v1");
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
        return std::move(plan_);
    }

private:
    using Env = std::unordered_map<const Object*, Leaves>;

    Function function_;
    runtime::ControlPlan plan_;
    runtime::ValueId next_value_{0};
    runtime::RegionId next_region_{0};
    runtime::TaskId next_task_{0};
    std::unordered_map<runtime::RegionId, std::size_t> region_index_;
    std::unordered_map<runtime::RegionId, RegionState> region_state_;
    std::unordered_map<const Object*, Leaves> constants_;

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

    Leaves AddLeaves(const Expr& source, const Type& type,
                     const std::string& path) {
        std::vector<Type> leaf_types;
        FlattenTensorTypes(type, &leaf_types, path + ".checked_type");
        const Device device = DeviceFor(source, path);
        Leaves ids;
        ids.reserve(leaf_types.size());
        for (std::size_t i = 0; i < leaf_types.size(); ++i) {
            const auto* tensor = leaf_types[i].As<TensorTypeNode>();
            runtime::ControlValueSpec value;
            value.id = next_value_++;
            value.dtype = tensor->dtype;
            value.shape.assign(tensor->shape.begin(), tensor->shape.end());
            value.device = device;
            value.source_locator = path + ".leaf[" + std::to_string(i) + "]";
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
                ids = AddLeaves(expr, expr.checked_type(), path);
                constants_.emplace(expr.get(), ids);
                plan_.constant_values.insert(plan_.constant_values.end(), ids.begin(), ids.end());
            }
        } else {
            Fail(path, "ANF lowering expected an atomic Var or Constant");
        }
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

    void AddTask(runtime::RegionId region, runtime::ControlTask task) {
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
    }

    Leaves LowerCall(const Expr& expr, const CallNode* call, runtime::RegionId region,
                     const Env& environment, const std::string& path) {
        const auto* op = call->op.As<relay::OpNode>();
        if (!op || !op->has_spec || !relay::Op::TryGet(op->name) ||
            relay::Op::TryGet(op->name)->get() != call->op.get()) {
            Fail(path, "requires a registered OperatorSpec Call");
        }
        relay::ValidateOperatorSpec(op->spec);
        if (op->spec.effect != relay::OperatorEffectKind::kPure ||
            !op->spec.deterministic || op->spec.alias_contract != "none" ||
            !IsKernelLowering(op->spec.lowering_kind)) {
            Fail(path, "OperatorSpec is not a pure deterministic non-aliasing kernel lowering");
        }
        ValidateInputArity(op->spec, call->args.size(), path);
        ValidateCallAttrs(op->spec, call, path);
        ValidateOperatorBindings(op, call, expr, path);

        Leaves arguments;
        for (std::size_t i = 0; i < call->args.size(); ++i) {
            const Leaves leaves = ResolveAtomic(call->args[i], region, environment,
                                                path + ".args[" + std::to_string(i) + "]");
            arguments.insert(arguments.end(), leaves.begin(), leaves.end());
        }
        const Leaves outputs = AddLeaves(expr, expr.checked_type(), path);
        if (outputs.empty() ||
            (op->spec.output_arity >= 0 &&
             static_cast<std::size_t>(op->spec.output_arity) != outputs.size())) {
            Fail(path, "OperatorSpec output arity does not match flattened Call result");
        }
        runtime::ControlTask task;
        task.kind = runtime::ControlTaskKind::kKernel;
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
        task.kernel_ref = "relay.kernel.v1;" + relay::SerializeOperatorSpec(op->spec) +
                          ";attrs=" + relay::SerializeAttrs(relay::Attrs(call->attrs));
        task.source_locator = path;
        AddTask(region, std::move(task));
        return outputs;
    }

    Leaves LowerTupleGetItem(const TupleGetItemNode* get_item, runtime::RegionId region,
                             const Env& environment, const std::string& path) {
        const Leaves tuple = ResolveAtomic(get_item->tuple, region, environment,
                                           path + ".tuple");
        const auto* tuple_type = get_item->tuple.checked_type().As<TupleTypeNode>();
        if (!tuple_type || get_item->index < 0 ||
            static_cast<std::size_t>(get_item->index) >= tuple_type->fields.size()) {
            Fail(path, "TupleGetItem is outside its checked TupleType");
        }
        std::size_t begin = 0;
        for (int i = 0; i < get_item->index; ++i) {
            std::vector<Type> ignored;
            FlattenTensorTypes(tuple_type->fields[static_cast<std::size_t>(i)], &ignored,
                               path + ".tuple.checked_type");
            begin += ignored.size();
        }
        std::vector<Type> selected_types;
        FlattenTensorTypes(tuple_type->fields[static_cast<std::size_t>(get_item->index)],
                           &selected_types, path + ".checked_type");
        if (begin + selected_types.size() > tuple.size()) {
            Fail(path, "TupleGetItem flattening is inconsistent with its checked TupleType");
        }
        return Leaves(tuple.begin() + static_cast<std::ptrdiff_t>(begin),
                      tuple.begin() + static_cast<std::ptrdiff_t>(begin + selected_types.size()));
    }

    Leaves LowerIf(const Expr& expr, const IfNode* if_node, runtime::RegionId parent,
                   const Env& environment, const std::string& path) {
        const Leaves predicate = ResolveAtomic(if_node->cond, parent, environment,
                                               path + ".cond");
        if (predicate.size() != 1) Fail(path + ".cond", "requires one scalar bool predicate leaf");
        const Leaves results = AddLeaves(expr, expr.checked_type(), path);
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
        if (const auto* tuple = expr.As<TupleNode>()) {
            Leaves values;
            for (std::size_t i = 0; i < tuple->fields.size(); ++i) {
                const Leaves field = ResolveAtomic(tuple->fields[i], region, environment,
                                                   path + ".fields[" + std::to_string(i) + "]");
                values.insert(values.end(), field.begin(), field.end());
            }
            return values;
        }
        if (const auto* get_item = expr.As<TupleGetItemNode>()) {
            return LowerTupleGetItem(get_item, region, environment, path);
        }
        Fail(path, "encountered unsupported non-ANF Relay value");
    }

    Leaves LowerTerminal(const Expr& expr, runtime::RegionId region, Env* environment,
                         const std::string& path) {
        if (const auto* let = expr.As<LetNode>()) {
            const Leaves value = LowerValue(let->value, region, *environment, path + ".value");
            const auto existing = environment->find(let->var.get());
            const bool had_existing = existing != environment->end();
            const Leaves saved = had_existing ? existing->second : Leaves{};
            (*environment)[let->var.get()] = value;
            const Leaves result = LowerTerminal(let->body, region, environment, path + ".body");
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

runtime::ControlPlan LowerRelayToControlPlan(Function function) {
    return ControlPlanBuilder(std::move(function)).Build();
}

}  // namespace kxc::api
