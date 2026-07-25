/*! \file src/compiler/control_flow/executable_capability.cc
 * \brief Verification for the static-exact Relay executable subset.
 */

#include "../internal/executable_capability.h"
#include "../internal/resolved_relay_call.h"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace kxc::api::internal {
namespace {

std::string NodeKind(const Expr& expr) {
    if (!expr.defined()) return "Undefined";
    if (expr.As<VarNode>()) return "Var";
    if (expr.As<ConstantNode>()) return "Constant";
    if (expr.As<CallNode>()) return "Call";
    if (expr.As<FunctionNode>()) return "Function";
    if (expr.As<TupleNode>()) return "Tuple";
    if (expr.As<TupleGetItemNode>()) return "TupleGetItem";
    if (expr.As<IfNode>()) return "If";
    if (expr.As<WhileNode>()) return "While";
    if (expr.As<LetNode>()) return "Let";
    if (expr.As<relay::OpNode>()) return "Op";
    return "Unknown";
}

class CapabilityFailure final : public std::invalid_argument {
public:
    explicit CapabilityFailure(ExecutableCapabilityIssue issue)
        : std::invalid_argument(issue.detail), issue(std::move(issue)) {}

    ExecutableCapabilityIssue issue;
};

class CapabilityVerifier {
public:
    explicit CapabilityVerifier(const ExecutableCapabilityOptions& options)
        : options_(options) {}

    void Verify(const Function& function) {
        if (!function.defined() || !function->body.defined()) {
            Fail("function", "Function", "defined_function",
                 "Function is undefined or has no body");
        }
        RequireChecked(Expr(ObjectRef(function)), "function");
        if (options_.require_checked_types) {
            VerifyType(function.checked_type(), "function.checked_type");
        }
        for (size_t i = 0; i < function->params.size(); ++i) {
            const Var& parameter = function->params[i];
            const std::string path = "function.params[" + std::to_string(i) + "]";
            if (!parameter.defined() || !parameter->type_annotation.defined()) {
                Fail(path, "Var", "typed_parameter", "parameter annotation is missing");
            }
            RequireChecked(Expr(ObjectRef(parameter)), path);
            VerifyType(parameter->type_annotation, path + ".type_annotation");
            if (options_.require_checked_types) {
                VerifyType(parameter.checked_type(), path + ".checked_type");
                if (!TypeEqual(parameter->type_annotation, parameter.checked_type())) {
                    Fail(path, "Var", "typed_parameter",
                         "annotation and checked_type differ");
                }
            }
            if (!parameter->type_annotation.As<TensorTypeNode>() &&
                !options_.allow_tuple_parameters) {
                Fail(path, "Var", "tensor_parameter",
                     "static ValueGraph parameters must be TensorType");
            }
            bindings_[parameter.get()] += 1;
        }
        Visit(function->body, "function.body");
        if (options_.require_checked_types &&
            !TypeEqual(function.checked_type(), function->body.checked_type())) {
            Fail("function", "Function", "typed_function_result",
                 "Function checked_type differs from its body");
        }
    }

private:
    const ExecutableCapabilityOptions& options_;
    std::unordered_map<const Object*, size_t> bindings_;

    [[noreturn]] void Fail(const std::string& path, const std::string& kind,
                           const std::string& capability,
                           const std::string& detail) const {
        throw CapabilityFailure({path, kind, capability, detail});
    }

    void RequireChecked(const Expr& expr, const std::string& path) const {
        if (!expr.defined()) {
            Fail(path, "Undefined", "defined_typed_relay", "expression is undefined");
        }
        if (options_.require_checked_types && !expr.checked_type().defined()) {
            Fail(path, NodeKind(expr), "defined_typed_relay", "checked_type is missing");
        }
        const auto* relay_node = dynamic_cast<const RelayNode*>(expr.get());
        if (!relay_node || !relay_node->virtual_device_.defined()) return;
        const Device actual = relay_node->virtual_device_->device;
        if (!actual.defined() ||
            (actual.device_type() != kCPU && actual.device_type() != kCUDA)) {
            Fail(path, NodeKind(expr), "explicit_execution_device",
                 "VirtualDevice must define a CPU or CUDA device");
        }
        if (options_.execution_device.defined()) {
            if (actual != options_.execution_device) {
                Fail(path, NodeKind(expr), "matching_execution_device",
                     "Relay placement differs from the requested execution device");
            }
        } else if (!options_.allow_device_regions) {
            Fail(path, NodeKind(expr), "explicit_execution_device",
                 "explicit Relay placement requires an execution device");
        }
    }

    void VerifyType(const Type& type, const std::string& path) const {
        if (!type.defined()) {
            Fail(path, "Type", "defined_typed_relay", "type is undefined");
        }
        if (const auto* tensor = type.As<TensorTypeNode>()) {
            if (tensor->dtype.empty()) {
                Fail(path, "TensorType", "tensor_dtype", "tensor dtype must be explicit");
            }
            for (size_t i = 0; i < tensor->shape.size(); ++i) {
                if (tensor->shape[i] < 0) {
                    Fail(path + ".shape[" + std::to_string(i) + "]", "TensorType",
                         "static_exact_shape", "dimension is dynamic");
                }
            }
            return;
        }
        if (const auto* tuple = type.As<TupleTypeNode>()) {
            for (size_t i = 0; i < tuple->fields.size(); ++i) {
                VerifyType(tuple->fields[i], path + ".fields[" + std::to_string(i) + "]");
            }
            return;
        }
        Fail(path, "Type", "static_tensor_or_tuple_type", "unsupported Relay type");
    }

    void VerifyCallContract(const Expr& expr, const CallNode* call,
                            const std::string& path) const {
        (void)call;
        try {
            (void)ResolveRelayCall(
                expr,
                options_.allow_nested_tuple_call_outputs
                    ? OperatorCapabilityPolicy::StructuredControl(
                          options_.require_checked_types)
                    : OperatorCapabilityPolicy::StaticDataflow(
                          options_.require_checked_types),
                path);
        } catch (const RelayCallResolutionError& error) {
            Fail(error.issue().path, "Call", error.issue().capability,
                 error.issue().detail);
        }
    }

    void VerifyBoundVar(const Expr& expr, const VarNode* var,
                        const std::string& path) const {
        if (bindings_.count(expr.get()) == 0) {
            Fail(path, "Var", "lexically_bound_var",
                 "free or unbound variables include '" +
                     var->vid->name_hint + "'");
        }
    }

    Device Placement(const Expr& expr) const {
        const auto* relay_node = dynamic_cast<const RelayNode*>(expr.get());
        if (!relay_node || !relay_node->virtual_device_.defined()) return Device::CPU();
        return relay_node->virtual_device_->device;
    }

    void VerifyLoopPlacement(const Expr& result, const WhileNode* loop,
                             const std::string& path) const {
        const Device state_device = Placement(loop->initial_state);
        if (Placement(Expr(ObjectRef(loop->loop_var))) != state_device ||
            Placement(loop->body) != state_device || Placement(result) != state_device) {
            Fail(path, "While", "exact_loop_state_placement",
                 "While initial state, binder, body, and result must have one exact device placement");
        }
        if (Placement(loop->condition) != Device::CPU()) {
            Fail(path + ".condition", NodeKind(loop->condition),
                 "cpu_loop_condition_placement", "While condition must be placed on CPU:0");
        }
    }

    void Visit(const Expr& expr, const std::string& path) {
        RequireChecked(expr, path);
        if (options_.require_checked_types) VerifyType(expr.checked_type(), path + ".checked_type");
        if (const auto* var = expr.As<VarNode>()) {
            VerifyBoundVar(expr, var, path);
            return;
        }
        if (const auto* constant = expr.As<ConstantNode>()) {
            if (!constant->data.defined()) {
                Fail(path, "Constant", "constant_payload", "constant payload is undefined");
            }
            return;
        }
        if (const auto* call = expr.As<CallNode>()) {
            VerifyCallContract(expr, call, path);
            for (size_t i = 0; i < call->args.size(); ++i) {
                Visit(call->args[i], path + ".args[" + std::to_string(i) + "]");
            }
            return;
        }
        if (expr.As<FunctionNode>()) {
            Fail(path, "Function", "first_order_relay",
                 "Function values, closures, and recursion are unsupported");
        }
        if (const auto* tuple = expr.As<TupleNode>()) {
            for (size_t i = 0; i < tuple->fields.size(); ++i) {
                Visit(tuple->fields[i], path + ".fields[" + std::to_string(i) + "]");
            }
            return;
        }
        if (const auto* get_item = expr.As<TupleGetItemNode>()) {
            Visit(get_item->tuple, path + ".tuple");
            if (get_item->index < 0) {
                Fail(path, "TupleGetItem", "well_typed_tuple_get_item",
                     "tuple index is outside the checked tuple type");
            }
            if (options_.require_checked_types) {
                const auto* tuple_type = get_item->tuple.checked_type().As<TupleTypeNode>();
                if (!tuple_type || static_cast<size_t>(get_item->index) >= tuple_type->fields.size() ||
                    !TypeEqual(expr.checked_type(),
                               tuple_type->fields[static_cast<size_t>(get_item->index)])) {
                    Fail(path, "TupleGetItem", "well_typed_tuple_get_item",
                         "tuple index is outside the checked tuple type");
                }
            }
            return;
        }
        if (const auto* if_node = expr.As<IfNode>()) {
            if (!options_.allow_if) {
                Fail(path, "If", "if", "static-dataflow executable does not enable If");
            }
            Visit(if_node->cond, path + ".cond");
            Visit(if_node->true_branch, path + ".true_branch");
            Visit(if_node->false_branch, path + ".false_branch");
            const auto* predicate = if_node->cond.checked_type().As<TensorTypeNode>();
            if (!predicate || predicate->dtype != "bool" || !predicate->shape.empty()) {
                Fail(path + ".cond", NodeKind(if_node->cond), "scalar_bool_if_predicate",
                     "If predicate must have scalar bool TensorType");
            }
            if (!TypeEqual(if_node->true_branch.checked_type(), if_node->false_branch.checked_type()) ||
                !TypeEqual(expr.checked_type(), if_node->true_branch.checked_type())) {
                Fail(path, "If", "exact_if_branch_type",
                     "If branches and result must have exactly the same type");
            }
            return;
        }
        if (const auto* loop = expr.As<WhileNode>()) {
            if (!options_.allow_while) {
                Fail(path, "While", "control_flow.loop",
                     "static-dataflow executable does not enable While");
            }
            if (loop->max_trip_count < 0 || !loop->loop_var.defined()) {
                Fail(path, "While", "bounded_loop",
                     "While requires a defined binder and non-negative max_trip_count");
            }
            VerifyLoopPlacement(expr, loop, path);
            Visit(loop->initial_state, path + ".initial_state");
            RequireChecked(Expr(ObjectRef(loop->loop_var)), path + ".loop_var");
            VerifyType(loop->loop_var.checked_type(), path + ".loop_var.checked_type");
            if (!TypeEqual(loop->initial_state.checked_type(), loop->loop_var.checked_type())) {
                Fail(path + ".loop_var", "Var", "typed_loop_binding",
                     "loop binder must exactly match initial state");
            }
            bindings_[loop->loop_var.get()] += 1;
            Visit(loop->condition, path + ".condition");
            Visit(loop->body, path + ".body");
            auto binding = bindings_.find(loop->loop_var.get());
            if (--binding->second == 0) bindings_.erase(binding);
            const auto* predicate = loop->condition.checked_type().As<TensorTypeNode>();
            if (!predicate || predicate->dtype != "bool" || !predicate->shape.empty()) {
                Fail(path + ".condition", NodeKind(loop->condition),
                     "scalar_bool_loop_predicate", "While condition must be a scalar bool TensorType");
            }
            if (!TypeEqual(loop->initial_state.checked_type(), loop->body.checked_type()) ||
                !TypeEqual(expr.checked_type(), loop->initial_state.checked_type())) {
                Fail(path, "While", "exact_loop_state_type",
                     "initial state, body, and result must exactly match");
            }
            return;
        }
        if (const auto* let = expr.As<LetNode>()) {
            Visit(let->value, path + ".value");
            if (!let->var.defined()) {
                Fail(path + ".var", "Var", "lexical_let_binding", "Let binder is undefined");
            }
            RequireChecked(Expr(ObjectRef(let->var)), path + ".var");
            const Type binder_type = let->var.checked_type().defined()
                ? let->var.checked_type() : let->var->type_annotation;
            VerifyType(binder_type, path + ".var.type");
            if (options_.require_checked_types && let->var->type_annotation.defined() &&
                !TypeEqual(let->var->type_annotation, let->var.checked_type())) {
                Fail(path + ".var", "Var", "typed_let_binding",
                     "annotation and checked_type differ");
            }
            if (options_.require_checked_types &&
                !TypeEqual(let->var.checked_type(), let->value.checked_type())) {
                Fail(path + ".var", "Var", "typed_let_binding",
                     "binder checked_type differs from value type");
            }
            bindings_[let->var.get()] += 1;
            Visit(let->body, path + ".body");
            auto binding = bindings_.find(let->var.get());
            if (--binding->second == 0) bindings_.erase(binding);
            return;
        }
        Fail(path, NodeKind(expr), "supported_static_relay_node", "unsupported Relay node");
    }
};

std::string IssueDiagnostic(const ExecutableCapabilityIssue& issue) {
    return "Executable capability error: path=" + issue.path + "; node=" +
           issue.node_kind + "; required capability=" + issue.capability +
           "; detail=" + issue.detail;
}

}  // namespace

ExecutableCapabilityOptions StaticDataflowExecutableCapabilities(Device execution_device) {
    ExecutableCapabilityOptions options;
    options.execution_device = std::move(execution_device);
    return options;
}

std::vector<ExecutableCapabilityIssue> CollectExecutableCapabilityIssues(
    const Function& function, const ExecutableCapabilityOptions& options) {
    if (options.version != ExecutableCapabilityOptions::kVersion) {
        return {{"options.version", "Options", "static_options_v1",
                 "unsupported options version"}};
    }
    if (options.execution_device.defined() &&
        options.execution_device.device_type() != kCPU &&
        options.execution_device.device_type() != kCUDA) {
        return {{"options.execution_device", "Options", "cpu_or_cuda_device",
                 "unsupported execution device"}};
    }
    try {
        CapabilityVerifier(options).Verify(function);
    } catch (const CapabilityFailure& failure) {
        return {failure.issue};
    }
    return {};
}

void VerifyExecutableCapability(const Function& function,
                                const ExecutableCapabilityOptions& options) {
    const std::vector<ExecutableCapabilityIssue> issues =
        CollectExecutableCapabilityIssues(function, options);
    if (!issues.empty()) throw std::invalid_argument(IssueDiagnostic(issues.front()));
}

}  // namespace kxc::api::internal
