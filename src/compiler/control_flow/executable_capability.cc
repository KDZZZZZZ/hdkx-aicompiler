/*! \file src/compiler/control_flow/executable_capability.cc
 * \brief Verification for the static-exact Relay executable subset.
 */

#include "../internal/executable_capability.h"

#include <any>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include "kxc/relay/op.h"
#include "kxc/relay/op_attr_types.h"

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

bool IsOrdinaryOperator(relay::OperatorLoweringKind kind) {
    return kind == relay::OperatorLoweringKind::kSingleTE ||
           kind == relay::OperatorLoweringKind::kMultiTE;
}

size_t TensorLeafCount(const Type& type) {
    if (type.As<TensorTypeNode>()) return 1;
    if (const auto* tuple = type.As<TupleTypeNode>()) {
        size_t count = 0;
        for (const Type& field : tuple->fields) count += TensorLeafCount(field);
        return count;
    }
    return 0;
}

bool IsFlatTensorTuple(const Type& type) {
    const auto* tuple = type.As<TupleTypeNode>();
    if (!tuple) return false;
    for (const Type& field : tuple->fields) {
        if (!field.As<TensorTypeNode>()) return false;
    }
    return true;
}

class CapabilityVerifier {
public:
    explicit CapabilityVerifier(const ExecutableCapabilityOptions& options)
        : options_(options) {}

    void Verify(const Function& function) {
        if (!function.defined()) {
            Fail("function", "Function", "defined_function", "Function is undefined");
        }
        RequireChecked(Expr(ObjectRef(function)), "function");
        VerifyType(function.checked_type(), "function.checked_type");
        for (size_t i = 0; i < function->params.size(); ++i) {
            const Var& parameter = function->params[i];
            const std::string path = "function.params[" + std::to_string(i) + "]";
            if (!parameter.defined() || !parameter->type_annotation.defined()) {
                Fail(path, "Var", "typed_parameter", "parameter annotation is missing");
            }
            RequireChecked(Expr(ObjectRef(parameter)), path);
            VerifyType(parameter->type_annotation, path + ".type_annotation");
            VerifyType(parameter.checked_type(), path + ".checked_type");
            if (!parameter->type_annotation.As<TensorTypeNode>() &&
                !options_.allow_tuple_parameters) {
                Fail(path, "Var", "tensor_parameter",
                     "static ValueGraph parameters must be TensorType");
            }
            if (!TypeEqual(parameter->type_annotation, parameter.checked_type())) {
                Fail(path, "Var", "typed_parameter", "annotation and checked_type differ");
            }
            bindings_[parameter.get()] += 1;
        }
        Visit(function->body, "function.body");
        if (!TypeEqual(function.checked_type(), function->body.checked_type())) {
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
        throw std::invalid_argument("Executable capability error: path=" + path +
                                    "; node=" + kind + "; required capability=" +
                                    capability + "; detail=" + detail);
    }

    void RequireChecked(const Expr& expr, const std::string& path) const {
        if (!expr.defined()) {
            Fail(path, "Undefined", "defined_typed_relay", "expression is undefined");
        }
        if (!expr.checked_type().defined()) {
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
        const auto* op = call->op.As<relay::OpNode>();
        if (!op || !op->has_spec) {
            Fail(path, "Call", "registered_ordinary_op_call",
                 "Call target is not a specified Op");
        }
        const relay::Op* registered = relay::Op::TryGet(op->name);
        if (!registered || registered->get() != call->op.get()) {
            Fail(path, "Call", "registered_ordinary_op_call",
                 "Call target is not the registered Op instance");
        }
        try {
            relay::ValidateOperatorSpec(op->spec);
        } catch (const std::exception& error) {
            Fail(path, "Call", "registered_ordinary_op_call", error.what());
        }
        if (!IsOrdinaryOperator(op->spec.lowering_kind)) {
            Fail(path, "Call", "registered_ordinary_op_call",
                 "operator is not an ordinary static-dataflow operation");
        }
        if (op->spec.effect != relay::OperatorEffectKind::kPure ||
            !op->spec.deterministic || op->spec.alias_contract != "none") {
            Fail(path, "Call", "pure_deterministic_no_alias",
                 "operator is not a pure deterministic non-aliasing kernel");
        }
        const Type output_type = expr.checked_type();
        const size_t output_leaves = TensorLeafCount(output_type);
        if (op->spec.lowering_kind ==
                relay::OperatorLoweringKind::kSingleTE &&
            !output_type.As<TensorTypeNode>()) {
            Fail(path, "Call", "single_tensor_output",
                 "single-output lowering requires TensorType");
        }
        if (op->spec.lowering_kind ==
                relay::OperatorLoweringKind::kMultiTE &&
            (!output_type.As<TupleTypeNode>() ||
             (!options_.allow_nested_tuple_call_outputs &&
              !IsFlatTensorTuple(output_type)))) {
            Fail(path, "Call", "flat_multi_tensor_output",
                 "static ValueGraph multi-output Calls require a flat tensor tuple");
        }
        if (output_leaves == 0 ||
            (op->spec.output_arity >= 0 &&
             static_cast<size_t>(op->spec.output_arity) != output_leaves)) {
            Fail(path, "Call", "operator_output_arity",
                 "Call output leaves differ from OperatorSpec");
        }
        const size_t actual_arity = call->args.size();
        if ((op->spec.input_arity.num_inputs >= 0 &&
             actual_arity !=
                 static_cast<size_t>(op->spec.input_arity.num_inputs)) ||
            (op->spec.input_arity.num_inputs < 0 &&
             (actual_arity <
                  static_cast<size_t>(op->spec.input_arity.min_inputs) ||
              actual_arity >
                  static_cast<size_t>(op->spec.input_arity.max_inputs)))) {
            Fail(path, "Call", "operator_input_arity",
                 "Call input arity differs from OperatorSpec");
        }
        if (call->attrs.defined()) {
            if (op->spec.attrs_type_key.empty()) {
                Fail(path, "Call", "operator_attrs_schema",
                     "Call defines attrs outside its OperatorSpec schema");
            }
            const std::string actual(call->attrs.get()->GetTypeKey());
            if (actual != op->spec.attrs_type_key &&
                actual != op->spec.attrs_type_key + "Node") {
                Fail(path, "Call", "operator_attrs_schema",
                     "Call attrs type differs from OperatorSpec");
            }
        }
        const auto relation = op->attrs.find(op->spec.type_relation_key);
        const auto lowering = op->attrs.find(op->spec.lowering_key);
        if (relation == op->attrs.end() || lowering == op->attrs.end()) {
            Fail(path, "Call", "operator_implementation_binding",
                 "OperatorSpec implementation binding is missing");
        }
        const auto* infer =
            std::any_cast<relay::FInferType>(&relation->second);
        if (!infer || !*infer) {
            Fail(path, "Call", "operator_implementation_binding",
                 "type relation binding has the wrong type or is empty");
        }
        if (op->spec.lowering_kind ==
                relay::OperatorLoweringKind::kSingleTE) {
            if (op->spec.lowering_key != "FRelayToTE") {
                Fail(path, "Call", "operator_implementation_binding",
                     "single-output lowering must use FRelayToTE");
            }
            const auto* lower =
                std::any_cast<relay::FRelayToTE>(&lowering->second);
            if (!lower || !*lower) {
                Fail(path, "Call", "operator_implementation_binding",
                     "single-output lowering binding has the wrong type or is empty");
            }
        }
        if (op->spec.lowering_kind ==
                relay::OperatorLoweringKind::kMultiTE) {
            if (op->spec.lowering_key != "FRelayToTEMulti") {
                Fail(path, "Call", "operator_implementation_binding",
                     "multi-output lowering must use FRelayToTEMulti");
            }
            const auto* lower =
                std::any_cast<relay::FRelayToTEMulti>(&lowering->second);
            if (!lower || !*lower) {
                Fail(path, "Call", "operator_implementation_binding",
                     "multi-output lowering binding has the wrong type or is empty");
            }
        }
        Array<Type> input_types;
        for (const Expr& argument : call->args) {
            input_types.push_back(argument.checked_type());
        }
        const relay::Attrs attrs =
            call->attrs.defined() ? relay::Attrs(call->attrs) : relay::Attrs();
        Type inferred;
        try {
            inferred = (*infer)(attrs, input_types);
        } catch (const std::exception& error) {
            Fail(path, "Call", "operator_type_relation", error.what());
        }
        if (!TypeEqual(inferred, expr.checked_type())) {
            Fail(path, "Call", "operator_type_relation",
                 "Call checked_type is stale for OperatorSpec");
        }
    }

    void VerifyBoundVar(const Expr& expr, const VarNode* var,
                        const std::string& path) const {
        if (bindings_.count(expr.get()) == 0) {
            Fail(path, "Var", "lexically_bound_var", "free variable '" +
                var->vid->name_hint + "'");
        }
    }

    void Visit(const Expr& expr, const std::string& path) {
        RequireChecked(expr, path);
        VerifyType(expr.checked_type(), path + ".checked_type");
        if (const auto* var = expr.As<VarNode>()) {
            VerifyBoundVar(expr, var, path);
            return;
        }
        if (expr.As<ConstantNode>()) return;
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
            const auto* tuple_type = get_item->tuple.checked_type().As<TupleTypeNode>();
            if (!tuple_type || get_item->index < 0 ||
                static_cast<size_t>(get_item->index) >= tuple_type->fields.size()) {
                Fail(path, "TupleGetItem", "well_typed_tuple_get_item",
                     "tuple index is outside the checked tuple type");
            }
            if (!TypeEqual(expr.checked_type(),
                           tuple_type->fields[static_cast<size_t>(get_item->index)])) {
                Fail(path, "TupleGetItem", "well_typed_tuple_get_item",
                     "checked_type differs from selected tuple field");
            }
            return;
        }
        if (const auto* if_node = expr.As<IfNode>()) {
            if (!options_.allow_if) {
                Fail(path, "If", "if", "static-dataflow executable does not enable If");
            }
            Visit(if_node->cond, path + ".cond");
            const auto* predicate = if_node->cond.checked_type().As<TensorTypeNode>();
            if (!predicate || predicate->dtype != "bool" || !predicate->shape.empty()) {
                Fail(path + ".cond", NodeKind(if_node->cond), "scalar_bool_if_predicate",
                     "If predicate must have scalar bool TensorType");
            }
            Visit(if_node->true_branch, path + ".true_branch");
            Visit(if_node->false_branch, path + ".false_branch");
            if (!TypeEqual(if_node->true_branch.checked_type(),
                           if_node->false_branch.checked_type()) ||
                !TypeEqual(expr.checked_type(), if_node->true_branch.checked_type())) {
                Fail(path, "If", "exact_if_branch_type",
                     "If branches and result must have exactly the same type");
            }
            return;
        }
        if (const auto* while_node = expr.As<WhileNode>()) {
            if (!options_.allow_while) {
                Fail(path, "While", "control_flow.loop",
                     "static-dataflow executable does not enable While");
            }
            if (while_node->max_trip_count < 0 || !while_node->loop_var.defined()) {
                Fail(path, "While", "bounded_loop",
                     "While requires a defined binder and non-negative max_trip_count");
            }
            Visit(while_node->initial_state, path + ".initial_state");
            RequireChecked(Expr(ObjectRef(while_node->loop_var)), path + ".loop_var");
            VerifyType(while_node->loop_var.checked_type(), path + ".loop_var.checked_type");
            if (!TypeEqual(while_node->initial_state.checked_type(),
                           while_node->loop_var.checked_type())) {
                Fail(path + ".loop_var", "Var", "typed_loop_binding",
                     "loop binder must exactly match initial state");
            }
            bindings_[while_node->loop_var.get()] += 1;
            Visit(while_node->condition, path + ".condition");
            Visit(while_node->body, path + ".body");
            auto binding = bindings_.find(while_node->loop_var.get());
            if (--binding->second == 0) bindings_.erase(binding);
            const auto* predicate = while_node->condition.checked_type().As<TensorTypeNode>();
            if (!predicate || predicate->dtype != "bool" || !predicate->shape.empty()) {
                Fail(path + ".condition", NodeKind(while_node->condition),
                     "scalar_bool_loop_predicate", "While condition must be CPU scalar bool");
            }
            if (!TypeEqual(while_node->initial_state.checked_type(),
                           while_node->body.checked_type()) ||
                !TypeEqual(expr.checked_type(), while_node->initial_state.checked_type())) {
                Fail(path, "While", "exact_loop_state_type",
                     "initial state, body, and result must have exactly the same type");
            }
            return;
        }
        if (const auto* let = expr.As<LetNode>()) {
            Visit(let->value, path + ".value");
            if (!let->var.defined()) {
                Fail(path + ".var", "Var", "lexical_let_binding", "Let binder is undefined");
            }
            RequireChecked(Expr(ObjectRef(let->var)), path + ".var");
            VerifyType(let->var.checked_type(), path + ".var.checked_type");
            if (let->var->type_annotation.defined()) {
                VerifyType(let->var->type_annotation, path + ".var.type_annotation");
                if (!TypeEqual(let->var->type_annotation, let->var.checked_type())) {
                    Fail(path + ".var", "Var", "typed_let_binding",
                         "annotation and checked_type differ");
                }
            }
            if (!TypeEqual(let->var.checked_type(), let->value.checked_type())) {
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

}  // namespace

ExecutableCapabilityOptions StaticDataflowExecutableCapabilities(
    Device execution_device) {
    ExecutableCapabilityOptions options;
    options.execution_device = std::move(execution_device);
    return options;
}

void VerifyExecutableCapability(const Function& function,
                                const ExecutableCapabilityOptions& options) {
    if (options.version != ExecutableCapabilityOptions::kVersion) {
        throw std::invalid_argument(
            "Executable capability error: path=options.version; node=Options; "
            "required capability=static_options_v1; detail=unsupported options version");
    }
    if (options.execution_device.defined() &&
        options.execution_device.device_type() != kCPU &&
        options.execution_device.device_type() != kCUDA) {
        throw std::invalid_argument(
            "Executable capability error: path=options.execution_device; "
            "node=Options; required capability=cpu_or_cuda_device; "
            "detail=unsupported execution device");
    }
    CapabilityVerifier(options).Verify(function);
}

}  // namespace kxc::api::internal
