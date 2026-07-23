/*! \file src/compiler/control_flow/executable_capability.cc
 * \brief Verification for the static-exact Relay executable subset.
 */

#include "../internal/executable_capability.h"

#include <stdexcept>
#include <string>
#include <unordered_map>

#include "kxc/relay/op.h"

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
    if (expr.As<LetNode>()) return "Let";
    if (expr.As<relay::OpNode>()) return "Op";
    return "Unknown";
}

bool IsOrdinaryOperator(relay::OperatorLoweringKind kind) {
    return kind == relay::OperatorLoweringKind::kSingleTE ||
           kind == relay::OperatorLoweringKind::kMultiTE;
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

ExecutableCapabilityOptions StaticDataflowExecutableCapabilities() {
    return ExecutableCapabilityOptions{};
}

void VerifyExecutableCapability(const Function& function,
                                const ExecutableCapabilityOptions& options) {
    if (options.version != ExecutableCapabilityOptions::kVersion) {
        throw std::invalid_argument(
            "Executable capability error: path=options.version; node=Options; "
            "required capability=static_options_v1; detail=unsupported options version");
    }
    CapabilityVerifier(options).Verify(function);
}

}  // namespace kxc::api::internal
