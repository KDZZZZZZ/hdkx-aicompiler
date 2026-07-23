/*! \file src/compiler/capability.cc
 * \brief Implements the compiler's fail-closed executable dialect check.
 */

#include "kxc/compiler/capability.h"

#include <algorithm>
#include <any>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

#include "kxc/relay/op.h"
#include "kxc/relay/op_attr_types.h"

namespace kxc::api {
namespace {

class Verifier final {
public:
    explicit Verifier(const CapabilityRequest& request) : request_(request) {
        result_.requested_mode = request.requested_mode;
        result_.pipeline_fingerprint = request.pipeline_fingerprint;
        if (request.target.defined() && request.target.As<TargetNode>()) {
            result_.target_identity =
                request.target->kind + ":" +
                std::to_string(static_cast<int>(request.target->device_type)) +
                ":" + std::to_string(request.target->device_id);
        } else {
            result_.target_identity = "undefined";
        }
        result_.normalized_requirements = {
            "relay.static_exact.v1", "relay.registered_operator_calls",
            "relay.tensor_or_flat_tuple_values", "compiler.per_unit_lowering"};
    }

    CapabilityResult Run() {
        VerifyRequest();
        result_.supported = result_.issues.empty();
        if (!result_.supported) {
            result_.diagnostic_locator = result_.issues.front().diagnostic_locator;
        }
        return result_;
    }

private:
    const CapabilityRequest& request_;
    CapabilityResult result_;
    std::unordered_set<const Object*> visited_;
    std::unordered_set<const Object*> bound_vars_;

    void AddIssue(std::string locator, std::string node_kind,
                  std::string capability, std::string detail) {
        if (std::find(result_.missing_capabilities.begin(),
                      result_.missing_capabilities.end(), capability) ==
            result_.missing_capabilities.end()) {
            result_.missing_capabilities.push_back(capability);
        }
        result_.issues.push_back(CapabilityIssue{
            std::move(locator), std::move(node_kind), std::move(capability),
            std::move(detail)});
    }

    void VerifyRequest() {
        const std::string root = request_.graph_locator.empty()
                                     ? "graph"
                                     : request_.graph_locator;
        if (request_.requested_mode != CapabilityMode::kStaticExact) {
            AddIssue(root, "Function", "execution_mode",
                     std::string("requested mode '") +
                         ToString(request_.requested_mode) +
                         "' is not executable; only static_exact is supported");
        }
        if (!request_.target.defined() || !request_.target.As<TargetNode>()) {
            AddIssue(root, "Function", "target", "target is undefined or invalid");
        } else {
            const bool llvm_cpu = request_.target->kind == "llvm" &&
                                  request_.target->device_type == kCPU;
            const bool cuda_gpu = request_.target->kind == "cuda" &&
                                  request_.target->device_type == kCUDA;
            if (!llvm_cpu && !cuda_gpu) {
                AddIssue(root, "Function", "target",
                         "target kind and device type have no executable backend");
            }
        }
        if (!request_.function.defined() ||
            !request_.function.As<FunctionNode>() ||
            !request_.function->body.defined()) {
            AddIssue(root, "Function", "defined_function",
                     "validated relay must have a body");
            return;
        }
        for (size_t i = 0; i < request_.function->params.size(); ++i) {
            const Var& parameter = request_.function->params[i];
            const std::string locator = root + "/param[" +
                                        std::to_string(i) + "]";
            if (!parameter.defined() || !parameter.As<VarNode>()) {
                AddIssue(locator, "Var", "typed_parameter",
                         "function parameter is undefined or invalid");
                continue;
            }
            bound_vars_.insert(parameter.get());
            CheckType(parameter->type_annotation, locator, "Var", true);
        }
        Visit(request_.function->body, root + "/body");
    }

    void CheckType(const Type& type, const std::string& locator,
                   const std::string& node_kind, bool required) {
        if (!type.defined()) {
            if (required) {
                AddIssue(locator, node_kind, "checked_type",
                         "static-exact execution requires a complete type");
            }
            return;
        }
        if (const auto* tensor = type.As<TensorTypeNode>()) {
            if (tensor->dtype.empty()) {
                AddIssue(locator, node_kind, "tensor_dtype",
                         "tensor dtype must be explicit");
            }
            for (size_t i = 0; i < tensor->shape.size(); ++i) {
                if (tensor->shape[i] < 0) {
                    AddIssue(locator + "/dim[" + std::to_string(i) + "]",
                             node_kind, "static_exact_shape",
                             "symbolic or legacy -1 dimensions are not static-exact capability");
                }
            }
            return;
        }
        if (const auto* tuple = type.As<TupleTypeNode>()) {
            for (size_t i = 0; i < tuple->fields.size(); ++i) {
                if (!tuple->fields[i].As<TensorTypeNode>()) {
                    AddIssue(locator + "/field[" + std::to_string(i) + "]",
                             node_kind, "flat_tensor_tuple",
                             "only a flat tuple of tensor leaves is executable");
                } else {
                    CheckType(tuple->fields[i],
                              locator + "/field[" + std::to_string(i) + "]",
                              node_kind, true);
                }
            }
            return;
        }
        AddIssue(locator, node_kind, "tensor_value_type",
                 "only TensorType and flat TupleType values are executable");
    }

    void Visit(const Expr& expr, const std::string& locator) {
        if (!expr.defined()) {
            AddIssue(locator, "Expr", "defined_expression",
                     "expression is undefined");
            return;
        }
        if (!visited_.insert(expr.get()).second) return;
        if (const auto* var = expr.As<VarNode>()) {
            if (bound_vars_.count(expr.get()) == 0) {
                AddIssue(locator, "Var", "bound_variable",
                         "free or unbound variables are not executable");
            }
            CheckType(expr.checked_type().defined() ? expr.checked_type()
                                                    : var->type_annotation,
                      locator, "Var", request_.require_checked_types);
            return;
        }
        if (const auto* constant = expr.As<ConstantNode>()) {
            if (!constant->data.defined()) {
                AddIssue(locator, "Constant", "constant_payload",
                         "constant payload is undefined");
            }
            CheckType(expr.checked_type(), locator, "Constant",
                      request_.require_checked_types);
            return;
        }
        if (const auto* call = expr.As<CallNode>()) {
            VisitCall(expr, call, locator);
            return;
        }
        if (const auto* tuple = expr.As<TupleNode>()) {
            for (size_t i = 0; i < tuple->fields.size(); ++i) {
                Visit(tuple->fields[i], locator + "/field[" +
                                            std::to_string(i) + "]");
            }
            CheckType(expr.checked_type(), locator, "Tuple",
                      request_.require_checked_types);
            return;
        }
        if (const auto* get_item = expr.As<TupleGetItemNode>()) {
            Visit(get_item->tuple, locator + "/tuple");
            if (get_item->index < 0) {
                AddIssue(locator, "TupleGetItem", "tuple_index",
                         "tuple field index must be non-negative");
            }
            CheckType(expr.checked_type(), locator, "TupleGetItem",
                      request_.require_checked_types);
            return;
        }
        if (expr.As<IfNode>()) {
            AddIssue(locator, "If", "control_flow.if",
                     "If is representable Relay IR but not executable by the static plan");
            return;
        }
        if (expr.As<LetNode>()) {
            AddIssue(locator, "Let", "control_flow.let",
                     "Let is representable Relay IR but not executable by the value graph");
            return;
        }
        if (expr.As<FunctionNode>()) {
            AddIssue(locator, "Function", "function_value",
                     "nested function values are not executable");
            return;
        }
        AddIssue(locator, std::string(expr.get()->GetTypeKey()), "relay_node_kind",
                 "Relay node kind is outside the executable dialect");
    }

    void VisitCall(const Expr& ref, const CallNode* call,
                   const std::string& locator) {
        const auto* op = call->op.As<relay::OpNode>();
        const std::string op_name = op ? op->name : "<non-op>";
        const std::string call_locator = locator + "/call(" + op_name + ")";
        if (!op || !op->has_spec) {
            AddIssue(call_locator, "Call", "registered_operator",
                     "Call must reference an operator with an explicit OperatorSpec");
        } else {
            VerifyOperator(call, op, call_locator);
        }
        for (size_t i = 0; i < call->args.size(); ++i) {
            Visit(call->args[i], call_locator + "/arg[" +
                                     std::to_string(i) + "]");
        }
        CheckType(ref.checked_type(), call_locator, "Call",
                  request_.require_checked_types);
    }

    void VerifyOperator(const CallNode* call, const relay::OpNode* op,
                        const std::string& locator) {
        try {
            relay::ValidateOperatorSpec(op->spec);
        } catch (const std::exception& error) {
            AddIssue(locator, "Call", "operator_schema", error.what());
            return;
        }
        const relay::OperatorSpec& spec = op->spec;
        if (spec.input_arity.num_inputs >= 0) {
            if (call->args.size() !=
                static_cast<size_t>(spec.input_arity.num_inputs)) {
                AddIssue(locator, "Call", "operator_arity",
                         "Call argument count does not match OperatorSpec");
            }
        } else if (spec.input_arity.min_inputs < 0 ||
                   spec.input_arity.max_inputs < spec.input_arity.min_inputs ||
                   call->args.size() <
                       static_cast<size_t>(spec.input_arity.min_inputs) ||
                   call->args.size() >
                       static_cast<size_t>(spec.input_arity.max_inputs)) {
            AddIssue(locator, "Call", "operator_arity",
                     "Call argument count is outside OperatorSpec range");
        }
        if (spec.lowering_kind != relay::OperatorLoweringKind::kSingleTE &&
            spec.lowering_kind != relay::OperatorLoweringKind::kMultiTE) {
            AddIssue(locator, "Call", "ordinary_compute_lowering",
                     "only single-TE or multi-TE operators enter the per-unit compiler");
            return;
        }
        const auto relation = op->attrs.find(spec.type_relation_key);
        if (relation == op->attrs.end() ||
            std::any_cast<relay::FInferType>(&relation->second) == nullptr) {
            AddIssue(locator, "Call", "type_relation_binding",
                     "OperatorSpec type relation has no matching implementation binding");
        }
        const auto lowering = op->attrs.find(spec.lowering_key);
        const bool single_ok =
            spec.lowering_kind == relay::OperatorLoweringKind::kSingleTE &&
            lowering != op->attrs.end() &&
            std::any_cast<relay::FRelayToTE>(&lowering->second) != nullptr;
        const bool multi_ok =
            spec.lowering_kind == relay::OperatorLoweringKind::kMultiTE &&
            lowering != op->attrs.end() &&
            std::any_cast<relay::FRelayToTEMulti>(&lowering->second) != nullptr;
        if (!single_ok && !multi_ok) {
            AddIssue(locator, "Call", "lowering_binding",
                     "OperatorSpec lowering has no matching TE implementation binding");
        }
        if (call->attrs.defined()) {
            if (spec.attrs_type_key.empty()) {
                AddIssue(locator, "Call", "operator_attrs",
                         "Call supplies attrs outside its OperatorSpec");
            } else {
                const std::string actual(call->attrs.get()->GetTypeKey());
                const std::string expected_node = spec.attrs_type_key + "Node";
                if (actual != spec.attrs_type_key && actual != expected_node) {
                    AddIssue(locator, "Call", "operator_attrs",
                             "Call attrs runtime type does not match OperatorSpec");
                }
            }
        }
    }
};

}  // namespace

const char* ToString(CapabilityBoundary boundary) {
    switch (boundary) {
        case CapabilityBoundary::kCompilerEntry: return "compiler_entry";
        case CapabilityBoundary::kPostGraphPass: return "post_graph_pass";
        case CapabilityBoundary::kPrePartition: return "pre_partition";
    }
    return "unknown_boundary";
}

const char* ToString(CapabilityMode mode) {
    switch (mode) {
        case CapabilityMode::kStaticExact: return "static_exact";
        case CapabilityMode::kShapeSpecialization: return "shape_specialization";
        case CapabilityMode::kControlFlow: return "control_flow";
        case CapabilityMode::kRegion: return "region";
    }
    return "unknown_mode";
}

std::string CapabilityResult::Diagnostic() const {
    if (supported) return "supported";
    std::ostringstream stream;
    stream << "executable capability rejected (mode=" << ToString(requested_mode)
           << ", target=" << target_identity << ", pipeline="
           << (pipeline_fingerprint.empty() ? "<none>" : pipeline_fingerprint)
           << ")";
    for (const CapabilityIssue& issue : issues) {
        stream << "\n- " << issue.diagnostic_locator << " ["
               << issue.relay_node_kind << "] missing "
               << issue.missing_capability << ": " << issue.detail;
    }
    return stream.str();
}

CapabilityResult CapabilityVerifier::Verify(const CapabilityRequest& request) {
    return Verifier(request).Run();
}

void CapabilityVerifier::Require(const CapabilityRequest& request) {
    const CapabilityResult result = Verify(request);
    if (!result.supported) {
        throw std::invalid_argument(std::string("CapabilityVerifier[") +
                                    ToString(request.boundary) + "]: " +
                                    result.Diagnostic());
    }
}

}  // namespace kxc::api
