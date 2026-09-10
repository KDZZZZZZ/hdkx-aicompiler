/*! \file src/compiler/analysis/resolved_relay_call.cc
 * \brief Resolves and validates one ordinary Relay operator Call.
 */

#include "../internal/resolved_relay_call.h"

#include <any>
#include <sstream>
#include <utility>

namespace kxc::api::internal {
namespace {

[[noreturn]] void Fail(std::string path, std::string capability,
                       std::string detail) {
    throw RelayCallResolutionError(
        {std::move(path), std::move(capability), std::move(detail)});
}

bool IsOrdinaryOperator(relay::OperatorLoweringKind kind) {
    return kind == relay::OperatorLoweringKind::kSingleTE ||
           kind == relay::OperatorLoweringKind::kMultiTE;
}

void ValidateInputArity(const relay::OperatorSpec& spec, std::size_t actual,
                        const std::string& path) {
    if (spec.input_arity.num_inputs >= 0) {
        if (actual != static_cast<std::size_t>(spec.input_arity.num_inputs)) {
            Fail(path, "operator_input_arity",
                 "Call input arity differs from OperatorSpec");
        }
        return;
    }
    if (spec.input_arity.min_inputs < 0 ||
        spec.input_arity.max_inputs < 0 ||
        actual < static_cast<std::size_t>(spec.input_arity.min_inputs) ||
        actual > static_cast<std::size_t>(spec.input_arity.max_inputs)) {
        Fail(path, "operator_input_arity",
             "Call variable input arity differs from OperatorSpec");
    }
}

relay::Attrs ResolveAttrs(const relay::OperatorSpec& spec,
                          const CallNode* call,
                          const std::string& path) {
    if (!call->attrs.defined()) return relay::Attrs();
    if (spec.attrs_type_key.empty()) {
        Fail(path, "operator_attrs_schema",
             "Call defines attrs outside its OperatorSpec schema");
    }
    const std::string actual(call->attrs.get()->GetTypeKey());
    if (actual != spec.attrs_type_key &&
        actual != spec.attrs_type_key + "Node") {
        Fail(path, "operator_attrs_schema",
             "Call attrs type differs from OperatorSpec");
    }
    return relay::Attrs(call->attrs);
}

void FlattenTensorLeaves(const Type& type, bool allow_nested,
                         std::vector<Type>* leaves,
                         const std::string& path) {
    if (type.As<TensorTypeNode>()) {
        leaves->push_back(type);
        return;
    }
    const auto* tuple = type.As<TupleTypeNode>();
    if (!tuple) {
        Fail(path, "tensor_call_output",
             "Call output must contain only TensorType leaves");
    }
    for (std::size_t index = 0; index < tuple->fields.size(); ++index) {
        const Type& field = tuple->fields[index];
        if (!allow_nested && !field.As<TensorTypeNode>()) {
            Fail(path, "flat_multi_tensor_output",
                 "static dataflow multi-output Calls require a flat tensor tuple");
        }
        FlattenTensorLeaves(
            field, allow_nested, leaves,
            path + ".fields[" + std::to_string(index) + "]");
    }
}

}  // namespace

OperatorCapabilityPolicy OperatorCapabilityPolicy::StaticDataflow(
    bool require_checked_types) {
    return OperatorCapabilityPolicy{false, require_checked_types};
}

OperatorCapabilityPolicy OperatorCapabilityPolicy::StructuredControl(
    bool require_checked_types) {
    return OperatorCapabilityPolicy{true, require_checked_types};
}

RelayCallResolutionError::RelayCallResolutionError(
    RelayCallResolutionIssue issue)
    : std::invalid_argument(
          "ResolveRelayCall: path=" + issue.path +
          "; capability=" + issue.capability + "; " + issue.detail),
      issue_(std::move(issue)) {}

const RelayCallResolutionIssue&
RelayCallResolutionError::issue() const noexcept {
    return issue_;
}

ResolvedRelayCall ResolveRelayCall(
    const Expr& call_expr, const OperatorCapabilityPolicy& policy,
    std::string path) {
    const auto* call = call_expr.As<CallNode>();
    const auto* op = call ? call->op.As<relay::OpNode>() : nullptr;
    if (!call || !op || !op->has_spec) {
        Fail(std::move(path), "registered_ordinary_op_call",
             "Call target is not a specified Op");
    }
    const relay::Op* registered = relay::Op::TryGet(op->name);
    if (!registered || registered->get() != call->op.get()) {
        Fail(std::move(path), "registered_ordinary_op_call",
             "Call target is not the registered Op instance");
    }
    try {
        relay::ValidateOperatorSpec(op->spec);
    } catch (const std::exception& error) {
        Fail(std::move(path), "registered_ordinary_op_call", error.what());
    }
    if (!IsOrdinaryOperator(op->spec.lowering_kind)) {
        Fail(std::move(path), "registered_ordinary_op_call",
             "operator is not an ordinary kernel operation");
    }
    if (op->spec.effect != relay::OperatorEffectKind::kPure ||
        !op->spec.deterministic || op->spec.alias_contract != "none") {
        Fail(std::move(path), "pure_deterministic_no_alias",
             "operator is not a pure deterministic non-aliasing kernel");
    }

    ValidateInputArity(op->spec, call->args.size(), path);
    relay::Attrs attrs = ResolveAttrs(op->spec, call, path);

    const auto relation = op->attrs.find(op->spec.type_relation_key);
    const auto lowering = op->attrs.find(op->spec.lowering_key);
    if (relation == op->attrs.end() || lowering == op->attrs.end()) {
        Fail(std::move(path), "operator_implementation_binding",
             "OperatorSpec implementation binding is missing");
    }
    const auto* infer = std::any_cast<relay::FInferType>(&relation->second);
    if (!infer || !*infer) {
        Fail(std::move(path), "operator_implementation_binding",
             "type relation binding has the wrong type or is empty");
    }

    RelayOperatorLowering resolved_lowering;
    if (op->spec.lowering_kind ==
        relay::OperatorLoweringKind::kSingleTE) {
        const auto* lower =
            std::any_cast<relay::FRelayToTE>(&lowering->second);
        if (op->spec.lowering_key != "FRelayToTE" || !lower || !*lower) {
            Fail(std::move(path), "operator_implementation_binding",
                 "single-output lowering requires a non-empty FRelayToTE binding");
        }
        resolved_lowering = *lower;
    } else {
        const auto* lower =
            std::any_cast<relay::FRelayToTEMulti>(&lowering->second);
        if (op->spec.lowering_key != "FRelayToTEMulti" ||
            !lower || !*lower) {
            Fail(std::move(path), "operator_implementation_binding",
                 "multi-output lowering requires a non-empty FRelayToTEMulti binding");
        }
        resolved_lowering = *lower;
    }

    if (!policy.require_checked_types) {
        return ResolvedRelayCall{
            call_expr, op->spec, std::move(attrs), *infer,
            std::move(resolved_lowering), {}, {}};
    }
    if (!call_expr.checked_type().defined()) {
        Fail(std::move(path), "defined_typed_relay",
             "Call checked_type is missing");
    }
    Array<Type> input_types;
    for (std::size_t index = 0; index < call->args.size(); ++index) {
        const Type type = call->args[index].checked_type();
        if (!type.defined()) {
            Fail(path + ".args[" + std::to_string(index) + "]",
                 "defined_typed_relay",
                 "Call argument checked_type is missing");
        }
        input_types.push_back(type);
    }
    Type inferred;
    try {
        inferred = (*infer)(attrs, input_types);
    } catch (const std::exception& error) {
        Fail(std::move(path), "operator_type_relation", error.what());
    }
    if (!TypeEqual(inferred, call_expr.checked_type())) {
        Fail(std::move(path), "operator_type_relation",
             "Call checked_type is stale for OperatorSpec");
    }

    std::vector<Type> output_leaf_types;
    if (op->spec.lowering_kind ==
        relay::OperatorLoweringKind::kSingleTE) {
        if (!call_expr.checked_type().As<TensorTypeNode>()) {
            Fail(std::move(path), "single_tensor_output",
                 "single-output lowering requires TensorType");
        }
        output_leaf_types.push_back(call_expr.checked_type());
    } else {
        if (!call_expr.checked_type().As<TupleTypeNode>()) {
            Fail(std::move(path), "flat_multi_tensor_output",
                 "multi-output lowering requires TupleType");
        }
        FlattenTensorLeaves(
            call_expr.checked_type(), policy.allow_nested_tuple_outputs,
            &output_leaf_types, path + ".checked_type");
    }
    if (output_leaf_types.empty() ||
        (op->spec.output_arity > 0 &&
         static_cast<std::size_t>(op->spec.output_arity) !=
             output_leaf_types.size())) {
        Fail(std::move(path), "operator_output_arity",
             "Call output leaves differ from OperatorSpec");
    }

    return ResolvedRelayCall{
        call_expr, op->spec, std::move(attrs), *infer,
        std::move(resolved_lowering), std::move(input_types),
        std::move(output_leaf_types)};
}

}  // namespace kxc::api::internal
