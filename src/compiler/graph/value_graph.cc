/*! \file src/compiler/graph/value_graph.cc
 * \brief Builds deterministic stable value ids from checked Relay data flow.
 */

#include "../internal/value_graph.h"

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace kxc::api::internal {
namespace {

size_t TensorLeafCount(
    const Type& type, const BoundedLogicalShapeAdmission* bounded) {
    return bounded
        ? FlattenLogicalTensorTypes(type, "value_graph.checked_type", *bounded).size()
        : FlattenLogicalTensorTypes(type, "value_graph.checked_type").size();
}

class ValueGraphBuilder {
public:
    ValueGraphBuilder(Function function, Device execution_device,
                      const BoundedLogicalShapeAdmission* bounded = nullptr)
        : execution_device_(std::move(execution_device)), bounded_(bounded) {
        if (!function.defined()) {
            throw std::invalid_argument(
                "BuildValueGraph: capability=defined_typed_relay; requires a defined Function");
        }
        graph_.function = std::move(function);
    }

    ValueGraph Build() {
        if (!graph_.function->body.defined()) {
            throw std::invalid_argument(
                "BuildValueGraph: capability=defined_typed_relay; Function has no body");
        }
        RequireCheckedType(Expr(ObjectRef(graph_.function)), "Function");
        for (const auto& parameter : graph_.function->params) {
            const std::string path =
                "function.params[" + std::to_string(graph_.input_value_ids.size()) + "]";
            if (!parameter.defined() || !parameter->type_annotation.defined() ||
                !parameter->type_annotation.As<TensorTypeNode>()) {
                throw std::invalid_argument(
                    "BuildValueGraph: capability=tensor_parameter; " +
                    path + " requires a TensorType annotation");
            }
            if (!TypeEqual(parameter->type_annotation,
                           RequireCheckedType(parameter, "parameter"))) {
                throw std::invalid_argument(
                    "BuildValueGraph: capability=typed_parameter; " +
                    path + " annotation and checked_type differ");
            }
            const int64_t id = AddValue(
                parameter, ValueOrigin::kParameter, parameter->type_annotation,
                path);
            graph_.input_value_ids.push_back(id);
        }

        if (!TypeEqual(RequireCheckedType(Expr(ObjectRef(graph_.function)), "Function"),
                       RequireCheckedType(graph_.function->body, "Function body"))) {
            throw std::invalid_argument(
                "BuildValueGraph: capability=typed_function_result; Function and body checked_type differ");
        }
        const std::vector<int64_t> outputs = Resolve(graph_.function->body);
        if (outputs.empty()) {
            throw std::invalid_argument("BuildValueGraph requires graph outputs");
        }
        for (int64_t output_id : outputs) {
            if (output_id < 0 ||
                static_cast<size_t>(output_id) >= graph_.values.size()) {
                throw std::logic_error("BuildValueGraph produced an invalid output id");
            }
            graph_.output_value_ids.push_back(output_id);
        }
        return std::move(graph_);
    }

private:
    ValueGraph graph_;
    Device execution_device_;
    const BoundedLogicalShapeAdmission* bounded_{nullptr};
    std::unordered_map<const Object*, std::vector<int64_t>> bound_value_ids_;

    int64_t AddValue(const Expr& source, ValueOrigin origin, const Type& type,
                     const std::string& source_locator) {
        if (!source.defined() || !type.defined()) {
            throw std::invalid_argument(
                "Stable graph values require a source and checked type");
        }
        if (!type.As<TensorTypeNode>()) {
            throw std::invalid_argument(
                "Stable graph value leaves must have TensorType");
        }
        const int64_t id = static_cast<int64_t>(graph_.values.size());
        std::vector<LogicalValueContract> leaves = bounded_
            ? MakeLogicalValueLeaves(source, type, origin, id, execution_device_,
                                     source_locator, *bounded_)
            : MakeLogicalValueLeaves(source, type, origin, id, execution_device_,
                                     source_locator);
        if (leaves.size() != 1) {
            throw std::invalid_argument(
                "Stable graph AddValue requires exactly one TensorType leaf");
        }
        if (!execution_device_.defined()) {
            const auto* relay = dynamic_cast<const RelayNode*>(source.get());
            if (relay && relay->virtual_device_.defined()) {
                throw std::invalid_argument(
                    "BuildValueGraph: capability=explicit_execution_device; " +
                    source_locator + " has explicit Relay placement");
            }
        } else if (leaves.front().device != execution_device_) {
            throw std::invalid_argument(
                "BuildValueGraph: capability=matching_execution_device; " +
                source_locator + " placement differs from the execution device");
        }
        graph_.values.push_back(std::move(leaves.front()));
        graph_.value_ids_by_expr[source.get()].push_back(id);
        if (origin == ValueOrigin::kConstant) {
            graph_.constant_value_ids.push_back(id);
        }
        return id;
    }

    std::vector<int64_t> Resolve(const Expr& expr) {
        if (!expr.defined()) {
            throw std::invalid_argument(
                "BuildValueGraph: capability=defined_typed_relay; Relay Expr is undefined");
        }
        if (expr.As<VarNode>()) {
            (void)RequireCheckedType(expr, "Var");
            const auto bound_it = bound_value_ids_.find(expr.get());
            if (bound_it != bound_value_ids_.end()) {
                // Keep the Var identity queryable by unit lowering while resolving
                // its let value exactly once through the lexical environment.
                graph_.value_ids_by_expr[expr.get()] = bound_it->second;
                return bound_it->second;
            }
            const auto parameter_it = graph_.value_ids_by_expr.find(expr.get());
            if (parameter_it != graph_.value_ids_by_expr.end()) {
                return parameter_it->second;
            }
            throw std::invalid_argument(
                "BuildValueGraph: capability=lexically_bound_var; encountered a free or unbound Var");
        }
        const auto memo_it = graph_.value_ids_by_expr.find(expr.get());
        if (memo_it != graph_.value_ids_by_expr.end()) return memo_it->second;

        if (const auto* constant = expr.As<ConstantNode>()) {
            if (!constant->data.defined()) {
                throw std::invalid_argument(
                    "BuildValueGraph: capability=constant_payload; constant payload is undefined");
            }
            return {AddValue(expr, ValueOrigin::kConstant,
                             RequireCheckedType(expr, "Constant"),
                             "function.constant[" +
                                 std::to_string(graph_.constant_value_ids.size()) +
                                 "]")};
        }
        if (const auto* call = expr.As<CallNode>()) {
            return ResolveCall(expr, call);
        }
        if (expr.As<IfNode>()) {
            throw std::invalid_argument(
                "BuildValueGraph: capability=if; static dataflow does not support If");
        }
        if (expr.As<WhileNode>()) {
            throw std::invalid_argument(
                "BuildValueGraph: capability=control_flow.loop; static dataflow does not support While");
        }
        if (expr.As<FunctionNode>()) {
            throw std::invalid_argument(
                "BuildValueGraph: capability=first_order_relay; function values are unsupported");
        }
        if (const auto* let = expr.As<LetNode>()) {
            if (!let->var.defined()) {
                throw std::invalid_argument(
                    "BuildValueGraph: capability=lexical_let_binding; Let binder is undefined");
            }
            const Type binder_type = RequireCheckedType(let->var, "Let binder");
            const std::vector<int64_t> value_ids = Resolve(let->value);
            if (!TypeEqual(binder_type, RequireCheckedType(let->value, "Let value"))) {
                throw std::invalid_argument(
                    "BuildValueGraph: capability=typed_let_binding; Let binder and value differ");
            }
            const auto outer = bound_value_ids_.find(let->var.get());
            const std::optional<std::vector<int64_t>> saved =
                outer == bound_value_ids_.end()
                    ? std::nullopt
                    : std::optional<std::vector<int64_t>>(outer->second);
            bound_value_ids_[let->var.get()] = value_ids;
            const std::vector<int64_t> body_ids = Resolve(let->body);
            if (saved) {
                bound_value_ids_[let->var.get()] = *saved;
            } else {
                bound_value_ids_.erase(let->var.get());
            }
            graph_.value_ids_by_expr.emplace(expr.get(), body_ids);
            return body_ids;
        }
        if (const auto* tuple = expr.As<TupleNode>()) {
            (void)RequireCheckedType(expr, "Tuple");
            std::vector<int64_t> fields;
            for (const auto& field : tuple->fields) {
                const std::vector<int64_t> field_values = Resolve(field);
                fields.insert(fields.end(), field_values.begin(), field_values.end());
            }
            graph_.value_ids_by_expr.emplace(expr.get(), fields);
            return fields;
        }
        if (const auto* get_item = expr.As<TupleGetItemNode>()) {
            const std::vector<int64_t> tuple_values = Resolve(get_item->tuple);
            const Type result_type = RequireCheckedType(expr, "TupleGetItem");
            const auto* tuple_type =
                RequireCheckedType(get_item->tuple, "TupleGetItem tuple").As<TupleTypeNode>();
            if (!tuple_type || get_item->index < 0 ||
                static_cast<size_t>(get_item->index) >=
                    tuple_type->fields.size()) {
                throw std::invalid_argument(
                    "BuildValueGraph: capability=well_typed_tuple_get_item; "
                    "TupleGetItem index is outside its checked TupleType");
            }
            if (!TypeEqual(result_type,
                           tuple_type->fields[static_cast<size_t>(get_item->index)])) {
                throw std::invalid_argument(
                    "BuildValueGraph: capability=well_typed_tuple_get_item; "
                    "TupleGetItem checked type differs from its selected field");
            }
            size_t begin = 0;
            for (int index = 0; index < get_item->index; ++index) {
                begin += TensorLeafCount(
                    tuple_type->fields[static_cast<size_t>(index)], bounded_);
            }
            const size_t count = TensorLeafCount(
                tuple_type->fields[static_cast<size_t>(get_item->index)], bounded_);
            if (begin + count > tuple_values.size()) {
                throw std::invalid_argument(
                    "BuildValueGraph: capability=well_typed_tuple_get_item; "
                    "TupleGetItem checked type does not match flattened values");
            }
            std::vector<int64_t> selected(
                tuple_values.begin() + static_cast<std::ptrdiff_t>(begin),
                tuple_values.begin() +
                    static_cast<std::ptrdiff_t>(begin + count));
            graph_.value_ids_by_expr.emplace(expr.get(), selected);
            return selected;
        }
        throw std::invalid_argument(
            "BuildValueGraph supports parameters, constants, calls, lets, tuples, and tuple fields");
    }

    Type RequireCheckedType(const Expr& expr, const char* kind) const {
        const Type type = expr.checked_type();
        if (!type.defined()) {
            throw std::invalid_argument(
                std::string("BuildValueGraph: capability=defined_typed_relay; ") +
                kind + " checked_type is missing");
        }
        return type;
    }

    std::vector<int64_t> ResolveCall(const Expr& expr, const CallNode* call) {
        ResolvedRelayCall resolved = ResolveRelayCall(
            expr, OperatorCapabilityPolicy::StaticDataflow(),
            "function.call");
        Array<int64_t> argument_ids;
        Array<int64_t> unique_ids;
        std::unordered_set<int64_t> seen_inputs;
        for (const auto& argument : call->args) {
            for (int64_t id : Resolve(argument)) {
                argument_ids.push_back(id);
                if (seen_inputs.insert(id).second) unique_ids.push_back(id);
            }
        }
        // PrimFunc ABI is [non-constant inputs][constants][outputs]. Logical
        // argument_value_ids above retains original grouping/order/duplicates.
        Array<int64_t> input_ids;
        for (int64_t id : unique_ids) {
            if (graph_.values[static_cast<size_t>(id)].origin !=
                ValueOrigin::kConstant) {
                input_ids.push_back(id);
            }
        }
        for (int64_t id : unique_ids) {
            if (graph_.values[static_cast<size_t>(id)].origin ==
                ValueOrigin::kConstant) {
                input_ids.push_back(id);
            }
        }

        Array<int64_t> output_ids;
        std::vector<int64_t> result;
        for (size_t index = 0;
             index < resolved.output_leaf_types.size(); ++index) {
            const int64_t id = AddValue(expr, ValueOrigin::kPrimitiveOutput,
                                        resolved.output_leaf_types[index],
                                        "function.call[" +
                                            std::to_string(graph_.calls.size()) +
                                            "].output[" +
                                            std::to_string(index) + "]");
            output_ids.push_back(id);
            result.push_back(id);
        }
        const std::string operator_name = resolved.spec.name;
        const relay::OperatorLoweringKind lowering_kind =
            resolved.spec.lowering_kind;
        graph_.calls.push_back(CallInfo{
            expr, std::move(resolved), operator_name, lowering_kind,
            argument_ids, input_ids, output_ids});
        return result;
    }
};

}  // namespace

bool IsOrdinaryCompute(relay::OperatorLoweringKind kind) {
    return kind == relay::OperatorLoweringKind::kSingleTE ||
           kind == relay::OperatorLoweringKind::kMultiTE;
}

ValueGraph BuildValueGraph(const Function& function,
                           Device execution_device) {
    return ValueGraphBuilder(function, std::move(execution_device)).Build();
}

ValueGraph BuildBoundedValueGraph(
    const Function& function, Device execution_device,
    const BoundedLogicalShapeAdmission& admission) {
    return ValueGraphBuilder(function, std::move(execution_device),
                             &admission).Build();
}

}  // namespace kxc::api::internal
