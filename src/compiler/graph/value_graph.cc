/*! \file src/compiler/graph/value_graph.cc
 * \brief Builds deterministic stable value ids from checked Relay data flow.
 */

#include "../internal/value_graph.h"
#include "../internal/executable_capability.h"

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace kxc::api::internal {
namespace {

size_t TensorLeafCount(const Type& type) {
    if (type.As<TensorTypeNode>()) return 1;
    if (const auto* tuple = type.As<TupleTypeNode>()) {
        size_t count = 0;
        for (const Type& field : tuple->fields) count += TensorLeafCount(field);
        return count;
    }
    throw std::invalid_argument(
        "BuildValueGraph requires TensorType or nested TupleType leaves");
}

class ValueGraphBuilder {
public:
    ValueGraphBuilder(Function function, Device execution_device)
        : execution_device_(std::move(execution_device)) {
        if (!function.defined()) {
            throw std::invalid_argument("BuildValueGraph requires a defined Function");
        }
        graph_.function = std::move(function);
    }

    ValueGraph Build() {
        VerifyExecutableCapability(
            graph_.function,
            StaticDataflowExecutableCapabilities(execution_device_));
        for (const auto& parameter : graph_.function->params) {
            if (!parameter->type_annotation.As<TensorTypeNode>()) {
                throw std::invalid_argument(
                    "BuildValueGraph requires TensorType function parameters");
            }
            const int64_t id = AddValue(parameter, ValueOrigin::kParameter, 0,
                                        parameter->type_annotation);
            graph_.input_value_ids.push_back(id);
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
            graph_.values[static_cast<size_t>(output_id)].is_graph_output = true;
            graph_.output_value_ids.push_back(output_id);
        }
        return std::move(graph_);
    }

private:
    ValueGraph graph_;
    Device execution_device_;
    std::unordered_map<const Object*, std::vector<int64_t>> bound_value_ids_;

    int64_t AddValue(const Expr& source, ValueOrigin origin, int64_t output_index,
                     const Type& type) {
        if (!source.defined() || !type.defined()) {
            throw std::invalid_argument(
                "Stable graph values require a source and checked type");
        }
        if (!type.As<TensorTypeNode>()) {
            throw std::invalid_argument(
                "Stable graph value leaves must have TensorType");
        }
        const int64_t id = static_cast<int64_t>(graph_.values.size());
        graph_.values.push_back(
            ValueInfo{id, origin, source, output_index, type, false});
        graph_.value_ids_by_expr[source.get()].push_back(id);
        if (origin == ValueOrigin::kConstant) {
            graph_.constant_value_ids.push_back(id);
        }
        return id;
    }

    std::vector<int64_t> Resolve(const Expr& expr) {
        if (!expr.defined()) {
            throw std::invalid_argument("BuildValueGraph encountered undefined Relay Expr");
        }
        if (expr.As<VarNode>()) {
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
                "BuildValueGraph encountered a free or unbound Var");
        }
        const auto memo_it = graph_.value_ids_by_expr.find(expr.get());
        if (memo_it != graph_.value_ids_by_expr.end()) return memo_it->second;

        if (expr.As<ConstantNode>()) {
            return {AddValue(expr, ValueOrigin::kConstant, 0,
                             RequireCheckedType(expr, "Constant"))};
        }
        if (const auto* call = expr.As<CallNode>()) {
            return ResolveCall(expr, call);
        }
        if (expr.As<WhileNode>()) {
            throw std::invalid_argument(
                "BuildValueGraph rejects While; required capability=control_flow.loop");
        }
        if (const auto* let = expr.As<LetNode>()) {
            const std::vector<int64_t> value_ids = Resolve(let->value);
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
            const auto* tuple_type =
                get_item->tuple.checked_type().As<TupleTypeNode>();
            if (!tuple_type || get_item->index < 0 ||
                static_cast<size_t>(get_item->index) >=
                    tuple_type->fields.size()) {
                throw std::invalid_argument(
                    "TupleGetItem index is outside its checked TupleType");
            }
            size_t begin = 0;
            for (int index = 0; index < get_item->index; ++index) {
                begin += TensorLeafCount(
                    tuple_type->fields[static_cast<size_t>(index)]);
            }
            const size_t count = TensorLeafCount(
                tuple_type->fields[static_cast<size_t>(get_item->index)]);
            if (begin + count > tuple_values.size()) {
                throw std::invalid_argument(
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
            throw std::invalid_argument(std::string("BuildValueGraph requires checked_type for ") +
                                        kind);
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
            const int64_t id = AddValue(expr, ValueOrigin::kCallOutput,
                                        static_cast<int64_t>(index),
                                        resolved.output_leaf_types[index]);
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

}  // namespace kxc::api::internal
