/*! \file src/compiler/graph/partition.cc
 * \brief Partitions ordinary compute Calls into proved primitive units.
 */

#include "../internal/compilation_unit.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace kxc::api::internal {
namespace {

bool SameIds(const Array<int64_t>& lhs, const Array<int64_t>& rhs) {
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i] != rhs[i]) return false;
    }
    return true;
}

bool InternalSingleConsumer(const ValueGraph& graph, ValueId id) {
    if (std::find(graph.output_value_ids.begin(), graph.output_value_ids.end(), id) !=
        graph.output_value_ids.end()) return false;
    size_t uses = 0;
    for (const auto& call : graph.calls) {
        for (const ValueId argument : call.argument_value_ids) uses += argument == id;
    }
    return uses == 1;
}

}  // namespace

PartitionedGraph PartitionValueGraph(ValueGraph value_graph, bool fuse_static_add_sqrt) {
    PartitionedGraph result;
    result.input_value_ids = value_graph.input_value_ids;
    result.constant_value_ids = value_graph.constant_value_ids;
    result.output_value_ids = value_graph.output_value_ids;

    int64_t next_unit_id = 0;
    for (size_t call_index = 0; call_index < value_graph.calls.size(); ++call_index) {
        const auto& call = value_graph.calls[call_index];
        if (!IsOrdinaryCompute(call.lowering_kind)) continue;
        PrimitiveUnit unit = BuildPrimitiveUnit(
            next_unit_id, call.resolved, call.argument_value_ids,
            call.output_value_ids, value_graph.values);
        if (!SameIds(unit.boundary_input_value_ids, call.input_value_ids)) {
            throw std::invalid_argument(
                "ValueGraph and PrimitiveUnit boundary ordering drifted");
        }
        if (fuse_static_add_sqrt && unit.call.spec.name == "add" &&
            call_index + 1 < value_graph.calls.size()) {
            const auto& next = value_graph.calls[call_index + 1];
            if (IsOrdinaryCompute(next.lowering_kind) && next.resolved.spec.name == "sqrt") {
                const PrimitiveUnit consumer = BuildPrimitiveUnit(next_unit_id + 1,
                    next.resolved, next.argument_value_ids, next.output_value_ids, value_graph.values);
                if (CanFuseAddSqrt(unit, consumer, value_graph.values) &&
                    InternalSingleConsumer(value_graph, unit.output_value_ids[0])) {
                    unit = FuseAddSqrt(unit, consumer, value_graph.values);
                    ++call_index;
                }
            }
        }
        result.calls.push_back(
            runtime::KernelCall(unit.symbol, unit.boundary_input_value_ids,
                                unit.output_value_ids));
        result.units.push_back(std::move(unit));
        ++next_unit_id;
    }
    result.value_graph = std::move(value_graph);
    ValidatePartition(result);
    return result;
}

void ValidatePartition(const PartitionedGraph& partitioned) {
    if (!SameIds(partitioned.input_value_ids,
                 partitioned.value_graph.input_value_ids) ||
        !SameIds(partitioned.constant_value_ids,
                 partitioned.value_graph.constant_value_ids) ||
        !SameIds(partitioned.output_value_ids,
                 partitioned.value_graph.output_value_ids)) {
        throw std::invalid_argument(
            "PartitionedGraph boundary value lists drifted from the value graph");
    }
    std::unordered_map<const Object*, const CallInfo*> compute_calls;
    for (const auto& call : partitioned.value_graph.calls) {
        if (IsOrdinaryCompute(call.lowering_kind)) {
            if (!compute_calls.emplace(call.call.get(), &call).second) {
                throw std::invalid_argument(
                    "Value graph contains duplicate ordinary compute Call records");
            }
        }
    }
    if (partitioned.calls.size() != partitioned.units.size()) {
        throw std::invalid_argument(
            "Every PrimitiveUnit must own exactly one KernelCall");
    }

    std::unordered_set<const Object*> owned_calls;
    std::unordered_set<std::string> symbols;
    std::unordered_set<ValueId> available;
    for (const ValueId id : partitioned.input_value_ids) available.insert(id);
    for (const ValueId id : partitioned.constant_value_ids) available.insert(id);
    for (size_t index = 0; index < partitioned.units.size(); ++index) {
        const PrimitiveUnit& unit = partitioned.units[index];
        if (unit.id != static_cast<int64_t>(index)) {
            throw std::invalid_argument(
                "PrimitiveUnit ids must be dense and topologically ordered");
        }
        ValidatePrimitiveUnit(unit, partitioned.value_graph.values);
        const auto* call_node = unit.call.call.As<CallNode>();
        if (!call_node) {
            throw std::invalid_argument(
                "PrimitiveUnit must contain a root Relay Call");
        }
        const auto call_it = compute_calls.find(unit.call.call.get());
        if (call_it == compute_calls.end()) {
            throw std::invalid_argument(
                "PrimitiveUnit owns a non-compute or unknown Call");
        }
        if (!owned_calls.insert(unit.call.call.get()).second) {
            throw std::invalid_argument(
                "Ordinary compute Call belongs to more than one PrimitiveUnit");
        }
        if (std::string(unit.symbol).empty() ||
            !symbols.insert(std::string(unit.symbol)).second ||
            !unit.semantic_key.defined()) {
            throw std::invalid_argument(
                "PrimitiveUnit symbol and semantic key must be valid");
        }
        if (!SameIds(unit.argument_value_ids, call_it->second->argument_value_ids) ||
            !SameIds(unit.output_value_ids, call_it->second->output_value_ids)) {
            throw std::invalid_argument(
                "PrimitiveUnit value ids do not match its Call record");
        }

        const CallInfo* boundary_call = call_it->second;
        if (unit.producer) {
            const auto& producer = *unit.producer;
            const auto found = compute_calls.find(producer.call.call.get());
            if (found == compute_calls.end() || !owned_calls.insert(producer.call.call.get()).second ||
                !SameIds(producer.argument_value_ids, found->second->argument_value_ids) ||
                !SameIds(producer.output_value_ids, found->second->output_value_ids) ||
                !InternalSingleConsumer(partitioned.value_graph, producer.output_value_ids[0])) {
                throw std::invalid_argument("PrimitiveUnit region producer is shared, observable or inconsistent");
            }
            boundary_call = found->second;
            // The initial policy accepts adjacent source Calls only.
            const auto& calls = partitioned.value_graph.calls;
            bool adjacent = false;
            for (size_t i = 1; i < calls.size(); ++i) {
                adjacent |= &calls[i - 1] == boundary_call && &calls[i] == call_it->second;
            }
            if (!adjacent) throw std::invalid_argument("PrimitiveUnit region Calls are not adjacent");
        }
        if (!SameIds(unit.boundary_input_value_ids, boundary_call->input_value_ids)) {
            throw std::invalid_argument("PrimitiveUnit external boundary differs from its source Calls");
        }
        for (const ValueId id : unit.boundary_input_value_ids) {
            if (!available.count(id)) throw std::invalid_argument("PrimitiveUnit order has an unavailable input");
        }
        for (const ValueId id : unit.output_value_ids) {
            if (!available.insert(id).second) throw std::invalid_argument("PrimitiveUnit output is not fresh");
        }

        const runtime::KernelCall& plan_call = partitioned.calls[index];
        if (!(plan_call->symbol == unit.symbol) ||
            !SameIds(plan_call.input_value_ids(),
                     unit.boundary_input_value_ids) ||
            !SameIds(plan_call.output_value_ids(), unit.output_value_ids)) {
            throw std::invalid_argument(
                "PrimitiveUnit and KernelCall draft must stay one-to-one");
        }
    }
    if (owned_calls.size() != compute_calls.size()) {
        throw std::invalid_argument("Every ordinary compute Call must belong to exactly one PrimitiveUnit");
    }
}

}  // namespace kxc::api::internal
