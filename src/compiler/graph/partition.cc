/*! \file src/compiler/graph/partition.cc
 * \brief Partitions each ordinary compute Call into exactly one compilation unit.
 */

#include "../internal/compilation_unit.h"

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

}  // namespace

PartitionedGraph PartitionValueGraph(ValueGraph value_graph) {
    PartitionedGraph result;
    result.input_value_ids = value_graph.input_value_ids;
    result.constant_value_ids = value_graph.constant_value_ids;
    result.output_value_ids = value_graph.output_value_ids;

    int64_t next_unit_id = 0;
    for (const auto& call : value_graph.calls) {
        if (!IsOrdinaryCompute(call.lowering_kind)) continue;
        PrimitiveUnit unit = BuildPrimitiveUnit(
            next_unit_id, call.resolved, call.argument_value_ids,
            call.output_value_ids, value_graph.values);
        if (!SameIds(unit.boundary_input_value_ids, call.input_value_ids)) {
            throw std::invalid_argument(
                "ValueGraph and PrimitiveUnit boundary ordering drifted");
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
    if (partitioned.units.size() != compute_calls.size() ||
        partitioned.calls.size() != partitioned.units.size()) {
        throw std::invalid_argument(
            "Every ordinary compute Call must own exactly one PrimitiveUnit and KernelCall");
    }

    std::unordered_set<const Object*> owned_calls;
    std::unordered_set<std::string> symbols;
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
                "PrimitiveUnit must contain exactly one root Relay Call");
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
        if (!SameIds(unit.argument_value_ids,
                     call_it->second->argument_value_ids) ||
            !SameIds(unit.boundary_input_value_ids,
                     call_it->second->input_value_ids) ||
            !SameIds(unit.output_value_ids, call_it->second->output_value_ids)) {
            throw std::invalid_argument(
                "PrimitiveUnit value ids do not match its Call record");
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
}

}  // namespace kxc::api::internal
