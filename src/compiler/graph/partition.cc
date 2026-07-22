/*! \file src/compiler/graph/partition.cc
 * \brief Partitions each ordinary compute Call into exactly one compilation unit.
 */

#include "../internal/compilation_unit.h"

#include <cctype>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "kxc/profiling/profiling.h"

namespace kxc::api::internal {
namespace {

std::string SanitizeSymbolPart(const std::string& name) {
    std::string result;
    result.reserve(name.size());
    for (unsigned char ch : name) {
        result.push_back(std::isalnum(ch) ? static_cast<char>(ch) : '_');
    }
    return result.empty() ? "op" : result;
}

std::string BuildUnitSymbol(int64_t unit_id, const std::string& operator_name) {
    return "kxc_unit_" + std::to_string(unit_id) + "_" +
           SanitizeSymbolPart(operator_name);
}

std::string BuildStructuralHash(const CallInfo& call,
                                const ValueGraph& graph) {
    const auto* call_node = call.call.As<CallNode>();
    const auto* op = call_node ? call_node->op.As<relay::OpNode>() : nullptr;
    if (!call_node || !op) {
        throw std::invalid_argument(
            "Structural hash requires an operator Call");
    }
    std::ostringstream canonical;
    canonical << relay::SerializeOperatorSpec(op->spec) << ";inputs=";
    for (int64_t id : call.argument_value_ids) {
        if (id < 0 || static_cast<size_t>(id) >= graph.values.size()) {
            throw std::invalid_argument(
                "Structural hash references an invalid input value id");
        }
        canonical << id << ':'
                  << TypeToString(graph.values[static_cast<size_t>(id)].checked_type)
                  << ',';
    }
    canonical << ";outputs=";
    for (int64_t id : call.output_value_ids) {
        if (id < 0 || static_cast<size_t>(id) >= graph.values.size()) {
            throw std::invalid_argument(
                "Structural hash references an invalid output value id");
        }
        canonical << id << ':'
                  << TypeToString(graph.values[static_cast<size_t>(id)].checked_type)
                  << ',';
    }
    canonical << ";attrs=";
    if (call_node->attrs.defined()) {
        canonical << relay::SerializeAttrs(relay::Attrs(call_node->attrs));
    } else {
        canonical << "<none>";
    }
    return profiling::HashText(canonical.str());
}

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
        const std::string symbol = BuildUnitSymbol(next_unit_id, call.operator_name);
        CompilationUnit unit{next_unit_id,
                             String(symbol),
                             call.call,
                             call.input_value_ids,
                             call.output_value_ids,
                             String(BuildStructuralHash(call, value_graph))};
        result.calls.push_back(
            runtime::KernelCall(unit.symbol, unit.input_value_ids,
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
            "Every ordinary compute Call must own exactly one CompilationUnit and KernelCall");
    }

    std::unordered_set<const Object*> owned_calls;
    std::unordered_set<std::string> symbols;
    for (size_t index = 0; index < partitioned.units.size(); ++index) {
        const CompilationUnit& unit = partitioned.units[index];
        if (unit.unit_id != static_cast<int64_t>(index)) {
            throw std::invalid_argument(
                "CompilationUnit ids must be dense and topologically ordered");
        }
        const auto* call_node = unit.call.As<CallNode>();
        if (!call_node) {
            throw std::invalid_argument(
                "CompilationUnit must contain exactly one root Relay Call");
        }
        const auto call_it = compute_calls.find(unit.call.get());
        if (call_it == compute_calls.end()) {
            throw std::invalid_argument(
                "CompilationUnit owns a non-compute or unknown Call");
        }
        if (!owned_calls.insert(unit.call.get()).second) {
            throw std::invalid_argument(
                "Ordinary compute Call belongs to more than one CompilationUnit");
        }
        if (std::string(unit.symbol).empty() ||
            !symbols.insert(std::string(unit.symbol)).second ||
            std::string(unit.structural_hash).empty()) {
            throw std::invalid_argument(
                "CompilationUnit symbol and structural hash must be non-empty and unique");
        }
        if (!SameIds(unit.input_value_ids, call_it->second->input_value_ids) ||
            !SameIds(unit.output_value_ids, call_it->second->output_value_ids)) {
            throw std::invalid_argument(
                "CompilationUnit boundary ids do not match its Call record");
        }

        const runtime::KernelCall& plan_call = partitioned.calls[index];
        if (!(plan_call->symbol == unit.symbol) ||
            !SameIds(plan_call.input_value_ids(), unit.input_value_ids) ||
            !SameIds(plan_call.output_value_ids(), unit.output_value_ids)) {
            throw std::invalid_argument(
                "CompilationUnit and KernelCall draft must stay one-to-one");
        }
    }
}

}  // namespace kxc::api::internal
