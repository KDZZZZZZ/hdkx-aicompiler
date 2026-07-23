/*! \file src/compiler/internal/compilation_unit.h
 * \brief Internal one-operator compilation unit and plan draft contracts.
 */

#pragma once

#include <cstdint>
#include <vector>

#include "kxc/compiler/identity.h"
#include "kxc/runtime/executable_plan.h"
#include "value_graph.h"

namespace kxc::api::internal {

struct CompilationUnit {
    int64_t unit_id{-1};
    String symbol;
    Expr call;
    Array<int64_t> input_value_ids;
    Array<int64_t> output_value_ids;
    UnitSemanticKey semantic_key;
};

struct PartitionedGraph {
    ValueGraph value_graph;
    std::vector<CompilationUnit> units;
    Array<runtime::KernelCall> calls;
    Array<int64_t> input_value_ids;
    Array<int64_t> constant_value_ids;
    Array<int64_t> output_value_ids;
};

PartitionedGraph PartitionValueGraph(ValueGraph value_graph);
void ValidatePartition(const PartitionedGraph& partitioned);

}  // namespace kxc::api::internal
