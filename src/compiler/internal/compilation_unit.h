/*! \file src/compiler/internal/compilation_unit.h
 * \brief Internal compilation unit and plan draft contracts.
 */

#pragma once

#include <cstdint>
#include <vector>

#include "kxc/runtime/executable_plan.h"
#include "primitive_unit.h"
#include "value_graph.h"

namespace kxc::api::internal {

struct PartitionedGraph {
    ValueGraph value_graph;
    std::vector<PrimitiveUnit> units;
    Array<runtime::KernelCall> calls;
    Array<int64_t> input_value_ids;
    Array<int64_t> constant_value_ids;
    Array<int64_t> output_value_ids;
};

PartitionedGraph PartitionValueGraph(ValueGraph value_graph, bool fuse_static_add_sqrt = false);
void ValidatePartition(const PartitionedGraph& partitioned);

}  // namespace kxc::api::internal
