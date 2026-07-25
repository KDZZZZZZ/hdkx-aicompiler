/*! \file src/compiler/internal/lowered_graph.h
 * \brief Internal multi-PrimFunc lowering result for one checked Relay graph.
 */

#pragma once

#include <vector>

#include "prepared_static_graph.h"
#include "kxc/compiler/lowering/relay_to_tir.h"

namespace kxc::api::internal {

struct LoweredPrimitive {
    int64_t unit_id{-1};
    String symbol;
    String operator_identity;
    UnitSemanticKey semantic_key;
    relay::LoweredFunction lowered;
};

struct LoweredGraph {
    PartitionedGraph partitioned;
    std::vector<LoweredPrimitive> primitives;
    runtime::ExecutablePlan plan;
    Map<String, runtime::NDArray> constants;
};

relay::LoweredFunction LowerPrimitiveUnit(
    const ValueGraph& graph, const PrimitiveUnit& unit);
PreparedStaticGraph PrepareStaticGraph(Function function, Device device,
                                       Target target,
                                       String pipeline_fingerprint);
LoweredGraph LowerPreparedStaticGraph(const PreparedStaticGraph& prepared);
LoweredGraph LowerGraph(Function function,
                        Device device = Device::CPU(),
                        Target target = Target(),
                        String pipeline_fingerprint = String());
void ValidateLoweredGraph(const LoweredGraph& graph);

}  // namespace kxc::api::internal
