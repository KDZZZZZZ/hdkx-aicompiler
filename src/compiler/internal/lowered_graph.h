/*! \file src/compiler/internal/lowered_graph.h
 * \brief Internal multi-PrimFunc lowering result for one checked Relay graph.
 */

#pragma once

#include <vector>

#include "lowered_function.h"
#include "prepared_static_graph.h"

namespace kxc::api::internal {

relay::LoweredFunction LowerPrimitiveUnit(
    const std::vector<LogicalValueContract>& values,
    const PrimitiveUnit& unit, const Target& target);
runtime::ExecutablePlan BuildStaticExecutablePlan(
    const PreparedStaticGraph& prepared);
PreparedStaticGraph PrepareStaticGraph(Function function, Device device,
                                       Target target,
                                       String pipeline_fingerprint);

}  // namespace kxc::api::internal
