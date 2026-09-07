/*! \file src/compiler/internal/lowered_graph.h
 * \brief Internal multi-PrimFunc lowering result for one checked Relay graph.
 */

#pragma once

#include <vector>

#include "lowered_function.h"
#include "prepared_static_graph.h"

namespace kxc::api::internal {

class DynamicUnitShapeContract;

relay::LoweredFunction LowerPrimitiveUnit(
    const std::vector<LogicalValueContract>& values,
    const PrimitiveUnit& unit, const Target& target);
relay::LoweredFunction LowerPrimitiveUnit(
    const std::vector<LogicalValueContract>& values,
    const PrimitiveUnit& unit, const Target& target,
    const DynamicUnitShapeContract& shape_contract);
runtime::ExecutablePlan BuildStaticExecutablePlan(
    const PreparedStaticGraph& prepared);
PreparedStaticGraph PrepareStaticGraph(Function function, Device device,
                                       Target target,
                                       String pipeline_fingerprint);

}  // namespace kxc::api::internal
