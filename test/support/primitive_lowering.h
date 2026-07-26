/*! \file test/support/primitive_lowering.h
 * \brief Focused test helpers over the production per-primitive lowering path.
 */

#pragma once

#include <stdexcept>
#include <utility>
#include <vector>

#include "../../src/compiler/internal/lowered_graph.h"
#include "kxc/relay/transforms/infer_type.h"

namespace kxc::test_support {

struct PrimitiveLoweringFixture final {
    api::internal::PreparedStaticGraph prepared;
    std::vector<relay::LoweredFunction> lowered;
};

inline PrimitiveLoweringFixture LowerPrimitivesForTest(Function function) {
    function = relay::InferTypePass(std::move(function));
    PrimitiveLoweringFixture result{
        api::internal::PrepareStaticGraph(
            std::move(function), Device::CPU(), BuildTarget(Device::CPU()),
            String()),
        {}};
    result.lowered.reserve(result.prepared.partitioned.units.size());
    for (const api::internal::PrimitiveUnit& unit :
         result.prepared.partitioned.units) {
        result.lowered.push_back(api::internal::LowerPrimitiveUnit(
            result.prepared.partitioned.value_graph.values, unit,
            result.prepared.target));
    }
    return result;
}

inline std::vector<relay::LoweredFunction> LowerPrimitiveUnits(
    Function function) {
    return LowerPrimitivesForTest(std::move(function)).lowered;
}

inline relay::LoweredFunction LowerFirstPrimitive(Function function) {
    std::vector<relay::LoweredFunction> lowered =
        LowerPrimitiveUnits(std::move(function));
    if (lowered.empty()) {
        throw std::invalid_argument(
            "LowerFirstPrimitive requires at least one production primitive");
    }
    return lowered.front();
}

}  // namespace kxc::test_support
