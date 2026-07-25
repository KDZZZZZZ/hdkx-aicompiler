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

inline std::vector<relay::LoweredFunction> LowerPrimitiveUnits(
    Function function) {
    function = relay::InferTypePass(std::move(function));
    const api::internal::LoweredGraph graph =
        api::internal::LowerGraph(std::move(function));
    std::vector<relay::LoweredFunction> result;
    result.reserve(graph.primitives.size());
    for (const auto& primitive : graph.primitives) {
        result.push_back(primitive.lowered);
    }
    return result;
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
