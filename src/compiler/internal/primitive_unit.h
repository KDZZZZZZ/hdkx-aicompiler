/*! \file src/compiler/internal/primitive_unit.h
 * \brief Compiler-authoritative one-operator primitive unit.
 */

#pragma once

#include <cstdint>
#include <vector>

#include "kxc/compiler/identity.h"
#include "logical_value.h"
#include "resolved_relay_call.h"

namespace kxc::api::internal {

using PrimitiveUnitId = std::int64_t;

struct PrimitiveUnit {
    PrimitiveUnitId id{-1};
    String symbol;
    ResolvedRelayCall call;
    Array<ValueId> argument_value_ids;
    Array<ValueId> boundary_input_value_ids;
    Array<ValueId> output_value_ids;
    Device device{Device::CPU()};
    UnitSemanticKey semantic_key;
};

PrimitiveUnit BuildPrimitiveUnit(
    PrimitiveUnitId id, ResolvedRelayCall call,
    Array<ValueId> argument_value_ids, Array<ValueId> output_value_ids,
    const std::vector<LogicalValueContract>& values);

void ValidatePrimitiveUnit(
    const PrimitiveUnit& unit,
    const std::vector<LogicalValueContract>& values);

}  // namespace kxc::api::internal
