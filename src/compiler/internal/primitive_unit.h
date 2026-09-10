/*! \file src/compiler/internal/primitive_unit.h
 * \brief Compiler-authoritative primitive units and proved static regions.
 */

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "kxc/compiler/identity.h"
#include "logical_value.h"
#include "resolved_relay_call.h"

namespace kxc::api::internal {

using PrimitiveUnitId = std::int64_t;

// The initial region has exactly one internal producer before the root Call.
// It is admitted only as static, pure, same-shape add -> sqrt.
struct PrimitiveUnitProducer {
    ResolvedRelayCall call;
    Array<ValueId> argument_value_ids;
    Array<ValueId> output_value_ids;
};

struct PrimitiveUnit {
    PrimitiveUnitId id{-1};
    String symbol;
    ResolvedRelayCall call;
    Array<ValueId> argument_value_ids;
    Array<ValueId> boundary_input_value_ids;
    Array<ValueId> output_value_ids;
    Device device{Device::CPU()};
    UnitSemanticKey semantic_key;
    std::optional<PrimitiveUnitProducer> producer;
};

PrimitiveUnit BuildPrimitiveUnit(
    PrimitiveUnitId id, ResolvedRelayCall call,
    Array<ValueId> argument_value_ids, Array<ValueId> output_value_ids,
    const std::vector<LogicalValueContract>& values);

bool CanFuseAddSqrt(const PrimitiveUnit& producer, const PrimitiveUnit& consumer,
                   const std::vector<LogicalValueContract>& values);
PrimitiveUnit FuseAddSqrt(const PrimitiveUnit& producer, const PrimitiveUnit& consumer,
                         const std::vector<LogicalValueContract>& values);

void ValidatePrimitiveUnit(
    const PrimitiveUnit& unit,
    const std::vector<LogicalValueContract>& values);

std::string PrimitiveUnitOperatorIdentity(const PrimitiveUnit& unit);

}  // namespace kxc::api::internal
