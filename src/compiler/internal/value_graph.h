/*! \file src/compiler/internal/value_graph.h
 * \brief Internal deterministic Relay value graph used by unit partitioning.
 */

#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "logical_value.h"
#include "resolved_relay_call.h"
#include "kxc/relay/op.h"
#include "kxc/relay/relay.h"

namespace kxc::api::internal {

using ValueInfo = LogicalValueContract;
using ValueOrigin = LogicalValueOrigin;

struct CallInfo {
    Expr call;
    ResolvedRelayCall resolved;
    std::string operator_name;
    relay::OperatorLoweringKind lowering_kind{relay::OperatorLoweringKind::kNone};
    Array<int64_t> argument_value_ids;
    Array<int64_t> input_value_ids;
    Array<int64_t> output_value_ids;
};

struct ValueGraph {
    Function function;
    std::vector<ValueInfo> values;
    std::vector<CallInfo> calls;
    Array<int64_t> input_value_ids;
    Array<int64_t> constant_value_ids;
    Array<int64_t> output_value_ids;

    // Object identity is used only for lookup while building the graph. Stable ids
    // are assigned exclusively by deterministic parameter and post-order traversal.
    std::unordered_map<const Object*, std::vector<int64_t>> value_ids_by_expr;
};

bool IsOrdinaryCompute(relay::OperatorLoweringKind kind);
ValueGraph BuildValueGraph(const Function& function,
                           Device execution_device = Device());
ValueGraph BuildBoundedValueGraph(
    const Function& function, Device execution_device,
    const BoundedLogicalShapeAdmission& admission);

}  // namespace kxc::api::internal
