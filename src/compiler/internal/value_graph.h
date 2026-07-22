/*! \file src/compiler/internal/value_graph.h
 * \brief Internal deterministic Relay value graph used by unit partitioning.
 */

#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "kxc/relay/op.h"
#include "kxc/relay/relay.h"

namespace kxc::api::internal {

enum class ValueOrigin : int {
    kParameter = 0,
    kConstant = 1,
    kCallOutput = 2,
};

struct ValueInfo {
    int64_t value_id{-1};
    ValueOrigin origin{ValueOrigin::kCallOutput};
    Expr source;
    int64_t output_index{0};
    Type checked_type;
    bool is_graph_output{false};
};

struct CallInfo {
    Expr call;
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
ValueGraph BuildValueGraph(const Function& function);

}  // namespace kxc::api::internal
