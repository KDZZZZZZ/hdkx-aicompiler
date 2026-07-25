/*! \file src/compiler/internal/resolved_relay_call.h
 * \brief One authoritative resolved contract for ordinary Relay operator Calls.
 */

#pragma once

#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "kxc/relay/op_attr_types.h"

namespace kxc::api::internal {

struct OperatorCapabilityPolicy final {
    bool allow_nested_tuple_outputs{false};
    bool require_checked_types{true};

    static OperatorCapabilityPolicy StaticDataflow(
        bool require_checked_types = true);
    static OperatorCapabilityPolicy StructuredControl(
        bool require_checked_types = true);
};

struct RelayCallResolutionIssue final {
    std::string path;
    std::string capability;
    std::string detail;
};

class RelayCallResolutionError final : public std::invalid_argument {
public:
    explicit RelayCallResolutionError(RelayCallResolutionIssue issue);

    const RelayCallResolutionIssue& issue() const noexcept;

private:
    RelayCallResolutionIssue issue_;
};

using RelayOperatorLowering =
    std::variant<relay::FRelayToTE, relay::FRelayToTEMulti>;

struct ResolvedRelayCall final {
    Expr call;
    relay::OperatorSpec spec;
    relay::Attrs attrs;
    relay::FInferType infer_type;
    RelayOperatorLowering lowering;
    Array<Type> input_types;
    std::vector<Type> output_leaf_types;
};

ResolvedRelayCall ResolveRelayCall(
    const Expr& call_expr, const OperatorCapabilityPolicy& policy,
    std::string path);

}  // namespace kxc::api::internal
