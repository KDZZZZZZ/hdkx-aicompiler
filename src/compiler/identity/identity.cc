/*! \file src/compiler/identity.cc
 * \brief Implements length-delimited compiler identity canonicalization.
 */

#include "kxc/compiler/identity.h"

#include "../internal/identity_canonical.h"
#include "../internal/identity_private.h"

#include <functional>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kxc/relay/op.h"
#include "kxc/relay/relay.h"

namespace kxc::api {
namespace {

using internal::AppendField;
using internal::AppendInteger;
using internal::Digest;
using internal::RequireNonEmpty;

template <typename Node>
const Node* AsExactExprNode(const Expr& expr) noexcept {
    if (!expr.defined() || expr.get()->GetTypeId() != Node::_type_index ||
        typeid(*expr.get()) != typeid(Node)) {
        return nullptr;
    }
    return expr.As<Node>();
}

}  // namespace

GraphSemanticKey::GraphSemanticKey(std::string canonical_bytes)
    : canonical_bytes_(std::make_shared<const std::string>(std::move(canonical_bytes))) {
    RequireNonEmpty(*canonical_bytes_, "graph semantic canonical bytes");
    digest_ = Digest(*canonical_bytes_, {});
}

bool GraphSemanticKey::defined() const noexcept {
    return canonical_bytes_ && !canonical_bytes_->empty() && !digest_.empty();
}

const std::string& GraphSemanticKey::canonical_bytes() const noexcept {
    static const std::string empty;
    return canonical_bytes_ ? *canonical_bytes_ : empty;
}

const std::string& GraphSemanticKey::digest() const noexcept {
    return digest_;
}

bool GraphSemanticKey::operator==(
    const GraphSemanticKey& other) const noexcept {
    return canonical_bytes_ == other.canonical_bytes_ || canonical_bytes() == other.canonical_bytes();
}

bool GraphSemanticKey::operator!=(
    const GraphSemanticKey& other) const noexcept {
    return !(*this == other);
}

bool GraphSemanticKey::operator<(
    const GraphSemanticKey& other) const noexcept {
    return canonical_bytes() < other.canonical_bytes();
}

GraphSemanticKey internal::IdentityAccess::Graph(
    std::string canonical_bytes) {
    return GraphSemanticKey(std::move(canonical_bytes));
}

GraphSemanticKey internal::BuildGraphSemanticKey(const Function& function) {
    if (!AsExactExprNode<FunctionNode>(function)) {
        throw std::invalid_argument(
            "graph semantic identity requires an exact Function node");
    }
    std::string canonical;
    AppendField(&canonical, "kind", "graph-semantic-key-v5-canonical-device");
    std::unordered_map<const Object*, size_t> node_ids;
    std::function<void(const Expr&)> visit = [&](const Expr& expr) {
        if (!expr.defined()) {
            throw std::invalid_argument(
                "graph semantic identity has an undefined Relay Expr");
        }
        const auto existing = node_ids.find(expr.get());
        if (existing != node_ids.end()) {
            AppendField(&canonical, "node_ref",
                        std::to_string(existing->second));
            return;
        }
        const size_t node_id = node_ids.size();
        node_ids.emplace(expr.get(), node_id);
        AppendField(&canonical, "node_id", std::to_string(node_id));
        if (const auto* relay_node =
                dynamic_cast<const RelayNode*>(expr.get())) {
            AppendField(&canonical, "virtual_device",
                        relay::SerializeVirtualDeviceLogicalPlacement(
                            relay_node->virtual_device_));
        }
        if (const auto* constant = AsExactExprNode<ConstantNode>(expr)) {
            AppendField(&canonical, "node_kind", "constant");
            if (!constant->data.defined()) {
                throw std::invalid_argument(
                    "graph semantic identity has an undefined Constant");
            }
            const DLDataType dtype = constant->data.dtype();
            AppendInteger(&canonical, "constant_dtype_code", dtype.code);
            AppendInteger(&canonical, "constant_dtype_bits", dtype.bits);
            AppendInteger(&canonical, "constant_dtype_lanes", dtype.lanes);
            for (int64_t dimension : constant->data.shape()) {
                AppendInteger(&canonical, "constant_dimension", dimension);
            }
            AppendInteger(&canonical, "constant_device_type",
                          static_cast<int>(constant->data.device().device_type()));
            AppendInteger(&canonical, "constant_device_id",
                          constant->data.device().device_id());
            std::string bytes(constant->data.NBytes(), '\0');
            if (!bytes.empty()) {
                constant->data.CopyToBytes(bytes.data(), bytes.size());
            }
            AppendField(&canonical, "constant_bytes", bytes);
            return;
        }
        if (const auto* variable = AsExactExprNode<VarNode>(expr)) {
            AppendField(&canonical, "node_kind", "var");
            AppendField(&canonical, "type_annotation",
                        TypeToString(variable->type_annotation));
            return;
        }
        if (const auto* op = AsExactExprNode<relay::OpNode>(expr)) {
            AppendField(&canonical, "node_kind", "op");
            AppendField(&canonical, "op_name", op->name);
            AppendField(&canonical, "op_spec",
                        op->has_spec
                            ? relay::SerializeOperatorSpec(op->spec)
                            : "<unspecified>");
            return;
        }
        if (const auto* call = AsExactExprNode<CallNode>(expr)) {
            AppendField(&canonical, "node_kind", "call");
            AppendField(
                &canonical, "call_attrs",
                call->attrs.defined()
                    ? relay::SerializeAttrs(relay::Attrs(call->attrs))
                    : "<none>");
            visit(call->op);
            for (const Expr& argument : call->args) visit(argument);
            return;
        }
        if (const auto* function_node =
                AsExactExprNode<FunctionNode>(expr)) {
            AppendField(&canonical, "node_kind", "function");
            for (const Var& parameter : function_node->params) {
                visit(parameter);
            }
            visit(function_node->body);
            return;
        }
        if (const auto* branch = AsExactExprNode<IfNode>(expr)) {
            AppendField(&canonical, "node_kind", "if");
            visit(branch->cond);
            visit(branch->true_branch);
            visit(branch->false_branch);
            return;
        }
        if (const auto* loop = AsExactExprNode<WhileNode>(expr)) {
            AppendField(&canonical, "node_kind", "while");
            AppendInteger(&canonical, "max_trip_count",
                          loop->max_trip_count);
            visit(loop->initial_state);
            visit(loop->loop_var);
            visit(loop->condition);
            visit(loop->body);
            return;
        }
        if (const auto* let = AsExactExprNode<LetNode>(expr)) {
            AppendField(&canonical, "node_kind", "let");
            visit(let->var);
            visit(let->value);
            visit(let->body);
            return;
        }
        if (const auto* tuple = AsExactExprNode<TupleNode>(expr)) {
            AppendField(&canonical, "node_kind", "tuple");
            for (const Expr& field : tuple->fields) visit(field);
            return;
        }
        if (const auto* item = AsExactExprNode<TupleGetItemNode>(expr)) {
            AppendField(&canonical, "node_kind", "tuple_get");
            AppendInteger(&canonical, "tuple_index", item->index);
            visit(item->tuple);
            return;
        }
        throw std::invalid_argument(
            "graph semantic identity does not support Relay node type '" +
            std::string(expr.get()->GetTypeKey()) + "'");
    };
    visit(function);
    return internal::IdentityAccess::Graph(std::move(canonical));
}

UnitSemanticKey::UnitSemanticKey(std::string canonical_bytes)
    : UnitSemanticKey(std::move(canonical_bytes), {}) {}

UnitSemanticKey::UnitSemanticKey(std::string canonical_bytes,
                                 std::string index_digest)
    : canonical_bytes_(std::move(canonical_bytes)) {
    RequireNonEmpty(canonical_bytes_, "unit semantic canonical bytes");
    digest_ = Digest(canonical_bytes_, std::move(index_digest));
}

bool UnitSemanticKey::defined() const noexcept {
    return !canonical_bytes_.empty() && !digest_.empty();
}

const std::string& UnitSemanticKey::canonical_bytes() const noexcept {
    return canonical_bytes_;
}

const std::string& UnitSemanticKey::digest() const noexcept { return digest_; }

bool UnitSemanticKey::operator==(const UnitSemanticKey& other) const noexcept {
    return canonical_bytes_ == other.canonical_bytes_;
}

bool UnitSemanticKey::operator!=(const UnitSemanticKey& other) const noexcept {
    return !(*this == other);
}

bool UnitSemanticKey::operator<(const UnitSemanticKey& other) const noexcept {
    return canonical_bytes_ < other.canonical_bytes_;
}

PrimitiveArtifactKey::PrimitiveArtifactKey(
    UnitSemanticKey unit_semantic_key,
    std::string target_capability_fingerprint,
    std::string pipeline_fingerprint, int abi_version,
    std::string schedule_contract,
    std::string backend_version)
    : PrimitiveArtifactKey(
          std::move(unit_semantic_key),
          std::move(target_capability_fingerprint),
          std::move(pipeline_fingerprint), abi_version,
          std::move(schedule_contract), std::move(backend_version), {}) {}

PrimitiveArtifactKey::PrimitiveArtifactKey(
    UnitSemanticKey unit_semantic_key,
    std::string target_capability_fingerprint,
    std::string pipeline_fingerprint, int abi_version,
    std::string schedule_contract,
    std::string backend_version,
    std::string index_digest)
    : unit_semantic_key_(std::move(unit_semantic_key)),
      target_capability_fingerprint_(std::move(target_capability_fingerprint)) {
    if (!unit_semantic_key_.defined()) {
        throw std::invalid_argument(
            "artifact identity requires a unit semantic key");
    }
    RequireNonEmpty(target_capability_fingerprint_, "target fingerprint");
    RequireNonEmpty(pipeline_fingerprint, "pipeline fingerprint");
    RequireNonEmpty(schedule_contract, "schedule contract");
    RequireNonEmpty(backend_version, "backend version");
    if (abi_version <= 0) {
        throw std::invalid_argument(
            "artifact identity requires a positive ABI version");
    }
    AppendField(&canonical_bytes_, "kind", "primitive-artifact-key-v4");
    AppendField(&canonical_bytes_, "unit_semantic",
                unit_semantic_key_.canonical_bytes());
    AppendField(&canonical_bytes_, "target",
                target_capability_fingerprint_);
    AppendField(&canonical_bytes_, "pipeline", pipeline_fingerprint);
    AppendField(&canonical_bytes_, "abi", std::to_string(abi_version));
    AppendField(&canonical_bytes_, "te_candidate", schedule_contract);
    AppendField(&canonical_bytes_, "backend", backend_version);
    digest_ = Digest(canonical_bytes_, std::move(index_digest));
}

bool PrimitiveArtifactKey::defined() const noexcept {
    return unit_semantic_key_.defined() && !canonical_bytes_.empty() &&
           !digest_.empty();
}

const UnitSemanticKey& PrimitiveArtifactKey::unit_semantic_key() const noexcept {
    return unit_semantic_key_;
}

const std::string& PrimitiveArtifactKey::target_capability_fingerprint() const noexcept {
    return target_capability_fingerprint_;
}

const std::string& PrimitiveArtifactKey::canonical_bytes() const noexcept {
    return canonical_bytes_;
}

const std::string& PrimitiveArtifactKey::digest() const noexcept {
    return digest_;
}

bool PrimitiveArtifactKey::operator==(
    const PrimitiveArtifactKey& other) const noexcept {
    return canonical_bytes_ == other.canonical_bytes_;
}

bool PrimitiveArtifactKey::operator!=(
    const PrimitiveArtifactKey& other) const noexcept {
    return !(*this == other);
}

bool PrimitiveArtifactKey::operator<(
    const PrimitiveArtifactKey& other) const noexcept {
    return canonical_bytes_ < other.canonical_bytes_;
}


}  // namespace kxc::api
