/*! \file include/kxc/compiler/identity.h
 * \brief Canonical, collision-safe identities shared by compiler control tracks.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace kxc {
class Function;
}

namespace kxc::runtime {
class ExecutablePlan;
}

namespace kxc::api {

class CompiledModule;

/*! \brief Whole-graph Relay semantics, independent of target and compiler policy. */
class GraphSemanticKey final {
public:
    GraphSemanticKey() = default;

    bool defined() const noexcept;
    const std::string& canonical_bytes() const noexcept;
    const std::string& digest() const noexcept;

    bool operator==(const GraphSemanticKey& other) const noexcept;
    bool operator!=(const GraphSemanticKey& other) const noexcept;
    bool operator<(const GraphSemanticKey& other) const noexcept;

private:
    explicit GraphSemanticKey(std::string canonical_bytes,
                              std::string index_digest = {});

    std::string canonical_bytes_;
    std::string digest_;

    friend GraphSemanticKey BuildGraphSemanticKey(const Function& function);
};

/*! \brief Canonicalizes one Relay graph without target or pipeline policy. */
GraphSemanticKey BuildGraphSemanticKey(const Function& function);

/*! \brief Plan-local graph value locator; never a semantic cache key. */
struct GraphValueLocator final {
    std::string graph_revision;
    int64_t value_id{-1};
    int64_t output_index{0};

    std::string CanonicalBytes() const;
};

/*! \brief Unit semantics independent of graph numbering, symbol, and storage. */
class UnitSemanticKey final {
public:
    UnitSemanticKey() = default;
    explicit UnitSemanticKey(std::string canonical_bytes,
                             std::string index_digest = {});

    bool defined() const noexcept;
    const std::string& canonical_bytes() const noexcept;
    const std::string& digest() const noexcept;

    bool operator==(const UnitSemanticKey& other) const noexcept;
    bool operator!=(const UnitSemanticKey& other) const noexcept;
    bool operator<(const UnitSemanticKey& other) const noexcept;

private:
    std::string canonical_bytes_;
    std::string digest_;
};

/*! \brief Ready primitive artifact identity, including target, pipeline, ABI and backend. */
class PrimitiveArtifactKey final {
public:
    PrimitiveArtifactKey() = default;
    PrimitiveArtifactKey(UnitSemanticKey unit_semantic_key,
                         std::string target_capability_fingerprint,
                         std::string pipeline_fingerprint, int abi_version,
                         std::string schedule_version,
                         std::string backend_version,
                         std::string index_digest = {});

    bool defined() const noexcept;
    const UnitSemanticKey& unit_semantic_key() const noexcept;
    const std::string& canonical_bytes() const noexcept;
    const std::string& digest() const noexcept;

    bool operator==(const PrimitiveArtifactKey& other) const noexcept;
    bool operator!=(const PrimitiveArtifactKey& other) const noexcept;
    bool operator<(const PrimitiveArtifactKey& other) const noexcept;

private:
    UnitSemanticKey unit_semantic_key_;
    std::string canonical_bytes_;
    std::string digest_;
};

/*! \brief One ordered plan-call to immutable primitive artifact mapping. */
struct OrderedArtifactIdentity final {
    size_t call_index{0};
    std::string link_symbol;
    PrimitiveArtifactKey artifact_key;

    std::string CanonicalBytes() const;
    bool operator==(const OrderedArtifactIdentity& other) const noexcept;
    bool operator!=(const OrderedArtifactIdentity& other) const noexcept;
};

/*! \brief Shape applicability for one graph, independent of target/backend identity. */
class ShapeProfileKey final {
public:
    ShapeProfileKey() = default;

    bool defined() const noexcept;
    const GraphSemanticKey& graph_semantic_key() const noexcept;
    const std::string& canonical_bytes() const noexcept;
    const std::string& digest() const noexcept;
    bool operator==(const ShapeProfileKey& other) const noexcept;
    bool operator!=(const ShapeProfileKey& other) const noexcept;
    bool operator<(const ShapeProfileKey& other) const noexcept;

private:
    ShapeProfileKey(GraphSemanticKey graph_semantic_key,
                    std::string canonical_bytes,
                    std::string index_digest = {});

    GraphSemanticKey graph_semantic_key_;
    std::string canonical_bytes_;
    std::string digest_;

    friend ShapeProfileKey BuildShapeProfileKey(
        const GraphSemanticKey&, const std::string&, const std::string&,
        const std::string&, uint32_t);
    friend ShapeProfileKey BuildStaticExactShapeProfileKey(
        const GraphSemanticKey&, const runtime::ExecutablePlan&);
};

/*! \brief Builds a profile from canonical shape program, bindings, and policy. */
ShapeProfileKey BuildShapeProfileKey(
    const GraphSemanticKey& graph_semantic_key,
    const std::string& shape_program_canonical,
    const std::string& bindings_canonical,
    const std::string& specialization_policy,
    uint32_t shape_abi_version);

/*! \brief Builds the static-exact input profile represented by a real plan. */
ShapeProfileKey BuildStaticExactShapeProfileKey(
    const GraphSemanticKey& graph_semantic_key,
    const runtime::ExecutablePlan& plan);

/*! \brief Shape/layout applicability identity, separate from primitive semantics. */
class DispatchKey final {
public:
    DispatchKey() = default;
    DispatchKey(std::string artifact_family,
                std::string shape_layout_valid_extent,
                std::string variant_policy_version,
                std::string index_digest = {});

    bool defined() const noexcept;
    const std::string& canonical_bytes() const noexcept;
    const std::string& digest() const noexcept;
    bool operator==(const DispatchKey& other) const noexcept;
    bool operator!=(const DispatchKey& other) const noexcept;
    bool operator<(const DispatchKey& other) const noexcept;

private:
    std::string canonical_bytes_;
    std::string digest_;
};

/*! \brief Opaque, versioned static-exact ABI identity derived from a real plan. */
class PlanAbiFingerprint final {
public:
    PlanAbiFingerprint() = default;

    bool defined() const noexcept;
    const std::string& canonical_bytes() const noexcept;
    const std::string& digest() const noexcept;
    bool operator==(const PlanAbiFingerprint& other) const noexcept;
    bool operator!=(const PlanAbiFingerprint& other) const noexcept;

private:
    explicit PlanAbiFingerprint(std::string canonical_bytes,
                                std::string index_digest = {});

    std::string canonical_bytes_;
    std::string digest_;

    friend PlanAbiFingerprint BuildPlanAbiFingerprint(
        const CompiledModule& module,
        const runtime::ExecutablePlan& plan,
        const std::vector<OrderedArtifactIdentity>& ordered_artifacts);
};

/*! \brief Canonicalizes module, plan, and ordered primitive artifact contracts. */
PlanAbiFingerprint BuildPlanAbiFingerprint(
    const CompiledModule& module,
    const runtime::ExecutablePlan& plan,
    const std::vector<OrderedArtifactIdentity>& ordered_artifacts);

/*! \brief Builds exact input applicability from a graph artifact family and plan. */
DispatchKey BuildStaticExactDispatchKey(
    const GraphSemanticKey& graph_semantic_key,
    const ShapeProfileKey& shape_profile_key);

/*! \brief One ordered primitive selection in a frozen plan variant. */
struct OrderedArtifactSelectionIdentity final {
    size_t call_index{0};
    std::string link_symbol;
    PrimitiveArtifactKey artifact_key;
    uint64_t generation{0};

    std::string CanonicalBytes() const;
};

/*! \brief Frozen graph/profile/artifact-generation and memory-plan identity. */
class PlanVariantKey final {
public:
    PlanVariantKey() = default;

    bool defined() const noexcept;
    const GraphSemanticKey& graph_semantic_key() const noexcept;
    const ShapeProfileKey& shape_profile_key() const noexcept;
    const std::string& canonical_bytes() const noexcept;
    const std::string& digest() const noexcept;
    bool operator==(const PlanVariantKey& other) const noexcept;
    bool operator!=(const PlanVariantKey& other) const noexcept;
    bool operator<(const PlanVariantKey& other) const noexcept;

private:
    PlanVariantKey(GraphSemanticKey graph_semantic_key,
                   ShapeProfileKey shape_profile_key,
                   std::string canonical_bytes,
                   std::string index_digest = {});

    GraphSemanticKey graph_semantic_key_;
    ShapeProfileKey shape_profile_key_;
    std::string canonical_bytes_;
    std::string digest_;

    friend PlanVariantKey BuildPlanVariantKey(
        const GraphSemanticKey&, const ShapeProfileKey&,
        const std::vector<OrderedArtifactSelectionIdentity>&,
        const std::string&);
};

/*! \brief Builds one whole-plan selection identity from formal component keys. */
PlanVariantKey BuildPlanVariantKey(
    const GraphSemanticKey& graph_semantic_key,
    const ShapeProfileKey& shape_profile_key,
    const std::vector<OrderedArtifactSelectionIdentity>& ordered_artifacts,
    const std::string& memory_plan_version);

/*! \brief Backend module lookup identity, explicitly not cache equivalence. */
struct LinkSymbol final {
    std::string value;

    std::string CanonicalBytes() const;
};

/*! \brief Physical allocation identity scoped to one frozen plan. */
struct StorageId final {
    int64_t value{-1};

    std::string CanonicalBytes() const;
};

}  // namespace kxc::api
