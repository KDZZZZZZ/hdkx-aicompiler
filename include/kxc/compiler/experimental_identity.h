/*! \file include/kxc/compiler/experimental_identity.h
 * \brief Non-installed Shape/adaptive identity contracts.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "kxc/compiler/identity.h"

namespace kxc::runtime {
class ExecutablePlan;
}

namespace kxc::api {

class CompiledModule;

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
                    std::string canonical_bytes);

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
                std::string variant_policy_version);

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
    explicit PlanAbiFingerprint(std::string canonical_bytes);

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
                   std::string canonical_bytes);

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

}  // namespace kxc::api
