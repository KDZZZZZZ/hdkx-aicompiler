/*! \file include/kxc/compiler/shape_control.h
 * \brief Caller-owned finite exact-profile routing control plane.
 *
 * This source-tree experimental API binds named direct symbolic input axes and
 * selects only caller-published CompiledGraphs. It never compiles, touches the
 * primitive cache, allocates runtime buffers, or executes.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "kxc/compiler/compiler.h"
#include "kxc/compiler/experimental_identity.h"
#include "kxc/compiler/shape_specialization.h"

namespace kxc::api::experimental::shape_control::v1 {

namespace specialization =
    kxc::api::experimental::shape_specialization::v1;

struct ConcreteInputShape final {
    std::string name;
    std::vector<int64_t> logical;
};

// Binds direct Symbol or Const input axes only, then evaluates the complete
// ShapeProgram. Arithmetic input expressions are intentionally not inverted.
[[nodiscard]] specialization::BindingSet BindExactInputShapes(
    const specialization::GraphTemplate& graph_template,
    const std::vector<ConcreteInputShape>& inputs);

// Immutable publication result. Returning this value copies the lightweight
// CompiledGraph handle, so its artifact pins remain alive independently of
// later table growth or destruction.
class PublishedExactVariant final {
public:
    [[nodiscard]] const CompiledGraph& compiled_graph() const noexcept;
    [[nodiscard]] const DispatchKey& dispatch_key() const noexcept;
    [[nodiscard]] const ShapeProfileKey& shape_profile_key() const noexcept;
    [[nodiscard]] const PlanVariantKey& plan_variant_key() const noexcept;
    [[nodiscard]] const PlanAbiFingerprint& plan_abi() const noexcept;

private:
    PublishedExactVariant(CompiledGraph compiled, DispatchKey dispatch,
                          ShapeProfileKey profile, PlanVariantKey variant,
                          PlanAbiFingerprint abi,
                          std::vector<ConcreteInputShape> inputs);

    CompiledGraph compiled_;
    DispatchKey dispatch_;
    ShapeProfileKey profile_;
    PlanVariantKey variant_;
    PlanAbiFingerprint abi_;
    std::vector<ConcreteInputShape> inputs_;

    friend class ExactProfileRouteTable;
};

// A finite, single-template, single-target set of exact profiles. The caller
// must validate compiled against the oracle with its producer-specific exact
// adapter before Publish. expected_plan_abi is checked against the canonical
// BuildPlanAbiFingerprint result. Lookup has no compile/cache side effect.
class ExactProfileRouteTable final {
public:
    ExactProfileRouteTable(
        specialization::GraphTemplate graph_template,
        std::string target_capability_fingerprint);

    void Publish(const specialization::ExactOracle& oracle,
                 CompiledGraph compiled,
                 PlanAbiFingerprint expected_plan_abi);
    [[nodiscard]] PublishedExactVariant Lookup(
        const specialization::ExactOracle& oracle) const;
    [[nodiscard]] std::size_t size() const noexcept;

private:
    specialization::GraphTemplate graph_template_;
    std::string target_capability_fingerprint_;
    std::vector<PublishedExactVariant> variants_;
};

}  // namespace kxc::api::experimental::shape_control::v1
