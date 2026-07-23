/*! \file include/kxc/compiler/shape_exact.h
 * \brief Default-off production bridge for shape experimental-v1 exact profiles.
 *
 * \warning The bridge executes the production compiler/cache path, but this
 * installed C++ surface remains experimental v1. Source and binary ABI
 * compatibility are not promised while it exposes experimental Shape types.
 */
#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "kxc/compiler/compiler.h"
#include "kxc/shape/specialization.h"

namespace kxc::api::experimental::shape_exact::v1 {

struct ShapeExactPreparationCounters final {
    size_t execution_contract_resolutions{0};
    size_t relay_graph_pipelines{0};
    size_t capability_boundary_checks{0};
    size_t value_graph_builds{0};
    size_t partitions{0};
};

class PreparedGraphTemplate final {
public:
    PreparedGraphTemplate();
    ~PreparedGraphTemplate();
    PreparedGraphTemplate(const PreparedGraphTemplate&);
    PreparedGraphTemplate& operator=(const PreparedGraphTemplate&);
    PreparedGraphTemplate(PreparedGraphTemplate&&) noexcept;
    PreparedGraphTemplate& operator=(PreparedGraphTemplate&&) noexcept;

    const shape::experimental::v1::GraphTemplate& graph_template() const;
    const ShapeExactPreparationCounters& counters() const;
    size_t unit_count() const;
    // Relay currently has concrete-only dimensions; non-empty bindings are
    // rejected rather than silently treated as another profile.
    bool multi_profile_supported() const noexcept;

private:
    struct Impl;
    explicit PreparedGraphTemplate(std::shared_ptr<const Impl> impl);
    std::shared_ptr<const Impl> impl_;
    friend class ProductionExactShapeAdapter;
};

class ExactPlanVariant final {
public:
    ExactPlanVariant();
    ~ExactPlanVariant();
    ExactPlanVariant(const ExactPlanVariant&);
    ExactPlanVariant& operator=(const ExactPlanVariant&);
    ExactPlanVariant(ExactPlanVariant&&) noexcept;
    ExactPlanVariant& operator=(ExactPlanVariant&&) noexcept;

    const CompiledModule& module() const;
    const runtime::ExecutablePlan& plan() const;
    const shape::experimental::v1::ShapeProfileKey& shape_profile_key() const;
    const shape::experimental::v1::PlanVariantKey& plan_variant_key() const;
    const std::vector<ArtifactPin>& artifact_pins() const;

private:
    struct Impl;
    explicit ExactPlanVariant(std::shared_ptr<const Impl> impl);
    std::shared_ptr<const Impl> impl_;
    friend class ProductionExactShapeAdapter;
};

class ProductionExactShapeAdapter final {
public:
    static bool IsEnabled() noexcept;
    static PreparedGraphTemplate PrepareGraphTemplate(Function function,
                                                       CompileConfig config);
    static shape::experimental::v1::ExactOracle InstantiateExactProfile(
        const PreparedGraphTemplate& prepared,
        const shape::experimental::v1::BindingSet& bindings);
    static ExactPlanVariant AssembleExactPlan(
        const PreparedGraphTemplate& prepared,
        const shape::experimental::v1::ExactOracle& oracle);
};

}  // namespace kxc::api::experimental::shape_exact::v1
