/*! \file include/kxc/compiler/shape_exact.h
 * \brief Default-off production bridge for shape experimental-v1 exact profiles.
 *
 * \warning Executes production compiler/cache path, but C++ surface remains
 * experimental v1 — no source/binary ABI compatibility promise.
 */
#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "kxc/compiler/compiler.h"
#include "kxc/compiler/shape_specialization.h"

// =============================================================================
// 轨 02 W2 — production exact 桥（default-OFF）
// -----------------------------------------------------------------------------
// 把 experimental Shape 接到真实 Compiler / primitive cache / RuntimeSession。
// 能力边界：concrete Relay + 单一 empty profile + 静态 session；非空绑定拒绝。
// 典型用法：
//   if (!ProductionExactShapeAdapter::IsEnabled()) { /* gate OFF */ }
//   auto prep = ProductionExactShapeAdapter::PrepareGraphTemplate(fn, cfg);
//   auto oracle = ProductionExactShapeAdapter::InstantiateExactProfile(
//       prep, shape::experimental::v1::BindingSet{});  // 仅 empty
//   auto plan = ProductionExactShapeAdapter::AssembleExactPlan(prep, oracle);
//   // plan.module() / plan.plan() / plan.artifact_pins() → RuntimeSession
// CMake：KXC_ENABLE_SHAPE_PRODUCTION_EXACT=ON
// =============================================================================
namespace kxc::api::experimental::shape_exact::v1 {

namespace specialization =
    kxc::api::experimental::shape_specialization::v1;

// ---------------------------------------------------------------------------
// ShapeExactPreparationCounters — 准备阶段观测计数
// 用法：读 prep.counters() 确认走了 pipeline / ValueGraph / partition 等步骤
// ---------------------------------------------------------------------------
struct ShapeExactPreparationCounters final {
    size_t execution_contract_resolutions{0};
    size_t relay_graph_pipelines{0};
    size_t capability_boundary_checks{0};
    size_t value_graph_builds{0};
    size_t partitions{0};
};

// ---------------------------------------------------------------------------
// PreparedGraphTemplate — 已冻结的图模板 + 计数
// 用法：仅由 PrepareGraphTemplate 得到；再交给 Instantiate / Assemble
// multi_profile_supported() 当前恒为 false（concrete-only）
// ---------------------------------------------------------------------------
class PreparedGraphTemplate final {
public:
    PreparedGraphTemplate();
    ~PreparedGraphTemplate();
    PreparedGraphTemplate(const PreparedGraphTemplate&);
    PreparedGraphTemplate& operator=(const PreparedGraphTemplate&);
    PreparedGraphTemplate(PreparedGraphTemplate&&) noexcept;
    PreparedGraphTemplate& operator=(PreparedGraphTemplate&&) noexcept;

    const specialization::GraphTemplate& graph_template() const;
    const ShapeExactPreparationCounters& counters() const;
    size_t unit_count() const;
    // 非空绑定会被 Instantiate 拒绝，不静默当另一 profile
    bool multi_profile_supported() const noexcept;

private:
    struct Impl;
    explicit PreparedGraphTemplate(std::shared_ptr<const Impl> impl);
    std::shared_ptr<const Impl> impl_;
    friend class ProductionExactShapeAdapter;
};

// ---------------------------------------------------------------------------
// ExactPlanVariant — 可执行 exact 变体（module + plan + pins）
// 用法：
//   const CompiledModule& m = variant.module();
//   const ExecutablePlan& p = variant.plan();
//   // pins 保活 artifact；session 只消费冻结对象
// ---------------------------------------------------------------------------
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
    const ShapeProfileKey& shape_profile_key() const;
    const PlanVariantKey& plan_variant_key() const;
    const std::vector<ArtifactPin>& artifact_pins() const;

private:
    struct Impl;
    explicit ExactPlanVariant(std::shared_ptr<const Impl> impl);
    std::shared_ptr<const Impl> impl_;
    friend class ProductionExactShapeAdapter;
};

// ---------------------------------------------------------------------------
// ProductionExactShapeAdapter — 三段式生产入口
// Prepare → InstantiateExactProfile → AssembleExactPlan
// ---------------------------------------------------------------------------
class ProductionExactShapeAdapter final {
public:
    // gate 是否打开（编译期宏 + 运行时可查）
    static bool IsEnabled() noexcept;
    // Relay Function → 走生产 compiler 准备 → 冻结 GraphTemplate
    static PreparedGraphTemplate PrepareGraphTemplate(Function function,
                                                       CompileConfig config);
    // 当前只接受 empty BindingSet；非空 fail closed
    static specialization::ExactOracle InstantiateExactProfile(
        const PreparedGraphTemplate& prepared,
        const specialization::BindingSet& bindings);
    // oracle → 真实 module/plan + production pin
    static ExactPlanVariant AssembleExactPlan(
        const PreparedGraphTemplate& prepared,
        const specialization::ExactOracle& oracle);
};

}  // namespace kxc::api::experimental::shape_exact::v1
