/*! \file include/kxc/compiler/restricted_symbolic_shape.h
 * \brief Default-off W3 restricted symbolic Shape control plane.
 *
 * Mints validated dispatch decisions only. Does not allocate, compile/cache,
 * execute, or lower control flow. Requires the exact representative gate.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kxc/compiler/guarded_shape_specialization.h"
#include "kxc/compiler/shape_exact.h"

// =============================================================================
// 轨 02 W3 — 受限符号 Shape 控制面（default-OFF）
// -----------------------------------------------------------------------------
// 只铸造 dispatch 决策（exact / bucket / polymorphic 契约），不执行。
// 典型用法：
//   auto prep = RestrictedSymbolicShapeAdapter::Prepare(fn, cfg, {
//       InputAxisSymbol{0, 0, "N", 1, 1024, 1}});
//   auto d = RestrictedSymbolicShapeAdapter::MintExact(prep, bindings);
//   // d.exact_requests() → 交给后续 compiler/cache（本 API 不调用它们）
//   auto changed = RestrictedSymbolicShapeAdapter::ChangedUnitIndices(prev, d);
// CMake：KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE（且依赖 exact gate）
// 子集：固定 rank relu/sqrt、等形 add/mul；广播/控制流/常量等 fail closed
// =============================================================================
namespace kxc::api::experimental::restricted_symbolic_shape::v1 {

namespace specialization =
    kxc::api::experimental::shape_specialization::v1;

inline constexpr uint32_t kRestrictedSymbolicShapeVersion = 1;

// ---------------------------------------------------------------------------
// InputAxisSymbol — 输入轴 ↔ 符号 + range/divisibility
// 用法：Prepare 时声明；Mint* 时 BindingSet 必须满足这些约束
// ---------------------------------------------------------------------------
struct InputAxisSymbol final {
    size_t parameter_index{0};  // 函数第几个参数
    size_t axis{0};             // 该参数第几轴
    std::string symbol;         // BindingSet 中的符号名
    int64_t lower{0};
    int64_t upper{0};
    int64_t divisible_by{1};
};

// exact：可走 production exact 特化；bucket/polymorphic：仅 guarded 契约（无 launch）
enum class DispatchKind { kExact, kBucket, kPolymorphic };

// ---------------------------------------------------------------------------
// RestrictedDispatchDecision — 不可变决策快照
// 用法：读 kind()；按 kind 取 exact_oracle/exact_requests 或 guarded_*
// 改返回的 vector 不影响内部 authority（每次重建快照）
// ---------------------------------------------------------------------------
class RestrictedDispatchDecision final {
public:
    [[nodiscard]] DispatchKind kind() const noexcept;
    [[nodiscard]] const specialization::ExactOracle& exact_oracle() const;
    [[nodiscard]] std::vector<specialization::UnitSpecializationRequest> exact_requests() const;
    [[nodiscard]] std::vector<specialization::GuardedUnitSpecializationRequest> guarded_requests() const;
    [[nodiscard]] const specialization::GraphTemplate& graph_template() const;
    // exact 决策时以下指针为空
    [[nodiscard]] const specialization::GuardedShapeProfile* guarded_profile() const;
    [[nodiscard]] const specialization::BucketPolicy* bucket_policy() const;
    [[nodiscard]] const specialization::PolymorphicPolicy* polymorphic_policy() const;

    struct Impl;

private:
    explicit RestrictedDispatchDecision(std::shared_ptr<const Impl> impl);

    std::shared_ptr<const Impl> impl_;
    friend class RestrictedSymbolicShapeAdapter;
};

// ---------------------------------------------------------------------------
// PreparedRestrictedSymbolicTemplate — Prepare 的产物
// 用法：持有 graph_template / 轴符号 / exact representative（仅 static fallback）
// ---------------------------------------------------------------------------
class PreparedRestrictedSymbolicTemplate final {
public:
    PreparedRestrictedSymbolicTemplate();
    ~PreparedRestrictedSymbolicTemplate();
    PreparedRestrictedSymbolicTemplate(const PreparedRestrictedSymbolicTemplate&);
    PreparedRestrictedSymbolicTemplate& operator=(const PreparedRestrictedSymbolicTemplate&);
    PreparedRestrictedSymbolicTemplate(PreparedRestrictedSymbolicTemplate&&) noexcept;
    PreparedRestrictedSymbolicTemplate& operator=(PreparedRestrictedSymbolicTemplate&&) noexcept;

    const specialization::GraphTemplate& graph_template() const;
    const std::vector<InputAxisSymbol>& input_axis_symbols() const;
    // 不是动态 RuntimeSession plan
    const kxc::api::experimental::shape_exact::v1::PreparedGraphTemplate& representative() const;

private:
    struct Impl;
    explicit PreparedRestrictedSymbolicTemplate(std::shared_ptr<const Impl> impl);
    std::shared_ptr<const Impl> impl_;
    friend class RestrictedSymbolicShapeAdapter;
};

// ---------------------------------------------------------------------------
// RestrictedSymbolicShapeAdapter — Prepare + Mint* + 差分
// ---------------------------------------------------------------------------
class RestrictedSymbolicShapeAdapter final {
public:
    static bool IsEnabled() noexcept;

    // 深冻结代表；不支持的 Relay 结构直接失败
    static PreparedRestrictedSymbolicTemplate Prepare(
        Function representative, CompileConfig config,
        std::vector<InputAxisSymbol> input_axis_symbols);

    // 在 lowering/cache 之前铸造完整请求；guard miss / 非法 proof → 抛错
    static RestrictedDispatchDecision MintExact(
        const PreparedRestrictedSymbolicTemplate& prepared,
        const specialization::BindingSet& bindings);
    static RestrictedDispatchDecision MintBucket(
        const PreparedRestrictedSymbolicTemplate& prepared,
        const specialization::BindingSet& bindings,
        const specialization::BucketPolicy& policy);
    static RestrictedDispatchDecision MintPolymorphic(
        const PreparedRestrictedSymbolicTemplate& prepared,
        const specialization::BindingSet& bindings,
        const specialization::PolymorphicPolicy& policy);

    // 语义 artifact/边界变化的 unit 下标；不是 cache 比较
    static std::vector<size_t> ChangedUnitIndices(
        const RestrictedDispatchDecision& previous,
        const RestrictedDispatchDecision& next);
};

}  // namespace kxc::api::experimental::restricted_symbolic_shape::v1
