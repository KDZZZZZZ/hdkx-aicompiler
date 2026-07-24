/*! \file include/kxc/compiler/restricted_symbolic_shape.h
 * \brief Default-off restricted symbolic Shape exact-decision control plane.
 *
 * Mints validated exact decisions only. Does not allocate, compile/cache,
 * execute, or lower control flow. Requires the exact representative gate.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kxc/compiler/shape_exact.h"
#include "kxc/compiler/shape_specialization.h"

// =============================================================================
// 轨 02 W3 — 受限符号 Shape 控制面（default-OFF）
// -----------------------------------------------------------------------------
// 只铸造 exact dispatch 决策，不执行。
// 典型用法：
//   auto prep = RestrictedSymbolicShapeAdapter::Prepare(fn, cfg, {
//       InputAxisSymbol{0, 0, "N", 1, 1024, 1}});
//   auto d = RestrictedSymbolicShapeAdapter::MintExact(prep, bindings);
//   // d.exact_requests() → 交给后续 compiler/cache（本 API 不调用它们）
// CMake：KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE（且依赖 exact gate）
// 子集：固定 rank relu/sqrt、等形 add/mul；广播/控制流/常量等 fail closed
// =============================================================================
namespace kxc::api::experimental::restricted_symbolic_shape::v1 {

namespace specialization =
    kxc::api::experimental::shape_specialization::v1;

inline constexpr uint32_t kRestrictedSymbolicShapeVersion = 1;

struct InputAxisSymbol final {
    size_t parameter_index{0};
    size_t axis{0};
    std::string symbol;
    int64_t lower{0};
    int64_t upper{0};
    int64_t divisible_by{1};
};

// 不可变 exact 决策快照。返回的 request vector 不影响内部 authority。
class RestrictedDispatchDecision final {
public:
    [[nodiscard]] const specialization::ExactOracle& exact_oracle() const;
    [[nodiscard]] std::vector<specialization::UnitSpecializationRequest> exact_requests() const;
    [[nodiscard]] const specialization::GraphTemplate& graph_template() const;

    struct Impl;

private:
    explicit RestrictedDispatchDecision(std::shared_ptr<const Impl> impl);

    std::shared_ptr<const Impl> impl_;
    friend class RestrictedSymbolicShapeAdapter;
};

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
    const kxc::api::experimental::shape_exact::v1::PreparedGraphTemplate& representative() const;

private:
    struct Impl;
    explicit PreparedRestrictedSymbolicTemplate(std::shared_ptr<const Impl> impl);
    std::shared_ptr<const Impl> impl_;
    friend class RestrictedSymbolicShapeAdapter;
};

class RestrictedSymbolicShapeAdapter final {
public:
    static bool IsEnabled() noexcept;

    static PreparedRestrictedSymbolicTemplate Prepare(
        Function representative, CompileConfig config,
        std::vector<InputAxisSymbol> input_axis_symbols);

    static RestrictedDispatchDecision MintExact(
        const PreparedRestrictedSymbolicTemplate& prepared,
        const specialization::BindingSet& bindings);

    // 语义 artifact/boundary 变化的 unit 下标；不是 cache 比较。
    static std::vector<size_t> ChangedUnitIndices(
        const RestrictedDispatchDecision& previous,
        const RestrictedDispatchDecision& next);
};

}  // namespace kxc::api::experimental::restricted_symbolic_shape::v1
