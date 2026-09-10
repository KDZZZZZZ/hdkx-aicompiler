/*! \file include/kxc/compiler/restricted_symbolic_shape.h
 * \brief Default-off restricted symbolic Shape compile-admission control plane.
 *
 * Mints validated exact decisions and bounded compile admission only. Does not
 * allocate, compile/cache, execute, or lower control flow. Requires the exact
 * representative gate.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "kxc/compiler/compiler.h"
#include "kxc/compiler/experimental_identity.h"
#include "kxc/compiler/shape_exact.h"
#include "kxc/compiler/shape_specialization.h"

// =============================================================================
// 轨 02 — 受限符号 Shape 控制面（default-OFF）
// -----------------------------------------------------------------------------
// 只铸造 exact dispatch 决策或 bounded compile admission，不执行。
// 典型用法：
//   auto prep = RestrictedSymbolicShapeAdapter::Prepare(fn, cfg, {
//       InputAxisSymbol{0, 0, "N", 1, 1024, 1}});
//   auto d = RestrictedSymbolicShapeAdapter::MintExact(prep, bindings);
//   // d.exact_requests() → 交给后续 compiler/cache（本 API 不调用它们）
// CMake：KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE（且依赖 exact gate）
// 子集：固定 rank relu/sqrt/cast、可证明广播的 add/mul/divide/pow 与
// rank >= 2 matmul、transpose、静态归约轴 reduce_mean、正归约域 softmax/masked_softmax
// 和受限 shape 值链；静态常量、共享 DAG、重复实参及非空 tuple 结果。
// 无法证明的广播、控制流、动态常量轴及直接返回常量/重复输出 fail closed。
// =============================================================================
namespace kxc::api::experimental::restricted_symbolic_shape::v1 {

namespace specialization =
    kxc::api::experimental::shape_specialization::v1;

// 由私有 M3 shape-value resolver 定义的编码三元组；仅前向声明。
namespace shape_resolution {
struct EncodedExpr;
}

inline constexpr uint32_t kRestrictedSymbolicShapeVersion = 1;
inline constexpr uint32_t kBoundedCompileApplicabilityVersion = 14;

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

// Adapter-minted immutable bounded compilation authority.  The representative
// is a detached Relay snapshot; logical_boundary_function() is the same
// fixed-rank graph with direct symbolic axes represented as -1.  A caller
// cannot construct this type from a bare Function or BindingSet.
class BoundedCompileRequest final {
public:
    ~BoundedCompileRequest();
    BoundedCompileRequest(const BoundedCompileRequest&);
    BoundedCompileRequest& operator=(const BoundedCompileRequest&);
    BoundedCompileRequest(BoundedCompileRequest&&) noexcept;
    BoundedCompileRequest& operator=(BoundedCompileRequest&&) noexcept;

    [[nodiscard]] Function representative() const;
    [[nodiscard]] Function logical_boundary_function() const;
    [[nodiscard]] const specialization::GraphTemplate& graph_template() const;
    [[nodiscard]] const specialization::ExactOracle& representative_oracle() const;
    // 与模板 unit 顺序平行的形状值 value 表达式覆盖（编码三元组，相对
    // 单元局部输入 0）；nullopt 表示由 unit 合同推导。
    [[nodiscard]] const std::vector<
        std::optional<shape_resolution::EncodedExpr>>&
    unit_value_expressions() const;
    [[nodiscard]] const CompileConfig& compile_config() const;
    [[nodiscard]] const Target& target() const;
    [[nodiscard]] uint32_t applicability_version() const noexcept;

    struct Impl;

private:
    explicit BoundedCompileRequest(std::shared_ptr<const Impl> impl);

    std::shared_ptr<const Impl> impl_;
    friend class RestrictedSymbolicShapeAdapter;
};

class RestrictedSymbolicShapeAdapter final {
public:
    static bool IsEnabled() noexcept;

    static PreparedRestrictedSymbolicTemplate Prepare(
        Function representative, CompileConfig config,
        std::vector<InputAxisSymbol> input_axis_symbols,
        // Optional declared outputs from an imported representative. Checked
        // after shape-chain resolution and before compiler/cache preparation.
        std::vector<TensorType> representative_outputs = {});

    static RestrictedDispatchDecision MintExact(
        const PreparedRestrictedSymbolicTemplate& prepared,
        const specialization::BindingSet& bindings);

    // 铸造 bounded compilation admission。请求绑定 detached representative、
    // GraphTemplate、representative exact proof、CompileConfig/Target snapshot
    // 与 applicability version；不 lower、不编译 backend、不创建 route。
    static BoundedCompileRequest MintBoundedCompileRequest(
        const PreparedRestrictedSymbolicTemplate& prepared);

    // 语义 artifact/boundary 变化的 unit 下标；不是 cache 比较。
    static std::vector<size_t> ChangedUnitIndices(
        const RestrictedDispatchDecision& previous,
        const RestrictedDispatchDecision& next);

    // 按决策把 representative 物化为可编译的 concrete Function。
    // 重注类型到 detached representative 快照，保留共享调用、实参与
    // tuple 输出顺序；参数类型携带决策求值后的 concrete shape 与原 dtype。
    // 不编译、不缓存、不执行；
    // 编译意图必须由调用方显式调用 Compiler::Compile。决策不属于该模板时
    // fail closed。
    static Function MaterializeExactFunction(
        const PreparedRestrictedSymbolicTemplate& prepared,
        const RestrictedDispatchDecision& decision);

    // 铸造该决策的 static-exact route identity：
    // BuildStaticExactDispatchKey(模板 semantic key, 决策 profile key)。
    // 同一模板下的所有决策构成同一 route family。键属于 oracle 键空间；
    // 与 adaptive plan 派生键的对齐是独立计划（见 issue #46）。
    static DispatchKey ExactDispatchKey(
        const PreparedRestrictedSymbolicTemplate& prepared,
        const RestrictedDispatchDecision& decision);

    // 验证编译产物与决策边界契约一致（plan I/O shape/dtype、unit 数量）。
    // 决策授权 route identity，编译产物提供 artifact，两者由本函数绑定。
    // 同时绑定物化 Function 的 concrete semantic key（与 family key 不同），
    // 拒绝相同边界但算子、接线或结果顺序不同的产物。不匹配抛出。
    static void VerifyCompiledExactVariant(
        const PreparedRestrictedSymbolicTemplate& prepared,
        const RestrictedDispatchDecision& decision,
        const CompiledGraph& compiled);

    // 把实际输入 shape 换算成 canonical BindingSet。冲突的共享 symbol、
    // 输入个数/rank 不匹配、非 overlay 静态轴不一致均 fail closed；
    // 范围/整除约束由随后的 MintExact 检查。
    static specialization::BindingSet BindingsFromInputShapes(
        const PreparedRestrictedSymbolicTemplate& prepared,
        const std::vector<std::vector<int64_t>>& input_shapes);
};

}  // namespace kxc::api::experimental::restricted_symbolic_shape::v1
