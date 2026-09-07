/*! \file src/compiler/internal/dynamic_shape_contract.h
 * \brief Compiler-only bounded Relay admission and unit-local shape contract.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "compilation_unit.h"
#include "kxc/compiler/restricted_symbolic_shape.h"

namespace kxc::api {
class ModuleInvocationContract;
}

namespace kxc::api::internal {

namespace restricted =
    kxc::api::experimental::restricted_symbolic_shape::v1;
namespace specialization =
    kxc::api::experimental::shape_specialization::v1;
// 私有 M3 resolver 编码三元组（定义见 shape_value_resolver.h）。
// 注意：受限符号头已在其真实命名空间前向声明该类型，这里用别名引用；
// 不能在 kxc::api::internal 内重开限定命名空间，否则会产生
// kxc::api::internal::kxc::... 影子命名空间。
namespace shape_resolution =
    kxc::api::experimental::restricted_symbolic_shape::v1::shape_resolution;

inline constexpr std::uint32_t kBoundedDynamicGraphVersion = 1;
inline constexpr std::uint32_t kDynamicUnitShapeContractVersion = 3;
inline constexpr std::uint32_t kBoundedCompilePreparationVersion = 2;

struct DynamicInputAxisReference final {
    std::size_t input_index{0};
    std::size_t axis{0};
};

struct DynamicInputAxisGuard final {
    std::size_t axis{0};
    std::uint64_t lower{0};
    std::uint64_t upper{0};
    std::uint64_t divisible_by{1};
    std::optional<std::uint64_t> exact;
    std::optional<DynamicInputAxisReference> equal_to;
};

class DynamicShapeExpr final {
public:
    enum class Kind : std::uint8_t { kConst = 0, kInputAxis = 1 };

    static DynamicShapeExpr Const(std::uint64_t value);
    static DynamicShapeExpr InputAxis(std::size_t input_index,
                                      std::size_t axis);

    [[nodiscard]] Kind kind() const noexcept;
    [[nodiscard]] std::uint64_t constant() const;
    [[nodiscard]] std::size_t input_index() const;
    [[nodiscard]] std::size_t axis() const;
    [[nodiscard]] bool operator==(const DynamicShapeExpr& other) const noexcept;
    [[nodiscard]] bool operator!=(const DynamicShapeExpr& other) const noexcept {
        return !(*this == other);
    }

private:
    DynamicShapeExpr(Kind kind, std::uint64_t constant,
                     std::size_t input_index, std::size_t axis);

    Kind kind_;
    std::uint64_t constant_{0};
    std::size_t input_index_{0};
    std::size_t axis_{0};
};

/*! \brief One unit's sole bounded shape authority.
 *
 * Input guards are grouped in physical boundary-input order. Output
 * expressions reference only those local inputs; runtime extents are exactly
 * the ordered, de-duplicated dynamic expressions the unit consumes: first the
 * non-Const output shape expressions in output-axis order, then any remaining
 * dynamic boundary-input axes in input/axis order, then the non-Const shape
 * value element expressions of shape-producing outputs. Shape-value outputs
 * (M3 shape-as-value producers such as shape_of) additionally carry
 * output_value_expressions: one expression per element of the materialized
 * int64 vector, rooted at the same local input axes. Construction is
 * private so every instance comes from GraphTemplate + ordered UnitSkeleton.
 */
class DynamicUnitShapeContract final {
public:
    [[nodiscard]] std::uint32_t version() const noexcept;
    [[nodiscard]] const UnitSemanticKey& representative_unit_semantic_key()
        const noexcept;
    [[nodiscard]] const std::vector<std::vector<DynamicInputAxisGuard>>&
    local_input_guards() const noexcept;
    [[nodiscard]] const std::vector<std::vector<DynamicShapeExpr>>&
    output_shape_expressions() const noexcept;
    [[nodiscard]] const std::vector<std::vector<DynamicShapeExpr>>&
    output_value_expressions() const noexcept;
    [[nodiscard]] const std::vector<DynamicShapeExpr>&
    runtime_extent_expressions() const noexcept;
    [[nodiscard]] const std::string& canonical_bytes() const noexcept;

private:
    DynamicUnitShapeContract(
        UnitSemanticKey representative_unit_semantic_key,
        std::vector<std::vector<DynamicInputAxisGuard>> local_input_guards,
        std::vector<std::vector<DynamicShapeExpr>> output_shape_expressions,
        std::vector<std::vector<DynamicShapeExpr>> output_value_expressions,
        std::vector<DynamicShapeExpr> runtime_extent_expressions);

    std::uint32_t version_{kDynamicUnitShapeContractVersion};
    UnitSemanticKey representative_unit_semantic_key_;
    std::vector<std::vector<DynamicInputAxisGuard>> local_input_guards_;
    std::vector<std::vector<DynamicShapeExpr>> output_shape_expressions_;
    std::vector<std::vector<DynamicShapeExpr>> output_value_expressions_;
    std::vector<DynamicShapeExpr> runtime_extent_expressions_;
    std::string canonical_bytes_;

    friend DynamicUnitShapeContract BuildDynamicUnitShapeContract(
        const specialization::GraphTemplate&, std::size_t,
        const std::string&, const std::optional<std::vector<DynamicShapeExpr>>&);
};

[[nodiscard]] DynamicUnitShapeContract BuildDynamicUnitShapeContract(
    const specialization::GraphTemplate& graph_template,
    std::size_t ordered_unit_index, const std::string& operator_name,
    const std::optional<std::vector<DynamicShapeExpr>>&
        value_expression_override = std::nullopt);
[[nodiscard]] std::vector<DynamicUnitShapeContract>
BuildDynamicUnitShapeContracts(
    const specialization::GraphTemplate& graph_template,
    const std::vector<std::string>& operator_names,
    const std::vector<std::optional<std::vector<DynamicShapeExpr>>>&
        value_expression_overrides = {});

// 把受限解析器的编码三元组解码为合同层 DynamicShapeExpr 覆盖。
[[nodiscard]] std::vector<std::optional<std::vector<DynamicShapeExpr>>>
DecodeValueExpressionOverrides(
    const std::vector<std::optional<
        kxc::api::experimental::restricted_symbolic_shape::v1::
            shape_resolution::EncodedExpr>>& encoded);

/*! \brief Immutable pre-lowering result consumed by Compiler::CompileBounded.
 *
 * Preparation itself performs no TE/TIR, backend compilation, module
 * assembly, Runtime plan construction, or routing. Its partition carries
 * fixed-rank -1 boundaries only under adapter-minted authority.
 */
class BoundedCompilePreparation final {
public:
    [[nodiscard]] std::uint32_t version() const noexcept;
    [[nodiscard]] const restricted::BoundedCompileRequest& request() const
        noexcept;
    [[nodiscard]] const PartitionedGraph& partitioned_graph() const noexcept;
    [[nodiscard]] const std::vector<DynamicUnitShapeContract>&
    unit_shape_contracts() const noexcept;
    [[nodiscard]] const std::vector<runtime::GraphInputAxisGuard>&
    graph_input_guards() const noexcept;

private:
    BoundedCompilePreparation(
        restricted::BoundedCompileRequest request,
        PartitionedGraph partitioned_graph,
        std::vector<DynamicUnitShapeContract> unit_shape_contracts,
        std::vector<runtime::GraphInputAxisGuard> graph_input_guards);

    std::uint32_t version_{kBoundedCompilePreparationVersion};
    restricted::BoundedCompileRequest request_;
    PartitionedGraph partitioned_graph_;
    std::vector<DynamicUnitShapeContract> unit_shape_contracts_;
    std::vector<runtime::GraphInputAxisGuard> graph_input_guards_;

    friend BoundedCompilePreparation PrepareBoundedCompile(
        const restricted::BoundedCompileRequest&);
};

[[nodiscard]] BoundedCompilePreparation PrepareBoundedCompile(
    const restricted::BoundedCompileRequest& request);

[[nodiscard]] std::shared_ptr<const ModuleInvocationContract>
BuildDynamicModuleInvocationContract(
    const DynamicUnitShapeContract& shape_contract,
    const PrimitiveUnit& unit,
    const std::vector<LogicalValueContract>& values);

[[nodiscard]] runtime::ExecutablePlan BuildDynamicExecutablePlan(
    const BoundedCompilePreparation& preparation);

}  // namespace kxc::api::internal
