/*! \file src/compiler/internal/dynamic_shape_contract.h
 * \brief Compiler-only bounded Relay admission and unit-local shape contract.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "compilation_unit.h"
#include "kxc/compiler/restricted_symbolic_shape.h"

namespace kxc::api::internal {

namespace restricted =
    kxc::api::experimental::restricted_symbolic_shape::v1;
namespace specialization =
    kxc::api::experimental::shape_specialization::v1;

inline constexpr std::uint32_t kDynamicUnitShapeContractVersion = 1;
inline constexpr std::uint32_t kBoundedCompilePreparationVersion = 1;

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
 * Input guards are grouped in physical boundary-input order. Output and
 * runtime-extent expressions reference only those local inputs. Construction
 * is private so every instance comes from GraphTemplate + ordered UnitSkeleton.
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
    [[nodiscard]] const std::vector<DynamicShapeExpr>&
    runtime_extent_expressions() const noexcept;
    [[nodiscard]] const std::string& canonical_bytes() const noexcept;

private:
    DynamicUnitShapeContract(
        UnitSemanticKey representative_unit_semantic_key,
        std::vector<std::vector<DynamicInputAxisGuard>> local_input_guards,
        std::vector<std::vector<DynamicShapeExpr>> output_shape_expressions,
        std::vector<DynamicShapeExpr> runtime_extent_expressions);

    std::uint32_t version_{kDynamicUnitShapeContractVersion};
    UnitSemanticKey representative_unit_semantic_key_;
    std::vector<std::vector<DynamicInputAxisGuard>> local_input_guards_;
    std::vector<std::vector<DynamicShapeExpr>> output_shape_expressions_;
    std::vector<DynamicShapeExpr> runtime_extent_expressions_;
    std::string canonical_bytes_;

    friend DynamicUnitShapeContract BuildDynamicUnitShapeContract(
        const specialization::GraphTemplate&, std::size_t);
};

[[nodiscard]] DynamicUnitShapeContract BuildDynamicUnitShapeContract(
    const specialization::GraphTemplate& graph_template,
    std::size_t ordered_unit_index);
[[nodiscard]] std::vector<DynamicUnitShapeContract>
BuildDynamicUnitShapeContracts(
    const specialization::GraphTemplate& graph_template);

/*! \brief Immutable pre-lowering result consumed by future CompileBounded.
 *
 * No TE/TIR, backend compilation, module assembly, Runtime plan, or route is
 * produced here. The partition carries fixed-rank -1 logical boundaries only
 * because preparation requires an adapter-minted BoundedCompileRequest.
 */
class BoundedCompilePreparation final {
public:
    [[nodiscard]] std::uint32_t version() const noexcept;
    [[nodiscard]] const restricted::BoundedCompileRequest& request() const
        noexcept;
    [[nodiscard]] const PartitionedGraph& partitioned_graph() const noexcept;
    [[nodiscard]] const std::vector<DynamicUnitShapeContract>&
    unit_shape_contracts() const noexcept;

private:
    BoundedCompilePreparation(
        restricted::BoundedCompileRequest request,
        PartitionedGraph partitioned_graph,
        std::vector<DynamicUnitShapeContract> unit_shape_contracts);

    std::uint32_t version_{kBoundedCompilePreparationVersion};
    restricted::BoundedCompileRequest request_;
    PartitionedGraph partitioned_graph_;
    std::vector<DynamicUnitShapeContract> unit_shape_contracts_;

    friend BoundedCompilePreparation PrepareBoundedCompile(
        const restricted::BoundedCompileRequest&);
};

[[nodiscard]] BoundedCompilePreparation PrepareBoundedCompile(
    const restricted::BoundedCompileRequest& request);

}  // namespace kxc::api::internal
