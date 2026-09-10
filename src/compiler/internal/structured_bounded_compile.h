/*! \file src/compiler/internal/structured_bounded_compile.h
 * \brief Region-aware bounded admission for structured control graphs.
 *
 * PR3 of the M10 C3 plan. Builds a dynamic fresh-output ExecutablePlan with an
 * optional structured_schedule directly from a control-capable representative,
 * reusing the restricted shape resolver for symbolic proofs and the control
 * lowerer for units/regions. It does not go through BuildValueGraph, which
 * rejects If/While.
 */
#pragma once

#include <vector>

#include "dynamic_shape_contract.h"
#include "compilation_unit.h"
#include "kxc/compiler/compile_config.h"
#include "kxc/compiler/identity.h"
#include "kxc/compiler/restricted_symbolic_shape.h"
#include "kxc/runtime/executable_plan.h"

namespace kxc::api::internal {

/*! \brief Region-aware bounded preparation result. */
class StructuredBoundedCompilation final {
public:
    StructuredBoundedCompilation(
        PartitionedGraph partitioned_graph,
        std::vector<DynamicUnitShapeContract> unit_shape_contracts,
        std::vector<runtime::GraphInputAxisGuard> graph_input_guards,
        runtime::StructuredSchedule schedule, GraphSemanticKey graph_semantic_key,
        CompileConfig config);

    [[nodiscard]] const PartitionedGraph& partitioned_graph() const noexcept;
    [[nodiscard]] const std::vector<DynamicUnitShapeContract>&
    unit_shape_contracts() const noexcept;
    [[nodiscard]] const std::vector<runtime::GraphInputAxisGuard>&
    graph_input_guards() const noexcept;
    [[nodiscard]] const runtime::StructuredSchedule& schedule() const noexcept;
    [[nodiscard]] const GraphSemanticKey& graph_semantic_key() const noexcept;
    [[nodiscard]] const CompileConfig& config() const noexcept;

private:
    PartitionedGraph partitioned_graph_;
    std::vector<DynamicUnitShapeContract> unit_shape_contracts_;
    std::vector<runtime::GraphInputAxisGuard> graph_input_guards_;
    runtime::StructuredSchedule schedule_;
    GraphSemanticKey graph_semantic_key_;
    CompileConfig config_;
};

/*! \brief Prepare a bounded structured compilation from a representative with
 *  residual If/While. Fails closed on any unsupported shape or topology. */
[[nodiscard]] StructuredBoundedCompilation PrepareStructuredBoundedCompile(
    Function representative, CompileConfig config,
    const std::vector<
        kxc::api::experimental::restricted_symbolic_shape::v1::InputAxisSymbol>&
        input_axis_symbols);

/*! \brief Build the dynamic fresh-output plan (with structured_schedule) for a
 *  structured bounded preparation. Mirrors BuildDynamicExecutablePlan. */
[[nodiscard]] runtime::ExecutablePlan BuildStructuredBoundedExecutablePlan(
    const StructuredBoundedCompilation& compilation);

}  // namespace kxc::api::internal
