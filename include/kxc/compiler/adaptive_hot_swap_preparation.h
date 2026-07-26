/*! \file include/kxc/compiler/adaptive_hot_swap_preparation.h
 * \brief Internal preparation contracts for adaptive hot-swap.
 *
 * This header is enabled only with the hot-swap lifecycle. It prepares and validates
 * static-exact candidates; routing, generation, leases, health, quarantine,
 * and execution remain owned by the hot-swap controller.
 */
#pragma once

#ifndef KXC_ENABLE_ADAPTIVE_HOT_SWAP
#define KXC_ENABLE_ADAPTIVE_HOT_SWAP 0
#endif

#if KXC_ENABLE_ADAPTIVE_HOT_SWAP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kxc/compiler/compiler.h"
#include "kxc/compiler/experimental_identity.h"
#include "kxc/runtime/session.h"

namespace kxc::api::adaptive::hot_swap::preparation {

inline constexpr uint32_t kAdaptivePreparationContractVersion = 1;

// Immutable static-exact compiler input. Construction snapshots config and
// verifies the baseline callable contract; Validate rechecks that snapshot.
class ProductionCompileRequest final {
public:
    ProductionCompileRequest(Function graph, CompileConfig config,
                             CompiledGraph baseline_graph,
                             std::vector<std::int64_t> requested_unit_ids);

    const Function& graph() const noexcept;
    CompileConfig config() const;
    const GraphSemanticKey& graph_semantic_key() const noexcept;
    const ShapeProfileKey& shape_profile_key() const noexcept;
    const DispatchKey& dispatch_key() const noexcept;
    const PlanAbiFingerprint& plan_abi() const noexcept;
    const std::vector<OrderedArtifactIdentity>& ordered_artifacts() const noexcept;
    const CompiledGraph& baseline_graph() const noexcept;
    const std::vector<std::int64_t>& requested_unit_ids() const noexcept;
    void Validate() const;

private:
    Function graph_;
    CompileConfig config_;
    GraphSemanticKey graph_semantic_key_;
    ShapeProfileKey shape_profile_key_;
    DispatchKey dispatch_key_;
    PlanAbiFingerprint plan_abi_;
    std::vector<OrderedArtifactIdentity> ordered_artifacts_;
    CompiledGraph baseline_graph_;
    std::vector<std::int64_t> requested_unit_ids_;
};

// Exact data-plane lookup input. It cannot carry a graph or trigger compile.
class ProductionExecutionRequest final {
public:
    ProductionExecutionRequest(DispatchKey dispatch_key,
                               PlanAbiFingerprint plan_abi);

    const DispatchKey& dispatch_key() const noexcept;
    const PlanAbiFingerprint& plan_abi() const noexcept;
    void Validate() const;

private:
    DispatchKey dispatch_key_;
    PlanAbiFingerprint plan_abi_;
};

// Immutable, structurally validated result ready for a lifecycle authority.
// It owns the compiled graph, its pins, and the corresponding static session.
class PreparedCandidate final {
public:
    const CompiledGraph& compiled_graph() const noexcept;
    const std::shared_ptr<const runtime::RuntimeSession>& session() const noexcept;
    const PlanVariantKey& selection_plan_key() const noexcept;
    const std::vector<OrderedArtifactIdentity>& selected_artifacts() const noexcept;
    const std::string& validation_receipt() const noexcept;

private:
    PreparedCandidate(CompiledGraph graph,
                      std::shared_ptr<const runtime::RuntimeSession> session,
                      PlanVariantKey selection_plan_key,
                      std::vector<OrderedArtifactIdentity> selected_artifacts,
                      std::string validation_receipt);
    CompiledGraph graph_;
    std::shared_ptr<const runtime::RuntimeSession> session_;
    PlanVariantKey selection_plan_key_;
    std::vector<OrderedArtifactIdentity> selected_artifacts_;
    std::string validation_receipt_;
    friend std::shared_ptr<const PreparedCandidate> PrepareCandidate(
        const ProductionCompileRequest&, CompiledGraph, std::string);
};

// Validates graph structure and exact request compatibility, then creates the
// pinned session. It has no generation, routing, publication, or lease effect.
std::shared_ptr<const PreparedCandidate> PrepareCandidate(
    const ProductionCompileRequest& request, CompiledGraph graph,
    std::string validation_receipt);

}  // namespace kxc::api::adaptive::hot_swap::preparation

#endif  // KXC_ENABLE_ADAPTIVE_HOT_SWAP
