/*! \file include/kxc/compiler/adaptive_production_experimental.h
 * \brief Default-off preparation contracts for production adaptive compilers.
 *
 * This header prepares and validates static-exact candidates. Routing,
 * generation, leases, health, quarantine, and execution belong to injected
 * control planes such as adaptive hot-swap v2.
 */
#pragma once

#ifndef KXC_ENABLE_EXPERIMENTAL_ADAPTIVE_PRODUCTION
#define KXC_ENABLE_EXPERIMENTAL_ADAPTIVE_PRODUCTION 0
#endif

#if KXC_ENABLE_EXPERIMENTAL_ADAPTIVE_PRODUCTION

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kxc/compiler/compiler.h"
#include "kxc/compiler/identity.h"
#include "kxc/runtime/session.h"

namespace kxc::api::adaptive::experimental::production_path {

inline constexpr uint32_t kAdaptivePreparationContractVersion = 1;

// Immutable static-exact compiler input. Construction snapshots config and
// verifies the baseline callable contract; Validate rechecks that snapshot.
class ProductionCompileRequest final {
public:
    ProductionCompileRequest(Function graph, CompileConfig config,
                             const CompiledGraph& expected_contract);

    const Function& graph() const noexcept;
    CompileConfig config() const;
    const GraphSemanticKey& graph_semantic_key() const noexcept;
    const ShapeProfileKey& shape_profile_key() const noexcept;
    const DispatchKey& dispatch_key() const noexcept;
    const PlanAbiFingerprint& plan_abi() const noexcept;
    const std::vector<OrderedArtifactIdentity>& ordered_artifacts() const noexcept;
    const std::vector<ArtifactPin>& verified_artifact_pins() const noexcept;
    void Validate() const;

private:
    Function graph_;
    CompileConfig config_;
    GraphSemanticKey graph_semantic_key_;
    ShapeProfileKey shape_profile_key_;
    DispatchKey dispatch_key_;
    PlanAbiFingerprint plan_abi_;
    std::vector<OrderedArtifactIdentity> ordered_artifacts_;
    std::vector<ArtifactPin> verified_artifact_pins_;
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

// Injection seam for a production compiler. Implementations must be
// thread-safe; preparation and all lifecycle authority are external.
class ProductionPathCompilerAdapter {
public:
    virtual ~ProductionPathCompilerAdapter() = default;
    virtual CompiledGraph Compile(const ProductionCompileRequest& request) = 0;
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

}  // namespace kxc::api::adaptive::experimental::production_path

#endif  // KXC_ENABLE_EXPERIMENTAL_ADAPTIVE_PRODUCTION
