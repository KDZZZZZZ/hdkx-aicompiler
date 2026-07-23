/*! \file include/kxc/compiler/adaptive_production_experimental.h
 * \brief Default-off experimental adapter for a production compiler path.
 *
 * This adapter is not production-ready: it has no cancellation, deadline,
 * negative-cache/retry authority, or automatic/one-shot health authority.
 */

#pragma once

#ifndef KXC_ENABLE_EXPERIMENTAL_ADAPTIVE_PRODUCTION
#define KXC_ENABLE_EXPERIMENTAL_ADAPTIVE_PRODUCTION 0
#endif

#if KXC_ENABLE_EXPERIMENTAL_ADAPTIVE_PRODUCTION

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "kxc/compiler/compiler.h"
#include "kxc/compiler/identity.h"
#include "kxc/runtime/session.h"

namespace kxc::api::adaptive::experimental::production_path {

inline constexpr uint32_t kExperimentalProductionPathContractVersion = 1;

/*! \brief Strong request for one whole-plan static-exact compilation.
 *
 * CompileConfig is deep-frozen. Relay Function nodes are shared immutable IR;
 * callers must not mutate them after construction or race mutation with use.
 */
class ProductionCompileRequest final {
public:
    /*! \brief Derives all canonical identities from graph/config and a typed baseline. */
    ProductionCompileRequest(Function graph, CompileConfig config,
                             const CompiledGraph& expected_contract);

    const Function& graph() const noexcept;
    /*! \brief Returns an independent deep copy of the frozen config snapshot. */
    CompileConfig config() const;
    const ArtifactKey& artifact_key() const noexcept;
    const DispatchKey& dispatch_key() const noexcept;
    const PlanAbiFingerprint& plan_abi() const noexcept;
    const std::vector<OrderedArtifactIdentity>& ordered_artifacts() const noexcept;
    /*! \brief Verified baseline pins that freeze launcher object identity. */
    const std::vector<ArtifactPin>& verified_artifact_pins() const noexcept;
    void Validate() const;

private:
    Function graph_;
    // Deep-cloned once; all subsequent compiler reads use this frozen snapshot.
    CompileConfig config_;
    ArtifactKey artifact_key_;
    DispatchKey dispatch_key_;
    PlanAbiFingerprint plan_abi_;
    std::vector<OrderedArtifactIdentity> ordered_artifacts_;
    std::vector<ArtifactPin> verified_artifact_pins_;
};

/*! \brief Exact routing request; it cannot carry a graph or trigger compilation. */
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

/*! \brief Injection seam for a separately hosted production compiler.
 *
 * Different exact keys may invoke Compile concurrently. Implementations must
 * therefore be thread-safe; same-key calls are singleflighted by the controller.
 */
class ProductionPathCompilerAdapter {
public:
    virtual ~ProductionPathCompilerAdapter() = default;
    virtual CompiledGraph Compile(const ProductionCompileRequest& request) = 0;
};

/*! \brief Generation-bound strong ownership of every production artifact pin. */
/*! \brief Prepare-only result: structurally validated and immutable, never routed.
 *
 * The receipt is supplied by the caller's validation authority.  It is selection
 * evidence, not Plan ABI compatibility.  `PrepareCandidate` deliberately has no
 * generation, route-head, history, or publication side effect.
 */
class FrozenPlanVariant;
class PreparedCandidate final {
public:
    const CompiledGraph& compiled_graph() const noexcept;
    const std::shared_ptr<const runtime::RuntimeSession>& session() const noexcept;
    const ArtifactKey& selection_artifact_key() const noexcept;
    const std::vector<OrderedArtifactIdentity>& selected_artifacts() const noexcept;
    const std::string& validation_receipt() const noexcept;

private:
    PreparedCandidate(CompiledGraph graph,
                      std::shared_ptr<const runtime::RuntimeSession> session,
                      ArtifactKey selection_artifact_key,
                      std::vector<OrderedArtifactIdentity> selected_artifacts,
                      std::string validation_receipt);
    CompiledGraph graph_;
    std::shared_ptr<const runtime::RuntimeSession> session_;
    ArtifactKey selection_artifact_key_;
    std::vector<OrderedArtifactIdentity> selected_artifacts_;
    std::string validation_receipt_;
    friend std::shared_ptr<const PreparedCandidate> PrepareCandidate(
        const ProductionCompileRequest&, CompiledGraph, std::string);
};

/*! \brief Structural/full typed preparation only; never publishes or mints a generation. */
std::shared_ptr<const PreparedCandidate> PrepareCandidate(
    const ProductionCompileRequest& request, CompiledGraph graph,
    std::string validation_receipt);

/*! \brief Materializes an immutable selected PlanVariant; no route mutation. */
std::shared_ptr<const FrozenPlanVariant> FreezePreparedCandidate(
    uint64_t generation, DispatchKey dispatch_key, PlanAbiFingerprint plan_abi,
    std::shared_ptr<const PreparedCandidate> candidate);

class ArtifactLease final {
public:
    ArtifactLease() = default;

    bool valid() const noexcept;
    uint64_t generation() const noexcept;
    const ArtifactKey& artifact_key() const noexcept;
    const std::vector<ArtifactPin>& pins() const noexcept;

private:
    ArtifactLease(uint64_t generation, ArtifactKey artifact_key,
                  std::vector<ArtifactPin> pins);

    uint64_t generation_{0};
    ArtifactKey artifact_key_;
    std::vector<ArtifactPin> pins_;
    friend class AdaptiveController;
    friend std::shared_ptr<const FrozenPlanVariant> FreezePreparedCandidate(
        uint64_t, DispatchKey, PlanAbiFingerprint,
        std::shared_ptr<const PreparedCandidate>);
};

/*! \brief Immutable whole CompiledModule+ExecutablePlan+session generation. */
class FrozenPlanVariant final {
public:
    uint64_t generation() const noexcept;
    const PlanVariantKey& key() const noexcept;
    const DispatchKey& dispatch_key() const noexcept;
    const PlanAbiFingerprint& plan_abi() const noexcept;
    const ArtifactLease& artifact_lease() const noexcept;
    const CompiledGraph& compiled_graph() const noexcept;
    const std::shared_ptr<const runtime::RuntimeSession>& session() const noexcept;

private:
    FrozenPlanVariant(uint64_t generation, PlanVariantKey key,
                      DispatchKey dispatch_key,
                      PlanAbiFingerprint plan_abi,
                      ArtifactLease artifact_lease,
                      CompiledGraph compiled_graph,
                      std::shared_ptr<const runtime::RuntimeSession> session);

    uint64_t generation_{0};
    PlanVariantKey key_;
    DispatchKey dispatch_key_;
    PlanAbiFingerprint plan_abi_;
    ArtifactLease artifact_lease_;
    CompiledGraph compiled_graph_;
    std::shared_ptr<const runtime::RuntimeSession> session_;
    friend class AdaptiveController;
    friend std::shared_ptr<const FrozenPlanVariant> FreezePreparedCandidate(
        uint64_t, DispatchKey, PlanAbiFingerprint,
        std::shared_ptr<const PreparedCandidate>);
};

/*! \brief Trusted control-plane administrative quarantine request.
 *
 * This is not runtime health proof or one-shot authority. The caller is
 * responsible for authentication, authorization, replay control, and evidence
 * provenance before invoking ForTrustedControlPlane.
 */
class AdministrativeQuarantineRequest final {
public:
    static AdministrativeQuarantineRequest ForTrustedControlPlane(
        std::shared_ptr<const FrozenPlanVariant> variant,
        std::string reason);

    const std::shared_ptr<const FrozenPlanVariant>& variant() const noexcept;
    const std::string& reason() const noexcept;

private:
    AdministrativeQuarantineRequest(
        std::shared_ptr<const FrozenPlanVariant> variant,
        std::string reason);

    std::shared_ptr<const FrozenPlanVariant> variant_;
    std::string reason_;
};

enum class AdaptiveControllerEventKind : uint8_t {
    kCompileStarted = 0,
    kCompileMerged = 1,
    kValidated = 2,
    kPublished = 3,
    kAcquired = 4,
    kRunSubmitted = 5,
    kRejected = 6,
    kQuarantined = 7,
    kRolledBack = 8,
};

/*! \brief Stable structured observability fields; digests are never equality. */
struct AdaptiveControllerEvent final {
    AdaptiveControllerEventKind kind{AdaptiveControllerEventKind::kCompileStarted};
    uint64_t generation{0};
    uint64_t predecessor_generation{0};
    size_t in_flight_compiles{0};
    std::string artifact_key_digest;
    std::string dispatch_key_digest;
    std::string plan_abi_digest;
    std::string plan_variant_digest;
    std::string diagnostic;
};

/*! \brief Synchronous best-effort observer.
 *
 * It runs outside controller locks and exceptions are ignored. While a callback
 * is active, every API entry on that controller fails fast from every thread,
 * including unrelated callers; defer work until the callback returns. Other
 * controller instances are unaffected.
 */
using AdaptiveControllerObserver =
    std::function<void(const AdaptiveControllerEvent&)>;

struct AdaptiveControllerOptions final {
    size_t max_in_flight_compiles{64};
    size_t max_slots{64};
    // Bounds discoverable controller history only, not external leases,
    // completion-held variants, executable/device bytes, or cache pins.
    size_t max_discoverable_generations{8};
    AdaptiveControllerObserver observer;
};

/*! \brief Result of an atomic quarantine/rollback routing handoff. */
struct AdaptiveHandoffResult final {
    bool changed{false};
    uint64_t generation{0};
    uint64_t predecessor_generation{0};
    std::string diagnostic;
};

struct AdaptiveControllerSnapshot final {
    uint64_t next_generation{1};
    uint64_t compile_requests{0};
    uint64_t merged_compiles{0};
    uint64_t published{0};
    uint64_t acquired{0};
    uint64_t run_requests{0};
    uint64_t rejected{0};
    uint64_t quarantined{0};
    // Controller history entries only; not total live variants/resident bytes.
    size_t discoverable_history_variants{0};
    size_t in_flight_compiles{0};
};

/*! \brief Completion handoff explicitly exposes both retained owners. */
struct AdaptiveRunAsyncResult final {
    Array<runtime::NDArray> outputs;
    AsyncOperation completion;
    ArtifactLease artifact_lease;
    std::shared_ptr<const FrozenPlanVariant> variant;
};

/*! \brief Upper-layer whole-plan controller; RuntimeSession stays data-plane only. */
class AdaptiveController final {
public:
    explicit AdaptiveController(
        std::shared_ptr<ProductionPathCompilerAdapter> compiler = nullptr,
        AdaptiveControllerOptions options = {});
    ~AdaptiveController();

    AdaptiveController(const AdaptiveController&) = delete;
    AdaptiveController& operator=(const AdaptiveController&) = delete;

    /*! \brief Compiles, validates, and atomically publishes one whole plan. */
    std::shared_ptr<const FrozenPlanVariant> CompileAndPublish(
        const ProductionCompileRequest& request);

    /*! \brief Acquires the exact current generation without compiling. */
    std::shared_ptr<const FrozenPlanVariant> Acquire(
        const ProductionExecutionRequest& request) const;

    /*! \brief Freezes one exact variant at entry and submits only that session. */
    AdaptiveRunAsyncResult RunAsync(
        const ProductionExecutionRequest& request,
        const Array<runtime::NDArray>& inputs,
        const DeviceStream& stream) const;

    /*! \brief Applies a trusted administrative quarantine and future rollback. */
    AdaptiveHandoffResult RollbackAdministrative(
        const AdministrativeQuarantineRequest& quarantine,
        uint64_t target_generation = 0);

    AdaptiveControllerSnapshot Snapshot() const;

private:
    class State;
    std::shared_ptr<State> state_;
};

}  // namespace kxc::api::adaptive::experimental::production_path

#endif  // KXC_ENABLE_EXPERIMENTAL_ADAPTIVE_PRODUCTION
