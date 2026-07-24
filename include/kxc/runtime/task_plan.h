/*! \file include/kxc/runtime/task_plan.h
 * \brief Frozen, runtime-neutral Region and single-stream task-DAG contracts.
 */

#pragma once

#include <cstdint>
#include <memory>

#include "kxc/runtime/compiled_module.h"
#include "kxc/runtime/executable_plan.h"

namespace kxc::runtime {

constexpr int32_t kFrozenTaskPlanVersion = 1;
constexpr int32_t kSelectedArtifactManifestVersion = 1;

/*! \brief Whether a selected artifact is bound to an ordered call or task. */
enum class ArtifactBindingKind : int32_t {
    kCall = 0,
    kTask = 1,
};

class SelectedArtifactBindingNode final : public Object {
public:
    ArtifactBindingKind kind{ArtifactBindingKind::kCall};
    int64_t invocation_id{-1};
    String artifact_identity;
    uint64_t generation{0};
    String exact_abi_fingerprint;
    String entry_symbol;
    String entry_binding;

    KXC_OBJECT_DECLARE
};

/*!
 * \brief Caller-supplied artifact observability and ABI/entry binding fields.
 *
 * Builder helpers derive ABI/entry fields; direct construction can supply them.
 * RuntimeSession recomputes structural consistency with its module and plan but
 * does not authenticate artifact_identity, generation, or their provenance.
 */
class SelectedArtifactBinding : public ObjectRef {
public:
    SelectedArtifactBinding(ArtifactBindingKind kind, int64_t invocation_id,
                            String artifact_identity, uint64_t generation,
                            String exact_abi_fingerprint, String entry_symbol,
                            String entry_binding);
    explicit SelectedArtifactBinding(const ObjectRef& ref);

    void Validate() const;
    const SelectedArtifactBindingNode* operator->() const;
};

class SelectedArtifactManifestNode final : public Object {
public:
    int32_t version{kSelectedArtifactManifestVersion};
    String plan_fingerprint;

    KXC_OBJECT_DECLARE

private:
    friend class SelectedArtifactManifest;
    Array<SelectedArtifactBinding> bindings_;
    std::shared_ptr<const void> retention_lease_;
};

/*!
 * \brief Trusted upper-control-plane declaration with an opaque retention lease.
 *
 * The lease provides type-erased lifetime retention only. Runtime neither
 * authenticates it nor proves that it owns the artifacts named by bindings.
 */
class SelectedArtifactManifest : public ObjectRef {
public:
    SelectedArtifactManifest() = default;
    SelectedArtifactManifest(
        String plan_fingerprint, Array<SelectedArtifactBinding> bindings,
        std::shared_ptr<const void> retention_lease = nullptr);
    explicit SelectedArtifactManifest(const ObjectRef& ref);

    Array<SelectedArtifactBinding> bindings() const;
    std::shared_ptr<const void> retention_lease() const;
    void Validate() const;
    const SelectedArtifactManifestNode* operator->() const;
};

/*! \brief Runtime-only ordered plan with a trusted observability declaration. */
class PlanVariantNode final : public Object {
public:
    KXC_OBJECT_DECLARE

private:
    friend class PlanVariant;
    ExecutablePlan plan_;
    SelectedArtifactManifest manifest_;
};

class PlanVariant : public ObjectRef {
public:
    PlanVariant() = default;
    PlanVariant(ExecutablePlan plan, SelectedArtifactManifest manifest);
    explicit PlanVariant(const ObjectRef& ref);

    ExecutablePlan plan() const;
    SelectedArtifactManifest manifest() const;
    void Validate() const;
    const PlanVariantNode* operator->() const;
};

/*!
 * \brief Trusted upper-layer declaration assigned to one call/task locator.
 *
 * This data is for observability and lease propagation. Runtime does not use it
 * to select an executable or prove that it came from an ArtifactPin.
 */
struct ArtifactSelection final {
    int64_t invocation_id{-1};
    String artifact_identity;
    uint64_t generation{0};
};

/*! \brief Runtime-derived exact module/call ABI and memory-contract fingerprint. */
String ComputeCallExactAbiFingerprint(const api::CompiledModule& module,
                                      const ExecutablePlan& plan,
                                      int64_t call_index);

/*! \brief Runtime-derived exact module/task ABI and memory-contract fingerprint. */
String ComputeTaskExactAbiFingerprint(const api::CompiledModule& module,
                                      const class FrozenTaskPlan& plan,
                                      int64_t task_id);

/*! \brief Canonicalizes a declared identity with runtime ABI and locator data. */
String ComputeEntryBindingFingerprint(ArtifactBindingKind kind,
                                      int64_t invocation_id,
                                      const String& artifact_identity,
                                      uint64_t generation,
                                      const String& exact_abi_fingerprint,
                                      const String& entry_symbol);

String ComputePlanVariantFingerprint(
    const ExecutablePlan& plan,
    const Array<SelectedArtifactBinding>& bindings);
String ComputeFrozenTaskPlanFingerprint(
    const class FrozenTaskPlan& plan,
    const Array<SelectedArtifactBinding>& bindings);

/*!
 * \brief Builds a structurally checked declaration for every ordered call.
 *
 * selections and retention_lease come from a trusted upper control plane; this
 * function does not authenticate their origin or their association.
 */
PlanVariant MakePlanVariant(
    const api::CompiledModule& module, const ExecutablePlan& plan,
    const Array<ArtifactSelection>& selections,
    std::shared_ptr<const void> retention_lease = nullptr);

/*! \brief Per-call region category produced by the compiler. */
enum class RegionKind : int32_t {
    kPerCall = 0,
};

/*! \brief Conservative effect ordering contract for a region. */
enum class RegionEffect : int32_t {
    kPure = 0,
    kOrdered = 1,
};

/*! \brief Alias knowledge available at a region boundary. */
enum class RegionAlias : int32_t {
    kNoAlias = 0,
    kConservative = 1,
};

/*! \brief Actions represented by frozen task plan version 1. */
enum class TaskKind : int32_t {
    kKernel = 0,
    kAllocate = 4,
};

class RegionSpecNode final : public Object {
public:
    int64_t region_id{-1};
    RegionKind kind{RegionKind::kPerCall};
    String semantic_key;
    RegionEffect effect{RegionEffect::kPure};
    RegionAlias alias{RegionAlias::kNoAlias};

    KXC_OBJECT_DECLARE

private:
    friend class RegionSpec;
    Array<int64_t> task_ids_;
    Array<int64_t> live_in_value_ids_;
    Array<int64_t> live_out_value_ids_;
    Array<int64_t> constant_value_ids_;
};

/*! \brief Immutable region boundary and identity metadata. */
class RegionSpec : public ObjectRef {
public:
    RegionSpec(int64_t region_id, RegionKind kind, String semantic_key,
               Array<int64_t> task_ids, Array<int64_t> live_in_value_ids,
               Array<int64_t> live_out_value_ids,
               Array<int64_t> constant_value_ids,
               RegionEffect effect = RegionEffect::kPure,
               RegionAlias alias = RegionAlias::kNoAlias);
    explicit RegionSpec(const ObjectRef& ref);

    Array<int64_t> task_ids() const;
    Array<int64_t> live_in_value_ids() const;
    Array<int64_t> live_out_value_ids() const;
    Array<int64_t> constant_value_ids() const;
    void Validate() const;
    const RegionSpecNode* operator->() const;
};

class TaskSpecNode final : public Object {
public:
    int64_t task_id{-1};
    TaskKind kind{TaskKind::kKernel};
    Device device;
    int64_t stream_id{0};
    String symbol;
    int64_t artifact_generation{0};
    uint64_t alignment{0};

    KXC_OBJECT_DECLARE

private:
    friend class TaskSpec;
    Array<int64_t> input_value_ids_;
    Array<int64_t> output_value_ids_;
    Array<int64_t> dependency_task_ids_;
};

/*! \brief Immutable task action and its explicit dependencies. */
class TaskSpec : public ObjectRef {
public:
    TaskSpec(int64_t task_id, TaskKind kind, Device device,
             Array<int64_t> input_value_ids,
             Array<int64_t> output_value_ids,
             Array<int64_t> dependency_task_ids, String symbol = String(),
             int64_t artifact_generation = 0, int64_t stream_id = 0,
             uint64_t alignment = 0);
    explicit TaskSpec(const ObjectRef& ref);

    Array<int64_t> input_value_ids() const;
    Array<int64_t> output_value_ids() const;
    Array<int64_t> dependency_task_ids() const;
    void Validate() const;
    const TaskSpecNode* operator->() const;
};

class FrozenTaskPlanNode final : public Object {
public:
    int32_t version{kFrozenTaskPlanVersion};

    KXC_OBJECT_DECLARE

private:
    friend class FrozenTaskPlan;
    Array<ValueSpec> values_;
    Array<TaskSpec> tasks_;
    Array<RegionSpec> regions_;
    Array<int64_t> input_value_ids_;
    Array<int64_t> constant_value_ids_;
    Array<int64_t> output_value_ids_;
    SelectedArtifactManifest manifest_;
};

/*! \brief Validated immutable exact-shape Region/task-DAG execution contract. */
class FrozenTaskPlan : public ObjectRef {
public:
    FrozenTaskPlan() = default;
    FrozenTaskPlan(int32_t version, Array<ValueSpec> values,
                   Array<TaskSpec> tasks, Array<RegionSpec> regions,
                   Array<int64_t> input_value_ids,
                   Array<int64_t> constant_value_ids,
                   Array<int64_t> output_value_ids,
                   SelectedArtifactManifest manifest = {});
    explicit FrozenTaskPlan(const ObjectRef& ref);

    Array<ValueSpec> values() const;
    Array<TaskSpec> tasks() const;
    Array<RegionSpec> regions() const;
    Array<int64_t> input_value_ids() const;
    Array<int64_t> constant_value_ids() const;
    Array<int64_t> output_value_ids() const;
    SelectedArtifactManifest manifest() const;
    FrozenTaskPlan WithManifest(SelectedArtifactManifest manifest) const;
    Array<int64_t> topological_task_ids() const;
    void Validate() const;
    const FrozenTaskPlanNode* operator->() const;
};

/*!
 * \brief Attaches a trusted observability declaration to every kernel task.
 *
 * Runtime derives the ABI fields but does not authenticate the caller identity
 * or prove that retention_lease owns the declared artifacts.
 */
FrozenTaskPlan AttachSelectedArtifacts(
    const api::CompiledModule& module, const FrozenTaskPlan& plan,
    const Array<ArtifactSelection>& selections,
    std::shared_ptr<const void> retention_lease = nullptr);

}  // namespace kxc::runtime
