/*! \file include/kxc/runtime/task_plan.h
 * \brief Frozen, runtime-neutral Region and single-stream task-DAG contracts.
 */

#pragma once

#include <cstdint>

#include "kxc/runtime/executable_plan.h"

namespace kxc::runtime {

constexpr int32_t kFrozenTaskPlanVersion = 1;

/*! \brief Runtime-neutral region category; PerCall remains the fallback policy. */
enum class RegionKind : int32_t {
    kPerCall = 0,
    kFusion = 1,
    kLibrary = 2,
    kControlFlow = 3,
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
    kCopy = 1,
    kEvent = 2,
    kShapeEval = 3,
    kAllocate = 4,
    kSync = 5,
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
};

/*! \brief Validated immutable exact-shape Region/task-DAG execution contract. */
class FrozenTaskPlan : public ObjectRef {
public:
    FrozenTaskPlan() = default;
    FrozenTaskPlan(int32_t version, Array<ValueSpec> values,
                   Array<TaskSpec> tasks, Array<RegionSpec> regions,
                   Array<int64_t> input_value_ids,
                   Array<int64_t> constant_value_ids,
                   Array<int64_t> output_value_ids);
    explicit FrozenTaskPlan(const ObjectRef& ref);

    Array<ValueSpec> values() const;
    Array<TaskSpec> tasks() const;
    Array<RegionSpec> regions() const;
    Array<int64_t> input_value_ids() const;
    Array<int64_t> constant_value_ids() const;
    Array<int64_t> output_value_ids() const;
    Array<int64_t> topological_task_ids() const;
    void Validate() const;
    const FrozenTaskPlanNode* operator->() const;
};

}  // namespace kxc::runtime
