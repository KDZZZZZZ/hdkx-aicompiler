/*! \file include/kxc/runtime/task_executor.h
 * \brief Dependency-aware task memory planning and deterministic fake execution.
 */

#pragma once

#include <functional>

#include "kxc/runtime/task_plan.h"

namespace kxc::runtime {

/*! \brief Auditable reason for selecting the per-Call oracle or rejecting a DAG. */
enum class FallbackReason : int32_t {
    kNone = 0,
    kFeatureDisabled = 1,
    kMissingArtifactManifest = 2,
    kUnsupportedDynamicInput = 3,
    kUnsupportedAlias = 4,
    kAdapterBug = 9,
};

/*! \brief Stable synchronous runtime observability schema. */
enum class RuntimeEventKind : int32_t {
    kTaskStart = 0,
    kAllocation = 1,
    kTaskComplete = 2,
    kRelease = 3,
    kTaskWait = 4,
    kGeneration = 5,
    kFallback = 6,
    kTaskLaunch = 7,
    // Source-compatible W1 spelling.
    kAllocate = kAllocation,
};

struct RuntimeEvent final {
    RuntimeEventKind kind{RuntimeEventKind::kTaskStart};
    int64_t task_id{-1};
    TaskKind task_kind{TaskKind::kKernel};
    int64_t value_id{-1};
    int64_t storage_id{-1};
    int64_t dependency_task_id{-1};
    uint64_t bytes{0};
    uint64_t alignment{0};
    uint64_t generation{0};
    String artifact_identity;
    String entry_symbol;
    FallbackReason fallback_reason{FallbackReason::kNone};
    String diagnostic;
};

using RuntimeObserver = std::function<void(const RuntimeEvent&)>;
using TaskTraceEventKind = RuntimeEventKind;
using TaskTraceEvent = RuntimeEvent;

/*! \brief Deterministically assigns reusable storage using dependency reachability. */
FrozenTaskPlan PlanTaskMemory(const FrozenTaskPlan& plan);

/*! \brief Peak produced-value bytes for deterministic topological execution. */
size_t EstimateTaskPeakLiveBytes(const FrozenTaskPlan& plan);

/*! \brief Synchronous callback used by the backend-free deterministic executor. */
using DeterministicTaskAction = std::function<void(const TaskSpec&)>;

/*!
 * \brief Executes every task once in deterministic topological order.
 *
 * This executor is a contract fake: it has no worker threads, devices, compiler,
 * cache, or RuntimeSession dependency. The action supplies task semantics.
 */
Array<RuntimeEvent> ExecuteTasksDeterministically(
    const FrozenTaskPlan& plan, const DeterministicTaskAction& action,
    const RuntimeObserver& observer = {});

}  // namespace kxc::runtime
