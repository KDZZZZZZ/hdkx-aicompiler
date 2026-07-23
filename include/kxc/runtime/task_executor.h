/*! \file include/kxc/runtime/task_executor.h
 * \brief Dependency-aware task memory planning and deterministic fake execution.
 */

#pragma once

#include <functional>

#include "kxc/runtime/task_plan.h"

namespace kxc::runtime {

/*! \brief Observable steps emitted by the deterministic single-stream executor. */
enum class TaskTraceEventKind : int32_t {
    kTaskStart = 0,
    kAllocate = 1,
    kTaskComplete = 2,
    kRelease = 3,
};

struct TaskTraceEvent final {
    TaskTraceEventKind kind{TaskTraceEventKind::kTaskStart};
    int64_t task_id{-1};
    TaskKind task_kind{TaskKind::kKernel};
    int64_t value_id{-1};
    int64_t storage_id{-1};
};

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
Array<TaskTraceEvent> ExecuteTasksDeterministically(
    const FrozenTaskPlan& plan, const DeterministicTaskAction& action);

}  // namespace kxc::runtime
