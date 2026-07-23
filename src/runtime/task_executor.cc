/*! \file src/runtime/task_executor.cc
 * \brief Dependency-aware memory planning and deterministic task execution.
 */

#include "kxc/runtime/task_executor.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace kxc::runtime {
namespace {

bool IsDataProducer(TaskKind kind) {
    return kind == TaskKind::kKernel || kind == TaskKind::kCopy ||
           kind == TaskKind::kShapeEval;
}

bool IsSourceValue(const ValueSpec& value) {
    return value->is_input || value->is_constant;
}

bool Reusable(const ValueSpec& value) {
    return !IsSourceValue(value) && !value->is_output && !value->is_alias &&
           !value->is_async_live;
}

bool SameShape(const Array<int64_t>& lhs, const Array<int64_t>& rhs) {
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i] != rhs[i]) return false;
    }
    return true;
}

bool SameContract(const ValueSpec& lhs, const ValueSpec& rhs) {
    return lhs->dtype.code == rhs->dtype.code &&
           lhs->dtype.bits == rhs->dtype.bits &&
           lhs->dtype.lanes == rhs->dtype.lanes &&
           lhs->device == rhs->device && SameShape(lhs.shape(), rhs.shape());
}

size_t ValueBytes(const ValueSpec& value) {
    const DLDataType dtype = value->dtype;
    if (dtype.bits == 0 || dtype.bits % 8 != 0 || dtype.lanes == 0) {
        throw std::invalid_argument("Task value dtype has no byte width");
    }
    size_t bytes = dtype.bits / 8;
    if (bytes > std::numeric_limits<size_t>::max() / dtype.lanes) {
        throw std::overflow_error("Task value dtype byte width overflow");
    }
    bytes *= dtype.lanes;
    bool zero = false;
    for (int64_t dimension : value.shape()) {
        if (dimension < 0) {
            throw std::invalid_argument(
                "Produced task values require exact static shapes");
        }
        zero = zero || dimension == 0;
    }
    if (zero) return 0;
    for (int64_t dimension : value.shape()) {
        const size_t extent = static_cast<size_t>(dimension);
        if (bytes > std::numeric_limits<size_t>::max() / extent) {
            throw std::overflow_error("Task value byte size overflow");
        }
        bytes *= extent;
    }
    return bytes;
}

struct PlanIndex final {
    std::unordered_map<int64_t, ValueSpec> values;
    std::unordered_map<int64_t, TaskSpec> tasks;
    std::unordered_map<int64_t, int64_t> producer_by_value;
    std::unordered_map<int64_t, int64_t> allocation_by_value;
    std::unordered_map<int64_t, std::unordered_set<int64_t>> consumers_by_value;
    std::unordered_map<int64_t, std::unordered_set<int64_t>> ancestors_by_task;
    std::unordered_set<int64_t> conservative_alias_values;
};

PlanIndex IndexPlan(const FrozenTaskPlan& plan) {
    PlanIndex result;
    for (const auto& value : plan.values()) {
        result.values.emplace(value->value_id, value);
    }
    for (const auto& task : plan.tasks()) {
        result.tasks.emplace(task->task_id, task);
        for (int64_t input : task.input_value_ids()) {
            result.consumers_by_value[input].insert(task->task_id);
        }
        for (int64_t output : task.output_value_ids()) {
            if (task->kind == TaskKind::kAllocate) {
                result.allocation_by_value.emplace(output, task->task_id);
            } else if (IsDataProducer(task->kind)) {
                result.producer_by_value.emplace(output, task->task_id);
            }
        }
    }
    for (const auto& region : plan.regions()) {
        if (region->alias != RegionAlias::kConservative) continue;
        for (int64_t task_id : region.task_ids()) {
            const TaskSpec& task = result.tasks.at(task_id);
            for (int64_t input : task.input_value_ids()) {
                result.conservative_alias_values.insert(input);
            }
            for (int64_t output : task.output_value_ids()) {
                result.conservative_alias_values.insert(output);
            }
        }
    }
    for (int64_t task_id : plan.topological_task_ids()) {
        auto& ancestors = result.ancestors_by_task[task_id];
        for (int64_t dependency :
             result.tasks.at(task_id).dependency_task_ids()) {
            ancestors.insert(dependency);
            const auto& transitive = result.ancestors_by_task.at(dependency);
            ancestors.insert(transitive.begin(), transitive.end());
        }
    }
    return result;
}

bool LifetimeEndsBeforeAllocation(const PlanIndex& index, int64_t value_id,
                                  int64_t allocation_task_id) {
    const auto& ancestors = index.ancestors_by_task.at(allocation_task_id);
    const auto consumers = index.consumers_by_value.find(value_id);
    if (consumers == index.consumers_by_value.end() ||
        consumers->second.empty()) {
        return ancestors.count(index.producer_by_value.at(value_id)) != 0;
    }
    for (int64_t consumer : consumers->second) {
        if (!ancestors.count(consumer)) return false;
    }
    return true;
}

struct Slot final {
    int64_t storage_id{-1};
    int64_t latest_value_id{-1};
    uint64_t alignment{0};
};

}  // namespace

FrozenTaskPlan PlanTaskMemory(const FrozenTaskPlan& plan) {
    plan.Validate();
    if (plan.manifest().defined()) {
        throw std::invalid_argument(
            "Task memory planning must precede artifact declaration attachment");
    }
    const PlanIndex index = IndexPlan(plan);
    std::unordered_map<int64_t, int64_t> storage_by_value;
    for (const auto& value : plan.values()) {
        if (!Reusable(value) ||
            index.conservative_alias_values.count(value->value_id)) {
            storage_by_value.emplace(value->value_id, value->value_id);
        }
    }

    std::vector<Slot> slots;
    for (int64_t task_id : plan.topological_task_ids()) {
        const TaskSpec& allocation = index.tasks.at(task_id);
        if (allocation->kind != TaskKind::kAllocate) continue;
        const int64_t value_id = allocation.output_value_ids()[0];
        const ValueSpec& value = index.values.at(value_id);
        if (!Reusable(value) ||
            index.conservative_alias_values.count(value_id)) {
            continue;
        }

        Slot* selected = nullptr;
        for (auto& slot : slots) {
            const ValueSpec& previous = index.values.at(slot.latest_value_id);
            if (slot.alignment >= allocation->alignment &&
                SameContract(previous, value) &&
                LifetimeEndsBeforeAllocation(index, slot.latest_value_id,
                                              task_id) &&
                (!selected || slot.storage_id < selected->storage_id)) {
                selected = &slot;
            }
        }
        if (selected) {
            storage_by_value.emplace(value_id, selected->storage_id);
            selected->latest_value_id = value_id;
        } else {
            storage_by_value.emplace(value_id, value_id);
            slots.push_back(Slot{value_id, value_id, allocation->alignment});
        }
    }

    Array<ValueSpec> planned_values;
    for (const auto& value : plan.values()) {
        const auto storage = storage_by_value.find(value->value_id);
        if (storage == storage_by_value.end()) {
            throw std::logic_error(
                "Task memory planner did not assign every value");
        }
        planned_values.push_back(ValueSpec(
            value->value_id, storage->second, value.shape(), value->dtype,
            value->device, value->is_input, value->is_constant,
            value->is_output, value->is_alias, value->is_async_live));
    }
    return FrozenTaskPlan(
        plan->version, std::move(planned_values), plan.tasks(), plan.regions(),
        plan.input_value_ids(), plan.constant_value_ids(),
        plan.output_value_ids());
}

Array<RuntimeEvent> ExecuteTasksDeterministically(
    const FrozenTaskPlan& plan, const DeterministicTaskAction& action,
    const RuntimeObserver& observer) {
    plan.Validate();
    if (!action) {
        throw std::invalid_argument(
            "Deterministic task execution requires an action");
    }
    const PlanIndex index = IndexPlan(plan);
    std::unordered_map<int64_t, SelectedArtifactBinding> artifacts;
    if (plan.manifest().defined()) {
        for (const auto& binding : plan.manifest().bindings()) {
            artifacts.emplace(binding->invocation_id, binding);
        }
    }
    std::unordered_map<int64_t, size_t> remaining_consumers;
    for (const auto& value : index.values) {
        const auto consumers = index.consumers_by_value.find(value.first);
        remaining_consumers[value.first] =
            consumers == index.consumers_by_value.end()
                ? 0
                : consumers->second.size();
    }
    std::unordered_set<int64_t> active_storage;
    std::unordered_set<int64_t> completed_tasks;
    std::unordered_set<int64_t> released_values;
    Array<RuntimeEvent> trace;
    const auto emit = [&](RuntimeEvent event) {
        trace.push_back(event);
        if (!observer) return;
        try {
            observer(event);
        } catch (...) {
            // Observability is synchronous but must never change execution.
        }
    };

    const auto release = [&](int64_t task_id, TaskKind task_kind,
                             int64_t value_id) {
        const ValueSpec& value = index.values.at(value_id);
        if (IsSourceValue(value) || value->is_output || value->is_async_live ||
            !released_values.insert(value_id).second) {
            return;
        }
        if (!active_storage.erase(value->storage_id)) {
            throw std::logic_error(
                "Task executor released storage that was not active");
        }
        RuntimeEvent event;
        event.kind = RuntimeEventKind::kRelease;
        event.task_id = task_id;
        event.task_kind = task_kind;
        event.value_id = value_id;
        event.storage_id = value->storage_id;
        event.bytes = ValueBytes(value);
        emit(std::move(event));
    };

    for (int64_t task_id : plan.topological_task_ids()) {
        const TaskSpec& task = index.tasks.at(task_id);
        for (int64_t dependency : task.dependency_task_ids()) {
            if (!completed_tasks.count(dependency)) {
                throw std::logic_error(
                    "Deterministic task scheduler observed an unmet dependency");
            }
            RuntimeEvent wait;
            wait.kind = RuntimeEventKind::kTaskWait;
            wait.task_id = task_id;
            wait.task_kind = task->kind;
            wait.dependency_task_id = dependency;
            emit(std::move(wait));
        }
        if (task->kind == TaskKind::kKernel) {
            RuntimeEvent generation;
            generation.kind = RuntimeEventKind::kGeneration;
            generation.task_id = task_id;
            generation.task_kind = task->kind;
            generation.generation =
                static_cast<uint64_t>(task->artifact_generation);
            generation.entry_symbol = task->symbol;
            const auto artifact = artifacts.find(task_id);
            if (artifact != artifacts.end()) {
                generation.artifact_identity =
                    artifact->second->artifact_identity;
            }
            emit(std::move(generation));
        }
        int64_t allocated_value_id = -1;
        int64_t allocated_storage_id = -1;
        if (task->kind == TaskKind::kAllocate) {
            allocated_value_id = task.output_value_ids()[0];
            allocated_storage_id =
                index.values.at(allocated_value_id)->storage_id;
            if (!active_storage.insert(allocated_storage_id).second) {
                throw std::logic_error(
                    "Task executor allocated a storage slot before retirement");
            }
        }

        action(task);
        RuntimeEvent launch;
        launch.kind = RuntimeEventKind::kTaskLaunch;
        launch.task_id = task_id;
        launch.task_kind = task->kind;
        launch.entry_symbol = task->symbol;
        const auto artifact = artifacts.find(task_id);
        if (artifact != artifacts.end()) {
            launch.generation = artifact->second->generation;
            launch.artifact_identity = artifact->second->artifact_identity;
        }
        emit(std::move(launch));
        if (task->kind == TaskKind::kAllocate) {
            RuntimeEvent allocation;
            allocation.kind = RuntimeEventKind::kAllocation;
            allocation.task_id = task_id;
            allocation.task_kind = task->kind;
            allocation.value_id = allocated_value_id;
            allocation.storage_id = allocated_storage_id;
            allocation.bytes =
                ValueBytes(index.values.at(allocated_value_id));
            allocation.alignment = task->alignment;
            emit(std::move(allocation));
        }
        completed_tasks.insert(task_id);
        RuntimeEvent complete;
        complete.kind = RuntimeEventKind::kTaskComplete;
        complete.task_id = task_id;
        complete.task_kind = task->kind;
        emit(std::move(complete));

        std::unordered_set<int64_t> unique_inputs;
        for (int64_t input : task.input_value_ids()) {
            if (!unique_inputs.insert(input).second) continue;
            size_t& remaining = remaining_consumers.at(input);
            if (remaining == 0) {
                throw std::logic_error(
                    "Task executor consumed a value more than once");
            }
            if (--remaining == 0 && index.producer_by_value.count(input)) {
                release(task_id, task->kind, input);
            }
        }
        if (IsDataProducer(task->kind)) {
            for (int64_t output : task.output_value_ids()) {
                if (remaining_consumers.at(output) == 0) {
                    release(task_id, task->kind, output);
                }
            }
        }
    }
    return trace;
}

size_t EstimateTaskPeakLiveBytes(const FrozenTaskPlan& plan) {
    const std::unordered_map<int64_t, ValueSpec> values = [&] {
        std::unordered_map<int64_t, ValueSpec> result;
        for (const auto& value : plan.values()) {
            result.emplace(value->value_id, value);
        }
        return result;
    }();
    const Array<TaskTraceEvent> trace = ExecuteTasksDeterministically(
        plan, [](const TaskSpec&) {});
    size_t live_bytes = 0;
    size_t peak_bytes = 0;
    for (const auto& event : trace) {
        if (event.kind == TaskTraceEventKind::kAllocate) {
            const size_t bytes = ValueBytes(values.at(event.value_id));
            if (live_bytes > std::numeric_limits<size_t>::max() - bytes) {
                throw std::overflow_error("Task peak live byte count overflow");
            }
            live_bytes += bytes;
            peak_bytes = std::max(peak_bytes, live_bytes);
        } else if (event.kind == TaskTraceEventKind::kRelease) {
            const size_t bytes = ValueBytes(values.at(event.value_id));
            if (bytes > live_bytes) {
                throw std::logic_error("Task live byte count underflow");
            }
            live_bytes -= bytes;
        }
    }
    return peak_bytes;
}

}  // namespace kxc::runtime
