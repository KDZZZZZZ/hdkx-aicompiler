/*! \file test/task_executor_test.cpp
 * \brief Verifies dependency-aware task memory planning and fake execution.
 */

#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kxc/runtime/task_executor.h"

namespace {

#define TEST_CHECK(condition, message)                                           \
    do {                                                                          \
        if (!(condition)) {                                                        \
            std::cerr << "[FAIL] " << __FUNCTION__ << ": " << (message) << '\n'; \
            return false;                                                         \
        }                                                                         \
    } while (false)

bool Throws(const std::function<void()>& function) {
    try {
        function();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

DLDataType Float32() { return DLDataType{kDLFloat, 32, 1}; }

static_assert(kxc::runtime::RuntimeEventKind::kTaskStart !=
                  kxc::runtime::RuntimeEventKind::kTaskLaunch,
              "task start and successful launch must be distinct events");

kxc::runtime::FrozenTaskPlan MakeChainPlan(
    bool async_first = false, bool conservative_first = false) {
    using namespace kxc;
    using namespace kxc::runtime;
    const Device cpu = Device::CPU();
    return FrozenTaskPlan(
        kFrozenTaskPlanVersion,
        {ValueSpec(0, 0, {4}, Float32(), cpu, true),
         ValueSpec(1, 1, {4}, Float32(), cpu, false, false, false, false,
                   async_first),
         ValueSpec(2, 2, {4}, Float32(), cpu),
         ValueSpec(3, 3, {4}, Float32(), cpu),
         ValueSpec(4, 4, {4}, Float32(), cpu, false, false, true)},
        {TaskSpec(10, TaskKind::kAllocate, cpu, {}, {1}, {}, String(), 0, 0,
                  64),
         TaskSpec(11, TaskKind::kKernel, cpu, {0}, {1}, {10}, "k1"),
         TaskSpec(20, TaskKind::kAllocate, cpu, {}, {2}, {11}, String(), 0, 0,
                  32),
         TaskSpec(21, TaskKind::kKernel, cpu, {1}, {2}, {20, 11}, "k2"),
         TaskSpec(30, TaskKind::kAllocate, cpu, {}, {3}, {21}, String(), 0, 0,
                  16),
         TaskSpec(31, TaskKind::kKernel, cpu, {2}, {3}, {30, 21}, "k3"),
         TaskSpec(40, TaskKind::kAllocate, cpu, {}, {4}, {31}, String(), 0, 0,
                  16),
         TaskSpec(41, TaskKind::kKernel, cpu, {3}, {4}, {40, 31}, "k4")},
        {RegionSpec(0, RegionKind::kPerCall, "", {10, 11}, {0}, {1}, {},
                    RegionEffect::kPure,
                    conservative_first ? RegionAlias::kConservative
                                       : RegionAlias::kNoAlias),
         RegionSpec(1, RegionKind::kPerCall, "", {20, 21}, {1}, {2}, {}),
         RegionSpec(2, RegionKind::kPerCall, "", {30, 31}, {2}, {3}, {}),
         RegionSpec(3, RegionKind::kPerCall, "", {40, 41}, {3}, {4}, {})},
        {0}, {}, {4});
}

kxc::runtime::FrozenTaskPlan WithDeclaredManifest(
    const kxc::runtime::FrozenTaskPlan& plan) {
    using namespace kxc;
    using namespace kxc::runtime;
    Array<SelectedArtifactBinding> bindings;
    for (const auto& task : plan.tasks()) {
        if (task->kind != TaskKind::kKernel) continue;
        const String identity(
            task->task_id == 11 ? "artifact:kernel:canonical-v1"
                                : "artifact:" + std::to_string(task->task_id));
        const String abi("exact-abi:" + std::to_string(task->task_id));
        bindings.push_back(SelectedArtifactBinding(
            ArtifactBindingKind::kTask, task->task_id, identity,
            task->artifact_generation, abi, task->symbol,
            ComputeEntryBindingFingerprint(
                ArtifactBindingKind::kTask, task->task_id, identity,
                task->artifact_generation, abi, task->symbol)));
    }
    return plan.WithManifest(SelectedArtifactManifest(
        ComputeFrozenTaskPlanFingerprint(plan, bindings), bindings));
}

bool TestDependencyAwareMemoryReuse() {
    using namespace kxc;
    using namespace kxc::runtime;
    const FrozenTaskPlan planned = PlanTaskMemory(MakeChainPlan());
    const Array<ValueSpec> values = planned.values();
    TEST_CHECK(values[1]->storage_id == values[3]->storage_id,
               "a slot must be reused only after every prior consumer completes");
    TEST_CHECK(values[1]->storage_id != values[2]->storage_id &&
                   values[2]->storage_id != values[3]->storage_id,
               "producer input and output lifetimes overlap at a kernel");
    TEST_CHECK(values[4]->storage_id == values[4]->value_id,
               "graph outputs require dedicated retained storage");
    TEST_CHECK(EstimateTaskPeakLiveBytes(planned) == 2 * 4 * sizeof(float),
               "linear dependency peak should contain two adjacent values");

    const FrozenTaskPlan guarded = PlanTaskMemory(MakeChainPlan(true));
    TEST_CHECK(guarded.values()[1]->storage_id != guarded.values()[3]->storage_id,
               "async-live values must never enter reusable storage slots");
    const FrozenTaskPlan conservative =
        PlanTaskMemory(MakeChainPlan(false, true));
    TEST_CHECK(conservative.values()[1]->storage_id !=
                   conservative.values()[3]->storage_id,
               "conservative alias regions must keep boundary storage dedicated");
    return true;
}

bool TestIndependentBranchesDoNotReuse() {
    using namespace kxc;
    using namespace kxc::runtime;
    const Device cpu = Device::CPU();
    const FrozenTaskPlan plan(
        kFrozenTaskPlanVersion,
        {ValueSpec(0, 0, {1}, Float32(), cpu, true),
         ValueSpec(1, 1, {1}, Float32(), cpu),
         ValueSpec(2, 2, {1}, Float32(), cpu),
         ValueSpec(3, 3, {1}, Float32(), cpu, false, false, true)},
        {TaskSpec(10, TaskKind::kAllocate, cpu, {}, {1}, {}, String(), 0, 0, 1),
         TaskSpec(11, TaskKind::kKernel, cpu, {0}, {1}, {10}, "left"),
         TaskSpec(20, TaskKind::kAllocate, cpu, {}, {2}, {}, String(), 0, 0, 1),
         TaskSpec(21, TaskKind::kKernel, cpu, {0}, {2}, {20}, "right"),
         TaskSpec(30, TaskKind::kAllocate, cpu, {}, {3}, {11, 21}, String(), 0,
                  0, 1),
         TaskSpec(31, TaskKind::kKernel, cpu, {1, 2}, {3}, {30, 11, 21},
                  "join")},
        {RegionSpec(0, RegionKind::kPerCall, "", {10, 11, 20, 21, 30, 31},
                    {0}, {3}, {})},
        {0}, {}, {3});
    const FrozenTaskPlan planned = PlanTaskMemory(plan);
    TEST_CHECK(planned.values()[1]->storage_id != planned.values()[2]->storage_id,
               "incomparable branch lifetimes must remain distinct");
    return true;
}

bool TestDeterministicExecutorEvents() {
    using namespace kxc;
    using namespace kxc::runtime;
    const FrozenTaskPlan plan =
        WithDeclaredManifest(PlanTaskMemory(MakeChainPlan()));
    std::unordered_map<int64_t, int> values{{0, 3}};
    std::vector<int64_t> action_order;
    std::vector<RuntimeEvent> observed;
    const Array<RuntimeEvent> trace = ExecuteTasksDeterministically(
        plan, [&](const TaskSpec& task) {
            action_order.push_back(task->task_id);
            if (task->kind == TaskKind::kKernel) {
                values[task.output_value_ids()[0]] =
                    values.at(task.input_value_ids()[0]) + 1;
            }
        },
        [&](const RuntimeEvent& event) { observed.push_back(event); });
    const std::vector<int64_t> expected{10, 11, 20, 21, 30, 31, 40, 41};
    TEST_CHECK(action_order == expected && values.at(4) == 7,
               "kernel and allocation actions must execute deterministically");
    TEST_CHECK(observed.size() == trace.size(),
               "synchronous observer must see every returned event");

    for (size_t i = 0; i < trace.size(); ++i) {
        TEST_CHECK(observed[i].kind == trace[i].kind &&
                       observed[i].task_id == trace[i].task_id &&
                       observed[i].dependency_task_id ==
                           trace[i].dependency_task_id &&
                       observed[i].value_id == trace[i].value_id,
                   "observer event order must be deterministic");
    }

    size_t exact_generation = 0;
    std::vector<int64_t> allocations;
    std::vector<int64_t> releases;
    for (const auto& event : trace) {
        if (event.kind == RuntimeEventKind::kGeneration && event.task_id == 11 &&
            event.generation == 0 &&
            std::string(event.artifact_identity) == "artifact:kernel:canonical-v1" &&
            std::string(event.entry_symbol) == "k1") {
            ++exact_generation;
        }
        if (event.kind == RuntimeEventKind::kAllocation) {
            allocations.push_back(event.value_id);
        } else if (event.kind == RuntimeEventKind::kRelease) {
            releases.push_back(event.value_id);
        }
    }
    TEST_CHECK(exact_generation == 1,
               "generation must retain the declared task identity and symbol");
    TEST_CHECK(allocations == std::vector<int64_t>({1, 2, 3, 4}) &&
                   releases == std::vector<int64_t>({1, 2, 3}),
               "allocation/release trace must follow consumer completion");

    const Array<RuntimeEvent> repeated = ExecuteTasksDeterministically(
        plan, [](const TaskSpec&) {});
    TEST_CHECK(repeated.size() == trace.size(),
               "deterministic observer trace size changed between runs");
    for (size_t i = 0; i < trace.size(); ++i) {
        TEST_CHECK(repeated[i].kind == trace[i].kind &&
                       repeated[i].task_id == trace[i].task_id &&
                       repeated[i].dependency_task_id ==
                           trace[i].dependency_task_id &&
                       repeated[i].value_id == trace[i].value_id &&
                       repeated[i].storage_id == trace[i].storage_id,
                   "deterministic observer trace changed between runs");
    }
    return true;
}

bool TestTaskLifecycleFailureAndOverflow() {
    using namespace kxc;
    using namespace kxc::runtime;
    const FrozenTaskPlan plan =
        WithDeclaredManifest(PlanTaskMemory(MakeChainPlan()));
    std::vector<RuntimeEventKind> success_events;
    ExecuteTasksDeterministically(
        plan, [](const TaskSpec&) {},
        [&](const RuntimeEvent& event) {
            if (event.task_id == 11) success_events.push_back(event.kind);
        });
    TEST_CHECK(success_events == std::vector<RuntimeEventKind>({
                                     RuntimeEventKind::kTaskWait,
                                     RuntimeEventKind::kGeneration,
                                     RuntimeEventKind::kTaskStart,
                                     RuntimeEventKind::kTaskLaunch,
                                     RuntimeEventKind::kTaskComplete}),
               "kernel events must be wait -> generation -> start -> launch -> complete");
    TEST_CHECK(Throws([&] {
                   ExecuteTasksDeterministically(plan, DeterministicTaskAction());
               }),
               "empty fake action must fail before execution");

    std::vector<RuntimeEventKind> failure_events;
    TEST_CHECK(Throws([&] {
                   ExecuteTasksDeterministically(
                       plan,
                       [](const TaskSpec& task) {
                           if (task->task_id == 11) {
                               throw std::runtime_error("fake kernel submission failed");
                           }
                       },
                       [&](const RuntimeEvent& event) {
                           if (event.task_id == 11) failure_events.push_back(event.kind);
                       });
               }) &&
                   failure_events == std::vector<RuntimeEventKind>({
                                         RuntimeEventKind::kTaskWait,
                                         RuntimeEventKind::kGeneration,
                                         RuntimeEventKind::kTaskStart}),
               "failed actions must not emit launch or completion events");

    const Device cpu = Device::CPU();
    const int64_t huge = std::numeric_limits<int64_t>::max();
    const FrozenTaskPlan overflow(
        kFrozenTaskPlanVersion,
        {ValueSpec(0, 0, {1}, Float32(), cpu, true),
         ValueSpec(1, 1, {huge}, Float32(), cpu, false, false, true)},
        {TaskSpec(0, TaskKind::kAllocate, cpu, {}, {1}, {}, String(), 0, 0, 1),
         TaskSpec(1, TaskKind::kKernel, cpu, {0}, {1}, {0}, "huge")},
        {RegionSpec(0, RegionKind::kPerCall, "", {0, 1}, {0}, {1}, {})},
        {0}, {}, {1});
    TEST_CHECK(Throws([&] { (void)EstimateTaskPeakLiveBytes(overflow); }),
               "peak-byte overflow must fail closed");
    return true;
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"dependency_aware_memory_reuse", TestDependencyAwareMemoryReuse},
        {"independent_branches_do_not_reuse", TestIndependentBranchesDoNotReuse},
        {"deterministic_executor_events", TestDeterministicExecutorEvents},
        {"task_lifecycle_failure_and_overflow", TestTaskLifecycleFailureAndOverflow},
    };
    int failures = 0;
    for (const auto& test : tests) {
        try {
            if (!test.second()) {
                ++failures;
                continue;
            }
            std::cout << "[PASS] " << test.first << '\n';
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << test.first << ": " << error.what() << '\n';
            ++failures;
        }
    }
    return failures == 0 ? 0 : 1;
}
