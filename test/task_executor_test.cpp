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

kxc::runtime::FrozenTaskPlan MakeChainPlan(bool async_first = false) {
    using namespace kxc;
    using namespace kxc::runtime;
    const Device cpu = Device::CPU();
    Array<ValueSpec> values{
        ValueSpec(0, 0, {4}, Float32(), cpu, true),
        ValueSpec(1, 1, {4}, Float32(), cpu, false, false, false, false,
                  async_first),
        ValueSpec(2, 2, {4}, Float32(), cpu),
        ValueSpec(3, 3, {4}, Float32(), cpu),
        ValueSpec(4, 4, {4}, Float32(), cpu, false, false, true),
    };
    Array<TaskSpec> tasks{
        TaskSpec(10, TaskKind::kAllocate, cpu, {}, {1}, {}, String(), 0, 0,
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
        TaskSpec(41, TaskKind::kKernel, cpu, {3}, {4}, {40, 31}, "k4"),
    };
    Array<RegionSpec> regions{
        RegionSpec(0, RegionKind::kPerCall, "", {10, 11}, {0}, {1}, {}),
        RegionSpec(1, RegionKind::kPerCall, "", {20, 21}, {1}, {2}, {}),
        RegionSpec(2, RegionKind::kPerCall, "", {30, 31}, {2}, {3}, {}),
        RegionSpec(3, RegionKind::kPerCall, "", {40, 41}, {3}, {4}, {}),
    };
    return FrozenTaskPlan(kFrozenTaskPlanVersion, std::move(values),
                          std::move(tasks), std::move(regions), {0}, {}, {4});
}

kxc::runtime::FrozenTaskPlan MakeAllKindsPlan() {
    using namespace kxc;
    using namespace kxc::runtime;
    const Device cpu = Device::CPU();
    return FrozenTaskPlan(
        kFrozenTaskPlanVersion,
        {ValueSpec(0, 0, {1}, Float32(), cpu, true),
         ValueSpec(1, 1, {1}, Float32(), cpu),
         ValueSpec(2, 2, {1}, Float32(), cpu),
         ValueSpec(3, 3, {1}, Float32(), cpu, false, false, true)},
        {TaskSpec(10, TaskKind::kAllocate, cpu, {}, {1}, {}, String(), 0, 0,
                  16),
         TaskSpec(11, TaskKind::kKernel, cpu, {0}, {1}, {10}, "kernel"),
         TaskSpec(12, TaskKind::kEvent, cpu, {}, {}, {11}),
         TaskSpec(20, TaskKind::kAllocate, cpu, {}, {2}, {12}, String(), 0, 0,
                  16),
         TaskSpec(21, TaskKind::kCopy, cpu, {1}, {2}, {20, 12}),
         TaskSpec(30, TaskKind::kAllocate, cpu, {}, {3}, {21}, String(), 0, 0,
                  16),
         TaskSpec(31, TaskKind::kShapeEval, cpu, {1, 2}, {3}, {30, 11, 21},
                  "shape.program"),
         TaskSpec(32, TaskKind::kSync, cpu, {}, {}, {31})},
        {RegionSpec(0, RegionKind::kFusion, "fixture.key",
                    {10, 11, 12, 20, 21, 30, 31, 32}, {0}, {3}, {})},
        {0}, {}, {3});
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
    TEST_CHECK(guarded.values()[1]->storage_id !=
                   guarded.values()[3]->storage_id,
               "async-live values must never enter reusable storage slots");
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
        {RegionSpec(0, RegionKind::kFusion, "", {10, 11, 20, 21, 30, 31},
                    {0}, {3}, {})},
        {0}, {}, {3});
    const FrozenTaskPlan planned = PlanTaskMemory(plan);
    TEST_CHECK(planned.values()[1]->storage_id !=
                   planned.values()[2]->storage_id,
               "incomparable branch lifetimes must remain distinct");
    return true;
}

bool TestDeterministicExecutorCoversTaskKinds() {
    using namespace kxc;
    using namespace kxc::runtime;
    const FrozenTaskPlan plan = PlanTaskMemory(MakeAllKindsPlan());
    std::unordered_map<int64_t, int> values{{0, 3}};
    std::vector<int64_t> action_order;
    const Array<TaskTraceEvent> trace = ExecuteTasksDeterministically(
        plan, [&](const TaskSpec& task) {
            action_order.push_back(task->task_id);
            switch (task->kind) {
                case TaskKind::kKernel:
                    values[task.output_value_ids()[0]] =
                        values.at(task.input_value_ids()[0]) + 1;
                    break;
                case TaskKind::kCopy:
                    values[task.output_value_ids()[0]] =
                        values.at(task.input_value_ids()[0]);
                    break;
                case TaskKind::kShapeEval:
                    values[task.output_value_ids()[0]] =
                        values.at(task.input_value_ids()[0]) +
                        values.at(task.input_value_ids()[1]);
                    break;
                case TaskKind::kAllocate:
                case TaskKind::kEvent:
                case TaskKind::kSync:
                    break;
            }
        });
    const std::vector<int64_t> expected{10, 11, 12, 20, 21, 30, 31, 32};
    TEST_CHECK(action_order == expected,
               "fake executor action order must be deterministic");
    TEST_CHECK(values.at(3) == 8,
               "kernel/copy/shape-eval fake semantics did not compose");

    std::vector<int64_t> allocations;
    std::vector<int64_t> releases;
    for (const auto& event : trace) {
        if (event.kind == TaskTraceEventKind::kAllocate) {
            allocations.push_back(event.value_id);
        } else if (event.kind == TaskTraceEventKind::kRelease) {
            releases.push_back(event.value_id);
        }
    }
    TEST_CHECK(allocations == std::vector<int64_t>({1, 2, 3}) &&
                   releases == std::vector<int64_t>({1, 2}),
               "allocation/release trace must follow consumer completion");
    return true;
}

bool TestExecutorFailureAndOverflowGuards() {
    using namespace kxc;
    using namespace kxc::runtime;
    const FrozenTaskPlan plan = MakeAllKindsPlan();
    int calls = 0;
    TEST_CHECK(Throws([&] {
                   ExecuteTasksDeterministically(
                       plan, [&](const TaskSpec& task) {
                           ++calls;
                           if (task->task_id == 21) {
                               throw std::runtime_error("fake copy failed");
                           }
                       });
               }) &&
                   calls == 5,
               "post-start failure must stop without replaying prior tasks");
    TEST_CHECK(Throws([&] {
                   ExecuteTasksDeterministically(plan,
                                                 DeterministicTaskAction());
               }),
               "empty fake action must fail before execution");

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
        {"deterministic_executor_covers_task_kinds",
         TestDeterministicExecutorCoversTaskKinds},
        {"executor_failure_and_overflow_guards",
         TestExecutorFailureAndOverflowGuards},
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
