/*! \file test/task_plan_test.cpp
 * \brief Verifies frozen Region/task-DAG DTO and dependency validation.
 */

#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "kxc/runtime/task_plan.h"

namespace {

#define TEST_CHECK(condition, message)                                           \
    do {                                                                          \
        if (!(condition)) {                                                        \
            std::cerr << "[FAIL] " << __FUNCTION__ << ": " << (message) << '\n'; \
            return false;                                                         \
        }                                                                         \
    } while (false)

bool Throws(const std::function<void()>& function,
            std::string* message = nullptr) {
    try {
        function();
    } catch (const std::exception& error) {
        if (message) *message = error.what();
        return true;
    }
    return false;
}

DLDataType Float32() { return DLDataType{kDLFloat, 32, 1}; }

kxc::runtime::FrozenTaskPlan MakeValidPlan() {
    using namespace kxc;
    using namespace kxc::runtime;
    const Device cpu = Device::CPU();
    Array<ValueSpec> values{
        ValueSpec(0, 0, {4}, Float32(), cpu, true),
        ValueSpec(1, 1, {4}, Float32(), cpu, false, true),
        ValueSpec(2, 2, {4}, Float32(), cpu),
        ValueSpec(3, 3, {4}, Float32(), cpu),
        ValueSpec(4, 4, {4}, Float32(), cpu, false, false, true),
    };
    Array<TaskSpec> tasks{
        TaskSpec(32, TaskKind::kSync, cpu, {}, {}, {31, 22}),
        TaskSpec(20, TaskKind::kAllocate, cpu, {}, {3}, {}, String(), 0, 0,
                 16),
        TaskSpec(11, TaskKind::kKernel, cpu, {0, 1, 0}, {2}, {10},
                 "kernel_a", 7),
        TaskSpec(10, TaskKind::kAllocate, cpu, {}, {2}, {}, String(), 0, 0,
                 32),
        TaskSpec(31, TaskKind::kShapeEval, cpu, {2, 3}, {4}, {30, 11, 21},
                 "shape_program.0"),
        TaskSpec(22, TaskKind::kEvent, cpu, {}, {}, {21}),
        TaskSpec(21, TaskKind::kCopy, cpu, {2}, {3}, {20, 11}),
        TaskSpec(30, TaskKind::kAllocate, cpu, {}, {4}, {11, 21}, String(),
                 0, 0, 64),
    };
    Array<RegionSpec> regions{
        RegionSpec(0, RegionKind::kPerCall, "", {10, 11}, {0, 1}, {2}, {1}),
        RegionSpec(1, RegionKind::kLibrary, "library.key", {20, 21, 22}, {2},
                   {3}, {}, RegionEffect::kOrdered),
        RegionSpec(2, RegionKind::kFusion, "fusion.key", {30, 31, 32}, {2, 3},
                   {4}, {}),
    };
    return FrozenTaskPlan(kFrozenTaskPlanVersion, std::move(values),
                          std::move(tasks), std::move(regions), {0}, {1}, {4});
}

bool TestFrozenDtoAndDeterministicTopology() {
    using namespace kxc;
    using namespace kxc::runtime;
    FrozenTaskPlan plan = MakeValidPlan();
    const Array<int64_t> topology = plan.topological_task_ids();
    const std::vector<int64_t> expected{10, 11, 20, 21, 22, 30, 31, 32};
    TEST_CHECK(topology.size() == expected.size(),
               "topological order has the wrong size");
    for (size_t i = 0; i < expected.size(); ++i) {
        TEST_CHECK(topology[i] == expected[i],
                   "topological order must use task id as deterministic tie-breaker");
    }
    TEST_CHECK(plan.tasks()[2].input_value_ids().size() == 3,
               "repeated logical operands must remain in the frozen task ABI");
    TEST_CHECK(plan.regions()[0]->semantic_key == "",
               "an unavailable semantic key must remain explicitly unkeyed");
    TEST_CHECK(plan.get()->GetTypeKey() == "kxc.runtime.FrozenTaskPlanNode",
               "frozen plan type key must be stable");

    Array<TaskSpec> copied_tasks = plan.tasks();
    copied_tasks.push_back(TaskSpec(99, TaskKind::kEvent, Device::CPU(), {}, {},
                                    {}));
    TEST_CHECK(plan.tasks().size() == 8,
               "public arrays must not mutate the frozen plan");
    return true;
}

bool TestTaskKindContracts() {
    using namespace kxc;
    using namespace kxc::runtime;
    const Device cpu = Device::CPU();
    TEST_CHECK(Throws([&] {
                   TaskSpec task(0, TaskKind::kKernel, cpu, {}, {1}, {});
               }),
               "kernel task without a symbol must fail");
    TEST_CHECK(Throws([&] {
                   TaskSpec task(0, TaskKind::kCopy, cpu, {0, 1}, {2}, {});
               }),
               "copy task arity must be exact");
    TEST_CHECK(Throws([&] {
                   TaskSpec task(0, TaskKind::kAllocate, cpu, {}, {1}, {},
                                 String(), 0, 0, 0);
               }),
               "allocate task requires a positive alignment");
    TEST_CHECK(Throws([&] {
                   TaskSpec task(0, TaskKind::kAllocate, cpu, {}, {1}, {},
                                 String(), 0, 0, 3);
               }),
               "allocate alignment must be a power of two");
    TEST_CHECK(Throws([&] {
                   TaskSpec task(0, TaskKind::kEvent, cpu, {0}, {}, {});
               }),
               "event task cannot hide value dependencies");
    TEST_CHECK(Throws([&] {
                   TaskSpec task(0, TaskKind::kShapeEval, cpu, {}, {1}, {});
               }),
               "shape-eval task requires an explicit program key");
    return true;
}

bool TestCyclesAndMissingDependencies() {
    using namespace kxc;
    using namespace kxc::runtime;
    const Device cpu = Device::CPU();
    const Array<ValueSpec> values{
        ValueSpec(0, 0, {1}, Float32(), cpu, true, false, true),
    };
    TEST_CHECK(Throws([&] {
                   FrozenTaskPlan plan(
                       kFrozenTaskPlanVersion, values,
                       {TaskSpec(0, TaskKind::kEvent, cpu, {}, {}, {1}),
                        TaskSpec(1, TaskKind::kSync, cpu, {}, {}, {0})},
                       {RegionSpec(0, RegionKind::kPerCall, "", {0, 1}, {0},
                                   {}, {})},
                       {0}, {}, {0});
               }),
               "task dependency cycle must fail");
    TEST_CHECK(Throws([&] {
                   FrozenTaskPlan plan(
                       kFrozenTaskPlanVersion, values,
                       {TaskSpec(0, TaskKind::kEvent, cpu, {}, {}, {99})},
                       {RegionSpec(0, RegionKind::kPerCall, "", {0}, {0}, {},
                                   {})},
                       {0}, {}, {0});
               }),
               "unknown dependency must fail");
    return true;
}

bool TestProducerAllocationAndBoundaryValidation() {
    using namespace kxc;
    using namespace kxc::runtime;
    const Device cpu = Device::CPU();
    const Array<ValueSpec> values{
        ValueSpec(0, 0, {1}, Float32(), cpu, true),
        ValueSpec(1, 1, {1}, Float32(), cpu, false, false, true),
    };
    TEST_CHECK(Throws([&] {
                   FrozenTaskPlan plan(
                       kFrozenTaskPlanVersion, values,
                       {TaskSpec(1, TaskKind::kKernel, cpu, {0}, {1}, {}, "k")},
                       {RegionSpec(0, RegionKind::kPerCall, "", {1}, {0}, {1},
                                   {})},
                       {0}, {}, {1});
               }),
               "produced values require an Allocate task");
    TEST_CHECK(Throws([&] {
                   FrozenTaskPlan plan(
                       kFrozenTaskPlanVersion, values,
                       {TaskSpec(0, TaskKind::kAllocate, cpu, {}, {1}, {},
                                 String(), 0, 0, 1),
                        TaskSpec(1, TaskKind::kKernel, cpu, {0}, {1}, {0}, "k")},
                       {RegionSpec(0, RegionKind::kPerCall, "", {0, 1}, {}, {1},
                                   {})},
                       {0}, {}, {1});
               }),
               "region live-ins must be complete");
    return true;
}

bool TestDataOrderAndSingleStreamGuards() {
    using namespace kxc;
    using namespace kxc::runtime;
    const Device cpu = Device::CPU();
    const Array<ValueSpec> values{
        ValueSpec(0, 0, {1}, Float32(), cpu, true),
        ValueSpec(1, 1, {1}, Float32(), cpu),
        ValueSpec(2, 2, {1}, Float32(), cpu, false, false, true),
    };
    TEST_CHECK(Throws([&] {
                   FrozenTaskPlan plan(
                       kFrozenTaskPlanVersion, values,
                       {TaskSpec(0, TaskKind::kAllocate, cpu, {}, {1}, {},
                                 String(), 0, 0, 1),
                        TaskSpec(1, TaskKind::kKernel, cpu, {0}, {1}, {0}, "a"),
                        TaskSpec(2, TaskKind::kAllocate, cpu, {}, {2}, {},
                                 String(), 0, 0, 1),
                        TaskSpec(3, TaskKind::kKernel, cpu, {1}, {2}, {2}, "b")},
                       {RegionSpec(0, RegionKind::kPerCall, "", {0, 1}, {0}, {1},
                                   {}),
                        RegionSpec(1, RegionKind::kPerCall, "", {2, 3}, {1}, {2},
                                   {})},
                       {0}, {}, {2});
               }),
               "a data consumer must depend on its producer");
    TEST_CHECK(Throws([&] {
                   TaskSpec task(0, TaskKind::kEvent, cpu, {}, {}, {}, String(),
                                 0, 1);
                   FrozenTaskPlan plan(
                       kFrozenTaskPlanVersion,
                       {ValueSpec(0, 0, {1}, Float32(), cpu, true, false, true)},
                       {task},
                       {RegionSpec(0, RegionKind::kPerCall, "", {0}, {0}, {},
                                   {})},
                       {0}, {}, {0});
               }),
               "v1 must reject multiple streams before lifecycle support exists");
    return true;
}

bool TestDependencyAwareStorageSharingGuard() {
    using namespace kxc;
    using namespace kxc::runtime;
    const Device cpu = Device::CPU();
    TEST_CHECK(Throws([&] {
                   FrozenTaskPlan plan(
                       kFrozenTaskPlanVersion,
                       {ValueSpec(0, 0, {1}, Float32(), cpu, true),
                        ValueSpec(1, 7, {1}, Float32(), cpu),
                        ValueSpec(2, 7, {1}, Float32(), cpu),
                        ValueSpec(3, 3, {1}, Float32(), cpu, false, false,
                                  true)},
                       {TaskSpec(10, TaskKind::kAllocate, cpu, {}, {1}, {},
                                 String(), 0, 0, 1),
                        TaskSpec(11, TaskKind::kKernel, cpu, {0}, {1}, {10}, "a"),
                        TaskSpec(20, TaskKind::kAllocate, cpu, {}, {2}, {},
                                 String(), 0, 0, 1),
                        TaskSpec(21, TaskKind::kKernel, cpu, {0}, {2}, {20}, "b"),
                        TaskSpec(30, TaskKind::kAllocate, cpu, {}, {3}, {11, 21},
                                 String(), 0, 0, 1),
                        TaskSpec(31, TaskKind::kKernel, cpu, {1, 2}, {3},
                                 {30, 11, 21}, "join")},
                       {RegionSpec(0, RegionKind::kFusion, "", {10, 11, 20, 21,
                                                               30, 31},
                                   {0}, {3}, {})},
                       {0}, {}, {3});
               }),
               "independent live values must not share storage");
    return true;
}

bool TestOrderedEffectRequiresDependencies() {
    using namespace kxc;
    using namespace kxc::runtime;
    const Device cpu = Device::CPU();
    TEST_CHECK(Throws([&] {
                   FrozenTaskPlan plan(
                       kFrozenTaskPlanVersion,
                       {ValueSpec(0, 0, {1}, Float32(), cpu, true, false,
                                  true)},
                       {TaskSpec(10, TaskKind::kEvent, cpu, {}, {}, {}),
                        TaskSpec(20, TaskKind::kEvent, cpu, {}, {}, {})},
                       {RegionSpec(0, RegionKind::kPerCall, "", {10}, {}, {},
                                   {}, RegionEffect::kOrdered),
                        RegionSpec(1, RegionKind::kPerCall, "", {20}, {}, {},
                                   {}, RegionEffect::kOrdered)},
                       {0}, {}, {0});
               }),
               "effectful regions require explicit total dependency order");
    return true;
}

bool TestObjectTypeChecks() {
    using namespace kxc;
    using namespace kxc::runtime;
    TEST_CHECK(Throws([&] { FrozenTaskPlan wrong(ObjectRef(Device::CPU())); }),
               "wrong ObjectRef type must fail safely");
    TEST_CHECK(Throws([&] {
                   FrozenTaskPlan undefined{ObjectRef()};
                   (void)undefined.tasks();
               }),
               "undefined frozen plan must fail on access");
    return true;
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"frozen_dto_and_deterministic_topology",
         TestFrozenDtoAndDeterministicTopology},
        {"task_kind_contracts", TestTaskKindContracts},
        {"cycles_and_missing_dependencies", TestCyclesAndMissingDependencies},
        {"producer_allocation_and_boundary_validation",
         TestProducerAllocationAndBoundaryValidation},
        {"data_order_and_single_stream_guards",
         TestDataOrderAndSingleStreamGuards},
        {"dependency_aware_storage_sharing_guard",
         TestDependencyAwareStorageSharingGuard},
        {"ordered_effect_requires_dependencies",
         TestOrderedEffectRequiresDependencies},
        {"object_type_checks", TestObjectTypeChecks},
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
