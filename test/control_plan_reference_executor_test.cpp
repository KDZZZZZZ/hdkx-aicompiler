/*! \file test/control_plan_reference_executor_test.cpp */

#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "support/control_plan_reference_executor.h"

namespace {
using kxc::Device;
using namespace kxc::runtime;
using namespace kxc::runtime::test_support;

#define CHECK(condition, message) do { if (!(condition)) { std::cerr << "[FAIL] " << __FUNCTION__ << ": " << message << "\n"; return false; } } while (0)

EffectSummary Reads(std::vector<ValueId> ids) { return EffectSummary{std::move(ids), {}, {}, false, false}; }
ControlValueSpec I64(ValueId id) {
    return ControlValueSpec{id, kxc::TensorType({}, "int64"), Device::CPU(),
                            kxc::api::internal::LogicalValueOrigin::kPrimitiveOutput,
                            kxc::Expr(),
                            "value" + std::to_string(id)};
}
ControlValueSpec Bool(ValueId id) {
    return ControlValueSpec{id, kxc::TensorType({}, "bool"), Device::CPU(),
                            kxc::api::internal::LogicalValueOrigin::kPrimitiveOutput,
                            kxc::Expr(),
                            "value" + std::to_string(id)};
}
ControlTask Kernel(TaskId id, std::vector<ValueId> in, std::vector<ValueId> out, const char* ref) {
    ControlTask task; task.id = id; task.kind = ControlTaskKind::kKernel; task.primitive_unit_id = id; task.inputs = std::move(in); task.argument_values = task.inputs; task.outputs = std::move(out); task.source_locator = ref; task.effect = Reads(task.inputs); return task;
}

ControlPlan BranchPlan() {
    ControlPlan plan; plan.values = {Bool(0), I64(1), I64(2), I64(3), I64(4)};
    plan.entry_region = 10; plan.region_order = {10, 11, 12}; plan.graph_inputs = {0, 1}; plan.graph_outputs = {4};
    ControlTask branch; branch.id = 20; branch.kind = ControlTaskKind::kBranch; branch.inputs = {0, 1}; branch.outputs = {4}; branch.effect = Reads({0, 1}); branch.source_locator = "branch"; branch.branch = {0, 11, 12, {{4, 2, 3}}};
    plan.regions = {{10, {0, 1}, {4}, {branch}, Reads({0, 1}), {}, "entry"},
                    {11, {1}, {2}, {Kernel(21, {1}, {2}, "then")}, Reads({1}), {}, "then"},
                    {12, {1}, {3}, {Kernel(22, {1}, {3}, "else")}, Reads({1}), {}, "else"}};
    return plan;
}

ControlPlan LoopPlan(std::int64_t max_trip_count = 3) {
    ControlPlan plan; plan.values = {I64(0), I64(1), Bool(2), I64(3), I64(4)};
    plan.entry_region = 10; plan.region_order = {10, 11, 12}; plan.graph_inputs = {0}; plan.graph_outputs = {4};
    ControlTask loop; loop.id = 30; loop.kind = ControlTaskKind::kLoop; loop.inputs = {0}; loop.outputs = {4}; loop.effect = Reads({0}); loop.source_locator = "loop"; loop.loop = {11, 12, 2, {{4, 0, 1, 3}}, max_trip_count};
    plan.regions = {{10, {0}, {4}, {loop}, Reads({0}), {}, "entry"},
                    {11, {1}, {2}, {Kernel(31, {1}, {2}, "condition")}, Reads({1}), {}, "condition"},
                    {12, {1}, {3}, {Kernel(32, {1}, {3}, "step")}, Reads({1}), {}, "body"}};
    return plan;
}

FakeKernelCallback Callback() {
    return [](const ControlTask& task, const std::vector<FakeValue>& values) {
        if (task.source_locator == "then") return std::vector<FakeValue>{FakeValue::I64(values[0].integer + 10)};
        if (task.source_locator == "else") return std::vector<FakeValue>{FakeValue::I64(values[0].integer + 20)};
        if (task.source_locator == "condition") return std::vector<FakeValue>{FakeValue::Bool(values[0].integer < 3)};
        if (task.source_locator == "step") return std::vector<FakeValue>{FakeValue::I64(values[0].integer + 1)};
        throw std::invalid_argument("unexpected fake kernel");
    };
}

bool Has(const ControlTrace& trace, const std::string& event) {
    for (const std::string& candidate : trace.events) if (candidate == event) return true;
    return false;
}

bool TestConstantSourceAndRepeatedOperand() {
    ControlPlan plan;
    plan.values = {I64(0), I64(1)};
    plan.entry_region = 10;
    plan.region_order = {10};
    plan.constant_values = {0};
    plan.graph_outputs = {1};
    ControlTask add = Kernel(20, {0}, {1}, "add");
    add.argument_values = {0, 0};
    plan.regions = {
        {10, {0}, {1}, {add}, Reads({0}), {}, "entry"},
    };
    ControlPlanReferenceExecutor executor(
        [](const ControlTask&, const std::vector<FakeValue>& values) {
            if (values.size() != 2) {
                throw std::invalid_argument("logical argument order was lost");
            }
            return std::vector<FakeValue>{
                FakeValue::I64(values[0].integer + values[1].integer)};
        });
    ReferenceExecution result = executor.Execute(plan, {{0, FakeValue::I64(3)}});
    CHECK(result.values.at(1).integer == 6,
          "constants and repeated logical operands must remain executable");
    return true;
}

bool TestTrueFalseAndUnselectedEffects() {
    ControlPlanReferenceExecutor executor(Callback());
    ReferenceExecution true_result = executor.Execute(BranchPlan(), {{0, FakeValue::Bool(true)}, {1, FakeValue::I64(7)}});
    CHECK(true_result.values.at(4).integer == 17, "true branch must select then phi source");
    CHECK(Has(true_result.trace, "task:21") && !Has(true_result.trace, "task:22"), "unselected else task must not execute");
    CHECK(Has(true_result.trace, "branch:20:then") && Has(true_result.trace, "write:20:4"), "branch trace must be structured");
    ReferenceExecution false_result = executor.Execute(BranchPlan(), {{0, FakeValue::Bool(false)}, {1, FakeValue::I64(7)}});
    CHECK(false_result.values.at(4).integer == 27, "false branch must select else phi source");
    CHECK(Has(false_result.trace, "task:22") && !Has(false_result.trace, "task:21"), "unselected then task must not execute");
    return true;
}

bool TestNestedIfAndMultiplePhi() {
    ControlPlan plan = BranchPlan();
    plan.values.push_back(I64(5)); plan.values.push_back(I64(6)); plan.values.push_back(I64(7));
    plan.graph_outputs = {4, 7}; plan.regions[0].live_outs = {4, 7};
    plan.regions[1].tasks[0] = Kernel(21, {1}, {2, 5}, "two-then");
    plan.regions[1].live_outs = {2, 5};
    plan.regions[2].tasks[0] = Kernel(22, {1}, {3, 6}, "two-else");
    plan.regions[2].live_outs = {3, 6};
    plan.regions[0].tasks[0].outputs = {4, 7};
    plan.regions[0].tasks[0].branch.phis.push_back({7, 5, 6});
    ControlPlanReferenceExecutor executor([](const ControlTask& task, const std::vector<FakeValue>& values) {
        if (task.source_locator == "two-then") return std::vector<FakeValue>{FakeValue::I64(values[0].integer), FakeValue::I64(values[0].integer + 1)};
        if (task.source_locator == "two-else") return std::vector<FakeValue>{FakeValue::I64(values[0].integer), FakeValue::I64(values[0].integer + 2)};
        throw std::invalid_argument("unexpected fake kernel");
    });
    ReferenceExecution result = executor.Execute(plan, {{0, FakeValue::Bool(false)}, {1, FakeValue::I64(4)}});
    CHECK(result.values.at(4).integer == 4 && result.values.at(7).integer == 6, "all phi bindings must select one side");

    ControlPlan nested = BranchPlan();
    nested.values.push_back(I64(5)); nested.values.push_back(I64(6));
    nested.region_order = {10, 11, 12, 13, 14};
    ControlTask inner; inner.id = 23; inner.kind = ControlTaskKind::kBranch; inner.inputs = {0, 1}; inner.outputs = {2}; inner.effect = Reads({0, 1}); inner.source_locator = "inner_branch"; inner.branch = {0, 13, 14, {{2, 5, 6}}};
    nested.regions[1] = {11, {0, 1}, {2}, {inner}, Reads({0, 1}), {}, "nested_then"};
    nested.regions.push_back({13, {1}, {5}, {Kernel(24, {1}, {5}, "then")}, Reads({1}), {}, "inner_then"});
    nested.regions.push_back({14, {1}, {6}, {Kernel(25, {1}, {6}, "else")}, Reads({1}), {}, "inner_else"});
    ReferenceExecution nested_result = ControlPlanReferenceExecutor(Callback()).Execute(nested, {{0, FakeValue::Bool(true)}, {1, FakeValue::I64(1)}});
    CHECK(Has(nested_result.trace, "task:23") && Has(nested_result.trace, "task:24") && !Has(nested_result.trace, "task:25"), "nested selected branch only must execute");
    return true;
}

bool TestLoopTripsAndDeterminism() {
    ControlPlanReferenceExecutor executor(Callback());
    for (const auto& case_value : std::vector<std::pair<std::int64_t, std::int64_t>>{{3, 0}, {2, 1}, {0, 3}}) {
        ReferenceExecution result = executor.Execute(LoopPlan(), {{0, FakeValue::I64(case_value.first)}});
        CHECK(result.values.at(4).integer == 3, "loop result must publish current carried value");
        std::int64_t iterations = 0;
        for (const std::string& event : result.trace.events) if (event.find("loop:30:iteration:") == 0) ++iterations;
        CHECK(iterations == case_value.second, "loop must have zero/one/multiple trip behavior");
    }
    ReferenceExecution first = executor.Execute(LoopPlan(), {{0, FakeValue::I64(0)}});
    ReferenceExecution second = executor.Execute(LoopPlan(), {{0, FakeValue::I64(0)}});
    CHECK(first.trace.events == second.trace.events, "reference trace must be deterministic");
    try { executor.Execute(LoopPlan(1), {{0, FakeValue::I64(0)}}); }
    catch (const std::runtime_error&) { return true; }
    CHECK(false, "true condition beyond max must exhaust");
    return false;
}

bool TestMultipleCarriedAndContracts() {
    ControlPlan plan; plan.values = {I64(0), I64(1), Bool(2), I64(3), I64(4), I64(5), I64(6), I64(7), I64(8)};
    plan.entry_region = 10; plan.region_order = {10, 11, 12}; plan.graph_inputs = {0, 5}; plan.graph_outputs = {4, 8};
    ControlTask loop; loop.id = 30; loop.kind = ControlTaskKind::kLoop; loop.inputs = {0, 5}; loop.outputs = {4, 8}; loop.effect = Reads({0, 5}); loop.source_locator = "loop"; loop.loop = {11, 12, 2, {{4, 0, 1, 3}, {8, 5, 6, 7}}, 3};
    plan.regions = {{10, {0, 5}, {4, 8}, {loop}, Reads({0, 5}), {}, "entry"},
                    {11, {1, 6}, {2}, {Kernel(31, {1, 6}, {2}, "condition")}, Reads({1, 6}), {}, "condition"},
                    {12, {1, 6}, {3, 7}, {Kernel(32, {1, 6}, {3, 7}, "two-step")}, Reads({1, 6}), {}, "body"}};
    ControlPlanReferenceExecutor executor([](const ControlTask& task, const std::vector<FakeValue>& values) {
        if (task.source_locator == "condition") return std::vector<FakeValue>{FakeValue::Bool(values[0].integer < 2)};
        if (task.source_locator == "two-step") return std::vector<FakeValue>{FakeValue::I64(values[0].integer + 1), FakeValue::I64(values[1].integer + 2)};
        throw std::invalid_argument("unexpected fake kernel");
    });
    ReferenceExecution result = executor.Execute(plan, {{0, FakeValue::I64(0)}, {5, FakeValue::I64(10)}});
    CHECK(result.values.at(4).integer == 2 && result.values.at(8).integer == 14, "multiple carried values must update independently");
    try { executor.Execute(plan, {{0, FakeValue::Bool(true)}, {5, FakeValue::I64(10)}}); }
    catch (const std::invalid_argument&) { return true; }
    CHECK(false, "input contract mismatch must fail");
    return false;
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"constant_source_repeated_operand", TestConstantSourceAndRepeatedOperand},
        {"true_false_unselected", TestTrueFalseAndUnselectedEffects},
        {"nested_if_multiple_phi", TestNestedIfAndMultiplePhi},
        {"loop_trips_determinism", TestLoopTripsAndDeterminism},
        {"multiple_carried_contracts", TestMultipleCarriedAndContracts},
    };
    int failures = 0;
    for (const auto& test : tests) {
        try { if (!test.second()) ++failures; else std::cout << "[PASS] " << test.first << "\n"; }
        catch (const std::exception& error) { std::cerr << "[FAIL] " << test.first << ": " << error.what() << "\n"; ++failures; }
    }
    return failures == 0 ? 0 : 1;
}
