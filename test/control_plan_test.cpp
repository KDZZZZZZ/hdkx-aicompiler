/*! \file test/control_plan_test.cpp */

#include <algorithm>
#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "../src/compiler/control_flow/control_plan.h"

namespace {
using kxc::Device;
using namespace kxc::runtime;

#define CHECK(condition, message) do { if (!(condition)) { std::cerr << "[FAIL] " << __FUNCTION__ << ": " << message << "\n"; return false; } } while (0)

bool Throws(const std::function<void()>& fn) {
    try { fn(); } catch (const std::exception&) { return true; }
    return false;
}

EffectSummary Reads(std::vector<ValueId> ids) { return EffectSummary{std::move(ids), {}, {}, false, false}; }
ControlValueSpec I64(ValueId id) {
    return ControlValueSpec{id, "int64", {}, Device::CPU(),
                            "value" + std::to_string(id)};
}
ControlValueSpec Bool(ValueId id) {
    return ControlValueSpec{id, "bool", {}, Device::CPU(),
                            "value" + std::to_string(id)};
}
ControlTask Kernel(TaskId id, std::vector<ValueId> in, std::vector<ValueId> out, const char* ref) {
    ControlTask task;
    task.id = id; task.kind = ControlTaskKind::kKernel; task.binding_state = KernelBindingState::kUnresolvedRelayKernel; task.inputs = std::move(in); task.argument_values = task.inputs; task.outputs = std::move(out);
    task.kernel_ref = ref; task.source_locator = ref;
    task.effect = Reads(task.inputs); return task;
}

ControlPlan BranchPlan() {
    ControlPlan plan;
    plan.values = {Bool(0), I64(1), I64(2), I64(3), I64(4)};
    plan.entry_region = 10; plan.region_order = {10, 11, 12}; plan.graph_inputs = {0, 1}; plan.graph_outputs = {4};
    ControlTask branch;
    branch.id = 20; branch.kind = ControlTaskKind::kBranch; branch.inputs = {0, 1}; branch.outputs = {4}; branch.effect = Reads({0, 1});
    branch.source_locator = "branch";
    branch.branch = BranchSpec{0, 11, 12, {{4, 2, 3}}};
    plan.regions = {
        ControlRegion{10, {0, 1}, {4}, {branch}, Reads({0, 1}), {}, "entry"},
        ControlRegion{11, {1}, {2}, {Kernel(21, {1}, {2}, "then")}, Reads({1}), {}, "then"},
        ControlRegion{12, {1}, {3}, {Kernel(22, {1}, {3}, "else")}, Reads({1}), {}, "else"},
    };
    return plan;
}

ControlPlan LinearPlan() {
    ControlPlan plan;
    plan.values = {I64(0), I64(1), I64(2)};
    plan.entry_region = 10;
    plan.region_order = {10};
    plan.graph_inputs = {0};
    plan.graph_outputs = {2};
    ControlTask first = Kernel(20, {0}, {1}, "first");
    ControlTask second = Kernel(21, {1}, {2}, "second");
    second.dependencies = {20};
    plan.regions = {
        ControlRegion{10, {0}, {2}, {first, second}, Reads({0}), {}, "entry"},
    };
    return plan;
}

ControlPlan LoopPlan(std::int64_t max_trip_count = 3) {
    ControlPlan plan;
    plan.values = {I64(0), I64(1), Bool(2), I64(3), I64(4)};
    plan.entry_region = 10; plan.region_order = {10, 11, 12}; plan.graph_inputs = {0}; plan.graph_outputs = {4};
    ControlTask loop;
    loop.id = 30; loop.kind = ControlTaskKind::kLoop; loop.inputs = {0}; loop.outputs = {4}; loop.effect = Reads({0});
    loop.source_locator = "loop";
    loop.loop = LoopSpec{11, 12, 2, {{4, 0, 1, 3}}, max_trip_count};
    plan.regions = {
        ControlRegion{10, {0}, {4}, {loop}, Reads({0}), {}, "entry"},
        ControlRegion{11, {1}, {2}, {Kernel(31, {1}, {2}, "condition")}, Reads({1}), {}, "condition"},
        ControlRegion{12, {1}, {3}, {Kernel(32, {1}, {3}, "step")}, Reads({1}), {}, "body"},
    };
    return plan;
}

bool TestValidAndCanonical() {
    ControlPlan plan = BranchPlan();
    plan.ValidateStaticExact();
    const std::string first = plan.CanonicalText();
    const std::string second = plan.CanonicalText();
    CHECK(first == second && first.find("ControlPlan/v2") == 0, "canonical text must be stable");
    CHECK(first.find("loc=\"entry\"") != std::string::npos, "locators are diagnostic text");
    ControlPlan reordered = BranchPlan();
    std::reverse(reordered.values.begin(), reordered.values.end());
    std::reverse(reordered.regions.begin(), reordered.regions.end());
    CHECK(reordered.CanonicalText() == first,
          "canonical text must follow ids and explicit region_order");
    LoopPlan().ValidateStaticExact();
    LinearPlan().ValidateStaticExact();
    ControlPlan repeated_operand = LinearPlan();
    repeated_operand.regions[0].tasks[0].argument_values = {0, 0};
    repeated_operand.ValidateStaticExact();
    ControlPlan constant_source = LinearPlan();
    constant_source.graph_inputs.clear();
    constant_source.constant_values = {0};
    constant_source.ValidateStaticExact();

    ControlPlan cuda_data = BranchPlan();
    for (std::size_t i = 1; i < cuda_data.values.size(); ++i) {
        cuda_data.values[i].device = Device::CUDA(1);
    }
    cuda_data.regions[1].tasks[0].device = Device::CUDA(1);
    cuda_data.regions[2].tasks[0].device = Device::CUDA(1);
    cuda_data.ValidateStaticExact();
    return true;
}

bool TestSchemaAndValueContracts() {
    ControlPlan plan = BranchPlan();
    plan.schema_version = 1;
    CHECK(Throws([&] { plan.ValidateStaticExact(); }), "legacy v1 schema must fail closed");
    plan = BranchPlan(); plan.values[2].shape = {-1};
    CHECK(Throws([&] { plan.ValidateStaticExact(); }), "dynamic dimensions must fail");
    plan = BranchPlan(); plan.values[2].dtype = "unknown";
    CHECK(Throws([&] { plan.ValidateStaticExact(); }), "malformed dtype must fail");
    plan = BranchPlan(); plan.values[2].device = Device();
    CHECK(Throws([&] { plan.ValidateStaticExact(); }), "undefined device must fail");
    plan = BranchPlan(); plan.values.push_back(plan.values[0]);
    CHECK(Throws([&] { plan.ValidateStaticExact(); }), "duplicate value must fail");
    plan = BranchPlan(); plan.values[0].source_locator.clear();
    CHECK(Throws([&] { plan.ValidateStaticExact(); }), "missing value locator must fail");
    plan = LinearPlan(); plan.values[1].device = Device::CUDA();
    CHECK(Throws([&] { plan.ValidateStaticExact(); }), "kernel device mismatch must fail");
    plan = LinearPlan(); plan.regions[0].tasks[0].stream = "async";
    CHECK(Throws([&] { plan.ValidateStaticExact(); }), "undeclared stream semantics must fail");
    return true;
}

bool TestTaskAndRegionClosureFailures() {
    ControlPlan plan = BranchPlan();
    plan.regions[0].tasks[0].dependencies = {21};
    CHECK(Throws([&] { plan.ValidateStaticExact(); }), "forward/outside dependency must fail");
    plan = BranchPlan(); plan.regions[0].tasks[0].source_locator.clear();
    CHECK(Throws([&] { plan.ValidateStaticExact(); }), "missing task locator must fail");
    plan = LinearPlan();
    plan.regions[0].tasks[0].binding_state = KernelBindingState::kNotApplicable;
    CHECK(Throws([&] { plan.ValidateStaticExact(); }),
          "kernel tasks must remain explicitly unresolved");
    plan = BranchPlan();
    plan.regions[0].tasks[0].binding_state =
        KernelBindingState::kUnresolvedRelayKernel;
    CHECK(Throws([&] { plan.ValidateStaticExact(); }),
          "branch tasks must not carry a kernel binding");
    plan = LinearPlan(); plan.regions[0].tasks[1].dependencies.clear();
    CHECK(Throws([&] { plan.ValidateStaticExact(); }),
          "data consumer must depend on its local producer");
    plan = BranchPlan(); plan.regions[1].live_ins.clear(); plan.regions[1].effect = Reads({});
    CHECK(Throws([&] { plan.ValidateStaticExact(); }), "missing live-in closure must fail");
    plan = BranchPlan(); plan.regions[1].live_outs.clear();
    CHECK(Throws([&] { plan.ValidateStaticExact(); }), "missing live-out closure must fail");
    plan = BranchPlan(); plan.regions[0].tasks[0].effect.allocates = {4};
    CHECK(Throws([&] { plan.ValidateStaticExact(); }), "allocation effect must fail");
    plan = BranchPlan(); plan.regions[0].tasks[0].effect.writes = {1};
    CHECK(Throws([&] { plan.ValidateStaticExact(); }), "live-in writes must fail");
    plan = BranchPlan(); plan.regions[0].tasks[0].effect.host_callback = true;
    CHECK(Throws([&] { plan.ValidateStaticExact(); }), "host callbacks must fail");
    plan = BranchPlan(); plan.regions[0].tasks[0].effect.device_sync = true;
    CHECK(Throws([&] { plan.ValidateStaticExact(); }), "implicit device sync must fail");
    plan = BranchPlan(); plan.regions[0].tasks[0].alias.may_alias = {{2, 3}};
    CHECK(Throws([&] { plan.ValidateStaticExact(); }), "may-alias must fail");
    plan = BranchPlan(); plan.regions[0].tasks[0].alias.no_alias = {{2, 2}};
    CHECK(Throws([&] { plan.ValidateStaticExact(); }), "invalid no-alias must fail");
    return true;
}

bool TestStructuredWiringFailures() {
    ControlPlan plan = BranchPlan();
    plan.values[4].dtype = "float32";
    CHECK(Throws([&] { plan.ValidateStaticExact(); }), "phi contract mismatch must fail");
    plan = BranchPlan(); plan.values[0].dtype = "int64";
    CHECK(Throws([&] { plan.ValidateStaticExact(); }), "non-bool predicate must fail");
    plan = BranchPlan(); plan.values[0].device = Device::CUDA();
    CHECK(Throws([&] { plan.ValidateStaticExact(); }), "device predicate without copy must fail");
    plan = BranchPlan(); plan.regions[1].live_outs = {1};
    CHECK(Throws([&] { plan.ValidateStaticExact(); }), "phi source must be a branch live-out");
    plan = BranchPlan(); plan.regions[0].tasks[0].branch.then_region = 12;
    CHECK(Throws([&] { plan.ValidateStaticExact(); }), "same branch child region must fail");
    plan = BranchPlan(); plan.regions[2].tasks[0].id = 21;
    CHECK(Throws([&] { plan.ValidateStaticExact(); }), "duplicate task id must fail");
    plan = BranchPlan(); plan.regions[0].tasks[0].branch.then_region = 99;
    CHECK(Throws([&] { plan.ValidateStaticExact(); }), "unknown child region must fail");
    plan = LoopPlan(-1);
    CHECK(Throws([&] { plan.ValidateStaticExact(); }), "negative trip count must fail");
    plan = LoopPlan(); plan.values[3].shape = {1};
    CHECK(Throws([&] { plan.ValidateStaticExact(); }), "shape-changing backedge must fail");
    return true;
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"valid_and_canonical", TestValidAndCanonical},
        {"schema_and_value_contracts", TestSchemaAndValueContracts},
        {"task_and_region_closure_failures", TestTaskAndRegionClosureFailures},
        {"structured_wiring_failures", TestStructuredWiringFailures},
    };
    int failures = 0;
    for (const auto& test : tests) {
        try { if (!test.second()) ++failures; else std::cout << "[PASS] " << test.first << "\n"; }
        catch (const std::exception& error) { std::cerr << "[FAIL] " << test.first << ": " << error.what() << "\n"; ++failures; }
    }
    return failures == 0 ? 0 : 1;
}
