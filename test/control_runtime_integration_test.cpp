/*! \file test/control_runtime_integration_test.cpp
 * \brief End-to-end binding and CPU execution checks for the control runtime.
 */

#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "kxc/compiler/compiler.h"
#include "kxc/compiler/control_flow.h"
#include "kxc/runtime/control_session.h"
#include "support/control_plan_reference_executor.h"
#include "../src/runtime/internal/compiled_module_node.h"

#ifndef KXC_ENABLE_CONTROL_RUNTIME
#define KXC_ENABLE_CONTROL_RUNTIME 0
#endif

namespace {
using kxc::Array;
using kxc::Device;
using kxc::DeviceStream;
using kxc::Map;
using kxc::String;
using namespace kxc::runtime;
using namespace kxc::runtime::test_support;

#define CHECK(condition, message) do { if (!(condition)) { std::cerr << "[FAIL] " << __FUNCTION__ << ": " << message << "\n"; return false; } } while (0)

bool Throws(const std::function<void()>& fn) {
    try { fn(); } catch (const std::exception&) { return true; }
    return false;
}

std::string ErrorText(const std::function<void()>& fn) {
    try { fn(); } catch (const std::exception& error) { return error.what(); }
    return {};
}

EffectSummary Reads(std::vector<ValueId> values) {
    return EffectSummary{std::move(values), {}, {}, false, false};
}

ControlValueSpec I64(ValueId id) {
    return ControlValueSpec{id, "int64", {}, Device::CPU(),
                            "value" + std::to_string(id)};
}

ControlValueSpec Bool(ValueId id) {
    return ControlValueSpec{id, "bool", {}, Device::CPU(),
                            "value" + std::to_string(id)};
}

ControlTask Kernel(TaskId id, std::vector<ValueId> inputs,
                   std::vector<ValueId> outputs, const char* locator) {
    ControlTask task;
    task.id = id;
    task.kind = ControlTaskKind::kKernel;
    task.binding_state = KernelBindingState::kUnresolvedRelayKernel;
    task.inputs = std::move(inputs);
    task.argument_values = task.inputs;
    task.outputs = std::move(outputs);
    task.kernel_ref = locator;
    task.source_locator = locator;
    task.effect = Reads(task.inputs);
    return task;
}

ControlPlan BranchPlan() {
    ControlPlan plan;
    plan.values = {Bool(0), I64(1), I64(2), I64(3), I64(4)};
    plan.entry_region = 10;
    plan.region_order = {10, 11, 12};
    plan.graph_inputs = {0, 1};
    plan.graph_outputs = {4};
    ControlTask branch;
    branch.id = 20;
    branch.kind = ControlTaskKind::kBranch;
    branch.inputs = {0, 1};
    branch.outputs = {4};
    branch.effect = Reads({0, 1});
    branch.source_locator = "branch";
    branch.branch = {0, 11, 12, {{4, 2, 3}}};
    plan.regions = {
        {10, {0, 1}, {4}, {branch}, Reads({0, 1}), {}, "entry"},
        {11, {1}, {2}, {Kernel(21, {1}, {2}, "decoy.then")}, Reads({1}), {}, "then"},
        {12, {1}, {3}, {Kernel(22, {1}, {3}, "decoy.else")}, Reads({1}), {}, "else"},
    };
    return plan;
}

ControlPlan LoopPlan(std::int64_t max_trips = 3) {
    ControlPlan plan;
    plan.values = {I64(0), I64(1), Bool(2), I64(3), I64(4)};
    plan.entry_region = 10;
    plan.region_order = {10, 11, 12};
    plan.graph_inputs = {0};
    plan.graph_outputs = {4};
    ControlTask loop;
    loop.id = 30;
    loop.kind = ControlTaskKind::kLoop;
    loop.inputs = {0};
    loop.outputs = {4};
    loop.effect = Reads({0});
    loop.source_locator = "loop";
    loop.loop = {11, 12, 2, {{4, 0, 1, 3}}, max_trips};
    plan.regions = {
        {10, {0}, {4}, {loop}, Reads({0}), {}, "entry"},
        {11, {1}, {2}, {Kernel(31, {1}, {2}, "decoy.condition")}, Reads({1}), {}, "condition"},
        {12, {1}, {3}, {Kernel(32, {1}, {3}, "decoy.step")}, Reads({1}), {}, "body"},
    };
    return plan;
}

ControlPlan MultiBranchPlan() {
    ControlPlan plan = BranchPlan();
    plan.values.push_back(I64(5));
    plan.values.push_back(I64(6));
    plan.values.push_back(I64(7));
    plan.graph_outputs = {4, 7};
    plan.regions[0].live_outs = {4, 7};
    plan.regions[0].tasks[0].outputs = {4, 7};
    plan.regions[0].tasks[0].branch.phis.push_back({7, 5, 6});
    plan.regions[1].live_outs = {2, 5};
    plan.regions[1].tasks.push_back(Kernel(23, {1}, {5}, "decoy.then.second"));
    plan.regions[2].live_outs = {3, 6};
    plan.regions[2].tasks.push_back(Kernel(24, {1}, {6}, "decoy.else.second"));
    return plan;
}

ControlPlan MultiLoopPlan() {
    ControlPlan plan;
    plan.values = {I64(0), I64(1), Bool(2), I64(3), I64(4),
                   I64(5), I64(6), I64(7), I64(8)};
    plan.entry_region = 10;
    plan.region_order = {10, 11, 12};
    plan.graph_inputs = {0, 5};
    plan.graph_outputs = {4, 8};
    ControlTask loop;
    loop.id = 30;
    loop.kind = ControlTaskKind::kLoop;
    loop.inputs = {0, 5};
    loop.outputs = {4, 8};
    loop.effect = Reads({0, 5});
    loop.source_locator = "multi-loop";
    loop.loop = {11, 12, 2, {{4, 0, 1, 3}, {8, 5, 6, 7}}, 3};
    plan.regions = {
        {10, {0, 5}, {4, 8}, {loop}, Reads({0, 5}), {}, "entry"},
        {11, {1, 6}, {2}, {Kernel(31, {1}, {2}, "decoy.condition")},
         Reads({1, 6}), {}, "condition"},
        {12, {1, 6}, {3, 7},
         {Kernel(32, {1}, {3}, "decoy.step.first"),
          Kernel(33, {6}, {7}, "decoy.step.second")},
         Reads({1, 6}), {}, "body"},
    };
    return plan;
}

ControlPlan TwoOutputKernelPlan() {
    ControlPlan plan;
    plan.values = {I64(0), I64(1), I64(2)};
    plan.entry_region = 10;
    plan.region_order = {10};
    plan.graph_inputs = {0};
    plan.graph_outputs = {1, 2};
    plan.regions = {
        {10, {0}, {1, 2}, {Kernel(50, {0}, {1, 2}, "decoy.two-output")},
         Reads({0}), {}, "entry"},
    };
    return plan;
}

ControlPlan ConstantOrderPlan() {
    ControlPlan plan;
    plan.values = {I64(0), I64(1), I64(2), I64(3)};
    plan.entry_region = 10;
    plan.region_order = {10};
    plan.graph_inputs = {0};
    plan.constant_values = {1, 2};
    plan.graph_outputs = {3};
    ControlTask kernel = Kernel(41, {0, 1, 2}, {3}, "decoy.constants");
    kernel.argument_values = {1, 0, 2};
    plan.regions = {
        {10, {0, 1, 2}, {3}, {kernel}, Reads({0, 1, 2}), {}, "entry"},
    };
    return plan;
}

ControlPlan RepeatedOperandPlan() {
    ControlPlan plan;
    plan.values = {I64(0), I64(1)};
    plan.entry_region = 10;
    plan.region_order = {10};
    plan.graph_inputs = {0};
    plan.graph_outputs = {1};
    ControlTask add = Kernel(40, {0}, {1}, "decoy.duplicate");
    add.argument_values = {0, 0};
    plan.regions = {{10, {0}, {1}, {add}, Reads({0}), {}, "entry"}};
    return plan;
}

enum class ScalarOp { kThen, kElse, kCondition, kStep, kSum, kConstantOrder };

class ScalarLauncher final : public kxc::codegen::KernelLauncher {
public:
    explicit ScalarLauncher(ScalarOp op, bool invalid_completion = false)
        : op_(op), invalid_completion_(invalid_completion) {}

    bool IsReady() const noexcept override { return true; }

    kxc::AsyncOperation Launch(const Array<NDArray>& arguments,
                               const DeviceStream& stream,
                               const kxc::ObjectRef&) const override {
        ++calls;
        last_arguments = arguments;
        if (invalid_completion_) return kxc::AsyncOperation();
        if (arguments.empty()) throw std::invalid_argument("scalar launcher has no arguments");
        const std::int64_t first = ReadI64(arguments[0]);
        NDArray output = arguments[arguments.size() - 1];
        switch (op_) {
            case ScalarOp::kThen: WriteI64(output, first + 10); break;
            case ScalarOp::kElse: WriteI64(output, first + 20); break;
            case ScalarOp::kCondition: WriteBool(output, first < 3); break;
            case ScalarOp::kStep: WriteI64(output, first + 1); break;
            case ScalarOp::kSum: WriteI64(output, first + ReadI64(arguments[1])); break;
            case ScalarOp::kConstantOrder:
                WriteI64(output, first + 10 * ReadI64(arguments[1]) +
                                     ReadI64(arguments[2]));
                break;
        }
        Array<kxc::Storage> retained;
        for (const NDArray& argument : arguments) retained.push_back(argument.storage());
        return kxc::AsyncOperation::Completed(stream, std::move(retained));
    }

    mutable int calls{0};
    mutable Array<NDArray> last_arguments;

private:
    static std::int64_t ReadI64(const NDArray& array) {
        std::int64_t value{0};
        array.CopyToBytes(&value, sizeof(value));
        return value;
    }
    static void WriteI64(const NDArray& array, std::int64_t value) {
        array.CopyFromBytes(&value, sizeof(value));
    }
    static void WriteBool(const NDArray& array, bool value) {
        const std::uint8_t byte = value ? 1 : 0;
        array.CopyFromBytes(&byte, sizeof(byte));
    }

    ScalarOp op_;
    bool invalid_completion_{false};
};

DLDataType Type(const char* name) { return DataTypeFromString(name); }

NDArray ScalarI64(std::int64_t value) {
    NDArray result = NDArray::Empty({}, Type("int64"), Device::CPU());
    result.CopyFromBytes(&value, sizeof(value));
    return result;
}

NDArray ScalarBool(bool value) {
    NDArray result = NDArray::Empty({}, Type("bool"), Device::CPU());
    const std::uint8_t byte = value ? 1 : 0;
    result.CopyFromBytes(&byte, sizeof(byte));
    return result;
}

std::int64_t ReadI64(const NDArray& value) {
    std::int64_t result{0};
    value.CopyToBytes(&result, sizeof(result));
    return result;
}

kxc::api::CompiledModule MakeModule(
    const String& symbol, const Array<kxc::codegen::KernelArgSpec>& arguments,
    const std::shared_ptr<ScalarLauncher>& launcher,
    const Map<String, NDArray>& constants = {}) {
    using namespace kxc;
    using namespace kxc::codegen;
    const KernelSignature signature(symbol, arguments);
    const KernelLaunchMetadata metadata(Device::CPU(), CodeGenBackend::kLLVM);
    return api::internal::BuildCompiledModule(
        BuildTarget(Device::CPU()),
        {api::internal::CompiledModuleEntry{
            tir::PrimFunc(), signature, metadata,
            CompiledKernel(signature, metadata, launcher)}}, constants);
}

struct Fixture {
    kxc::api::CompiledModule module;
    std::shared_ptr<ScalarLauncher> launcher;
};

Fixture MakeFixture(const char* symbol, ScalarOp op,
                    std::vector<kxc::codegen::KernelArgRole> roles,
                    std::vector<const char*> types = {},
                    bool invalid_completion = false) {
    using namespace kxc::codegen;
    Array<KernelArgSpec> arguments;
    for (std::size_t i = 0; i < roles.size(); ++i) {
        const char* type = types.empty() ? (op == ScalarOp::kCondition && i + 1 == roles.size() ? "bool" : "int64") : types[i];
        arguments.push_back(KernelArgSpec(
            "arg" + std::to_string(i), roles[i], Type(type), {}, Device::CPU(),
            1, roles[i] == KernelArgRole::kOutput));
    }
    auto launcher =
        std::make_shared<ScalarLauncher>(op, invalid_completion);
    return {MakeModule(symbol, arguments, launcher), std::move(launcher)};
}

Fixture MakeConstantOrderFixture() {
    using namespace kxc::codegen;
    Array<KernelArgSpec> arguments{
        KernelArgSpec("input", KernelArgRole::kInput, Type("int64"), {},
                      Device::CPU()),
        KernelArgSpec("constant_a", KernelArgRole::kConstant, Type("int64"),
                      {}, Device::CPU(), 1, false, "constant.a"),
        KernelArgSpec("constant_b", KernelArgRole::kConstant, Type("int64"),
                      {}, Device::CPU(), 1, false, "constant.b"),
        KernelArgSpec("output", KernelArgRole::kOutput, Type("int64"), {},
                      Device::CPU(), 1, true),
    };
    Map<String, NDArray> constants;
    constants.Set("constant.a", ScalarI64(2));
    constants.Set("constant.b", ScalarI64(3));
    auto launcher = std::make_shared<ScalarLauncher>(ScalarOp::kConstantOrder);
    return {MakeModule("constant_order_entry", arguments, launcher, constants),
            std::move(launcher)};
}

struct BranchFixture {
    Fixture then_kernel{MakeFixture("then_entry", ScalarOp::kThen,
                                    {kxc::codegen::KernelArgRole::kInput,
                                     kxc::codegen::KernelArgRole::kOutput})};
    Fixture else_kernel{MakeFixture("else_entry", ScalarOp::kElse,
                                    {kxc::codegen::KernelArgRole::kInput,
                                     kxc::codegen::KernelArgRole::kOutput})};
};

struct LoopFixture {
    Fixture condition{MakeFixture("condition_entry", ScalarOp::kCondition,
                                  {kxc::codegen::KernelArgRole::kInput,
                                   kxc::codegen::KernelArgRole::kOutput})};
    Fixture step{MakeFixture("step_entry", ScalarOp::kStep,
                             {kxc::codegen::KernelArgRole::kInput,
                              kxc::codegen::KernelArgRole::kOutput})};
};

std::vector<kxc::api::ControlKernelBinding> Bindings(const BranchFixture& fixture) {
    return {{21, fixture.then_kernel.module, "then_entry", 7, {1}},
            {22, fixture.else_kernel.module, "else_entry", 8, {1}}};
}

std::vector<kxc::api::ControlKernelBinding> Bindings(const LoopFixture& fixture) {
    return {{31, fixture.condition.module, "condition_entry", 9, {1}},
            {32, fixture.step.module, "step_entry", 10, {1}}};
}

std::int64_t ReferenceIterations(const ReferenceExecution& execution,
                                 TaskId task_id) {
    const std::string prefix = "loop:" + std::to_string(task_id) + ":iteration:";
    std::int64_t count{0};
    for (const std::string& event : execution.trace.events) {
        if (event.compare(0, prefix.size(), prefix) == 0) ++count;
    }
    return count;
}

FakeKernelCallback ReferenceKernel() {
    return [](const ControlTask& task, const std::vector<FakeValue>& values) {
        switch (task.id) {
            case 21:
            case 23: return std::vector<FakeValue>{FakeValue::I64(values[0].integer + 10)};
            case 22:
            case 24: return std::vector<FakeValue>{FakeValue::I64(values[0].integer + 20)};
            case 31: return std::vector<FakeValue>{FakeValue::Bool(values[0].integer < 3)};
            case 32:
            case 33: return std::vector<FakeValue>{FakeValue::I64(values[0].integer + 1)};
            case 40: return std::vector<FakeValue>{FakeValue::I64(values[0].integer + values[1].integer)};
            default: throw std::invalid_argument("unknown reference task id");
        }
    };
}

bool TestCompilerDefaultStillRejectsIf() {
    const kxc::TensorType boolean({}, "bool");
    const kxc::TensorType integer({}, "int64");
    kxc::Var predicate("predicate", boolean);
    kxc::Var lhs("lhs", integer);
    kxc::Var rhs("rhs", integer);
    const kxc::Function function({predicate, lhs, rhs},
                                 kxc::If(predicate, lhs, rhs));
    const std::string error = ErrorText([&] {
        (void)kxc::api::Compiler::Compile(
            function,
            kxc::api::CompileConfig::Create(kxc::BuildTarget(Device::CPU())));
    });
    CHECK(error.find("mode=static_exact") != std::string::npos &&
              error.find("missing control_flow.if") != std::string::npos,
          "Compiler::Compile must keep the default Relay If rejection");
    return true;
}

bool TestBindingAndBranchDifferential() {
    BranchFixture fixture;
    const ControlPlan original = BranchPlan();
    ControlPlan decoy_changed = original;
    decoy_changed.regions[1].tasks[0].kernel_ref = "unrelated-and-invalid-looking";
    decoy_changed.regions[2].tasks[0].kernel_ref = "also-not-an-entry";
    const auto bindings = Bindings(fixture);
    const ControlExecutionPlan bound = kxc::api::BindControlPlanForRuntime(original, bindings);
    const ControlExecutionPlan decoy_bound =
        kxc::api::BindControlPlanForRuntime(decoy_changed, bindings);
    CHECK(bound.spec().regions[1].tasks[0].kernel.signature()->symbol ==
              decoy_bound.spec().regions[1].tasks[0].kernel.signature()->symbol &&
              bound.spec().regions[2].tasks[0].kernel.signature()->symbol ==
              decoy_bound.spec().regions[2].tasks[0].kernel.signature()->symbol,
          "binding must use task id and supplied entry, never kernel_ref");

    const ControlPlanReferenceExecutor reference(ReferenceKernel());
    const ReferenceExecution expected_true = reference.Execute(
        original, {{0, FakeValue::Bool(true)}, {1, FakeValue::I64(7)}});
    const ReferenceExecution expected_false = reference.Execute(
        original, {{0, FakeValue::Bool(false)}, {1, FakeValue::I64(7)}});
#if KXC_ENABLE_CONTROL_RUNTIME
    auto* selected_module = const_cast<kxc::api::CompiledModuleNode*>(
        fixture.then_kernel.module.As<kxc::api::CompiledModuleNode>());
    selected_module->entries_.clear();
    ControlRuntimeSession session(bound);
    const ControlRunResult actual_true = session.Run({ScalarBool(true), ScalarI64(7)});
    CHECK(actual_true.outputs.size() == 1 && ReadI64(actual_true.outputs[0]) == expected_true.values.at(4).integer &&
              actual_true.events == expected_true.trace.events &&
              fixture.then_kernel.launcher->calls == 1 && fixture.else_kernel.launcher->calls == 0,
          "true branch must exactly match reference and not launch else");
    const ControlRunResult actual_false = session.Run({ScalarBool(false), ScalarI64(7)});
    CHECK(actual_false.outputs.size() == 1 && ReadI64(actual_false.outputs[0]) == expected_false.values.at(4).integer &&
              actual_false.events == expected_false.trace.events &&
              fixture.then_kernel.launcher->calls == 1 && fixture.else_kernel.launcher->calls == 1,
          "false branch must exactly match reference and not relaunch then");

    BranchFixture multi_fixture;
    auto multi_bindings = Bindings(multi_fixture);
    multi_bindings.push_back(
        {23, multi_fixture.then_kernel.module, "then_entry", 13, {1}});
    multi_bindings.push_back(
        {24, multi_fixture.else_kernel.module, "else_entry", 14, {1}});
    const ControlPlan multi_plan = MultiBranchPlan();
    const ControlExecutionPlan multi_bound =
        kxc::api::BindControlPlanForRuntime(multi_plan, multi_bindings);
    const ReferenceExecution multi_expected_true = reference.Execute(
        multi_plan, {{0, FakeValue::Bool(true)}, {1, FakeValue::I64(4)}});
    const ReferenceExecution multi_expected_false = reference.Execute(
        multi_plan, {{0, FakeValue::Bool(false)}, {1, FakeValue::I64(4)}});
    ControlRuntimeSession multi_session(multi_bound);
    const ControlRunResult multi_true =
        multi_session.Run({ScalarBool(true), ScalarI64(4)});
    CHECK(multi_true.outputs.size() == 2 &&
              ReadI64(multi_true.outputs[0]) == multi_expected_true.values.at(4).integer &&
              ReadI64(multi_true.outputs[1]) == multi_expected_true.values.at(7).integer &&
              multi_true.events == multi_expected_true.trace.events &&
              multi_fixture.then_kernel.launcher->calls == 2 &&
              multi_fixture.else_kernel.launcher->calls == 0,
          "all true-side Phi bindings must forward and else must stay unlaunched");
    const ControlRunResult multi_false =
        multi_session.Run({ScalarBool(false), ScalarI64(4)});
    CHECK(multi_false.outputs.size() == 2 &&
              ReadI64(multi_false.outputs[0]) == multi_expected_false.values.at(4).integer &&
              ReadI64(multi_false.outputs[1]) == multi_expected_false.values.at(7).integer &&
              multi_false.events == multi_expected_false.trace.events &&
              multi_fixture.then_kernel.launcher->calls == 2 &&
              multi_fixture.else_kernel.launcher->calls == 2,
          "all false-side Phi bindings must forward and then must not relaunch");
#else
    NDArray direct_output = ScalarI64(0);
    CHECK(Throws([&] {
              (void)bound.spec().regions[1].tasks[0].kernel.Launch(
                  {ScalarI64(1), direct_output},
                  DeviceStream::Default(Device::CPU()));
          }) && Throws([&] { ControlRuntimeSession disabled(bound); }) &&
              fixture.then_kernel.launcher->calls == 0 && fixture.else_kernel.launcher->calls == 0,
          "gate OFF must reject direct bound launch and session construction");
#endif
    return true;
}

bool TestLoopDifferentialAndBound() {
    LoopFixture fixture;
    const ControlPlanReferenceExecutor reference(ReferenceKernel());
    const ControlExecutionPlan bound = kxc::api::BindControlPlanForRuntime(LoopPlan(), Bindings(fixture));
#if KXC_ENABLE_CONTROL_RUNTIME
    ControlRuntimeSession session(bound);
    for (const auto& sample : std::vector<std::pair<std::int64_t, std::int64_t>>{{3, 0}, {2, 1}, {0, 3}}) {
        const ReferenceExecution expected = reference.Execute(LoopPlan(), {{0, FakeValue::I64(sample.first)}});
        const ControlRunResult actual = session.Run({ScalarI64(sample.first)});
        CHECK(actual.outputs.size() == 1 && ReadI64(actual.outputs[0]) == expected.values.at(4).integer &&
                  actual.events == expected.trace.events &&
                  actual.loop_iterations.size() == 1 && actual.loop_iterations[0].task_id == 30 &&
                  actual.loop_iterations[0].iterations == ReferenceIterations(expected, 30) &&
                  actual.loop_iterations[0].iterations == sample.second,
              "zero/one/multi-trip loop must match typed reference semantics and trace");
    }
    LoopFixture multi_fixture;
    auto multi_bindings = Bindings(multi_fixture);
    multi_bindings.push_back(
        {33, multi_fixture.step.module, "step_entry", 15, {6}});
    const ControlPlan multi_plan = MultiLoopPlan();
    const ControlExecutionPlan multi_bound =
        kxc::api::BindControlPlanForRuntime(multi_plan, multi_bindings);
    const ReferenceExecution multi_expected = reference.Execute(
        multi_plan, {{0, FakeValue::I64(0)}, {5, FakeValue::I64(10)}});
    const ControlRunResult multi_actual = ControlRuntimeSession(multi_bound).Run(
        {ScalarI64(0), ScalarI64(10)});
    CHECK(multi_actual.outputs.size() == 2 &&
              ReadI64(multi_actual.outputs[0]) == multi_expected.values.at(4).integer &&
              ReadI64(multi_actual.outputs[1]) == multi_expected.values.at(8).integer &&
              multi_actual.events == multi_expected.trace.events &&
              multi_actual.loop_iterations.size() == 1 &&
              multi_actual.loop_iterations[0].iterations == 3 &&
              multi_fixture.condition.launcher->calls == 4 &&
              multi_fixture.step.launcher->calls == 6,
          "multiple loop-carried values and backedges must match the reference");

    const ControlExecutionPlan exhausted =
        kxc::api::BindControlPlanForRuntime(LoopPlan(1), Bindings(fixture));
    CHECK(Throws([&] {
              (void)reference.Execute(LoopPlan(1), {{0, FakeValue::I64(0)}});
          }) && Throws([&] { ControlRuntimeSession(exhausted).Run({ScalarI64(0)}); }),
          "reference and runtime must both exhaust max_trip_count");
#else
    CHECK(Throws([&] { ControlRuntimeSession disabled(bound); }) &&
              fixture.condition.launcher->calls == 0 && fixture.step.launcher->calls == 0,
          "disabled control runtime must not dispatch loop kernels");
#endif
    return true;
}

bool TestRepeatedOperandAbiOrder() {
    Fixture sum = MakeFixture("sum_entry", ScalarOp::kSum,
                              {kxc::codegen::KernelArgRole::kInput,
                               kxc::codegen::KernelArgRole::kInput,
                               kxc::codegen::KernelArgRole::kOutput});
    const ControlPlan plan = RepeatedOperandPlan();
    const ControlExecutionPlan bound = kxc::api::BindControlPlanForRuntime(
        plan, {{40, sum.module, "sum_entry", 11, {0, 0}}});
#if KXC_ENABLE_CONTROL_RUNTIME
    const ControlRunResult result = ControlRuntimeSession(bound).Run({ScalarI64(6)});
    CHECK(result.outputs.size() == 1 && ReadI64(result.outputs[0]) == 12 && sum.launcher->calls == 1 &&
              sum.launcher->last_arguments.size() == 3 &&
              sum.launcher->last_arguments[0].get() == sum.launcher->last_arguments[1].get(),
          "duplicate logical operands must retain ordered duplicate ABI arguments");
#else
    CHECK(Throws([&] { ControlRuntimeSession disabled(bound); }) && sum.launcher->calls == 0,
          "disabled runtime must reject repeated-operand plan without launch");
#endif
    return true;
}

bool TestConstantAbiOrderAndResolvedValidation() {
    Fixture fixture = MakeConstantOrderFixture();
    const ControlPlan plan = ConstantOrderPlan();
    const kxc::api::ControlKernelBinding exact{
        41, fixture.module, "constant_order_entry", 12, {0, 1, 2}};
    const ControlExecutionPlan bound =
        kxc::api::BindControlPlanForRuntime(plan, {exact});
    auto swapped = exact;
    swapped.abi_non_output_value_ids = {0, 2, 1};
    CHECK(Throws([&] {
              (void)kxc::api::BindControlPlanForRuntime(plan, {swapped});
          }) && fixture.launcher->calls == 0,
          "adapter must reject reordered same-contract constants");

    Fixture malformed_constant = MakeConstantOrderFixture();
    auto* malformed_module = const_cast<kxc::api::CompiledModuleNode*>(
        malformed_constant.module.As<kxc::api::CompiledModuleNode>());
    auto* malformed_payload = const_cast<NDArrayNode*>(
        malformed_module->constants_.at("constant.a").As<NDArrayNode>());
    malformed_payload->byte_offset =
        malformed_payload->storage.capacity_bytes();
    const kxc::api::ControlKernelBinding malformed_constant_binding{
        41, malformed_constant.module, "constant_order_entry", 16, {0, 1, 2}};
    CHECK(Throws([&] {
              (void)kxc::api::BindControlPlanForRuntime(
                  plan, {malformed_constant_binding});
          }) && malformed_constant.launcher->calls == 0,
          "malformed constant storage must fail during immutable binding");

    ControlExecutionPlanSpec malformed = bound.spec();
    malformed.regions[0].tasks[0].kind =
        static_cast<ControlExecutionTaskKind>(999);
    CHECK(Throws([&] { (void)ControlExecutionPlan(malformed); }),
          "resolved verifier must reject unknown public task enum values");

    Fixture two_output = MakeFixture(
        "two_output_entry", ScalarOp::kThen,
        {kxc::codegen::KernelArgRole::kInput,
         kxc::codegen::KernelArgRole::kOutput,
         kxc::codegen::KernelArgRole::kOutput});
    const ControlExecutionPlan ordered_outputs =
        kxc::api::BindControlPlanForRuntime(
            TwoOutputKernelPlan(),
            {{50, two_output.module, "two_output_entry", 17, {0}}});
    ControlExecutionPlanSpec swapped_outputs = ordered_outputs.spec();
    swapped_outputs.regions[0].tasks[0].argument_values = {0, 2, 1};
    ControlExecutionPlanSpec duplicate_outputs = ordered_outputs.spec();
    duplicate_outputs.regions[0].tasks[0].argument_values = {0, 1, 1};
    CHECK(Throws([&] { (void)ControlExecutionPlan(swapped_outputs); }) &&
              Throws([&] { (void)ControlExecutionPlan(duplicate_outputs); }) &&
              two_output.launcher->calls == 0,
          "resolved kernel ABI must preserve exact output order and uniqueness");
#if KXC_ENABLE_CONTROL_RUNTIME
    const ControlRunResult result =
        ControlRuntimeSession(bound).Run({ScalarI64(5)});
    CHECK(result.outputs.size() == 1 && ReadI64(result.outputs[0]) == 28 &&
              fixture.launcher->calls == 1,
          "exact constant ABI order must preserve constant-key semantics");
#else
    CHECK(Throws([&] { ControlRuntimeSession disabled(bound); }) &&
              fixture.launcher->calls == 0,
          "gate-off constant plan must not launch");
#endif
    return true;
}

bool TestBindingAndValidationNegatives() {
    BranchFixture fixture;
    const ControlPlan branch = BranchPlan();
    const auto bindings = Bindings(fixture);
    CHECK(Throws([&] { (void)kxc::api::BindControlPlanForRuntime(branch, {bindings[0]}); }),
          "missing kernel binding must fail");
    auto duplicate = bindings;
    duplicate.push_back(bindings[0]);
    CHECK(Throws([&] { (void)kxc::api::BindControlPlanForRuntime(branch, duplicate); }),
          "duplicate task binding must fail");
    auto extra = bindings;
    extra.push_back({99, fixture.then_kernel.module, "then_entry", 12, {1}});
    CHECK(Throws([&] { (void)kxc::api::BindControlPlanForRuntime(branch, extra); }),
          "extra absent-task binding must fail");
    auto non_kernel = bindings;
    non_kernel.push_back({20, fixture.then_kernel.module, "then_entry", 12, {1}});
    CHECK(Throws([&] { (void)kxc::api::BindControlPlanForRuntime(branch, non_kernel); }),
          "non-kernel binding must fail");
    auto zero_generation = bindings;
    zero_generation[0].generation = 0;
    CHECK(Throws([&] { (void)kxc::api::BindControlPlanForRuntime(branch, zero_generation); }),
          "generation zero must fail");
    auto wrong_abi = bindings;
    wrong_abi[0].abi_non_output_value_ids = {0};
    CHECK(Throws([&] { (void)kxc::api::BindControlPlanForRuntime(branch, wrong_abi); }),
          "wrong explicit ABI ids must fail");

    Fixture wrong_signature = MakeFixture("wrong_signature", ScalarOp::kSum,
        {kxc::codegen::KernelArgRole::kInput, kxc::codegen::KernelArgRole::kInput,
         kxc::codegen::KernelArgRole::kOutput});
    auto signature_mismatch = bindings;
    signature_mismatch[0] = {21, wrong_signature.module, "wrong_signature", 12, {1, 1}};
    CHECK(Throws([&] { (void)kxc::api::BindControlPlanForRuntime(branch, signature_mismatch); }),
          "wrong module signature ABI must fail before runtime launch");
    Fixture bad_output = MakeFixture("bad_output", ScalarOp::kThen,
        {kxc::codegen::KernelArgRole::kInput, kxc::codegen::KernelArgRole::kOutput},
        {"int64", "bool"});
    auto bad_contract = bindings;
    bad_contract[0].module = bad_output.module;
    bad_contract[0].entry_symbol = "bad_output";
    CHECK(Throws([&] { (void)kxc::api::BindControlPlanForRuntime(branch, bad_contract); }) &&
              wrong_signature.launcher->calls == 0 && bad_output.launcher->calls == 0,
          "wrong module output contract must fail before runtime launch");

    ControlPlan dynamic = branch;
    dynamic.values[1].shape = {-1};
    ControlPlan effects = branch;
    effects.regions[0].tasks[0].effect.host_callback = true;
    ControlPlan aliases = branch;
    aliases.regions[0].tasks[0].alias.may_alias = {{2, 3}};
    CHECK(Throws([&] { (void)kxc::api::BindControlPlanForRuntime(dynamic, bindings); }) &&
              Throws([&] { (void)kxc::api::BindControlPlanForRuntime(effects, bindings); }) &&
              Throws([&] { (void)kxc::api::BindControlPlanForRuntime(aliases, bindings); }) &&
              fixture.then_kernel.launcher->calls == 0 && fixture.else_kernel.launcher->calls == 0,
          "invalid source dynamic/effect/alias plans must fail before dispatch");

    const ControlExecutionPlan valid =
        kxc::api::BindControlPlanForRuntime(branch, bindings);
    ControlExecutionPlanSpec bad_provenance = valid.spec();
    bad_provenance.source_control_plan_version = 1;
    ControlExecutionPlanSpec bad_predicate = valid.spec();
    bad_predicate.values[0].dtype = "int64";
    ControlExecutionPlanSpec bad_phi = valid.spec();
    bad_phi.regions[0].tasks[0].branch.phis[0].then_value = 1;
    ControlExecutionPlanSpec bad_placement = valid.spec();
    bad_placement.regions[0].tasks[0].stream = "borrowed";
    ControlExecutionPlanSpec bad_scope = valid.spec();
    bad_scope.regions[0].tasks[0].inputs = {0, 1, 3};
    bad_scope.regions[1].live_ins = {3};
    bad_scope.regions[1].tasks[0].inputs = {3};
    bad_scope.regions[1].tasks[0].argument_values = {3, 2};

    LoopFixture loop_fixture;
    const ControlExecutionPlan valid_loop = kxc::api::BindControlPlanForRuntime(
        LoopPlan(), Bindings(loop_fixture));
    ControlExecutionPlanSpec unbounded_loop = valid_loop.spec();
    unbounded_loop.regions[0].tasks[0].loop.max_trip_count = -1;
    ControlExecutionPlanSpec bad_backedge = valid_loop.spec();
    bad_backedge.regions[0].tasks[0].loop.carried[0].backedge = 0;
    CHECK(Throws([&] { (void)ControlExecutionPlan(bad_provenance); }) &&
              Throws([&] { (void)ControlExecutionPlan(bad_predicate); }) &&
              Throws([&] { (void)ControlExecutionPlan(bad_phi); }) &&
              Throws([&] { (void)ControlExecutionPlan(bad_placement); }) &&
              Throws([&] { (void)ControlExecutionPlan(bad_scope); }) &&
              Throws([&] { (void)ControlExecutionPlan(unbounded_loop); }) &&
              Throws([&] { (void)ControlExecutionPlan(bad_backedge); }) &&
              fixture.then_kernel.launcher->calls == 0 &&
              fixture.else_kernel.launcher->calls == 0 &&
              loop_fixture.condition.launcher->calls == 0 &&
              loop_fixture.step.launcher->calls == 0,
          "resolved verifier must close provenance, predicate, Phi, scope, placement, and loop invariants");

#if KXC_ENABLE_CONTROL_RUNTIME
    ControlRuntimeSession session(valid);
    CHECK(Throws([&] { (void)session.Run({ScalarBool(true)}); }) &&
              Throws([&] { (void)session.Run({ScalarI64(1), ScalarI64(2)}); }) &&
              fixture.then_kernel.launcher->calls == 0 && fixture.else_kernel.launcher->calls == 0,
          "wrong runtime input contract must fail before any launcher");

    NDArray malformed_predicate = ScalarBool(true);
    auto* predicate_node =
        const_cast<NDArrayNode*>(malformed_predicate.As<NDArrayNode>());
    predicate_node->byte_offset = predicate_node->storage.capacity_bytes();
    auto* borrowed_node = new kxc::DeviceStreamNode();
    DeviceStream borrowed_stream(borrowed_node);
    borrowed_node->device = Device::CPU();
    borrowed_node->backend_handle = reinterpret_cast<void*>(1);
    borrowed_node->owns_handle = false;
    CHECK(Throws([&] {
              (void)session.Run({malformed_predicate, ScalarI64(7)});
          }) && Throws([&] {
              (void)session.RunAsync(
                  {ScalarBool(true), ScalarI64(7)}, borrowed_stream);
          }) && fixture.then_kernel.launcher->calls == 0 &&
              fixture.else_kernel.launcher->calls == 0,
          "predicate range and canonical default stream must reject before launch");

    NDArray out_of_range = ScalarI64(7);
    auto* malformed = const_cast<NDArrayNode*>(out_of_range.As<NDArrayNode>());
    malformed->byte_offset = malformed->storage.capacity_bytes();
    CHECK(Throws([&] {
              (void)session.Run({ScalarBool(true), out_of_range});
          }) && fixture.then_kernel.launcher->calls == 0,
          "bound entry launch must retain CompiledModule range validation");

    Fixture invalid_completion = MakeFixture(
        "invalid_completion", ScalarOp::kThen,
        {kxc::codegen::KernelArgRole::kInput,
         kxc::codegen::KernelArgRole::kOutput}, {}, true);
    const ControlExecutionPlan invalid_completion_plan =
        kxc::api::BindControlPlanForRuntime(
            branch,
            {{21, invalid_completion.module, "invalid_completion", 18, {1}},
             bindings[1]});
    CHECK(Throws([&] {
              (void)ControlRuntimeSession(invalid_completion_plan)
                  .Run({ScalarBool(true), ScalarI64(7)});
          }) && invalid_completion.launcher->calls == 1 &&
              fixture.else_kernel.launcher->calls == 0,
          "launcher completion must be defined and match the default stream");
#endif
    return true;
}

bool TestAsyncCompletionRetention() {
#if KXC_ENABLE_CONTROL_RUNTIME
    kxc::AsyncOperation completion;
    std::weak_ptr<ScalarLauncher> selected_launcher;
    std::weak_ptr<ScalarLauncher> unselected_launcher;
    {
        BranchFixture fixture;
        selected_launcher = fixture.then_kernel.launcher;
        unselected_launcher = fixture.else_kernel.launcher;
        const ControlExecutionPlan resolved =
            kxc::api::BindControlPlanForRuntime(BranchPlan(), Bindings(fixture));
        ControlRuntimeSession session(resolved);
        ControlRunAsyncResult result = session.RunAsync(
            {ScalarBool(true), ScalarI64(5)}, DeviceStream::Default(Device::CPU()));
        CHECK(fixture.then_kernel.launcher->calls == 1 && fixture.else_kernel.launcher->calls == 0 &&
                  result.completion->retained_storage.size() == 3 &&
                  static_cast<bool>(result.completion->retained_context),
              "selected branch alone must allocate/launch and completion must retain its state/storage");
        completion = result.completion;
    }
    CHECK(!selected_launcher.expired() && !unselected_launcher.expired() &&
              completion->retained_storage.size() == 3,
          "completion must retain resolved plan modules and selected storage after local owners die");
    completion.Wait();
    completion = kxc::AsyncOperation();
    CHECK(selected_launcher.expired() && unselected_launcher.expired(),
          "releasing completion must release retained artifacts without an ownership cycle");
#else
    BranchFixture fixture;
    const ControlExecutionPlan resolved =
        kxc::api::BindControlPlanForRuntime(BranchPlan(), Bindings(fixture));
    CHECK(Throws([&] { ControlRuntimeSession disabled(resolved); }) &&
              fixture.then_kernel.launcher->calls == 0 && fixture.else_kernel.launcher->calls == 0,
          "gate-off construction must clearly reject without dispatch");
#endif
    return true;
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"compiler_default_rejects_if", TestCompilerDefaultStillRejectsIf},
        {"binding_and_branch_differential", TestBindingAndBranchDifferential},
        {"loop_differential_and_bound", TestLoopDifferentialAndBound},
        {"repeated_operand_abi_order", TestRepeatedOperandAbiOrder},
        {"constant_abi_order_and_resolved_validation",
         TestConstantAbiOrderAndResolvedValidation},
        {"binding_and_validation_negatives", TestBindingAndValidationNegatives},
        {"async_completion_retention", TestAsyncCompletionRetention},
    };
    int failures = 0;
    for (const auto& test : tests) {
        try {
            if (test.second()) std::cout << "[PASS] " << test.first << "\n";
            else ++failures;
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << test.first << ": " << error.what() << "\n";
            ++failures;
        }
    }
    return failures == 0 ? 0 : 1;
}
