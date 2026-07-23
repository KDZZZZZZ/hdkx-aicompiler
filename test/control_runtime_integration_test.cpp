/*! \file test/control_runtime_integration_test.cpp
 * \brief End-to-end binding and CPU execution checks for the control runtime.
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "kxc/compiler/compiler.h"
#include "kxc/compiler/control_flow.h"
#include "kxc/relay/op.h"
#include "kxc/runtime/control_session.h"
#include "support/control_plan_reference_executor.h"
#include "../src/compiler/control_flow/production_control_flow_test.h"
#include "../src/runtime/internal/compiled_module_node.h"

#ifndef KXC_ENABLE_CONTROL_RUNTIME
#define KXC_ENABLE_CONTROL_RUNTIME 0
#endif
#ifndef KXC_ENABLE_RELAY_CONTROL_FLOW_PRODUCTION
#define KXC_ENABLE_RELAY_CONTROL_FLOW_PRODUCTION 0
#endif
#ifndef KXC_USE_LLVM
#define KXC_USE_LLVM 0
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

ControlPlan SharedConstantBranchPlan() {
    ControlPlan plan = BranchPlan();
    plan.values.push_back(I64(5));
    plan.constant_values = {5};
    plan.regions[0].live_ins = {0, 1, 5};
    plan.regions[0].effect = Reads({0, 1, 5});
    plan.regions[0].tasks[0].inputs = {0, 1, 5};
    plan.regions[0].tasks[0].effect = Reads({0, 1, 5});
    for (std::size_t i = 1; i < plan.regions.size(); ++i) {
        plan.regions[i].live_ins = {1, 5};
        plan.regions[i].effect = Reads({1, 5});
        plan.regions[i].tasks[0].inputs = {1, 5};
        plan.regions[i].tasks[0].argument_values = {1, 5};
        plan.regions[i].tasks[0].effect = Reads({1, 5});
    }
    return plan;
}

ControlPlan ModuleLocalConstantBranchPlan() {
    ControlPlan plan = BranchPlan();
    plan.values.push_back(I64(5));
    plan.values.push_back(I64(6));
    plan.constant_values = {5, 6};
    plan.regions[0].live_ins = {0, 1, 5, 6};
    plan.regions[0].effect = Reads({0, 1, 5, 6});
    plan.regions[0].tasks[0].inputs = {0, 1, 5, 6};
    plan.regions[0].tasks[0].effect = Reads({0, 1, 5, 6});
    plan.regions[1].live_ins = {1, 5};
    plan.regions[1].effect = Reads({1, 5});
    plan.regions[1].tasks[0].inputs = {1, 5};
    plan.regions[1].tasks[0].argument_values = {1, 5};
    plan.regions[1].tasks[0].effect = Reads({1, 5});
    plan.regions[2].live_ins = {1, 6};
    plan.regions[2].effect = Reads({1, 6});
    plan.regions[2].tasks[0].inputs = {1, 6};
    plan.regions[2].tasks[0].argument_values = {1, 6};
    plan.regions[2].tasks[0].effect = Reads({1, 6});
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

ControlPlan AliasedInputsPlan() {
    ControlPlan plan;
    plan.values = {I64(0), I64(1), I64(2)};
    plan.entry_region = 10;
    plan.region_order = {10};
    plan.graph_inputs = {0, 1};
    plan.graph_outputs = {2};
    plan.regions = {
        {10, {0, 1}, {2}, {Kernel(60, {0, 1}, {2}, "decoy.alias")},
         Reads({0, 1}), {}, "entry"},
    };
    return plan;
}

enum class ScalarOp { kThen, kElse, kCondition, kStep, kSum, kDouble, kConstantOrder };

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
            case ScalarOp::kDouble: WriteI64(output, first * 2); break;
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

NDArray ScalarI64(std::int64_t value, std::size_t alignment = 0) {
    NDArray result =
        NDArray::Empty({}, Type("int64"), Device::CPU(), alignment);
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
                      {}, Device::CPU(), 64, false, "constant.a"),
        KernelArgSpec("constant_b", KernelArgRole::kConstant, Type("int64"),
                      {}, Device::CPU(), 32, false, "constant.b"),
        KernelArgSpec("output", KernelArgRole::kOutput, Type("int64"), {},
                      Device::CPU(), 1, true),
    };
    Map<String, NDArray> constants;
    constants.Set("constant.a", ScalarI64(2, 64));
    constants.Set("constant.b", ScalarI64(3, 32));
    auto launcher = std::make_shared<ScalarLauncher>(ScalarOp::kConstantOrder);
    return {MakeModule("constant_order_entry", arguments, launcher, constants),
            std::move(launcher)};
}

Fixture MakeSharedConstantFixture(
    const char* symbol, std::size_t alignment, std::int64_t value,
    Array<std::int64_t> shape = {}) {
    using namespace kxc::codegen;
    Array<KernelArgSpec> arguments{
        KernelArgSpec("input", KernelArgRole::kInput, Type("int64"), {},
                      Device::CPU()),
        KernelArgSpec("constant", KernelArgRole::kConstant, Type("int64"),
                      shape, Device::CPU(), alignment, false,
                      "constant.shared"),
        KernelArgSpec("output", KernelArgRole::kOutput, Type("int64"), {},
                      Device::CPU(), 1, true),
    };
    NDArray payload =
        NDArray::Empty(shape, Type("int64"), Device::CPU(), alignment);
    payload.CopyFromBytes(&value, sizeof(value));
    Map<String, NDArray> constants;
    constants.Set("constant.shared", std::move(payload));
    auto launcher = std::make_shared<ScalarLauncher>(ScalarOp::kSum);
    return {MakeModule(symbol, arguments, launcher, constants),
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

bool TestLeaseGenerationDoesNotWrap() {
    using kxc::api::internal::MintControlFlowLeaseGenerationForTest;
    using kxc::api::internal::SetControlFlowLeaseGenerationForTest;
    SetControlFlowLeaseGenerationForTest(
        std::numeric_limits<std::uint64_t>::max() - 1);
    const std::uint64_t generation = MintControlFlowLeaseGenerationForTest();
    const std::string first_failure = ErrorText([] {
        (void)MintControlFlowLeaseGenerationForTest();
    });
    const std::string second_failure = ErrorText([] {
        (void)MintControlFlowLeaseGenerationForTest();
    });
    SetControlFlowLeaseGenerationForTest(0);
    CHECK(generation == std::numeric_limits<std::uint64_t>::max() &&
              first_failure.find("generation overflowed") != std::string::npos &&
              second_failure.find("generation overflowed") != std::string::npos,
          "the terminal lease generation must be issued once and then fail closed without reuse");
    return true;
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

bool TestRelayWhileCompileGates() {
    const kxc::TensorType boolean({}, "bool");
    const kxc::TensorType integer({}, "int64");
    kxc::Var predicate("while_predicate", boolean);
    kxc::Var initial("while_initial", integer);
    kxc::Var increment("while_increment", integer);
    kxc::Var state("while_state", integer);
    const kxc::Function function(
        {predicate, initial, increment},
        kxc::While(initial, state, predicate,
                   kxc::Call(kxc::relay::Op::Get("add"), {state, increment}), 1));
    const auto config =
        kxc::api::CompileConfig::Create(kxc::BuildTarget(Device::CPU()));
    const std::string default_error = ErrorText([&] {
        (void)kxc::api::Compiler::Compile(function, config);
    });
    CHECK(default_error.find("missing control_flow.loop") != std::string::npos,
          "Compiler::Compile must reject Relay While before ValueGraph");
    const std::string control_error = ErrorText([&] {
        (void)kxc::api::Compiler::CompileControlFlowExact(function, config);
    });
#if !KXC_ENABLE_RELAY_CONTROL_FLOW_PRODUCTION
    CHECK(control_error.find("disabled by KXC_ENABLE_RELAY_CONTROL_FLOW_PRODUCTION") !=
              std::string::npos,
          "production Relay While API must remain default-OFF");
#elif !KXC_USE_LLVM
    CHECK(control_error.find("KXC_ENABLE_LLVM=ON") != std::string::npos,
          "enabled Relay While API must fail closed without LLVM");
#else
    CHECK(control_error.empty(), "enabled Relay While path must resolve real artifacts");
#endif
    return true;
}

bool TestProductionRelayWhileNumericE2E() {
#if KXC_ENABLE_RELAY_CONTROL_FLOW_PRODUCTION && KXC_USE_LLVM && KXC_ENABLE_CONTROL_RUNTIME
    const kxc::TensorType boolean({}, "bool");
    const kxc::TensorType integer({}, "int64");
    kxc::Var first("while_first", boolean), second("while_second", boolean);
    kxc::Var third("while_third", boolean), tail("while_tail", boolean);
    kxc::Var value("while_value", integer), increment("while_increment", integer);
    kxc::Var state("while_state");
    const auto make_loop = [&](std::int64_t max_trip_count) {
        const kxc::Expr initial = kxc::Tuple(
            {first, second, third, tail, value});
        const kxc::Expr condition = kxc::TupleGetItem(state, 0);
        const kxc::Expr body = kxc::Tuple(
            {kxc::TupleGetItem(state, 1), kxc::TupleGetItem(state, 2),
             kxc::TupleGetItem(state, 3), tail,
             kxc::Call(kxc::relay::Op::Get("add"),
                       {kxc::TupleGetItem(state, 4), increment})});
        return kxc::Function({first, second, third, tail, value, increment},
                             kxc::While(initial, state, condition, body,
                                        max_trip_count));
    };
    const auto config =
        kxc::api::CompileConfig::Create(kxc::BuildTarget(Device::CPU()));
    std::weak_ptr<const kxc::api::ControlFlowArtifactLease> lease;
    ControlRunAsyncResult asynchronous;
    {
        auto compiled = kxc::api::Compiler::CompileControlFlowExact(make_loop(3), config);
        CHECK(!compiled.artifact_pins.empty() && compiled.artifact_lease,
              "real Relay While must retain the compiled body artifact pin");
        lease = compiled.artifact_lease;
        const auto expected_events = [&compiled](std::int64_t trips) {
            const auto& spec = compiled.plan.spec();
            const auto& entry = spec.regions[0];
            const auto& loop = entry.tasks[0];
            std::vector<std::string> events;
            for (const auto input : loop.inputs) {
                events.push_back("read:" + std::to_string(loop.id) + ":" +
                                 std::to_string(input));
            }
            events.push_back("task:" + std::to_string(loop.id));
            const auto body = std::find_if(
                spec.regions.begin(), spec.regions.end(),
                [&loop](const auto& region) { return region.id == loop.loop.body_region; });
            for (std::int64_t iteration = 0; iteration < trips; ++iteration) {
                events.push_back("loop:" + std::to_string(loop.id) + ":iteration:" +
                                 std::to_string(iteration));
                for (const auto& task : body->tasks) {
                    for (const auto input : task.inputs) {
                        events.push_back("read:" + std::to_string(task.id) + ":" +
                                         std::to_string(input));
                    }
                    events.push_back("task:" + std::to_string(task.id));
                    for (const auto output : task.outputs) {
                        events.push_back("write:" + std::to_string(task.id) + ":" +
                                         std::to_string(output));
                    }
                }
            }
            for (const auto output : loop.outputs) {
                events.push_back("write:" + std::to_string(loop.id) + ":" +
                                 std::to_string(output));
            }
            return events;
        };
        ControlRuntimeSession session(compiled.plan);
        const auto run = [&](bool one, bool two, bool three,
                             std::int64_t expected_value,
                             std::int64_t expected_trips) {
            const ControlRunResult result = session.Run(
                {ScalarBool(one), ScalarBool(two), ScalarBool(three), ScalarBool(false),
                 ScalarI64(7), ScalarI64(1)});
            return result.outputs.size() == 5 && ReadI64(result.outputs[4]) == expected_value &&
                   result.loop_iterations.size() == 1 &&
                   result.loop_iterations[0].iterations == expected_trips &&
                   result.events == expected_events(expected_trips);
        };
        CHECK(run(false, false, false, 7, 0) && run(true, false, false, 8, 1) &&
                  run(true, true, true, 10, 3),
              "one real Relay While must execute exact zero, one, and multiple numeric trips");
        asynchronous = session.RunAsync(
            {ScalarBool(true), ScalarBool(true), ScalarBool(true), ScalarBool(false),
             ScalarI64(7), ScalarI64(1)}, DeviceStream::Default(Device::CPU()));
    }
    CHECK(!lease.expired(),
          "RunAsync completion must retain the compiler-minted artifact lease after owners die");
    asynchronous.completion.Wait();
    CHECK(asynchronous.outputs.size() == 5 && ReadI64(asynchronous.outputs[4]) == 10 &&
              asynchronous.loop_iterations.size() == 1 &&
              asynchronous.loop_iterations[0].iterations == 3,
          "async real Relay While must retain numeric outputs and iteration accounting");
    asynchronous = ControlRunAsyncResult{};
    CHECK(lease.expired(),
          "releasing RunAsync completion must release the retained production artifact lease");

    const auto exhausted = kxc::api::Compiler::CompileControlFlowExact(make_loop(1), config);
    CHECK(ErrorText([&] {
              (void)ControlRuntimeSession(exhausted.plan).Run(
                  {ScalarBool(true), ScalarBool(true), ScalarBool(false), ScalarBool(false),
                   ScalarI64(7), ScalarI64(1)});
          }).find("max_trip_count") != std::string::npos,
          "a true real Relay While condition beyond the bound must stop with max-trip exhaustion");
#else
    // Numeric Relay While execution requires the explicit production, LLVM, and runtime gates.
#endif
    return true;
}

bool TestProductionControlFlowGateAndArtifacts() {
    const kxc::TensorType boolean({}, "bool");
    const kxc::TensorType integer({}, "int64");
    kxc::Var predicate("production_predicate", boolean);
    kxc::Var lhs("production_lhs", integer);
    kxc::Var rhs("production_rhs", integer);
    const kxc::Function function(
        {predicate, lhs, rhs},
        kxc::If(predicate,
                kxc::Call(kxc::relay::Op::Get("add"), {lhs, rhs}),
                kxc::Call(kxc::relay::Op::Get("mul"), {lhs, rhs})));
    const auto config =
        kxc::api::CompileConfig::Create(kxc::BuildTarget(Device::CPU()));
    const std::string error = ErrorText([&] {
        (void)kxc::api::Compiler::CompileControlFlowExact(function, config);
    });
#if !KXC_ENABLE_RELAY_CONTROL_FLOW_PRODUCTION
    CHECK(error.find("disabled by KXC_ENABLE_RELAY_CONTROL_FLOW_PRODUCTION") !=
              std::string::npos,
          "production Relay If API must remain default-OFF");
#elif !KXC_USE_LLVM
    CHECK(error.find("KXC_ENABLE_LLVM=ON") != std::string::npos,
          "enabled production API must fail closed without a real LLVM backend");
#else
    CHECK(error.empty(), "enabled production API must resolve real compiler artifacts");
    const auto compiled = kxc::api::Compiler::CompileControlFlowExact(
        function, config);
    CHECK(!compiled.artifact_pins.empty() && compiled.artifact_lease &&
              compiled.artifact_lease->generation() != 0,
          "resolved production plan must retain real pins in a compiler-minted lease");
    for (const auto& region : compiled.plan.regions()) {
        for (const auto& task : region.tasks) {
            if (task.kind != ControlExecutionTaskKind::kKernel) continue;
            CHECK(task.kernel.binding_revision() == 0,
                  "production bindings must not use fixture revisions");
        }
    }
#if KXC_ENABLE_CONTROL_RUNTIME
    ControlRuntimeSession session(compiled.plan);
    const ControlRunResult yes =
        session.Run({ScalarBool(true), ScalarI64(2), ScalarI64(3)});
    const ControlRunResult no =
        session.Run({ScalarBool(false), ScalarI64(2), ScalarI64(3)});
    CHECK(yes.outputs.size() == 1 && no.outputs.size() == 1 &&
              ReadI64(yes.outputs[0]) == 5 && ReadI64(no.outputs[0]) == 6 &&
              yes.events != no.events,
          "real true/false compiler artifacts must execute distinct Relay If branches");
#endif
#endif
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

bool TestReadOnlyInputAliasingAndAbiOrder() {
    Fixture sum = MakeFixture("sum_entry", ScalarOp::kDouble,
                              {kxc::codegen::KernelArgRole::kInput,
                               kxc::codegen::KernelArgRole::kOutput});
    const ControlExecutionPlan repeated = kxc::api::BindControlPlanForRuntime(
        RepeatedOperandPlan(),
        {{40, sum.module, "sum_entry", 11, {0}}});
    Fixture aliased = MakeFixture("alias_entry", ScalarOp::kSum,
                                  {kxc::codegen::KernelArgRole::kInput,
                                   kxc::codegen::KernelArgRole::kInput,
                                   kxc::codegen::KernelArgRole::kOutput});
    const ControlExecutionPlan aliased_inputs =
        kxc::api::BindControlPlanForRuntime(
            AliasedInputsPlan(),
            {{60, aliased.module, "alias_entry", 12, {0, 1}}});
    CHECK(aliased_inputs.spec().effect_model ==
              ControlExecutionEffectModel::kPureFreshKernelOutputsV1,
          "effect model must not claim physical no-alias semantics");
#if KXC_ENABLE_CONTROL_RUNTIME
    const ControlRunResult repeated_result =
        ControlRuntimeSession(repeated).Run({ScalarI64(6)});
    CHECK(repeated_result.outputs.size() == 1 &&
              ReadI64(repeated_result.outputs[0]) == 12 &&
              sum.launcher->calls == 1 &&
              sum.launcher->last_arguments.size() == 2,
          "duplicate logical operands must collapse to one physical ABI input");

    const NDArray shared = ScalarI64(6);
    ControlRuntimeSession alias_session(aliased_inputs);
    const ControlRunResult first = alias_session.Run({shared, shared});
    const ControlRunResult second = alias_session.Run({shared, shared});
    CHECK(first.outputs.size() == 1 && second.outputs.size() == 1 &&
              ReadI64(first.outputs[0]) == 12 &&
              ReadI64(second.outputs[0]) == 12 &&
              aliased.launcher->calls == 2 &&
              aliased.launcher->last_arguments[0].get() ==
                  aliased.launcher->last_arguments[1].get() &&
              first.outputs[0].storage().get() != shared.storage().get() &&
              first.outputs[0].storage().get() !=
                  second.outputs[0].storage().get(),
          "read-only logical inputs may alias while every kernel output is fresh");
#else
    CHECK(Throws([&] { ControlRuntimeSession disabled(repeated); }) &&
              Throws([&] { ControlRuntimeSession disabled(aliased_inputs); }) &&
              sum.launcher->calls == 0 && aliased.launcher->calls == 0,
          "disabled runtime must reject alias fixtures without launch");
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
    const BoundControlKernel& bound_kernel =
        bound.spec().regions[0].tasks[0].kernel;
    CHECK(bound_kernel.binding_revision() == 12,
          "binding revision must preserve only the caller fixture label");

    NDArray original_constant = fixture.module.constants().at("constant.a");
    const std::int64_t changed_original = 100;
    original_constant.CopyFromBytes(&changed_original, sizeof(changed_original));
    NDArray returned_constant = bound_kernel.Constant("constant.a");
    CHECK(ReadI64(returned_constant) == 2 &&
              returned_constant.device() == Device::CPU() &&
              returned_constant.NBytes() == sizeof(std::int64_t) &&
              reinterpret_cast<std::uintptr_t>(
                  returned_constant.storage().data()) % 64 == 0,
          "binding must preserve an aligned private CPU constant snapshot");
    const std::int64_t changed_return = 200;
    returned_constant.CopyFromBytes(&changed_return, sizeof(changed_return));
    CHECK(ReadI64(bound_kernel.Constant("constant.a")) == 2,
          "mutating a returned constant copy must not alter the private snapshot");

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

    using kxc::codegen::KernelArgRole;
    using kxc::codegen::KernelArgSpec;
    Array<KernelArgSpec> zero_arguments{
        KernelArgSpec("empty", KernelArgRole::kConstant, Type("int64"),
                      {0}, Device::CPU(), 128, false, "constant.empty"),
        KernelArgSpec("output", KernelArgRole::kOutput, Type("int64"),
                      {0}, Device::CPU(), 128, true),
    };
    Map<String, NDArray> zero_constants;
    zero_constants.Set(
        "constant.empty",
        NDArray::Empty({0}, Type("int64"), Device::CPU(), 128));
    auto zero_launcher = std::make_shared<ScalarLauncher>(ScalarOp::kThen);
    BoundControlKernel zero_bound(
        MakeModule("zero_constant_entry", zero_arguments, zero_launcher,
                   zero_constants),
        "zero_constant_entry", 19);
    const NDArray empty_copy = zero_bound.Constant("constant.empty");
    CHECK(empty_copy.device() == Device::CPU() && empty_copy.NBytes() == 0 &&
              empty_copy.storage().data() == nullptr &&
              zero_bound.binding_revision() == 19 &&
              zero_launcher->calls == 0,
          "zero-byte CPU constants must bind and deep-copy without dereference");
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

bool TestSharedConstantAlignmentAcrossRegions() {
    const ControlPlan plan = SharedConstantBranchPlan();
    Fixture then_fixture =
        MakeSharedConstantFixture("shared_then_entry", 64, 2);
    Fixture else_fixture =
        MakeSharedConstantFixture("shared_else_entry", 32, 2);
    const std::vector<kxc::api::ControlKernelBinding> bindings{
        {21, then_fixture.module, "shared_then_entry", 20, {1, 5}},
        {22, else_fixture.module, "shared_else_entry", 21, {1, 5}},
    };
    const ControlExecutionPlan bound =
        kxc::api::BindControlPlanForRuntime(plan, bindings);

    Fixture wrong_bytes =
        MakeSharedConstantFixture("wrong_bytes_entry", 32, 3);
    auto wrong_bytes_bindings = bindings;
    wrong_bytes_bindings[1] =
        {22, wrong_bytes.module, "wrong_bytes_entry", 22, {1, 5}};
    Fixture wrong_contract = MakeSharedConstantFixture(
        "wrong_contract_entry", 32, 2, {1});
    auto wrong_contract_bindings = bindings;
    wrong_contract_bindings[1] =
        {22, wrong_contract.module, "wrong_contract_entry", 23, {1, 5}};
    CHECK(Throws([&] {
              (void)kxc::api::BindControlPlanForRuntime(
                  plan, wrong_bytes_bindings);
          }) && Throws([&] {
              (void)kxc::api::BindControlPlanForRuntime(
                  plan, wrong_contract_bindings);
          }) && then_fixture.launcher->calls == 0 &&
              else_fixture.launcher->calls == 0 &&
              wrong_bytes.launcher->calls == 0 &&
              wrong_contract.launcher->calls == 0,
          "shared constants must reject genuine byte and contract mismatches before launch");

    const auto& regions = bound.spec().regions;
    NDArray then_copy =
        regions[1].tasks[0].kernel.Constant("constant.shared");
    NDArray else_copy =
        regions[2].tasks[0].kernel.Constant("constant.shared");
    const std::int64_t changed = 99;
    then_fixture.module.constants().at("constant.shared")
        .CopyFromBytes(&changed, sizeof(changed));
    else_fixture.module.constants().at("constant.shared")
        .CopyFromBytes(&changed, sizeof(changed));
    then_copy.CopyFromBytes(&changed, sizeof(changed));
    else_copy.CopyFromBytes(&changed, sizeof(changed));
    CHECK(ReadI64(regions[1].tasks[0].kernel.Constant("constant.shared")) == 2 &&
              ReadI64(regions[2].tasks[0].kernel.Constant("constant.shared")) == 2,
          "module constants and returned copies must not mutate private snapshots");

    Fixture local_then = MakeSharedConstantFixture("local_then", 64, 2);
    Fixture local_else = MakeSharedConstantFixture("local_else", 32, 3);
    const ControlExecutionPlan module_local = kxc::api::BindControlPlanForRuntime(
        ModuleLocalConstantBranchPlan(),
        {{21, local_then.module, "local_then", 24, {1, 5}},
         {22, local_else.module, "local_else", 25, {1, 6}}});
#if KXC_ENABLE_CONTROL_RUNTIME
    const ControlRunResult local_true = ControlRuntimeSession(module_local).Run(
        {ScalarBool(true), ScalarI64(5)});
    const ControlRunResult local_false = ControlRuntimeSession(module_local).Run(
        {ScalarBool(false), ScalarI64(5)});
    CHECK(ReadI64(local_true.outputs[0]) == 7 && ReadI64(local_false.outputs[0]) == 8,
          "same module-local key may bind different logical constants by task");
#endif

#if KXC_ENABLE_CONTROL_RUNTIME
    ControlRuntimeSession session(bound);
    const ControlRunResult then_result =
        session.Run({ScalarBool(true), ScalarI64(5)});
    const ControlRunResult else_result =
        session.Run({ScalarBool(false), ScalarI64(5)});
    CHECK(then_result.outputs.size() == 1 &&
              ReadI64(then_result.outputs[0]) == 7 &&
              else_result.outputs.size() == 1 &&
              ReadI64(else_result.outputs[0]) == 7 &&
              then_fixture.launcher->calls == 1 &&
              else_fixture.launcher->calls == 1,
          "one logical constant must satisfy 64- and 32-byte consumers across branches");
#else
    CHECK(Throws([&] { ControlRuntimeSession disabled(bound); }) &&
              then_fixture.launcher->calls == 0 &&
              else_fixture.launcher->calls == 0,
          "gate-off shared constant plan must not launch");
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
    auto zero_revision = bindings;
    zero_revision[0].binding_revision = 0;
    CHECK(Throws([&] { (void)kxc::api::BindControlPlanForRuntime(branch, zero_revision); }),
          "binding_revision zero must fail");
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
                  !result.completion->retained_contexts.empty(),
              "selected branch alone must allocate/launch and completion must retain its state/storage");
        completion = result.completion;
    }
    CHECK(!selected_launcher.expired() && !unselected_launcher.expired() &&
              completion->retained_storage.size() == 3,
          "completion must retain resolved plan modules and selected storage after local owners die");
    completion.Wait();
    completion = kxc::AsyncOperation();
    CHECK(selected_launcher.expired() && unselected_launcher.expired(),
          "releasing completion must release retained fixture launchers without an ownership cycle");
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
        {"lease_generation_does_not_wrap", TestLeaseGenerationDoesNotWrap},
        {"compiler_default_rejects_if", TestCompilerDefaultStillRejectsIf},
        {"relay_while_compile_gates", TestRelayWhileCompileGates},
        {"production_relay_while_numeric_e2e", TestProductionRelayWhileNumericE2E},
        {"production_control_flow_gate_and_artifacts",
         TestProductionControlFlowGateAndArtifacts},
        {"binding_and_branch_differential", TestBindingAndBranchDifferential},
        {"loop_differential_and_bound", TestLoopDifferentialAndBound},
        {"read_only_input_aliasing_and_abi_order",
         TestReadOnlyInputAliasingAndAbiOrder},
        {"constant_abi_order_and_resolved_validation",
         TestConstantAbiOrderAndResolvedValidation},
        {"shared_constant_alignment_across_regions",
         TestSharedConstantAlignmentAcrossRegions},
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
