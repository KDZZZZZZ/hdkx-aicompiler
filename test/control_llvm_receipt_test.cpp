/*! \file test/control_llvm_receipt_test.cpp
 * \brief M10 C1 receipt evidence: Relay If/While on real LLVM artifacts with
 *     launch accounting, plus gate-off rejection, printed as M10_RECEIPT lines.
 */

#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kxc/compiler/compiler.h"
#include "kxc/relay/op.h"
#include "kxc/runtime/control_session.h"
#include "support/control_plan_reference_executor.h"
#include "../src/compiler/control_flow/internal_lowering.h"
#include "../src/runtime/internal/compiled_module_node.h"
#include "../src/runtime/internal/control_execution_plan_access.h"

#ifndef KXC_ENABLE_CONTROL_RUNTIME
#define KXC_ENABLE_CONTROL_RUNTIME 0
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
using kxc::api::Compiler;
using namespace kxc::api::internal;
using namespace kxc::runtime;
using namespace kxc::api::internal::test_support;
using PrivatePlanSpec = kxc::runtime::internal::ControlExecutionPlanSpec;
using PrivatePlanAccess = kxc::runtime::internal::ControlExecutionPlanAccess;

#define CHECK(condition, message) do { if (!(condition)) { std::cerr << "[FAIL] " << __FUNCTION__ << ": " << message << "\n"; return false; } } while (0)

bool Throws(const std::function<void()>& fn) {
    try { fn(); } catch (const std::exception&) { return true; }
    return false;
}

std::string ErrorText(const std::function<void()>& fn) {
    try { fn(); } catch (const std::exception& error) { return error.what(); }
    return {};
}

void ReceiptLine(const std::string& key, const std::string& value) {
    std::cout << "M10_RECEIPT " << key << "=" << value << "\n";
}

DLDataType Type(const char* name) { return DataTypeFromString(name); }

[[maybe_unused]] std::string Fnv1a(const void* data, std::size_t nbytes) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    std::uint64_t hash = 1469598103934665603ull;
    for (std::size_t i = 0; i < nbytes; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%016llx",
                  static_cast<unsigned long long>(hash));
    return buffer;
}

[[maybe_unused]] std::string InputChecksum(const Array<NDArray>& inputs) {
    std::string combined;
    for (const NDArray& input : inputs) {
        const std::size_t nbytes = input.NBytes();
        std::vector<std::uint8_t> bytes(nbytes);
        if (nbytes != 0) input.CopyToBytes(bytes.data(), nbytes);
        combined += Fnv1a(bytes.data(), bytes.size());
        combined += ":";
    }
    return combined;
}

[[maybe_unused]] NDArray ScalarI64(std::int64_t value) {
    NDArray result = NDArray::Empty({}, Type("int64"), Device::CPU());
    result.CopyFromBytes(&value, sizeof(value));
    return result;
}

[[maybe_unused]] NDArray ScalarBool(bool value) {
    NDArray result = NDArray::Empty({}, Type("bool"), Device::CPU());
    const std::uint8_t byte = value ? 1 : 0;
    result.CopyFromBytes(&byte, sizeof(byte));
    return result;
}

[[maybe_unused]] std::int64_t ReadScalarI64(const NDArray& value) {
    std::int64_t result{0};
    value.CopyToBytes(&result, sizeof(result));
    return result;
}

[[maybe_unused]] std::size_t TaskEvents(const ControlRunResult& result,
                       ControlExecutionTaskId task_id) {
    const std::string needle = "task:" + std::to_string(task_id);
    std::size_t count = 0;
    for (const std::string& event : result.events) {
        if (event == needle) ++count;
    }
    return count;
}

[[maybe_unused]] std::string JoinEvents(const ControlRunResult& result) {
    std::string joined;
    for (const std::string& event : result.events) {
        if (!joined.empty()) joined += ",";
        joined += event;
    }
    return joined;
}

[[maybe_unused]] FakeKernelCallback ArithmeticKernels(
    const std::vector<PrimitiveUnit>& units) {
    std::unordered_map<PrimitiveUnitId, std::string> operator_names;
    for (const auto& unit : units) {
        operator_names.emplace(unit.id, unit.call.spec.name);
    }
    return [operator_names = std::move(operator_names)](
               const ControlTask& task,
               const std::vector<FakeValue>& args) {
        const auto found = operator_names.find(task.primitive_unit_id);
        if (found == operator_names.end()) {
            throw std::invalid_argument("unknown fake PrimitiveUnit");
        }
        if (found->second == "add") {
            return std::vector<FakeValue>{FakeValue::I64(args[0].integer + args[1].integer)};
        }
        if (found->second == "mul") {
            return std::vector<FakeValue>{FakeValue::I64(args[0].integer * args[1].integer)};
        }
        throw std::invalid_argument("unexpected fake kernel reference");
    };
}

// --- Counting-launcher fixture used for zero-launch negative receipts. ------

enum class ScalarOp { kThen };

class CountingLauncher final : public kxc::codegen::KernelLauncher {
public:
    bool IsReady() const noexcept override { return true; }
    kxc::AsyncOperation Launch(const Array<NDArray>& arguments,
                               const DeviceStream& stream,
                               const kxc::ObjectRef&) const override {
        ++calls;
        const std::int64_t first = ReadScalarI64(arguments[0]);
        const std::int64_t result = first + 10;
        arguments[arguments.size() - 1].CopyFromBytes(&result, sizeof(result));
        Array<kxc::Storage> retained;
        for (const NDArray& argument : arguments) retained.push_back(argument.storage());
        return kxc::AsyncOperation::Completed(stream, std::move(retained));
    }
    mutable int calls{0};
};

struct Fixture {
    kxc::api::CompiledModule module;
    std::shared_ptr<CountingLauncher> launcher;
};

Fixture MakeFixture(const char* symbol) {
    using namespace kxc::codegen;
    const Array<KernelArgSpec> arguments{
        KernelArgSpec("input", KernelArgRole::kInput, Type("int64"), {},
                      Device::CPU()),
        KernelArgSpec("output", KernelArgRole::kOutput, Type("int64"), {},
                      Device::CPU(), 1, true)};
    auto launcher = std::make_shared<CountingLauncher>();
    const KernelSignature signature(symbol, arguments);
    const KernelLaunchMetadata metadata(Device::CPU(), CodeGenBackend::kLLVM);
    return {kxc::api::internal::BuildCompiledModule(
                BuildTarget(Device::CPU()),
                {kxc::api::internal::CompiledModuleEntry{
                    signature, metadata,
                    CompiledKernel(signature, metadata, launcher)}},
                {}),
            std::move(launcher)};
}

EffectSummary Reads(std::vector<ValueId> values) {
    return EffectSummary{std::move(values), {}, {}, false, false};
}

ControlValueSpec I64(ValueId id) {
    return ControlValueSpec{id, kxc::TensorType({}, "int64"), Device::CPU(),
                            LogicalValueOrigin::kPrimitiveOutput, kxc::Expr(),
                            "value" + std::to_string(id)};
}

ControlValueSpec Bool(ValueId id) {
    return ControlValueSpec{id, kxc::TensorType({}, "bool"), Device::CPU(),
                            LogicalValueOrigin::kPrimitiveOutput, kxc::Expr(),
                            "value" + std::to_string(id)};
}

ControlTask Kernel(TaskId id, std::vector<ValueId> inputs,
                   std::vector<ValueId> outputs, const char* locator) {
    ControlTask task;
    task.id = id;
    task.kind = ControlTaskKind::kKernel;
    task.primitive_unit_id = id;
    task.inputs = std::move(inputs);
    task.argument_values = task.inputs;
    task.outputs = std::move(outputs);
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
        {11, {1}, {2}, {Kernel(21, {1}, {2}, "then")}, Reads({1}), {}, "then"},
        {12, {1}, {3}, {Kernel(22, {1}, {3}, "else")}, Reads({1}), {}, "else"}};
    return plan;
}

// --- C1 evidence: production Relay If against the independent reference. ---

bool TestReceiptRelayIfBothBranches() {
#if KXC_ENABLE_CONTROL_RUNTIME && KXC_USE_LLVM
    using namespace kxc;
    const TensorType boolean({}, "bool");
    const TensorType integer({}, "int64");
    Var p("p", boolean), x("x", integer), y("y", integer);
    const Function function(
        {p, x, y},
        If(p, Call(relay::Op::Get("add"), {x, y}),
           Call(relay::Op::Get("mul"), {x, y})));
    const auto config =
        api::CompileConfig::Create(BuildTarget(Device::CPU()));
    const auto compiled = Compiler::CompileControlFlowExact(function, config);
    CHECK(compiled.defined() && compiled.plan().defined(),
          "Relay If must compile to a sealed control execution plan");

    const ControlExecutionPlan& plan = compiled.plan();
    const ControlExecutionTask* then_kernel = nullptr;
    const ControlExecutionTask* else_kernel = nullptr;
    for (const auto& region : plan.regions()) {
        for (const auto& task : region.tasks) {
            if (task.kind != ControlExecutionTaskKind::kKernel) continue;
            CHECK(task.kernel.defined(),
                  "every branch child kernel must bind a real artifact");
            if (then_kernel == nullptr) {
                then_kernel = &task;
            } else {
                else_kernel = &task;
            }
        }
    }
    CHECK(then_kernel != nullptr && else_kernel != nullptr &&
              then_kernel->kernel.signature()->symbol !=
                  else_kernel->kernel.signature()->symbol,
          "both If branches must bind distinct real primitive module entries");
    ReceiptLine("if_then_entry", then_kernel->kernel.signature()->symbol);
    ReceiptLine("if_else_entry", else_kernel->kernel.signature()->symbol);

    // Independent reference: standalone Relay lowering plus the fake executor.
    const ControlPlanLowering reference_lowering =
        LowerRelayToControlPlanWithSidecar(function);
    const ControlPlanReferenceExecutor reference(
        ArithmeticKernels(reference_lowering.primitive_units));
    const auto& reference_inputs = reference_lowering.plan.graph_inputs;
    const auto reference_run = [&](bool choice) {
        return reference.Execute(
            reference_lowering.plan,
            {{reference_inputs[0], FakeValue::Bool(choice)},
             {reference_inputs[1], FakeValue::I64(2)},
             {reference_inputs[2], FakeValue::I64(3)}});
    };

    ControlRuntimeSession session(plan);
    struct Case {
        bool choice;
        std::int64_t expected;
    };
    for (const Case& item : std::vector<Case>{{true, 5}, {false, 6}}) {
        const Array<NDArray> inputs = {ScalarBool(item.choice),
                                       ScalarI64(2), ScalarI64(3)};
        const ControlRunResult result = session.Run(inputs);
        CHECK(result.outputs.size() == 1 &&
                  ReadScalarI64(result.outputs[0]) == item.expected,
              "Relay If branch result must match the arithmetic oracle");
        const ReferenceExecution expected = reference_run(item.choice);
        CHECK(expected.values.at(reference_lowering.plan.graph_outputs[0])
                      .integer == item.expected,
              "independent reference must agree element-wise on the output");
        CHECK(result.events == expected.trace.events,
              "production branch events must match the independent reference trace");
        const std::size_t then_launches = TaskEvents(result, then_kernel->id);
        const std::size_t else_launches = TaskEvents(result, else_kernel->id);
        CHECK((item.choice ? then_launches == 1 && else_launches == 0
                           : then_launches == 0 && else_launches == 1),
              "exactly one branch kernel must launch per run");
        ReceiptLine("if_choice", item.choice ? "then" : "else");
        ReceiptLine("if_input_checksum", InputChecksum(inputs));
        ReceiptLine("if_output", std::to_string(item.expected));
        ReceiptLine("if_launch_then_else",
                    std::to_string(then_launches) + "/" +
                        std::to_string(else_launches));
        ReceiptLine("if_events", JoinEvents(result));
    }
#else
    ReceiptLine("if_case", "skipped: requires gate-on LLVM build");
#endif
    return true;
}

// --- C1 evidence: production Relay While 0/1/multi trips and bound. --------

[[maybe_unused]] kxc::Function ReceiptWhileFunction(std::int64_t max_trip_count) {
    using namespace kxc;
    const TensorType boolean({}, "bool");
    const TensorType integer({}, "int64");
    Var first("first", boolean), second("second", boolean);
    Var third("third", boolean), tail("tail", boolean);
    Var value("value", integer), increment("increment", integer);
    Var state("state");
    const Expr initial =
        Tuple({first, second, third, tail, value});
    const Expr condition = TupleGetItem(state, 0);
    const Expr body =
        Tuple({TupleGetItem(state, 1), TupleGetItem(state, 2),
               TupleGetItem(state, 3), tail,
               Call(relay::Op::Get("add"),
                    {TupleGetItem(state, 4), increment})});
    return Function({first, second, third, tail, value, increment},
                    While(initial, state, condition, body, max_trip_count));
}

bool TestReceiptRelayWhileTripsAndBound() {
#if KXC_ENABLE_CONTROL_RUNTIME && KXC_USE_LLVM
    using namespace kxc;
    const auto config =
        api::CompileConfig::Create(BuildTarget(Device::CPU()));
    const auto compiled =
        Compiler::CompileControlFlowExact(ReceiptWhileFunction(3), config);
    CHECK(compiled.defined() && compiled.plan().defined(),
          "Relay While must compile to a sealed control execution plan");
    const ControlExecutionPlan& plan = compiled.plan();

    // The loop condition is a carried scalar bool; the body add kernel is the
    // only kernel task.  Verify region wiring before execution.
    const ControlExecutionTask* loop_task = nullptr;
    const ControlExecutionTask* body_kernel = nullptr;
    for (const auto& region : plan.regions()) {
        for (const auto& task : region.tasks) {
            if (task.kind == ControlExecutionTaskKind::kLoop) {
                loop_task = &task;
            } else if (task.kind == ControlExecutionTaskKind::kKernel) {
                body_kernel = &task;
            }
        }
    }
    CHECK(loop_task != nullptr && body_kernel != nullptr &&
              loop_task->loop.carried.size() == 5 &&
              loop_task->loop.condition_region != loop_task->loop.body_region,
          "While must lower to distinct condition/body regions with carried leaves");
    CHECK(loop_task->loop.condition_region != -1 &&
              body_kernel != nullptr,
          "body region must depend on the carried loop state");
    ReceiptLine("while_body_entry", body_kernel->kernel.signature()->symbol);
    ReceiptLine("while_max_trip_count",
                std::to_string(loop_task->loop.max_trip_count));

    const ControlPlanLowering reference_lowering =
        LowerRelayToControlPlanWithSidecar(ReceiptWhileFunction(3));
    const ControlPlanReferenceExecutor reference(
        ArithmeticKernels(reference_lowering.primitive_units));
    const auto& reference_inputs = reference_lowering.plan.graph_inputs;
    const auto reference_input = [&](bool one, bool two, bool three) {
        return FakeValueTable{{reference_inputs[0], FakeValue::Bool(one)},
                              {reference_inputs[1], FakeValue::Bool(two)},
                              {reference_inputs[2], FakeValue::Bool(three)},
                              {reference_inputs[3], FakeValue::Bool(false)},
                              {reference_inputs[4], FakeValue::I64(7)},
                              {reference_inputs[5], FakeValue::I64(1)}};
    };
    const auto inputs_for = [](bool one, bool two, bool three) {
        return Array<NDArray>{ScalarBool(one), ScalarBool(two),
                              ScalarBool(three), ScalarBool(false),
                              ScalarI64(7), ScalarI64(1)};
    };

    ControlRuntimeSession session(plan);
    struct Case {
        bool one, two, three;
        std::int64_t trips;
    };
    for (const Case& item : std::vector<Case>{
             {false, false, false, 0}, {true, false, false, 1},
             {true, true, true, 3}}) {
        const Array<NDArray> inputs =
            inputs_for(item.one, item.two, item.three);
        const ControlRunResult result = session.Run(inputs);
        CHECK(result.outputs.size() == 5 &&
                  ReadScalarI64(result.outputs[4]) == 7 + item.trips,
              "While carried value must accumulate exactly one add per trip");
        CHECK(result.loop_iterations.size() == 1 &&
                  result.loop_iterations[0].iterations == item.trips,
              "While must report exact iteration accounting");
        CHECK(TaskEvents(result, body_kernel->id) ==
                  static_cast<std::size_t>(item.trips),
              "body kernel must launch once per executed trip");
        const ReferenceExecution expected = reference.Execute(
            reference_lowering.plan, reference_input(item.one, item.two, item.three));
        CHECK(expected.values.at(reference_lowering.plan.graph_outputs[4])
                      .integer == 7 + item.trips,
              "independent reference must agree on the carried result");
        CHECK(result.events == expected.trace.events,
              "production While events must match the independent reference trace");
        ReceiptLine("while_trips", std::to_string(item.trips));
        ReceiptLine("while_input_checksum", InputChecksum(inputs));
        ReceiptLine("while_carried_result",
                    std::to_string(7 + item.trips));
        ReceiptLine("while_body_launches",
                    std::to_string(TaskEvents(result, body_kernel->id)));
        ReceiptLine("while_events", JoinEvents(result));
    }

    // max_trip_count exhaustion must raise a clear execution-path error.
    const auto exhausted =
        Compiler::CompileControlFlowExact(ReceiptWhileFunction(1), config);
    const std::string bound_error = ErrorText([&] {
        (void)ControlRuntimeSession(exhausted.plan())
            .Run(inputs_for(true, true, false));
    });
    CHECK(bound_error.find("max_trip_count") != std::string::npos,
          "exceeding the trip bound must raise a clear max_trip_count error");
    ReceiptLine("while_bound_rejected", bound_error);
#else
    ReceiptLine("while_case", "skipped: requires gate-on LLVM build");
#endif
    return true;
}

// --- C1 evidence: rejected plans and inputs must launch nothing. -----------

bool TestReceiptInvalidPlansZeroLaunch() {
    Fixture fixture = MakeFixture("receipt_then_entry");
    const ControlPlan plan = BranchPlan();
    const std::vector<ControlKernelBinding> bindings{
        {21, fixture.module, "receipt_then_entry", {1}, std::make_shared<const int>(1)},
        {22, fixture.module, "receipt_then_entry", {1}, std::make_shared<const int>(1)}};
#if KXC_ENABLE_CONTROL_RUNTIME
    const ControlExecutionPlan bound =
        BindControlPlanForRuntime(plan, bindings);
    ControlRuntimeSession session(bound);
    CHECK(Throws([&] { (void)session.Run({ScalarI64(7), ScalarI64(7)}); }) &&
              Throws([&] { (void)session.Run({ScalarBool(true), ScalarI64(7), ScalarI64(7)}); }),
          "wrong input dtype/rank/count must fail before any launch");
    CHECK(fixture.launcher->calls == 0,
          "invalid runtime inputs must keep the launch count at zero");

    NDArray malformed = ScalarI64(7);
    auto* node = const_cast<NDArrayNode*>(malformed.As<NDArrayNode>());
    node->byte_offset = node->storage.capacity_bytes();
    CHECK(Throws([&] { (void)session.Run({ScalarBool(true), malformed}); }),
          "out-of-range storage must fail before any launch");
    CHECK(fixture.launcher->calls == 0,
          "malformed storage must keep the launch count at zero");

    auto* module_node = const_cast<kxc::api::CompiledModuleNode*>(
        fixture.module.As<kxc::api::CompiledModuleNode>());
    module_node->entries_.clear();
    CHECK(Throws([&] {
              (void)BindControlPlanForRuntime(
                  plan,
                  {{21, fixture.module, "receipt_then_entry", {1},
                    std::make_shared<const int>(1)},
                   {22, fixture.module, "receipt_then_entry", {1},
                    std::make_shared<const int>(1)}});
          }),
          "a plan whose kernel artifact disappeared must fail binding");
    CHECK(fixture.launcher->calls == 0,
          "missing kernel artifact must keep the launch count at zero");
#else
    const ControlExecutionPlan bound =
        BindControlPlanForRuntime(plan, bindings);
    CHECK(Throws([&] { ControlRuntimeSession disabled(bound); }),
          "gate-off construction must reject the session");
    CHECK(fixture.launcher->calls == 0,
          "gate-off rejection must keep the launch count at zero");
#endif

    // Runtime-extent and stream contracts fail closed on the resolved schema.
    Fixture probe = MakeFixture("receipt_probe_entry");
    const ControlExecutionPlan valid = BindControlPlanForRuntime(
        BranchPlan(),
        {{21, probe.module, "receipt_probe_entry", {1},
          std::make_shared<const int>(1)},
         {22, probe.module, "receipt_probe_entry", {1},
          std::make_shared<const int>(1)}});
    PrivatePlanSpec extent = PrivatePlanAccess::CopySpec(valid);
    const ValueSpec extent_spec = extent.values[0];
    extent.values[0] = ValueSpec(
        extent_spec->value_id, extent_spec->storage_id, extent_spec.shape(),
        extent_spec->dtype, extent_spec->device, extent_spec->is_input,
        extent_spec->is_constant, extent_spec->is_output,
        extent_spec->is_alias, extent_spec->is_async_live,
        extent_spec->is_state, extent_spec->alias_source_value_id,
        extent_spec->write_mode, 0);
    PrivatePlanSpec stream = PrivatePlanAccess::CopySpec(valid);
    stream.regions[1].tasks[0].stream = "borrowed";
    CHECK(Throws([&] { (void)PrivatePlanAccess::Create(std::move(extent)); }) &&
              Throws([&] { (void)PrivatePlanAccess::Create(std::move(stream)); }),
          "runtime extents and non-default streams must fail plan validation");
    CHECK(probe.launcher->calls == 0,
          "rejected plans must keep the launch count at zero");
    ReceiptLine("invalid_plan_launches", "0");
    return true;
}

// --- C1 evidence: gate-off entry points must reject before execution. ------

bool TestReceiptGateRejection() {
    using namespace kxc;
    const TensorType boolean({}, "bool");
    const TensorType integer({}, "int64");
    Var p("p", boolean), x("x", integer), y("y", integer);
    // 分支必须是真实 kernel（add/mul）。纯 passthrough 的 If(p, x, y) 不含
    // 任何 PrimitiveUnit，会在 CompilePrimitiveUnits 的"至少一个单元"合同下
    // 被拒绝——这是当前 fail-closed 边界，记录在 M10 receipt，不在本用例内。
    const Function function(
        {p, x, y},
        If(p, Call(relay::Op::Get("add"), {x, y}),
           Call(relay::Op::Get("mul"), {x, y})));
    const auto config =
        api::CompileConfig::Create(BuildTarget(Device::CPU()));
    const std::string compile_error = ErrorText([&] {
        (void)Compiler::CompileControlFlowExact(function, config);
    });
#if !KXC_ENABLE_CONTROL_RUNTIME
    CHECK(compile_error.find("disabled by KXC_ENABLE_CONTROL_RUNTIME") !=
              std::string::npos,
          "gate-off CompileControlFlowExact must reject before compilation");
    ReceiptLine("gate_off_compile_rejected", compile_error);
#else
    CHECK(compile_error.empty(),
          std::string("gate-on CompileControlFlowExact must resolve artifacts: ") +
              compile_error);
    ReceiptLine("gate_on_compile_accepted", "true");
#endif
    ReceiptLine("control_runtime_gate",
                KXC_ENABLE_CONTROL_RUNTIME ? "on" : "off");
    ReceiptLine("llvm_backend", KXC_USE_LLVM ? "on" : "off");
    return true;
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"receipt_gate_rejection", TestReceiptGateRejection},
        {"receipt_relay_if_both_branches", TestReceiptRelayIfBothBranches},
        {"receipt_relay_while_trips_and_bound",
         TestReceiptRelayWhileTripsAndBound},
        {"receipt_invalid_plans_zero_launch",
         TestReceiptInvalidPlansZeroLaunch},
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
