/*! \file test/m10_unified_control_llvm_test.cpp
 * \brief M10 C3 PR1/PR2: structured control through the ordinary
 *        Compiler::Compile -> CompiledGraph -> RuntimeSession chain.
 *
 *  Proves the unified path: a Relay If and a bounded While compile to a normal
 *  CompiledGraph (static ExecutablePlan plus an optional structured_schedule)
 *  and execute on the ordinary RuntimeSession, with the same module/invocation
 *  machinery as the linear dataflow path. Kernel launches are observed through
 *  the module's profiling observer so unselected branches can be shown to be
 *  zero-submit.
 */

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "kxc/compiler/compiler.h"
#include "kxc/compiler/experimental_identity.h"
#include "kxc/profiling/profiling.h"
#include "kxc/relay/op.h"
#include "kxc/runtime/session.h"
#include "support/control_plan_reference_executor.h"
#include "../src/compiler/control_flow/internal_lowering.h"

#ifndef KXC_ENABLE_CONTROL_RUNTIME
#define KXC_ENABLE_CONTROL_RUNTIME 0
#endif
#ifndef KXC_USE_LLVM
#define KXC_USE_LLVM 0
#endif

namespace {
using kxc::Array;
using kxc::Device;
using kxc::Expr;
using kxc::Function;
using kxc::TensorType;
using kxc::Tuple;
using kxc::TupleGetItem;
using kxc::Var;
using kxc::While;
using kxc::If;
using kxc::Call;
using kxc::api::Compiler;
using kxc::api::CompileConfig;
using kxc::api::CompiledGraph;
using namespace kxc::api::internal;
using namespace kxc::runtime;

#define CHECK(condition, message) do { if (!(condition)) { std::cerr << "[FAIL] " << __FUNCTION__ << ": " << message << "\n"; return false; } } while (0)

DLDataType Type(const char* name) {
    return DataTypeFromString(name);
}

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

std::int64_t ReadScalarI64(const NDArray& value) {
    std::int64_t result{0};
    value.CopyToBytes(&result, sizeof(result));
    return result;
}

/*! \brief Count kernel_submit events per (run_id, kernel_symbol). */
std::map<std::string, std::map<std::string, std::size_t>> SubmitsPerRun(
    const std::string& bundle_dir) {
    std::ifstream input(std::filesystem::path(bundle_dir) / "events.jsonl");
    std::map<std::string, std::map<std::string, std::size_t>> counts;
    std::string line;
    const std::string kind = "\"event_type\":\"kernel_submit\"";
    const auto value_after = [&](const std::string& key) -> std::string {
        const std::size_t at = line.find(key);
        if (at == std::string::npos) return {};
        const std::size_t begin = at + key.size();
        const std::size_t end = line.find('"', begin);
        return end == std::string::npos ? std::string()
                                        : line.substr(begin, end - begin);
    };
    while (std::getline(input, line)) {
        if (line.find(kind) == std::string::npos) continue;
        ++counts[value_after("\"run_id\":\"")][value_after("\"kernel_symbol\":\"")];
    }
    return counts;
}

std::string BundleDir(const std::string& name) {
    const auto base = std::filesystem::current_path() /
                      "m10_unified_control_output" / name;
    std::filesystem::remove_all(base);
    return base.string();
}

/*! \brief Compile under a caller-owned profiling context so runtime kernel
 *  submits land in the same bundle and can be counted after Flush(). */
CompiledGraph CompileProfiled(Function function, const std::string& bundle_dir,
                              std::shared_ptr<kxc::profiling::ProfileContext>* out) {
    kxc::profiling::ProfileOptions options;
    options.enabled = true;
    options.bundle_dir = bundle_dir;
    options.ir_capture_mode = kxc::profiling::IRCaptureMode::kDisabled;
    auto context = kxc::profiling::ProfileContext::Create(options);
    *out = context;
    kxc::profiling::ActivationScope activation(context, "m10_unified");
    return Compiler::Compile(std::move(function),
                             CompileConfig::Create(kxc::BuildTarget(Device::CPU()), 2,
                                                   options));
}

Function NestedIfFunction() {
    const TensorType boolean({}, "bool");
    const TensorType integer({}, "int64");
    Var p("p", boolean), q("q", boolean), x("x", integer), y("y", integer);
    const Expr inner = If(q, Call(kxc::relay::Op::Get("add"), {x, y}),
                          Call(kxc::relay::Op::Get("subtract"), {x, y}));
    return Function({p, q, x, y},
                    If(p, inner, Call(kxc::relay::Op::Get("mul"), {x, y})));
}

Function WhileFunction(std::int64_t max_trip_count) {
    const TensorType boolean({}, "bool");
    const TensorType integer({}, "int64");
    Var first("first", boolean), second("second", boolean);
    Var third("third", boolean), tail("tail", boolean);
    Var value("value", integer), increment("increment", integer);
    Var state("state");
    const Expr initial = Tuple({first, second, third, tail, value});
    const Expr condition = TupleGetItem(state, 0);
    const Expr body = Tuple({TupleGetItem(state, 1), TupleGetItem(state, 2),
                             TupleGetItem(state, 3), tail,
                             Call(kxc::relay::Op::Get("add"),
                                  {TupleGetItem(state, 4), increment})});
    return Function({first, second, third, tail, value, increment},
                    While(initial, state, condition, body, max_trip_count));
}

/*! \brief Loop whose body swaps two carried integers, so a naive in-order
 *  backedge install would corrupt the result. Body = (stop, b, a + 1). */
Function WhileSwapFunction() {
    const TensorType boolean({}, "bool");
    const TensorType integer({}, "int64");
    Var flag("flag", boolean), a("a", integer), b("b", integer);
    Var one("one", integer), stop("stop", boolean);
    Var state("state");
    const Expr initial = Tuple({flag, a, b});
    const Expr condition = TupleGetItem(state, 0);
    const Expr body =
        Tuple({stop, TupleGetItem(state, 2),
               Call(kxc::relay::Op::Get("add"),
                    {TupleGetItem(state, 1), one})});
    return Function({flag, a, b, one, stop},
                    While(initial, state, condition, body, 4));
}

/*! \brief A structured plan must be a normal static plan plus a schedule. */
bool CheckUnifiedShape(const CompiledGraph& compiled) {
    CHECK(compiled.defined(), "unified compile must return a defined CompiledGraph");
    const ExecutablePlan& plan = compiled.plan();
    CHECK(plan.mode() == ExecutablePlanMode::kStatic,
          "structured publish must keep the static allocation/state mode");
    CHECK(plan.structured_schedule().has_value(),
          "structured publish must carry a structured schedule");
    CHECK(!plan.calls().empty(), "structured plan must still expose kernel call points");
    return true;
}

bool TestUnifiedNestedIf() {
#if KXC_ENABLE_CONTROL_RUNTIME && KXC_USE_LLVM
    const std::string bundle = BundleDir("nested_if");
    std::shared_ptr<kxc::profiling::ProfileContext> context;
    const CompiledGraph compiled =
        CompileProfiled(NestedIfFunction(), bundle, &context);
    CHECK(CheckUnifiedShape(compiled), "nested If must publish the unified shape");

    const RuntimeSession session(compiled.module(), compiled.plan());
    struct Case {
        bool p, q;
        std::int64_t expected;
    };
    // p selects inner vs mul; q selects add vs subtract. add=5, sub=-1, mul=6.
    for (const Case& item : std::vector<Case>{
             {false, false, 6}, {true, true, 5}, {true, false, -1}}) {
        const Array<NDArray> inputs = {ScalarBool(item.p), ScalarBool(item.q),
                                       ScalarI64(2), ScalarI64(3)};
        const Array<NDArray> outputs = session.Run(inputs);
        CHECK(outputs.size() == 1 &&
                  ReadScalarI64(outputs[0]) == item.expected,
              "nested If result must follow the selected path");
    }
    context->Flush();
    // Each run must submit exactly one branch kernel, and all three distinct
    // arithmetic call points must have been selected exactly once overall.
    const auto per_run = SubmitsPerRun(bundle);
    CHECK(per_run.size() == 3, "nested If must have three observed runs");
    std::map<std::string, std::size_t> totals_by_symbol;
    for (const auto& run : per_run) {
        std::size_t total = 0;
        for (const auto& entry : run.second) {
            total += entry.second;
            totals_by_symbol[entry.first] += entry.second;
        }
        CHECK(total == 1, "nested If must submit exactly one kernel per run");
    }
    CHECK(totals_by_symbol.size() == 3 &&
              totals_by_symbol["kxc_unit_0_add"] == 1 &&
              totals_by_symbol["kxc_unit_1_subtract"] == 1 &&
              totals_by_symbol["kxc_unit_2_mul"] == 1,
          "each nested If path must be selected exactly once with no branch leaking");
#else
    std::cout << "[SKIP] unified nested If requires gate-on LLVM build\n";
#endif
    return true;
}

bool TestUnifiedWhile() {
#if KXC_ENABLE_CONTROL_RUNTIME && KXC_USE_LLVM
    const std::string bundle = BundleDir("while");
    std::shared_ptr<kxc::profiling::ProfileContext> context;
    const CompiledGraph compiled =
        CompileProfiled(WhileFunction(3), bundle, &context);
    CHECK(CheckUnifiedShape(compiled), "While must publish the unified shape");

    const RuntimeSession session(compiled.module(), compiled.plan());
    struct Case {
        bool one, two, three;
        int64_t trips;
    };
    for (const Case& item : std::vector<Case>{
             {false, false, false, 0}, {true, false, false, 1},
             {true, true, true, 3}}) {
        const Array<NDArray> inputs = {ScalarBool(item.one), ScalarBool(item.two),
                                       ScalarBool(item.three), ScalarBool(false),
                                       ScalarI64(7), ScalarI64(1)};
        const Array<NDArray> outputs = session.Run(inputs);
        CHECK(outputs.size() == 5 && ReadScalarI64(outputs[4]) == 7 + item.trips,
              "While carried value must accumulate one add per trip");
    }
    context->Flush();
    // One body call point re-executed per iteration: the 0-trip run submits no
    // kernel at all, the 1-trip run submits once, and the 3-trip run submits
    // three times -- so the multiset of per-run submits is {1, 3}.
    const auto per_run = SubmitsPerRun(bundle);
    std::vector<std::size_t> totals;
    for (const auto& run : per_run) {
        std::size_t total = 0;
        for (const auto& entry : run.second) {
            total += entry.second;
            CHECK(entry.first.find("add") != std::string::npos,
                  "only the body add kernel may be submitted by this fixture");
        }
        totals.push_back(total);
    }
    std::sort(totals.begin(), totals.end());
    CHECK(totals == std::vector<std::size_t>({1, 3}),
          "While body must launch exactly once per executed trip");
#else
    std::cout << "[SKIP] unified While requires gate-on LLVM build\n";
#endif
    return true;
}

bool TestUnifiedWhileBoundRejected() {
#if KXC_ENABLE_CONTROL_RUNTIME && KXC_USE_LLVM
    const std::string bundle = BundleDir("while_bound");
    std::shared_ptr<kxc::profiling::ProfileContext> context;
    const CompiledGraph compiled =
        CompileProfiled(WhileFunction(1), bundle, &context);
    CHECK(CheckUnifiedShape(compiled), "bounded While must publish the unified shape");
    const RuntimeSession session(compiled.module(), compiled.plan());
    bool threw = false;
    try {
        const Array<NDArray> inputs = {ScalarBool(true), ScalarBool(true),
                                       ScalarBool(true), ScalarBool(false),
                                       ScalarI64(7), ScalarI64(1)};
        (void)session.Run(inputs);
    } catch (const std::exception& error) {
        threw = std::string(error.what()).find("max_trip_count") !=
                std::string::npos;
    }
    CHECK(threw, "exceeding max_trip_count must fail through the ordinary session");
#endif
    return true;
}

bool TestUnifiedReferenceAgreement() {
#if KXC_ENABLE_CONTROL_RUNTIME && KXC_USE_LLVM
    using namespace kxc::api::internal::test_support;
    // The independent ControlPlan reference executor still agrees on the same
    // Relay source the unified compiler consumed.
    const ControlPlanLowering lowering = LowerRelayToControlPlanWithSidecar(
        WhileFunction(3));
    std::unordered_map<PrimitiveUnitId, std::string> names;
    for (const auto& unit : lowering.primitive_units) {
        names.emplace(unit.id, std::string(unit.call.spec.name));
    }
    const ControlPlanReferenceExecutor reference(
        [names = std::move(names)](const ControlTask& task,
                                   const std::vector<FakeValue>& args) {
            const std::string& op = names.at(task.primitive_unit_id);
            if (op == "add") {
                return std::vector<FakeValue>{
                    FakeValue::I64(args[0].integer + args[1].integer)};
            }
            throw std::runtime_error("unexpected reference op " + op);
        });
    const auto& inputs = lowering.plan.graph_inputs;
    const ReferenceExecution expected = reference.Execute(
        lowering.plan,
        FakeValueTable{{inputs[0], FakeValue::Bool(true)},
                       {inputs[1], FakeValue::Bool(false)},
                       {inputs[2], FakeValue::Bool(false)},
                       {inputs[3], FakeValue::Bool(false)},
                       {inputs[4], FakeValue::I64(7)},
                       {inputs[5], FakeValue::I64(1)}});
    CHECK(expected.values.at(lowering.plan.graph_outputs[4]).integer == 8,
          "independent reference must agree on the while carried result");
#else
    std::cout << "[SKIP] unified reference agreement requires gate-on LLVM build\n";
#endif
    return true;
}

bool TestUnifiedCarriedTupleSwap() {
#if KXC_ENABLE_CONTROL_RUNTIME && KXC_USE_LLVM
    using namespace kxc::relay;
    const std::string bundle = BundleDir("carried_swap");
    std::shared_ptr<kxc::profiling::ProfileContext> context;
    const CompiledGraph compiled =
        CompileProfiled(WhileSwapFunction(), bundle, &context);
    CHECK(CheckUnifiedShape(compiled), "swap loop must publish the unified shape");
    const RuntimeSession session(compiled.module(), compiled.plan());
    // initial: flag=true, a=1, b=2. body: (stop, b, a+1, ...) => (false, 2, 2).
    const Array<NDArray> inputs = {ScalarBool(true), ScalarI64(1), ScalarI64(2),
                                   ScalarI64(1), ScalarBool(false)};
    const Array<NDArray> outputs = session.Run(inputs);
    CHECK(outputs.size() == 3, "swap loop must expose three carried outputs");
    const bool flag = [&] {
        std::uint8_t byte{0};
        outputs[0].CopyToBytes(&byte, sizeof(byte));
        return byte != 0;
    }();
    CHECK(!flag, "swap loop must carry the body's false condition");
    CHECK(ReadScalarI64(outputs[1]) == 2 && ReadScalarI64(outputs[2]) == 2,
          "carried backedge install must use pre-update values, not sequential writes");
#else
    std::cout << "[SKIP] carried tuple swap requires gate-on LLVM build\n";
#endif
    return true;
}

bool TestUnifiedTopologyIdentity() {
#if KXC_ENABLE_CONTROL_RUNTIME && KXC_USE_LLVM
    const std::string bundle = BundleDir("identity");
    std::shared_ptr<kxc::profiling::ProfileContext> context;
    const CompiledGraph if_graph =
        CompileProfiled(NestedIfFunction(), bundle, &context);
    const CompiledGraph while_graph =
        CompileProfiled(WhileFunction(3), bundle, &context);
    const CompiledGraph while_graph_other =
        CompileProfiled(WhileFunction(2), bundle, &context);
    const auto if_fingerprint =
        kxc::api::BuildPlanAbiFingerprint(if_graph);
    const auto while_fingerprint =
        kxc::api::BuildPlanAbiFingerprint(while_graph);
    const auto while_other_fingerprint =
        kxc::api::BuildPlanAbiFingerprint(while_graph_other);
    CHECK(if_fingerprint.defined() && while_fingerprint.defined() &&
              while_other_fingerprint.defined(),
          "structured plans must have a defined plan ABI fingerprint");
    CHECK(if_fingerprint != while_fingerprint,
          "distinct structured topologies must have distinct plan ABI identity");
    CHECK(while_fingerprint != while_other_fingerprint,
          "a different loop bound must change plan ABI identity");
    CHECK(while_fingerprint == kxc::api::BuildPlanAbiFingerprint(while_graph),
          "the same structured plan must have stable identity");
#endif
    return true;
}

}  // namespace

int main() {
    struct Test {
        const char* name;
        bool (*run)();
    };
    const std::vector<Test> tests = {
        {"unified_nested_if", TestUnifiedNestedIf},
        {"unified_while", TestUnifiedWhile},
        {"unified_carried_tuple_swap", TestUnifiedCarriedTupleSwap},
        {"unified_while_bound_rejected", TestUnifiedWhileBoundRejected},
        {"unified_reference_agreement", TestUnifiedReferenceAgreement},
        {"unified_topology_identity", TestUnifiedTopologyIdentity},
    };
    int failures = 0;
    for (const Test& test : tests) {
        if (!test.run()) {
            ++failures;
        } else {
            std::cout << "[PASS] " << test.name << "\n";
        }
    }
    std::cout << (failures == 0 ? "[PASS] m10_unified_control_llvm\n"
                                : "[FAIL] m10_unified_control_llvm\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
