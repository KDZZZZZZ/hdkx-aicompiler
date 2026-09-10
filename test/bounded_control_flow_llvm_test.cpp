/*! \file test/bounded_control_flow_llvm_test.cpp
 * \brief PR3: bounded structured control through the ordinary
 *        Compiler::CompileBoundedStructured -> RuntimeSession chain.
 *
 * One compiled artifact serves a range of legal shapes while selecting an If
 * branch and running a bounded While, all through the single execution owner.
 */

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "kxc/compiler/compiler.h"
#include "kxc/relay/op.h"
#include "kxc/runtime/session.h"
#include "kxc/runtime/executable_plan.h"
#include "kxc/compiler/restricted_symbolic_shape.h"

#ifndef KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH
#define KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH 0
#endif
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
using kxc::Var;
using kxc::If;
using kxc::Call;
using kxc::api::Compiler;
using kxc::api::CompileConfig;
using kxc::api::CompiledGraph;
namespace restricted = kxc::api::experimental::restricted_symbolic_shape::v1;
using namespace kxc::runtime;

#define CHECK(c, m) do { if (!(c)) { std::cerr << "[FAIL] " << __FUNCTION__ << ": " << m << "\n"; return false; } } while (0)

DLDataType F32() { return DataTypeFromString("float32"); }
DLDataType Bool() { return DataTypeFromString("bool"); }

NDArray BoolScalar(bool value) {
    NDArray result = NDArray::Empty({}, Bool(), Device::CPU());
    const std::uint8_t byte = value ? 1 : 0;
    result.CopyFromBytes(&byte, sizeof(byte));
    return result;
}

NDArray Filled(const Array<int64_t>& shape, float value) {
    NDArray array = NDArray::Empty(shape, F32(), Device::CPU());
    const int64_t count = array.NBytes() / sizeof(float);
    std::vector<float> data(static_cast<size_t>(count), value);
    if (!data.empty()) array.CopyFromBytes(data.data(), data.size() * sizeof(float));
    return array;
}

std::vector<float> ReadFloats(const NDArray& array) {
    std::vector<float> data(array.NBytes() / sizeof(float));
    if (!data.empty()) array.CopyToBytes(data.data(), data.size() * sizeof(float));
    return data;
}

/*! \brief x[N,F]: if p then x+y else x*y. Used by the branch tests. */
Function ConstantIfFunction() {
    const TensorType f32({4, 4}, "float32");
    const TensorType boolean({}, "bool");
    Var x("x", f32), p("p", boolean), y("y", f32);
    return Function({p, x, y},
                    If(p, Call(kxc::relay::Op::Get("add"), {x, y}),
                       Call(kxc::relay::Op::Get("mul"), {x, y})));
}

/*! \brief Bounded loop over a symbolic-shape tensor value carried through.
 *  state = (go, x[N,F]); cond = go; body = (stop, add(x, one)). With go=true
 *  and stop=false the loop runs exactly one trip. */
Function WhileFunction() {
    const TensorType f32({4, 4}, "float32");
    const TensorType boolean({}, "bool");
    Var go("go", boolean), stop("stop", boolean), x("x", f32), one("one", f32);
    Var state("state");
    const Expr initial = kxc::Tuple({go, x});
    const Expr condition = kxc::TupleGetItem(state, 0);
    const Expr body = kxc::Tuple(
        {stop, Call(kxc::relay::Op::Get("add"),
                    {kxc::TupleGetItem(state, 1), one})});
    return Function({go, stop, x, one},
                    kxc::While(initial, state, condition, body, 3));
}

bool TestBoundedIfServesShapeRange() {
#if KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH && KXC_ENABLE_CONTROL_RUNTIME && KXC_USE_LLVM
    const auto config = CompileConfig::Create(kxc::BuildTarget(Device::CPU()));
    const std::vector<restricted::InputAxisSymbol> symbols = {
        {1, 0, "N", 1, 8, 1}, {2, 0, "N", 1, 8, 1}};
    const CompiledGraph compiled = Compiler::CompileBoundedStructured(
        ConstantIfFunction(), config, symbols);
    CHECK(compiled.defined() && compiled.plan().defined(),
          "bounded structured compile must return a defined graph");
    CHECK(compiled.plan().mode() == ExecutablePlanMode::kDynamicFreshOutputV1,
          "bounded structured plan must be a dynamic fresh-output plan");
    CHECK(compiled.plan().structured_schedule().has_value(),
          "bounded structured plan must carry a structured schedule");

    const RuntimeSession session(compiled.module(), compiled.plan());
    for (int64_t n : {1, 3, 8}) {
        const Array<NDArray> inputs = {BoolScalar(true), Filled({n, 4}, 2.0f),
                                       Filled({n, 4}, 1.0f)};
        const Array<NDArray> outputs = session.Run(inputs);
        CHECK(outputs.size() == 1 && outputs[0].shape().size() == 2 &&
                  outputs[0].shape()[0] == n && outputs[0].shape()[1] == 4,
              "one artifact must serve every legal N with the right shape");
        for (float value : ReadFloats(outputs[0])) {
            CHECK(value == 3.0f, "then branch must compute add(x, y) == 3");
        }
    }
    const Array<NDArray> else_inputs = {BoolScalar(false), Filled({5, 4}, 2.0f),
                                        Filled({5, 4}, 3.0f)};
    const Array<NDArray> else_outputs = session.Run(else_inputs);
    for (float value : ReadFloats(else_outputs[0])) {
        CHECK(value == 6.0f, "else branch must compute mul(x, y) == 6");
    }
#else
    std::cout << "[SKIP] bounded structured If needs gate-on LLVM build\n";
#endif
    return true;
}

bool TestBoundedIfRejectsOutOfRange() {
#if KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH && KXC_ENABLE_CONTROL_RUNTIME && KXC_USE_LLVM
    const auto config = CompileConfig::Create(kxc::BuildTarget(Device::CPU()));
    const std::vector<restricted::InputAxisSymbol> symbols = {
        {1, 0, "N", 2, 8, 1}, {2, 0, "N", 2, 8, 1}};
    const CompiledGraph compiled = Compiler::CompileBoundedStructured(
        ConstantIfFunction(), config, symbols);
    const RuntimeSession session(compiled.module(), compiled.plan());
    bool threw = false;
    try {
        (void)session.Run({BoolScalar(true), Filled({1, 4}, 2.0f),
                           Filled({1, 4}, 1.0f)});
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw, "a shape below the declared bound must reject before launch");
#endif
    return true;
}

bool TestBoundedWhileServesShapeRange() {
#if KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH && KXC_ENABLE_CONTROL_RUNTIME && KXC_USE_LLVM
    const auto config = CompileConfig::Create(kxc::BuildTarget(Device::CPU()));
    const std::vector<restricted::InputAxisSymbol> symbols = {
        {2, 0, "N", 1, 8, 1}, {3, 0, "N", 1, 8, 1}};
    const CompiledGraph compiled =
        Compiler::CompileBoundedStructured(WhileFunction(), config, symbols);
    CHECK(compiled.defined() && compiled.plan().structured_schedule().has_value(),
          "bounded While must publish a structured plan");
    const RuntimeSession session(compiled.module(), compiled.plan());
    struct Case { bool go; int64_t trips; };
    for (const Case& item : std::vector<Case>{{false, 0}, {true, 1}}) {
        const int64_t n = 3;
        // params are {go, stop, x, one}. x starts at 1.0; each trip adds 2.0.
        const Array<NDArray> inputs = {BoolScalar(item.go), BoolScalar(false),
                                       Filled({n, 4}, 1.0f),
                                       Filled({n, 4}, 2.0f)};
        const Array<NDArray> outputs = session.Run(inputs);
        CHECK(outputs.size() == 2, "While must expose both carried leaves");
        CHECK(outputs[0].shape().empty(), "the carried bool stays scalar");
        CHECK(outputs[1].shape()[0] == n && outputs[1].shape()[1] == 4,
              "the carried tensor keeps its symbolic shape");
        const float expected = 1.0f + static_cast<float>(item.trips) * 2.0f;
        for (float value : ReadFloats(outputs[1])) {
            CHECK(value == expected, "While must add exactly once per trip");
        }
    }
#else
    std::cout << "[SKIP] bounded structured While needs gate-on LLVM build\n";
#endif
    return true;
}

}  // namespace

int main() {
    struct Test { const char* name; bool (*run)(); };
    const std::vector<Test> tests = {
        {"bounded_if_serves_shape_range", TestBoundedIfServesShapeRange},
        {"bounded_if_rejects_out_of_range", TestBoundedIfRejectsOutOfRange},
        {"bounded_while_serves_shape_range", TestBoundedWhileServesShapeRange},
    };
    int failures = 0;
    for (const Test& test : tests) {
        if (!test.run()) ++failures;
        else { std::cout << "[PASS] " << test.name << "\n"; std::cout.flush(); }
    }
    std::cout << (failures == 0 ? "[PASS] bounded_control_flow_llvm\n"
                                : "[FAIL] bounded_control_flow_llvm\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
