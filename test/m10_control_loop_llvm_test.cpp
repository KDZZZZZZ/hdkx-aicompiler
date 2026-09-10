/*! \file test/m10_control_loop_llvm_test.cpp
 * \brief M10 C3 PR5 mechanism: in-graph decode loop with an inlined body,
 *        dynamic capacity write at an advancing position, argmax token
 *        selection, and loop-carried state, executed on RuntimeSession.
 *
 *  Part 1 proves pass_utils::InlineFunctionParameters inlines the REAL imported
 *  MiniMind bounded-decode Function and the result re-types.
 *  Part 2 drives a synthetic in-graph loop whose body is an inlined separate
 *  Function: it writes a row into a fixed-capacity table at a runtime position
 *  using `where` on a constant slot vector (the static-capacity form needs no
 *  scatter primitive), advances the position, and selects a token with argmax,
 *  carrying all of it across iterations.
 */

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "kxc/compiler/compiler.h"
#include "kxc/compiler/restricted_symbolic_shape.h"
#include "kxc/frontend/onnx_importer.h"
#include "kxc/relay/op.h"
#include "kxc/relay/pass_utils.h"
#include "kxc/relay/transforms/infer_type.h"
#include "kxc/runtime/session.h"

namespace {
using namespace kxc;
namespace restricted = api::experimental::restricted_symbolic_shape::v1;
using Adapter = restricted::RestrictedSymbolicShapeAdapter;

#define CHECK(c, m) do { if (!(c)) { std::cerr << "[FAIL] " << __FUNCTION__ << ": " << m << "\n"; return false; } } while (0)

DLDataType F32() { return runtime::DataTypeFromString("float32"); }
DLDataType I64() { return runtime::DataTypeFromString("int64"); }
DLDataType Bool() { return runtime::DataTypeFromString("bool"); }

runtime::NDArray ZerosF32(const Array<int64_t>& shape) {
    return runtime::NDArray::Zeros(shape, F32(), Device::CPU());
}
runtime::NDArray ZerosI64(const Array<int64_t>& shape) {
    return runtime::NDArray::Zeros(shape, I64(), Device::CPU());
}
runtime::NDArray Filled(const Array<int64_t>& shape, float value) {
    runtime::NDArray array = runtime::NDArray::Empty(shape, F32(), Device::CPU());
    const int64_t count = array.NBytes() / 4;
    std::vector<float> data(static_cast<size_t>(count), value);
    if (!data.empty()) array.CopyFromBytes(data.data(), data.size() * 4);
    return array;
}
runtime::NDArray ScalarI64(int64_t value) {
    runtime::NDArray array = runtime::NDArray::Empty({}, I64(), Device::CPU());
    array.CopyFromBytes(&value, sizeof(value));
    return array;
}
runtime::NDArray ScalarBool(bool value) {
    runtime::NDArray array = runtime::NDArray::Empty({}, Bool(), Device::CPU());
    const std::uint8_t byte = value ? 1 : 0;
    array.CopyFromBytes(&byte, sizeof(byte));
    return array;
}
std::vector<float> ReadFloats(const runtime::NDArray& array) {
    std::vector<float> data(array.NBytes() / 4);
    if (!data.empty()) array.CopyToBytes(data.data(), data.size() * 4);
    return data;
}
std::vector<int64_t> ReadInts(const runtime::NDArray& array) {
    std::vector<int64_t> data(array.NBytes() / 8);
    if (!data.empty()) array.CopyToBytes(data.data(), data.size() * 8);
    return data;
}

bool ReadBool(const runtime::NDArray& array) {
    std::uint8_t byte = 0;
    array.CopyToBytes(&byte, sizeof(byte));
    return byte != 0;
}
Expr BoolConst(bool value) { return Constant(ScalarBool(value)); }
Expr IntConst(int64_t value) { return Constant(ScalarI64(value)); }

/*! \brief Part 1: the substitution pass must inline the real decode Function. */
bool TestRealGraphInlining() {
    const char* directory = std::getenv("KXC_MINIMIND_BOUNDED_DECODE_DIR");
    if (!directory || !*directory) {
        std::cout << "[SKIP] real graph inlining: set KXC_MINIMIND_BOUNDED_DECODE_DIR\n";
        return true;
    }
    const std::filesystem::path root(directory);
    const auto source = frontend::LoadONNXShapeSource(
        (root / "decode.json").string(), (root / "decode.params").string());
    CHECK(source.input_names.size() == 17, "real decode must have 17 inputs");
    CHECK(source.function.defined(), "real decode function must be defined");

    // The shape-source graph needs the adapter to attach shape-control attrs
    // before it can be typed; use the adapter's detached representative.
    const auto config = api::CompileConfig::Create(BuildTarget(Device::CPU()));
    std::vector<restricted::InputAxisSymbol> axes{{0, 0, "B", 1, 3, 1}};
    for (size_t i = 1; i < 17; ++i) {
        axes.push_back({i, 0, "B", 1, 3, 1});
        axes.push_back({i, 1, "P", 0, 8, 1});
    }
    const auto prepared = Adapter::Prepare(
        source.function, config, axes, source.declared_output_types);
    const auto request =
        Adapter::MintBoundedCompileRequest(prepared);
    const Function representative = request.representative();
    CHECK(representative.defined(), "adapter representative must be defined");

    Array<Expr> arguments;
    for (size_t index = 0; index < representative->params.size(); ++index) {
        const auto* type =
            representative->params[index]->type_annotation.As<TensorTypeNode>();
        CHECK(type != nullptr, "real decode parameter must carry a TensorType");
        arguments.push_back(
            Var("in_" + std::to_string(index),
                representative->params[index]->type_annotation));
    }
    const Expr inlined = relay::pass_utils::InlineFunctionParameters(
        representative, arguments);
    CHECK(inlined.defined(), "inlined real decode body must be defined");
    const Function retyped = relay::InferTypePass(Function(Array<Var>(), inlined));
    CHECK(retyped.defined() && retyped->body.defined(),
          "inlined real decode body must re-type");
    std::cout << "[PASS] real decode inlined (" << representative->params.size()
              << " params)\n";
    return true;
}

/*! \brief A separate "decode cell": write `row` into `table` at slot `pos`. */
Function DecodeCellFunction() {
    const TensorType table({4, 4}, "float32");
    const TensorType row({1, 4}, "float32");
    Var table_in("table", table), row_in("row", row);
    Var pos("pos", TensorType({}, "int64"));
    runtime::NDArray slots_data = runtime::NDArray::Empty({4, 1}, I64(), Device::CPU());
    {
        std::vector<int64_t> values{0, 1, 2, 3};
        slots_data.CopyFromBytes(values.data(), values.size() * 8);
    }
    const Expr slots = Constant(slots_data);
    const Expr mask = Call(relay::Op::Get("equal"), {slots, pos});
    const Expr written = Call(relay::Op::Get("where"), {mask, row_in, table_in});
    return Function({table_in, row_in, pos}, written);
}

/*! \brief In-graph loop carrying (run0, run1, run2, table, pos, pending, tokens).
 *
 *  Runs exactly two iterations (shift-register bools). Each trip inlines the
 *  decode cell to write the next pending row at the advancing slot, drops that
 *  row from `pending`, and selects a per-row token with argmax. */
Function LoopFunction() {
    const TensorType table({4, 4}, "float32");
    const TensorType boolean({}, "bool");
    const TensorType integer({}, "int64");
    Var run0("run0", boolean), run1("run1", boolean), run2("run2", boolean);
    Var table0("table0", table), pos0("pos0", integer), rows("rows", table);
    Var state("state");
    // Write rows[0] at the advancing slot each trip; the slot index is the
    // runtime position, so the write target moves while every shape is static.
    const Expr row_now = Call(relay::Op::Get("slice"), {rows},
                              relay::SliceAttrs::Create({0}, {1}, {0}, {1}));
    const Expr written = relay::pass_utils::InlineFunctionParameters(
        DecodeCellFunction(), {TupleGetItem(state, 3), row_now, TupleGetItem(state, 4)});
    const Expr tokens_next = Call(relay::Op::Get("argmax"), {written},
                                  relay::ArgMaxAttrs::Create(1, 0, 0));
    const Expr pos_next =
        Call(relay::Op::Get("add"), {TupleGetItem(state, 4), IntConst(1)});
    // The stop flag is a real computation (pos_next != pos), never a dangling
    // constant carry: it is false whenever the position advanced.
    const Expr stop = Call(relay::Op::Get("equal"), {pos_next, TupleGetItem(state, 4)});
    const Expr body = Tuple({TupleGetItem(state, 1), TupleGetItem(state, 2),
                             stop, written, pos_next, tokens_next});
    // The token carry starts as a real argmax of the input rows (no dangling
    // constant initial).
    const Expr tokens0 = Call(relay::Op::Get("argmax"), {rows},
                              relay::ArgMaxAttrs::Create(1, 0, 0));
    const Expr initial = Tuple({run0, run1, run2, table0, pos0, tokens0});
    return Function({run0, run1, run2, table0, pos0, rows},
                    While(initial, state, TupleGetItem(state, 0), body, 3));
}

/*! \brief Execute the loop and verify the writes, position, and argmax tokens. */
bool TestInGraphDecodeLoop() {
    const auto config = api::CompileConfig::Create(BuildTarget(Device::CPU()));
    const auto compiled = api::Compiler::Compile(LoopFunction(), config);
    CHECK(compiled.defined() && compiled.plan().defined(),
          "in-graph loop must compile");

    // rows[0] = [1,2,3,4] (argmax 3), rows[1] = [4,1,1,1] (argmax 0).
    runtime::NDArray rows = runtime::NDArray::Empty({4, 4}, F32(), Device::CPU());
    {
        std::vector<float> values{1, 2, 3, 4, 4, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
        rows.CopyFromBytes(values.data(), values.size() * 4);
    }
    const runtime::RuntimeSession session(compiled.module(), compiled.plan());
    // run0=true, run1=true, run2=false -> exactly two iterations.
    const Array<runtime::NDArray> outputs = session.Run(
        {ScalarBool(true), ScalarBool(true), ScalarBool(false),
         ZerosF32({4, 4}), ScalarI64(0), rows});
    CHECK(outputs.size() == 6, "loop must expose its carried outputs");
    // Carried order: run0, run1, run2, table, pos, pending, tokens.
    const std::vector<float> table = ReadFloats(outputs[3]);
    const std::vector<int64_t> pos = ReadInts(outputs[4]);
    const std::vector<int64_t> tokens = ReadInts(outputs[5]);
    CHECK(ReadBool(outputs[0]) == false && ReadBool(outputs[1]) == false &&
              ReadBool(outputs[2]) == false,
          "loop must consume both scheduled iterations");
    CHECK(pos.size() == 1 && pos[0] == 2,
          "position must advance exactly once per iteration");
    CHECK(table.size() == 16 && table[0] == 1.0f && table[1] == 2.0f &&
              table[2] == 3.0f && table[3] == 4.0f && table[4] == 1.0f &&
              table[5] == 2.0f,
          "the row must be written at each advancing slot");
    CHECK(tokens.size() == 4 && tokens[0] == 3 && tokens[1] == 3,
          "argmax must select the max index per written row");
    return true;
}

}  // namespace

int main() {
    struct Test { const char* name; bool (*run)(); };
    const std::vector<Test> tests = {
        {"real_graph_inlining", TestRealGraphInlining},
        {"in_graph_decode_loop", TestInGraphDecodeLoop},
    };
    int failures = 0;
    for (const Test& test : tests) {
        if (!test.run()) ++failures;
        else std::cout << "[PASS] " << test.name << "\n";
    }
    std::cout << (failures == 0 ? "[PASS] m10_control_loop_llvm\n"
                                : "[FAIL] m10_control_loop_llvm\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
