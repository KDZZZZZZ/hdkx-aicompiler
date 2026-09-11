/*! \file test/m10_control_loop_llvm_test.cpp
 * \brief M10 C3 PR5: in-graph decode loop with an inlined real MiniMind body,
 *        dynamic capacity write, in-graph argmax token selection, and
 *        loop-carried KV state, executed on the ordinary RuntimeSession.
 *
 *  Part 1 proves pass_utils::InlineFunctionParameters inlines the REAL imported
 *  MiniMind bounded-decode representative and the result re-types.
 *  Part 2 drives a synthetic in-graph loop (inlined separate Function) that
 *  writes a row at a runtime position and selects a token with argmax.
 *  Part 3 runs the REAL fixed-capacity MiniMind decode as the While body: the
 *  loop carries token, position and all 16 KV tables, builds the additive mask
 *  from the advancing position, runs the real decode, picks the next token with
 *  argmax, and writes the fresh KV at the current position. Final state is
 *  compared elementwise with the fixture's ref_step*_state_*.bin.
 */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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
runtime::NDArray I64Scalar(int64_t value) {
    runtime::NDArray array = runtime::NDArray::Empty({}, I64(), Device::CPU());
    array.CopyFromBytes(&value, sizeof(value));
    return array;
}
runtime::NDArray I64Shape1(int64_t value) {
    runtime::NDArray array = runtime::NDArray::Empty({1}, I64(), Device::CPU());
    array.CopyFromBytes(&value, sizeof(value));
    return array;
}
runtime::NDArray I64Shape11(int64_t value) {
    runtime::NDArray array = runtime::NDArray::Empty({1, 1}, I64(), Device::CPU());
    array.CopyFromBytes(&value, sizeof(value));
    return array;
}
runtime::NDArray I64Vec(const std::vector<int64_t>& values,
                        const Array<int64_t>& shape) {
    runtime::NDArray array = runtime::NDArray::Empty(shape, I64(), Device::CPU());
    if (!values.empty()) array.CopyFromBytes(values.data(), values.size() * 8);
    return array;
}
runtime::NDArray F32Scalar(float value) {
    runtime::NDArray array = runtime::NDArray::Empty({}, F32(), Device::CPU());
    array.CopyFromBytes(&value, sizeof(value));
    return array;
}
runtime::NDArray Full(const Array<int64_t>& shape, float value) {
    runtime::NDArray array = runtime::NDArray::Empty(shape, F32(), Device::CPU());
    const int64_t count = array.NBytes() / 4;
    std::vector<float> data(static_cast<size_t>(count), value);
    if (!data.empty()) array.CopyFromBytes(data.data(), data.size() * 4);
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
std::vector<char> ReadFile(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.good()) throw std::runtime_error("cannot read " + path.string());
    return std::vector<char>((std::istreambuf_iterator<char>(in)),
                             std::istreambuf_iterator<char>());
}
std::vector<float> ReadFloatsFile(const std::filesystem::path& path) {
    const std::vector<char> bytes = ReadFile(path);
    std::vector<float> values(bytes.size() / 4);
    if (!values.empty()) std::memcpy(values.data(), bytes.data(), values.size() * 4);
    return values;
}

Expr BoolConst(bool value) { return Constant(ScalarBool(value)); }
Expr IntConst(int64_t value) { return Constant(I64Scalar(value)); }

/*! \brief Part 1: the substitution pass inlines the real decode Function. */
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

    const auto config = api::CompileConfig::Create(BuildTarget(Device::CPU()));
    std::vector<restricted::InputAxisSymbol> axes{{0, 0, "B", 1, 3, 1}};
    for (size_t i = 1; i < 17; ++i) {
        axes.push_back({i, 0, "B", 1, 3, 1});
        axes.push_back({i, 1, "P", 0, 8, 1});
    }
    const auto prepared = Adapter::Prepare(source.function, config, axes,
                                           source.declared_output_types);
    const auto request = Adapter::MintBoundedCompileRequest(prepared);
    const Function representative = request.representative();
    CHECK(representative.defined(), "adapter representative must be defined");

    Array<Expr> arguments;
    for (size_t index = 0; index < representative->params.size(); ++index) {
        arguments.push_back(Var("in_" + std::to_string(index),
                                representative->params[index]->type_annotation));
    }
    const Expr inlined =
        relay::pass_utils::InlineFunctionParameters(representative, arguments);
    CHECK(inlined.defined(), "inlined real decode body must be defined");
    const Function retyped = relay::InferTypePass(Function(Array<Var>(), inlined));
    CHECK(retyped.defined() && retyped->body.defined(),
          "inlined real decode body must re-type");
    std::cout << "[PASS] real decode inlined (" << representative->params.size()
              << " params)\n";
    return true;
}

/*! \brief A separate decode cell: write `row` into `table` at slot `pos`. */
Function DecodeCellFunction() {
    const TensorType table({4, 4}, "float32");
    const TensorType row({1, 4}, "float32");
    Var table_in("table", table), row_in("row", row);
    Var pos("pos", TensorType({}, "int64"));
    const Expr slots = Constant(
        I64Vec({0, 1, 2, 3}, {4, 1}));
    const Expr mask = Call(relay::Op::Get("equal"), {slots, pos});
    const Expr written = Call(relay::Op::Get("where"), {mask, row_in, table_in});
    return Function({table_in, row_in, pos}, written);
}

/*! \brief Part 2: synthetic in-graph loop with a dynamic-position write. */
Function LoopFunction() {
    const TensorType table({4, 4}, "float32");
    const TensorType boolean({}, "bool");
    const TensorType integer({}, "int64");
    Var run0("run0", boolean), run1("run1", boolean), run2("run2", boolean);
    Var table0("table0", table), pos0("pos0", integer), rows("rows", table);
    Var state("state");
    const Expr row_now = Call(relay::Op::Get("slice"), {rows},
                              relay::SliceAttrs::Create({0}, {1}, {0}, {1}));
    const Expr written = relay::pass_utils::InlineFunctionParameters(
        DecodeCellFunction(), {TupleGetItem(state, 3), row_now, TupleGetItem(state, 4)});
    const Expr tokens_next = Call(relay::Op::Get("argmax"), {written},
                                  relay::ArgMaxAttrs::Create(1, 0, 0));
    const Expr pos_next =
        Call(relay::Op::Get("add"), {TupleGetItem(state, 4), IntConst(1)});
    const Expr stop = Call(relay::Op::Get("equal"),
                           {pos_next, TupleGetItem(state, 4)});
    const Expr body = Tuple({TupleGetItem(state, 1), TupleGetItem(state, 2),
                             stop, written, pos_next, tokens_next});
    const Expr tokens0 = Call(relay::Op::Get("argmax"), {rows},
                              relay::ArgMaxAttrs::Create(1, 0, 0));
    const Expr initial = Tuple({run0, run1, run2, table0, pos0, tokens0});
    return Function({run0, run1, run2, table0, pos0, rows},
                    While(initial, state, TupleGetItem(state, 0), body, 3));
}

bool TestInGraphDecodeLoop() {
    const auto config = api::CompileConfig::Create(BuildTarget(Device::CPU()));
    const auto compiled = api::Compiler::Compile(LoopFunction(), config);
    CHECK(compiled.defined() && compiled.plan().defined(),
          "in-graph loop must compile");
    runtime::NDArray rows = runtime::NDArray::Empty({4, 4}, F32(), Device::CPU());
    {
        std::vector<float> values{1, 2, 3, 4, 4, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
        rows.CopyFromBytes(values.data(), values.size() * 4);
    }
    const runtime::RuntimeSession session(compiled.module(), compiled.plan());
    const Array<runtime::NDArray> outputs = session.Run(
        {ScalarBool(true), ScalarBool(true), ScalarBool(false),
         ZerosF32({4, 4}), I64Scalar(0), rows});
    CHECK(outputs.size() == 6, "loop must expose its carried outputs");
    const std::vector<int64_t> pos = ReadInts(outputs[4]);
    const std::vector<int64_t> tokens = ReadInts(outputs[5]);
    CHECK(pos.size() == 1 && pos[0] == 2, "position must advance per iteration");
    const std::vector<float> table = ReadFloats(outputs[3]);
    CHECK(table.size() == 16 && table[0] == 1.0f && table[1] == 2.0f &&
              table[2] == 3.0f && table[3] == 4.0f && table[4] == 1.0f,
          "row must be written at each advancing slot");
    CHECK(tokens.size() == 4 && tokens[0] == 3 && tokens[1] == 3,
          "argmax must select the max index per written row");
    CHECK(ReadBool(outputs[0]) == false, "loop must consume every scheduled trip");
    return true;
}

/*! \brief Part 3 builder: real capacity decode as the While body. */
Function RealLoopFunction(const frontend::ImportedONNXModel& decode,
                          int64_t capacity, int64_t kv_tables, int64_t trips,
                          const runtime::NDArray& slot_mask,
                          const runtime::NDArray& slot_write,
                          int64_t slots_len) {
    const TensorType i64v1({1}, "int64");
    const TensorType step_t({}, "int64");
    const TensorType table({1, capacity, 4, 96}, "float32");
    Var step0("step0", step_t);
    Var token("token", decode.function->params[0]->type_annotation);
    Var pos("pos", i64v1);
    std::vector<Var> past;
    for (int64_t i = 0; i < kv_tables; ++i) {
        past.push_back(Var("past_" + std::to_string(i), table));
    }
    Var state("state");
    const Expr step_expr = TupleGetItem(state, 0);
    const Expr token_expr = TupleGetItem(state, 1);
    const Expr pos_expr = TupleGetItem(state, 2);

    const Expr slots_m = Constant(slot_mask);       // [1, capacity+1] int64
    const Expr slots_w = Constant(slot_write);      // [1, capacity, 1, 1] int64
    const Expr one = Constant(F32Scalar(1.0f));
    const Expr zero = Constant(F32Scalar(0.0f));
    const Expr lt = Call(relay::Op::Get("less"), {slots_m, pos_expr});
    const Expr eq = Call(relay::Op::Get("equal"),
                         {slots_m, Constant(I64Vec({capacity}, {1}))});
    const Expr mask = Call(
        relay::Op::Get("where"),
        {lt, one, Call(relay::Op::Get("where"), {eq, one, zero})});
    (void)slots_len;

    Array<Expr> cell_args;
    cell_args.push_back(token_expr);
    cell_args.push_back(pos_expr);
    cell_args.push_back(mask);
    for (int64_t i = 0; i < kv_tables; ++i) {
        cell_args.push_back(TupleGetItem(state, 3 + i));
    }
    const Expr inlined = relay::pass_utils::InlineFunctionParameters(
        decode.function, cell_args);
    const Expr logits = TupleGetItem(inlined, 0);
    const Expr next_token = Call(relay::Op::Get("argmax"), {logits},
                                  relay::ArgMaxAttrs::Create(2, 0, 0));
    const Expr next_pos =
        Call(relay::Op::Get("add"), {pos_expr, Constant(I64Vec({1}, {1}))});
    const Expr next_step =
        Call(relay::Op::Get("add"), {step_expr, IntConst(1)});
    const Expr slot_match = Call(relay::Op::Get("equal"), {slots_w, pos_expr});

    Array<Expr> body_fields;
    body_fields.push_back(next_step);
    body_fields.push_back(next_token);
    body_fields.push_back(next_pos);
    for (int64_t i = 0; i < kv_tables; ++i) {
        const Expr present = TupleGetItem(inlined, 1 + i);
        const Expr fresh = Call(
            relay::Op::Get("slice"), {present},
            relay::SliceAttrs::Create({capacity}, {capacity + 1}, {1}, {1}));
        body_fields.push_back(Call(
            relay::Op::Get("where"),
            {slot_match, fresh, TupleGetItem(state, 3 + i)}));
    }
    const Expr body = Tuple(std::move(body_fields));

    Array<Expr> initial_fields;
    initial_fields.push_back(step0);
    initial_fields.push_back(token);
    initial_fields.push_back(pos);
    for (const Var& var : past) initial_fields.push_back(var);
    const Expr initial = Tuple(std::move(initial_fields));

    const Expr condition =
        Call(relay::Op::Get("less"), {step_expr, IntConst(trips)});
    Array<Var> params;
    params.push_back(step0);
    params.push_back(token);
    params.push_back(pos);
    for (const Var& var : past) params.push_back(var);
    return Function(params, While(initial, state, condition, body, trips));
}

/*! \brief Part 3: run the real decode in-loop and compare with the fixture. */
bool TestRealModelCapacityLoop() {
    const char* directory = std::getenv("KXC_MINIMIND_CONTROL_LOOP_DIR");
    if (!directory || !*directory) {
        directory = std::getenv("KXC_MINIMIND_DECODE_LOOP_DIR");
    }
    if (!directory || !*directory) {
        std::cout << "[SKIP] real capacity loop: set KXC_MINIMIND_CONTROL_LOOP_DIR "
                     "(or KXC_MINIMIND_DECODE_LOOP_DIR)\n";
        return true;
    }
    const std::filesystem::path root(directory);
    const auto decode = frontend::LoadONNXImportSpec(
        (root / "decode_capacity.json").string(),
        (root / "decode_capacity.params").string());
    CHECK(decode.input_names.size() == 19, "capacity decode must have 19 inputs");
    CHECK(decode.input_names[0] == "input_ids" &&
              decode.input_names[1] == "position" &&
              decode.input_names[2] == "attention_mask",
          "capacity decode input order drifted");

    int64_t capacity = 0, seed_extent = 0, trips = 0;
    {
        std::ifstream cap(root / "capacity.txt"), seed(root / "seed_extent.txt"),
            steps(root / "steps.txt");
        cap >> capacity;
        seed >> seed_extent;
        std::string line;
        while (std::getline(steps, line)) {
            if (!line.empty()) ++trips;
        }
    }
    CHECK(capacity > 0 && seed_extent > 0 && trips >= 3,
          "capacity decode fixture metadata must be valid");
    const int64_t kv_tables = static_cast<int64_t>(decode.input_names.size()) - 3;
    CHECK(kv_tables == 16, "capacity decode must carry 16 KV tables");

    // Seed past tables: valid prefix from the fixture, sentinel elsewhere.
    float sentinel = 0.0f;
    {
        std::ifstream in(root / "sentinel.txt");
        in >> sentinel;
    }
    Array<runtime::NDArray> initial_past;
    for (int64_t i = 0; i < kv_tables; ++i) {
        const std::string kind = (i % 2 == 0) ? "k" : "v";
        const std::string name = "seed_present_" + kind + "_" + std::to_string(i / 2) + ".bin";
        const auto bytes = ReadFile(root / name);
        const int64_t seed_bytes = seed_extent * 4 * 96 * 4;
        CHECK(static_cast<int64_t>(bytes.size()) == seed_bytes,
              "seed KV must match the seed extent (" + name + ")");
        runtime::NDArray table = Full({1, capacity, 4, 96}, sentinel);
        // Copy the valid seed prefix into a view of the first seed_extent rows.
        runtime::NDArray prefix =
            table.CreateView({1, seed_extent, 4, 96}, {}, 0);
        prefix.CopyFromBytes(bytes.data(), bytes.size());
        initial_past.push_back(table);
    }

    // First token from the fixture's host greedy sequence.
    std::vector<int64_t> step_tokens;
    std::vector<int64_t> step_extents;
    {
        std::ifstream steps(root / "steps.txt");
        int64_t step = 0, token = 0, extent = 0;
        while (steps >> step >> token >> extent) {
            step_tokens.push_back(token);
            step_extents.push_back(extent);
        }
    }
    CHECK(static_cast<int64_t>(step_tokens.size()) >= 3,
          "fixture must supply at least three decode steps");
    CHECK(step_extents[0] == seed_extent, "first step extent must equal the seed");

    const int64_t slots = capacity + 1;
    std::vector<int64_t> mask_values(static_cast<size_t>(slots));
    for (int64_t i = 0; i < slots; ++i) mask_values[static_cast<size_t>(i)] = i;
    const runtime::NDArray slot_mask = I64Vec(mask_values, {1, slots});
    std::vector<int64_t> write_values(static_cast<size_t>(capacity));
    for (int64_t i = 0; i < capacity; ++i) write_values[static_cast<size_t>(i)] = i;
    const runtime::NDArray slot_write = I64Vec(write_values, {1, capacity, 1, 1});

    const auto config = api::CompileConfig::Create(BuildTarget(Device::CPU()));
    const int64_t want_trips = std::min<int64_t>(trips, 4);
    double worst = 0.0;
    // Compile the loop for 1..want_trips trips and compare every step's full
    // 16-table state (including the untouched sentinel region) with the
    // fixture's ref_step<i>_state_*.bin, and the selected token with steps.txt.
    for (int64_t run_trips = 1; run_trips <= want_trips; ++run_trips) {
        const api::CompiledGraph compiled = api::Compiler::Compile(
            RealLoopFunction(decode, capacity, kv_tables, run_trips, slot_mask,
                             slot_write, slots),
            config);
        CHECK(compiled.defined() &&
                  compiled.plan().structured_schedule().has_value(),
              "real capacity loop must publish a structured plan");
        const runtime::RuntimeSession session(compiled.module(), compiled.plan());
        Array<runtime::NDArray> inputs;
        inputs.push_back(I64Scalar(0));               // step0 (int64 scalar)
        inputs.push_back(I64Shape11(step_tokens[0])); // token [1,1]
        inputs.push_back(I64Shape1(seed_extent));     // position [1]
        for (const auto& table : initial_past) inputs.push_back(table);
        const Array<runtime::NDArray> outputs = session.Run(inputs);
        CHECK(outputs.size() == static_cast<size_t>(3 + kv_tables),
              "loop must expose step, token, position and every KV table");

        const std::vector<int64_t> pos = ReadInts(outputs[2]);
        CHECK(pos.size() == 1 && pos[0] == seed_extent + run_trips,
              "position must advance once per trip");
        // The token produced at the end of trip i is the greedy token consumed
        // at step i+1; the fixture records one token per step, so the last
        // trip's output has no recorded successor.
        const std::vector<int64_t> token = ReadInts(outputs[1]);
        CHECK(token.size() == 1, "loop must expose one selected token");
        if (static_cast<size_t>(run_trips) < step_tokens.size()) {
            CHECK(token[0] == step_tokens[static_cast<size_t>(run_trips)],
                  "in-graph argmax token must match the fixture greedy sequence");
        }

        for (int64_t i = 0; i < kv_tables; ++i) {
            const std::vector<float> reference = ReadFloatsFile(
                root / ("ref_step" + std::to_string(run_trips - 1) +
                        "_state_" + std::to_string(i) + ".bin"));
            CHECK(reference.size() == static_cast<size_t>(capacity) * 4 * 96,
                  "reference capacity state size drifted");
            const std::vector<float> actual = ReadFloats(outputs[3 + i]);
            CHECK(actual.size() == reference.size(), "state size mismatch");
            for (size_t j = 0; j < actual.size(); ++j) {
                CHECK(std::isfinite(actual[j]), "state must be finite");
                worst = std::max(worst,
                                 std::abs(double(actual[j]) - reference[j]));
            }
        }
    }
    CHECK(worst < 1e-4, "in-graph loop state must match the reference");
    std::cout << "[PASS] real capacity loop: " << want_trips
              << " steps, 16 KV tables each, worst state diff " << worst << "\n";
    return true;
}

}  // namespace

int main() {
    struct Test { const char* name; bool (*run)(); };
    const std::vector<Test> tests = {
        {"real_graph_inlining", TestRealGraphInlining},
        {"in_graph_decode_loop", TestInGraphDecodeLoop},
        {"real_model_capacity_loop", TestRealModelCapacityLoop},
    };
    int failures = 0;
    for (const Test& test : tests) {
        if (!test.run()) ++failures;
        else std::cout << "[PASS] " << test.name << "\n";
    }
    if (!std::getenv("KXC_MINIMIND_CONTROL_LOOP_DIR") &&
        !std::getenv("KXC_MINIMIND_DECODE_LOOP_DIR")) {
        std::cout << "[SKIP] real capacity loop fixture unset\n";
    }
    std::cout << (failures == 0 ? "[PASS] m10_control_loop_llvm\n"
                                : "[FAIL] m10_control_loop_llvm\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
