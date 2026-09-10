// Source loading -> symbolic proof -> LLVM -> RuntimeSession, without model
// downloads. Actual MiniMind heads are exercised by bounded_projection_llvm_test.
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../src/compiler/internal/primitive_cache.h"
#include "kxc/compiler/compiler.h"
#include "kxc/compiler/restricted_symbolic_shape.h"
#include "kxc/frontend/onnx_importer.h"
#include "kxc/relay/op.h"
#include "kxc/runtime/session.h"

namespace {
using namespace kxc;
namespace ci = api::internal;
namespace restricted = api::experimental::restricted_symbolic_shape::v1;
using Adapter = restricted::RestrictedSymbolicShapeAdapter;
using runtime::NDArray;

void Check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
template <class Action>
void Rejects(Action action, const std::string& diagnostic) {
    try { action(); } catch (const std::exception& error) {
        Check(std::string(error.what()).find(diagnostic) != std::string::npos,
              "unexpected rejection: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("expected rejection: " + diagnostic);
}
bool SameStats(const ci::PrimitiveCacheStats& a, const ci::PrimitiveCacheStats& b) {
    return a.hits == b.hits && a.misses == b.misses && a.entries == b.entries &&
           a.accounted_bytes == b.accounted_bytes && a.evictions == b.evictions &&
           a.in_flight == b.in_flight && a.merged_waiters == b.merged_waiters &&
           a.failures == b.failures && a.rejections == b.rejections && a.active_pins == b.active_pins;
}
NDArray Input(int64_t batch, int64_t sequence) {
    std::vector<float> data(static_cast<size_t>(batch * sequence * 8));
    for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<float>(i) * 0.125f;
    NDArray result = NDArray::Empty({batch, sequence, 8}, runtime::DataTypeFromString("float32"), Device::CPU());
    if (!data.empty()) result.CopyFromBytes(data.data(), data.size() * sizeof(float));
    return result;
}
void Test() {
    ci::ClearPrimitiveCacheForTesting();
    const auto root = std::filesystem::current_path() / "out" / "onnx_shape_source";
    std::filesystem::create_directories(root);
    const auto json_path = root / "source.json", params_path = root / "source.params";
    // Scalar indices deliberately use two integer widths and negative values.
    const int32_t batch_index = -3;
    const int64_t sequence_index = -2, suffix[] = {2, 4};
    {
        std::ofstream params(params_path, std::ios::binary);
        params.write(reinterpret_cast<const char*>(&batch_index), sizeof(batch_index));
        params.write(reinterpret_cast<const char*>(&sequence_index), sizeof(sequence_index));
        params.write(reinterpret_cast<const char*>(suffix), sizeof(suffix));
        Check(params.good(), "cannot write source parameters");
    }
    const std::string source_json = R"({
      "format":"kxc.onnx_shape_source.v1",
      "function":{
        "inputs":[{"name":"x","shape":[1,4,8],"dtype":"float32"}],
        "outputs":[{"name":"heads","shape":[1,4,2,4],"dtype":"float32"},
                   {"name":"target","shape":[4],"dtype":"int64"}],
        "nodes":[
          {"name":"shape","op_name":"shape_of","inputs":["x"],"outputs":["shape"],"attrs":{}},
          {"name":"batch","op_name":"gather","inputs":["shape","ib"],"outputs":["b"],"attrs":{"axis":0}},
          {"name":"seq","op_name":"gather","inputs":["shape","is"],"outputs":["s"],"attrs":{"axis":-1}},
          {"name":"bv","op_name":"unsqueeze","inputs":["b"],"outputs":["bv"],"attrs":{"axes":[-1]}},
          {"name":"sv","op_name":"unsqueeze","inputs":["s"],"outputs":["sv"],"attrs":{"axes":[0]}},
          {"name":"bs","op_name":"concatenate","inputs":["bv","sv"],"outputs":["bs"],"attrs":{"axis":0}},
          {"name":"target","op_name":"concatenate","inputs":["bs","suffix"],"outputs":["target"],"attrs":{"axis":0}},
          {"name":"heads","op_name":"reshape_dynamic","inputs":["x","target"],"outputs":["heads"],"attrs":{}}
        ]},
      "params":[
        {"name":"ib","shape":[],"dtype":"int32","offset":0,"nbytes":4},
        {"name":"is","shape":[],"dtype":"int64","offset":4,"nbytes":8},
        {"name":"suffix","shape":[2],"dtype":"int64","offset":12,"nbytes":16}],
      "param_order":["ib","is","suffix"]})";
    { std::ofstream out(json_path); out << source_json; Check(out.good(), "cannot write source spec"); }
    const auto source = frontend::LoadONNXShapeSource(json_path.string(), params_path.string());
    Check(!source.function.checked_type().defined(), "a shape source must remain explicitly untyped");
    Rejects([&] { (void)frontend::LoadONNXImportSpec(json_path.string(), params_path.string()); },
            "Unsupported ONNX import spec format");
    profiling::ProfileOptions options;
    options.enabled = true;
    options.ir_capture_mode = profiling::IRCaptureMode::kDisabled;
    options.record_pass_ir = false;
    options.bundle_dir = (root / "profile").string();
    const auto context = profiling::ProfileContext::Create(options);
    const profiling::ActivationScope activation(context, "onnx_shape_source");
    const auto config = api::CompileConfig::Create(BuildTarget(Device::CPU()), 2, options);
    const std::vector<restricted::InputAxisSymbol> axes{{0,0,"B",1,3,1},{0,1,"S",1,8,1}};
    const auto before = ci::GetPrimitiveCacheStats();
    Rejects([&] { (void)api::Compiler::Compile(source.function, config); }, "reshape_dynamic");
    const auto wrong_outputs = [&](std::vector<TensorType> outputs) {
        Rejects([&] { (void)Adapter::Prepare(source.function, config, axes, outputs); },
                "declared imported output");
    };
    wrong_outputs({TensorType({1,4,4,2}, "float32"), TensorType({4}, "int64")});
    wrong_outputs({TensorType({1,4,2,4}, "int64"), TensorType({4}, "int64")});
    wrong_outputs({TensorType({1,4,2,4}, "float32")});
    const Var x = source.function->params[0];
    const Expr shape = Call(relay::Op::Get("shape_of"), {x});
    const Expr scalar = Call(relay::Op::Get("gather"), {shape, Constant(source.params.at("ib"))},
                             relay::GatherAttrs::Create(0));
    Rejects([&] { (void)Adapter::Prepare(Function({x}, scalar), config, axes); }, "scalar shape results");
    const Expr bad_unsqueeze = Call(relay::Op::Get("unsqueeze"), {scalar}, relay::UnsqueezeAttrs::Create({1}));
    Rejects([&] { (void)Adapter::Prepare(Function({x}, bad_unsqueeze), config, axes); }, "axis out of range");
    const Expr bad_concat = Call(relay::Op::Get("concatenate"), {scalar, scalar}, relay::ConcatenateAttrs::Create(0));
    Rejects([&] { (void)Adapter::Prepare(Function({x}, bad_concat), config, axes); }, "shape vector");
    const Var arbitrary("arbitrary_shape", TensorType({4}, "int64"));
    Rejects([&] { (void)Adapter::Prepare(Function({x, arbitrary},
        Call(relay::Op::Get("reshape_dynamic"), {x, arbitrary})), config, axes); }, "shape");
    Check(SameStats(before, ci::GetPrimitiveCacheStats()), "source rejections touched the cache");

    const auto prepared = Adapter::Prepare(source.function, config, axes, source.declared_output_types);
    const auto compiled = api::Compiler::CompileBounded(Adapter::MintBoundedCompileRequest(prepared));
    Check(compiled.plan().calls().size() == 2 && compiled.plan().constant_value_ids().empty(),
          "shape controls must fold to ShapeExpr and ReshapeDynamic");
    const runtime::RuntimeSession session(compiled.module(), compiled.plan());
    const auto stats = ci::GetPrimitiveCacheStats();
    const auto abi = api::BuildPlanAbiFingerprint(compiled);
    for (const auto& dims : std::vector<std::pair<int64_t,int64_t>>{{1,1},{1,4},{2,3},{3,8}}) {
        const auto [batch, sequence] = dims;
        const auto input = Input(batch, sequence);
        const auto output = session.Run({input}, {{"stage","onnx_heads"},{"batch",std::to_string(batch)},
            {"sequence",std::to_string(sequence)},{"plan_abi",abi.digest()}});
        Check(output.size() == 2, "data and shape results must be retained");
        const auto shape_out = output[0].shape();
        Check(shape_out.size() == 4 && shape_out[0] == batch && shape_out[1] == sequence &&
                  shape_out[2] == 2 && shape_out[3] == 4, "heads shape mismatch");
        std::vector<float> actual(input.NBytes() / sizeof(float)), expected(actual.size());
        output[0].CopyToBytes(actual.data(), input.NBytes());
        input.CopyToBytes(expected.data(), input.NBytes());
        Check(actual == expected && output[0]->dl_tensor.data != input->dl_tensor.data,
              "reshape must make a correct fresh copy");
        int64_t target[4] = {};
        output[1].CopyToBytes(target, sizeof(target));
        Check(target[0] == batch && target[1] == sequence && target[2] == 2 && target[3] == 4,
              "materialized shape froze the representative dimensions");
        Check(SameStats(stats, ci::GetPrimitiveCacheStats()), "source Run performed compile/cache work");
    }
    const auto counts = [&] {
        context->Flush();
        std::ifstream in(std::filesystem::path(context->bundle_dir()) / "events.jsonl");
        Check(in.good(), "missing source profile bundle");
        std::pair<size_t,size_t> result{0,0};
        std::string line;
        while (std::getline(in,line)) {
            if (line.find("\"event_type\":\"kernel_submit\"") != std::string::npos) ++result.first;
            if (line.find("\"event_type\":\"alloc\"") != std::string::npos) ++result.second;
        }
        return result;
    };
    const auto successful = counts();
    Check(successful.first == 8, "four source shapes must launch eight kernels");
    for (const auto& dims : std::vector<std::pair<int64_t,int64_t>>{{0,1},{4,1},{1,0},{1,9}}) {
        Rejects([&] { (void)session.Run({Input(dims.first,dims.second)}); }, "");
        Check(counts() == successful, "rejected source shape allocated or launched");
        Check(SameStats(stats, ci::GetPrimitiveCacheStats()), "rejected source shape compiled");
    }
    std::cout << "ONNX shape source: four shapes, 8 LLVM launches, exact data/shape results; "
                 "4 zero-launch runtime rejections and 9 preparation/format rejections\n";
}
}  // namespace

int main() {
    try { Test(); return 0; }
    catch (const std::exception& error) { std::cerr << "[FAIL] " << error.what() << '\n'; return 1; }
}
