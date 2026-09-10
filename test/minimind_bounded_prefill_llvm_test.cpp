// Full token-to-logits bounded prefill, with download-free embedding/FFN checks.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../src/compiler/internal/primitive_cache.h"
#include "kxc/compiler/compiler.h"
#include "kxc/compiler/restricted_symbolic_shape.h"
#include "kxc/frontend/onnx_importer.h"
#include "kxc/relay/op.h"
#include "kxc/relay/transforms/infer_type.h"
#include "kxc/runtime/session.h"
#include "kxc/runtime/device_api.h"

namespace {
using namespace kxc;
namespace ci = api::internal;
namespace restricted = api::experimental::restricted_symbolic_shape::v1;
using Adapter = restricted::RestrictedSymbolicShapeAdapter;
using runtime::NDArray;
bool cuda = false;
Device TestDevice() { return cuda ? Device::CUDA() : Device::CPU(); }
const char* Backend() { return cuda ? "CUDA" : "LLVM"; }
const std::vector<std::pair<int64_t, int64_t>> kCases{{1,1},{1,4},{2,3},{3,8}};
const std::vector<restricted::InputAxisSymbol> kAxes{{0,0,"B",1,3,1},{0,1,"S",1,8,1}};

void Check(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}
template<class Action> void Rejects(Action action, const std::string& diagnostic) {
    try { action(); } catch (const std::exception& error) {
        Check(std::string(error.what()).find(diagnostic) != std::string::npos,
              "wrong rejection: " + std::string(error.what()));
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
bool SameShape(const Array<int64_t>& a, const Array<int64_t>& b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}
Expr Op(const std::string& name, Array<Expr> args, relay::Attrs attrs = {}) {
    return Call(relay::Op::Get(name), std::move(args), std::move(attrs));
}
NDArray Empty(Array<int64_t> shape, const std::string& dtype = "float32") {
    return NDArray::Empty(shape, runtime::DataTypeFromString(dtype), TestDevice());
}
template<class T> std::vector<T> Read(const NDArray& array) {
    Check(array.NBytes() % sizeof(T) == 0, "invalid tensor byte length");
    std::vector<T> values(array.NBytes() / sizeof(T));
    if (!values.empty()) array.CopyToBytes(values.data(), array.NBytes());
    return values;
}
template<class T> std::vector<T> ReadFile(const std::filesystem::path& file, size_t count) {
    std::ifstream in(file, std::ios::binary);
    Check(in.good(), "missing fixture: " + file.string());
    std::vector<T> values(count);
    in.read(reinterpret_cast<char*>(values.data()), count * sizeof(T));
    Check(in.gcount() == static_cast<std::streamsize>(count * sizeof(T)) &&
          in.peek() == std::char_traits<char>::eof(), "fixture byte length mismatch: " + file.string());
    return values;
}
Expr Integer(int64_t value) {
    auto data = NDArray::Empty({1}, runtime::DataTypeFromString("int64"), Device::CPU());
    data.CopyFromBytes(&value, sizeof(value));
    return Constant(data);
}
NDArray Weights(Array<int64_t> shape, int seed) {
    auto data = NDArray::Empty(shape, runtime::DataTypeFromString("float32"), Device::CPU());
    std::vector<float> values(data.NBytes() / 4);
    for (size_t i = 0; i < values.size(); ++i) values[i] = std::sin(double(i + seed) * .31) * .5;
    data.CopyFromBytes(values.data(), data.NBytes());
    return data;
}
Expr Extent(Expr anchor, int64_t axis = 1) {
    return Op("gather", {Op("shape_of", {anchor}), Integer(axis)}, relay::GatherAttrs::Create(0));
}
Expr Prefix(Expr table, Expr anchor, int64_t start = 0, int64_t step = 1, int64_t axis = 0) {
    return Op("slice", {table, Integer(start), Extent(anchor), Integer(axis), Integer(step)});
}
std::pair<size_t, size_t> Counts(const std::shared_ptr<profiling::ProfileContext>& context) {
    context->Flush();
    std::ifstream in(std::filesystem::path(context->bundle_dir()) / "events.jsonl");
    Check(in.good(), "missing bounded prefill profile");
    std::pair<size_t, size_t> counts{};
    std::string line;
    while (std::getline(in, line)) {
        if (line.find("\"event_type\":\"kernel_submit\"") != std::string::npos) ++counts.first;
        if (line.find("\"event_type\":\"alloc\"") != std::string::npos) ++counts.second;
    }
    return counts;
}

template<class Index> void TestEmbeddingFFN(const api::CompileConfig& config,
    const std::shared_ptr<profiling::ProfileContext>& context) {
    const std::string dtype = sizeof(Index) == 8 ? "int64" : "int32";
    const Var ids("ids", TensorType({1,4}, dtype));
    const auto table = Weights({7,4}, 1), positions = Weights({8,4}, 7);
    const auto gate_weight = Weights({4,6}, 11), up_weight = Weights({4,6}, 17), down_weight = Weights({6,4}, 23);
    auto attrs = relay::GatherAttrs::Create(0);
    const auto embedding = Op("gather", {Constant(table), ids}, attrs);
    const auto position = Prefix(Constant(positions), ids);
    const auto x = Op("add", {embedding, position});
    const auto gate = Op("matmul", {x, Constant(gate_weight)});
    const auto silu = Op("mul", {gate, Op("sigmoid", {gate})});
    const auto up = Op("matmul", {x, Constant(up_weight)});
    const auto down = Op("matmul", {Op("mul", {silu, up}), Constant(down_weight)});
    const auto prepared = Adapter::Prepare(Function({ids}, Tuple({embedding, position, Op("add", {x, down})})), config, kAxes);
    const_cast<relay::GatherAttrsNode*>(attrs.operator->())->axis = 1;
    const auto compiled = api::Compiler::CompileBounded(Adapter::MintBoundedCompileRequest(prepared));
    Check(compiled.plan().calls().size() == 10, "embedding/position/FFN plan must have ten calls");
    const runtime::RuntimeSession session(compiled.module(), compiled.plan());
    const auto stats = ci::GetPrimitiveCacheStats();
    const auto e = Read<float>(table), p = Read<float>(positions), g = Read<float>(gate_weight),
        u = Read<float>(up_weight), d = Read<float>(down_weight);
    const std::vector<Index> pattern{0,6,-1,-7,7,-8,std::numeric_limits<Index>::max(),std::numeric_limits<Index>::min()};
    double worst = 0;
    for (const auto& [b,s] : kCases) {
        auto input = Empty({b,s}, dtype);
        std::vector<Index> tokens(b*s);
        for (size_t i = 0; i < tokens.size(); ++i) tokens[i] = pattern[i % pattern.size()];
        input.CopyFromBytes(tokens.data(), input.NBytes());
        const auto outputs = session.Run({input}, {{"stage","prefill_embedding_ffn"},{"batch",std::to_string(b)},
            {"sequence",std::to_string(s)},{"indices_dtype",dtype},{"plan_abi",api::BuildPlanAbiFingerprint(compiled).digest()}});
        Check(outputs.size() == 3 && SameShape(outputs[0].shape(), {b,s,4}) &&
            SameShape(outputs[1].shape(), {s,4}) && SameShape(outputs[2].shape(), {b,s,4}), "embedding/FFN shape mismatch");
        const auto actual_embedding = Read<float>(outputs[0]), actual_position = Read<float>(outputs[1]), actual = Read<float>(outputs[2]);
        Check(std::equal(actual_position.begin(), actual_position.end(), p.begin()), "position prefix changed table values");
        for (int64_t row = 0; row < b*s; ++row) {
            int64_t index = tokens[row];
            const bool valid = index >= -7 && index < 7;
            if (valid && index < 0) index += 7;
            std::vector<double> value(4), hidden(6);
            for (size_t col = 0; col < 4; ++col) {
                const float expected = valid ? e[index*4+col] : 0.f;
                Check(actual_embedding[row*4+col] == expected, "gather index guard or snapshot failed");
                value[col] = double(expected) + p[(row%s)*4+col];
            }
            for (size_t col = 0; col < 6; ++col) {
                double gate_value = 0, up_value = 0;
                for (size_t k = 0; k < 4; ++k) {
                    gate_value += value[k] * g[k*6+col];
                    up_value += value[k] * u[k*6+col];
                }
                hidden[col] = gate_value / (1 + std::exp(-gate_value)) * up_value;
            }
            for (size_t col = 0; col < 4; ++col) {
                double expected = value[col];
                for (size_t k = 0; k < 6; ++k) expected += hidden[k] * d[k*4+col];
                Check(std::isfinite(actual[row*4+col]), "nonfinite FFN output");
                worst = std::max(worst, std::abs(double(actual[row*4+col]) - expected));
            }
        }
        Check(SameStats(stats, ci::GetPrimitiveCacheStats()), "embedding/FFN Run touched cache");
    }
    Check(worst < 2e-6, "embedding/FFN differs from independent double reference");
    const auto counts = Counts(context);
    for (const auto& bad : Array<NDArray>{Empty({1,9},dtype), Empty({4,4},dtype),
        Empty({1,0},dtype), Empty({1,4,1},dtype)}) {
        Rejects([&] { (void)session.Run({bad}); }, "RuntimeSession");
        Check(Counts(context) == counts && SameStats(stats, ci::GetPrimitiveCacheStats()),
              "invalid embedding/prefix input allocated, launched, or touched cache");
    }
    std::cout << "embedding/position/SwiGLU/residual " << dtype << ": four B/S cases, 40 " << Backend() << " calls, max_abs_error="
        << worst << "; four preflight rejections\n";
}

void TestProofs(const api::CompileConfig& config) {
    const Var ids("ids", TensorType({1,4}, "int64"));
    const auto table = Constant(Weights({8,4}, 0));
    const auto before = ci::GetPrimitiveCacheStats();
    const auto prepare = [&](Expr body) { return Adapter::Prepare(Function({ids}, body), config, kAxes); };
    Rejects([&] { (void)Adapter::Prepare(Function({ids}, Prefix(table, ids)), config,
        {{0,0,"B",1,3,1},{0,1,"S",1,9,1}}); }, "upper bound exceeds table capacity");
    for (int64_t start : {-1,1}) Rejects([&] { (void)prepare(Prefix(table, ids, start)); }, "start 0");
    for (int64_t step : {-1,2}) Rejects([&] { (void)prepare(Prefix(table, ids, 0, step)); }, "step 1");
    Rejects([&] { (void)prepare(Prefix(table, ids, 0, 1, 2)); }, "axis out of range");
    Rejects([&] { (void)prepare(Op("slice", {table, ids}, relay::SliceAttrs::Create({}, {}, {}, {}, 0, 1))); }, "arity mismatch");
    Rejects([&] { (void)prepare(Op("slice", {table, Integer(0), Extent(ids), Integer(0), Integer(1)},
        relay::SliceAttrs::Create({0},{4},{0},{1}))); }, "without attrs");
    Rejects([&] { (void)prepare(Op("slice", {table}, relay::SliceAttrs::Create({}, {}, {}, {}, 0, 1))); }, "prepared shape anchor");
    Rejects([&] { (void)prepare(Op("slice", {table, Integer(0), ids, Integer(0), Integer(1)})); }, "proved int64");
    const Var dynamic_table("table", TensorType({8,4}, "float32"));
    auto table_axes = kAxes;
    table_axes.push_back({1,0,"P",1,8,1});
    Rejects([&] { (void)Adapter::Prepare(Function({ids,dynamic_table}, Prefix(dynamic_table, ids)), config, table_axes); }, "static table");
    Rejects([&] { (void)Adapter::Prepare(Function({ids,dynamic_table}, Op("gather", {dynamic_table,ids}, relay::GatherAttrs::Create(0))),
        config, table_axes); }, "static table");
    const Var floats("floats", TensorType({1,4}, "float32"));
    Rejects([&] { (void)Adapter::Prepare(Function({floats}, Op("gather", {table,floats}, relay::GatherAttrs::Create(0))),
        config, kAxes); }, "indices dtype");
    Rejects([&] { (void)prepare(Op("sigmoid", {ids})); }, "float32");
    for (const std::string name : {"sigmoid", "sqrt", "neg"}) {
        Rejects([&] { (void)Adapter::Prepare(Function({floats}, Op(name, {floats}, relay::GatherAttrs::Create(0))),
            config, kAxes); }, "without caller attrs");
    }
    Check(SameStats(before, ci::GetPrimitiveCacheStats()), "rejected embedding/prefix sources touched cache");
    std::cout << "embedding/prefix/sigmoid proof: 17 rejections before cache\n";
}

void TestSigmoid(const api::CompileConfig& config) {
    const Var x("x", TensorType({4}, "float32"));
    const auto compiled = api::Compiler::CompileBounded(Adapter::MintBoundedCompileRequest(
        Adapter::Prepare(Function({x}, Op("sigmoid", {x})), config, {{0,0,"S",1,8,1}})));
    const runtime::RuntimeSession session(compiled.module(), compiled.plan());
    const std::vector<float> values{-100.f,-20.f,-1.f,-0.f,0.f,1.f,20.f,100.f};
    double worst = 0;
    for (int64_t s : {1,4,8}) {
        auto input = Empty({s}); input.CopyFromBytes(values.data(), input.NBytes());
        const auto output = session.Run({input}, {{"stage","prefill_sigmoid"},{"sequence",std::to_string(s)}});
        const auto actual = Read<float>(output[0]);
        for (int64_t i = 0; i < s; ++i) {
            Check(std::isfinite(actual[i]) && actual[i] >= 0 && actual[i] <= 1, "invalid sigmoid output");
            worst = std::max(worst, std::abs(double(actual[i]) - 1/(1+std::exp(-double(values[i])))));
        }
    }
    Check(worst < 1e-7, "sigmoid differs from independent double reference");
    std::cout << "sigmoid [-100,100]: three " << Backend() << " shapes, max_abs_error=" << worst << '\n';
}

void TestActual(const api::CompileConfig& config, const std::shared_ptr<profiling::ProfileContext>& context) {
    const char* directory = std::getenv("KXC_MINIMIND_BOUNDED_PREFILL_DIR");
    if (!directory || !*directory) { std::cout << "[SKIP] full MiniMind: set KXC_MINIMIND_BOUNDED_PREFILL_DIR\n"; return; }
    const std::filesystem::path root(directory);
    const auto source = frontend::LoadONNXShapeSource((root/"prefill.json").string(), (root/"prefill.params").string());
    Check(source.input_names.size() == 1 && source.input_names[0] == "input_ids" && source.declared_output_types.size() == 17,
          "full prefill boundary drifted");
    std::cout << "full MiniMind source loaded\n" << std::flush;
    const auto prepared = Adapter::Prepare(source.function, config, kAxes, source.declared_output_types);
    std::cout << "full MiniMind shape preparation complete\n" << std::flush;
    const auto request = Adapter::MintBoundedCompileRequest(prepared);
    std::cout << "full MiniMind bounded request minted\n" << std::flush;
    const auto compiled = api::Compiler::CompileBounded(request);
    const size_t calls = compiled.plan().calls().size();
    Check(calls == 742, "full prefill must retain 742 calls after control proof");
    if (cuda) {
        for (const auto& call : compiled.plan().calls()) {
            const auto launch = compiled.module().launch_metadata(call->symbol);
            Check(launch->backend == codegen::CodeGenBackend::kCUDA && launch->device == TestDevice(),
                  "full bounded prefill changed backend or device");
            size_t extents = 0;
            for (const auto& arg : compiled.module().signature(call->symbol).arguments()) {
                if (arg->role != codegen::KernelArgRole::kRuntimeExtent) continue;
                ++extents;
                Check(arg->device == TestDevice() && arg->dtype.code == kDLUInt && arg->dtype.bits == 64 &&
                      arg.shape().size() == 1 && arg.shape()[0] == 1, "invalid bounded CUDA extent ABI");
            }
            Check(extents > 0, "full model silently specialized a bounded primitive");
        }
    }
    std::cout << "full MiniMind prepared and compiled: " << calls << " calls per Run\n" << std::flush;
    std::vector<runtime::RuntimeSession> sessions;
    std::vector<DeviceStream> streams;
    const size_t repeats = cuda ? 2 : 1;
    for (size_t repeat = 0; repeat < repeats; ++repeat) {
        sessions.emplace_back(compiled.module(), compiled.plan());
        if (cuda) streams.push_back(DeviceStream::Create(TestDevice()));
    }
    const auto stats = ci::GetPrimitiveCacheStats();
    std::ifstream receipt(root/"export_receipt.txt"), cases(root/"cases.txt");
    std::string sha; receipt >> sha;
    Check(sha.size() == 64 && sha.find_first_not_of("0123456789abcdef") == std::string::npos, "invalid prefill receipt");
    const auto abi = api::BuildPlanAbiFingerprint(compiled).digest();
    int64_t b = 0, s = 0;
    size_t case_index = 0;
    size_t reference_values = 0, causal_values = 0;
    double worst = 0;
    const auto run = [&](const std::vector<int64_t>& tokens, const std::string& stage) {
        std::vector<Array<NDArray>> outputs;
        std::vector<runtime::RunAsyncResult> pending;
        for (size_t repeat = 0; repeat < repeats; ++repeat) {
            auto input = Empty({b,s}, "int64");
            input.CopyFromBytes(tokens.data(), input.NBytes());
            const runtime::ExecutionMetadata metadata{{"model","minimind_bounded"},{"stage",stage},
                {"batch",std::to_string(b)},{"sequence",std::to_string(s)},{"sequence_length",std::to_string(s)},
                {"shape_case",std::to_string(case_index)},{"repeat",std::to_string(repeat)},
                {"layers","8"},{"export_receipt",sha},{"plan_abi",abi}};
            if (cuda) pending.push_back(sessions[repeat].RunAsync({input}, streams[repeat], metadata));
            else outputs.push_back(sessions[repeat].Run({input}, metadata));
            // GPU inputs lose their caller handle before either stream is waited.
        }
        for (auto it = pending.rbegin(); it != pending.rend(); ++it) it->completion.Wait();
        for (const auto& result : pending) outputs.push_back(result.outputs);
        return outputs;
    };
    Array<NDArray> retained;
    while (cases >> b >> s) {
        Check(case_index < kCases.size() && kCases[case_index] == std::make_pair(b,s), "unexpected full prefill case");
        auto tokens = ReadFile<int64_t>(root/("input_"+std::to_string(case_index)+".bin"), b*s);
        const auto repeated = run(tokens, "minimind_bounded_prefill");
        if (case_index == 0) retained = repeated[0];
        for (const auto& outputs : repeated) {
            Check(outputs.size() == 17, "full prefill must return logits and all KV outputs");
            for (size_t output = 0; output < outputs.size(); ++output) {
                Check(outputs[output].device() == TestDevice() &&
                      outputs[output].dtype().code == kDLFloat && outputs[output].dtype().bits == 32 &&
                      outputs[output].dtype().lanes == 1 &&
                      SameShape(outputs[output].shape(), output == 0 ? Array<int64_t>{b,s,6400} : Array<int64_t>{b,s,4,96}),
                      "full prefill output device, dtype or shape mismatch");
                const auto actual = Read<float>(outputs[output]);
                const auto reference = ReadFile<float>(root/("ref_"+std::to_string(case_index)+"_"+std::to_string(output)+".bin"), actual.size());
                for (size_t i = 0; i < actual.size(); ++i) {
                    Check(std::isfinite(actual[i]) && std::isfinite(reference[i]), "nonfinite full prefill output");
                    worst = std::max(worst, std::abs(double(actual[i])-reference[i]));
                }
                reference_values += actual.size();
            }
        }
        if (s == 8) {
            for (size_t i = 0; i < tokens.size(); ++i) if (i%s >= 4) tokens[i] = (tokens[i]+13)%6400;
            const auto future = run(tokens, "minimind_bounded_prefill_future");
            for (size_t repeat = 0; repeat < repeats; ++repeat) {
                Check(future[repeat].size() == 17, "future run lost an output");
                bool suffix_changed = false;
                for (size_t output = 0; output < repeated[repeat].size(); ++output) {
                    const auto a = Read<float>(repeated[repeat][output]), z = Read<float>(future[repeat][output]);
                    Check(a.size() == z.size(), "future output changed shape");
                    const size_t width = output == 0 ? 6400 : 384;
                    for (size_t i = 0; i < a.size(); ++i) {
                        Check(std::isfinite(z[i]), "future output is nonfinite");
                        if ((i/width)%s < 4) Check(a[i] == z[i], "future tokens changed the causal prefix");
                        else if (output == 0) suffix_changed = suffix_changed || std::abs(a[i]-z[i]) > 1e-3;
                    }
                    causal_values += a.size();
                }
                Check(suffix_changed, "future token perturbation did not change suffix logits");
            }
        }
        Check(SameStats(stats, ci::GetPrimitiveCacheStats()), "full prefill Run touched primitive cache");
        ++case_index;
    }
    Check(case_index == 4 && cases.eof() && worst < 5e-5, "full prefill differs from dynamic ONNX reference");
    const auto counts = Counts(context);
    const auto reject = [&](const Array<NDArray>& inputs, const DeviceStream& stream) {
        Rejects([&] {
            if (cuda) (void)sessions[0].RunAsync(inputs, stream, {{"model","minimind_bounded"},{"stage","reject"}});
            else (void)sessions[0].Run(inputs);
        }, "RuntimeSession");
        Check(Counts(context) == counts && SameStats(stats, ci::GetPrimitiveCacheStats()),
              "invalid full prefill input allocated, launched, or touched cache");
    };
    for (const auto& bad : Array<NDArray>{Empty({1,9},"int64"), Empty({4,4},"int64"), Empty({1,0},"int64"),
        Empty({0,4},"int64"), Empty({1,4,1},"int64"), Empty({1,4},"int32")}) {
        reject({bad}, cuda ? streams[0] : DeviceStream::Default(Device::CPU()));
    }
    if (cuda) {
        reject({}, streams[0]);
        reject({NDArray::Empty({1,4}, runtime::DataTypeFromString("int64"), Device::CPU())}, streams[0]);
        reject({Empty({1,4},"int64")}, DeviceStream::Default(Device::CPU()));
    }
    sessions.clear();
    for (size_t output = 0; output < retained.size(); ++output) {
        const auto actual = Read<float>(retained[output]);
        const auto reference = ReadFile<float>(root/("ref_0_"+std::to_string(output)+".bin"), actual.size());
        for (size_t i = 0; i < actual.size(); ++i)
            Check(std::isfinite(actual[i]) && std::abs(double(actual[i])-reference[i]) < 5e-5,
                  "retained output changed after later shapes or session destruction");
    }
    std::cout << "actual full eight-layer MiniMind prefill: four B/S references and future-token perturbation, "
        << calls*5*repeats << ' ' << Backend() << " calls, all 17 outputs, max_abs_error=" << worst
        << (cuda ? "; nine preflight rejections\n" : "; six preflight rejections\n");
    if (cuda) std::cout << "[PASS] bounded_minimind_prefill: runs=10 calls=742 reference_values=" << reference_values
                       << " causal_values=" << causal_values << " streams=2\n";
}
}  // namespace

int main(int argc, char** argv) {
    try {
        cuda = argc == 2 && std::string(argv[1]) == "--cuda";
        Check(argc == 1 || cuda, "usage: minimind_bounded_prefill_test [--cuda]");
        if (cuda && !CollectDeviceAttributes(TestDevice()).exists) {
            std::cout << "[SKIP] no CUDA device\n"; return 77;
        }
        const char* fixture = std::getenv("KXC_MINIMIND_BOUNDED_PREFILL_DIR");
        if (cuda && (!fixture || !*fixture)) {
            std::cout << "[SKIP] set KXC_MINIMIND_BOUNDED_PREFILL_DIR\n"; return 77;
        }
        ci::ClearPrimitiveCacheForTesting();
        profiling::ProfileOptions options;
        options.enabled = true; options.ir_capture_mode = profiling::IRCaptureMode::kDisabled; options.record_pass_ir = false;
        options.enable_cupti = cuda;
        options.bundle_dir = (std::filesystem::current_path()/"out"/"minimind_bounded_prefill_profile").string();
        options = profiling::ApplyEnvironmentOverrides(options);
        const std::filesystem::path bundle_root(options.bundle_dir);
        for (size_t stage = 0; stage < (cuda ? 2U : 1U); ++stage) {
            if (cuda) options.bundle_dir = (bundle_root/(stage == 0 ? "prefill" : "auxiliary")).string();
            const auto context = profiling::ProfileContext::Create(options);
            const profiling::ActivationScope activation(context, "minimind_bounded_prefill");
            const auto config = api::CompileConfig::Create(BuildTarget(TestDevice()), cuda ? 3 : 2, options);
            if (cuda && stage == 0) TestActual(config, context);
            else {
                TestProofs(config); TestEmbeddingFFN<int64_t>(config, context); TestEmbeddingFFN<int32_t>(config, context); TestSigmoid(config);
                if (!cuda) TestActual(config, context);
            }
        }
        if (cuda) std::cout << "[PASS] full_minimind_bounded_cuda_prefill_and_all_kv_no_runtime_compile\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << "[FAIL] " << error.what() << '\n'; return 1; }
}
