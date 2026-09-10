// One bounded artifact per graph, actual compact storage, no runtime compiler.
#include "../src/compiler/internal/primitive_cache.h"
#include "kxc/compiler/compiler.h"
#include "kxc/compiler/restricted_symbolic_shape.h"
#include "kxc/profiling/profiling.h"
#include "kxc/relay/op.h"
#include "kxc/runtime/device_api.h"
#include "kxc/runtime/session.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace kxc;
using runtime::NDArray;
namespace ci = api::internal;
namespace restricted = api::experimental::restricted_symbolic_shape::v1;
using Adapter = restricted::RestrictedSymbolicShapeAdapter;

void Check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
template <typename Action>
void Reject(Action action) {
    try { action(); } catch (const std::exception&) { return; }
    throw std::runtime_error("invalid bounded invocation was accepted");
}
bool SameStats(const ci::PrimitiveCacheStats& a, const ci::PrimitiveCacheStats& b) {
    return a.hits == b.hits && a.misses == b.misses && a.entries == b.entries &&
        a.accounted_bytes == b.accounted_bytes && a.evictions == b.evictions &&
        a.in_flight == b.in_flight && a.merged_waiters == b.merged_waiters &&
        a.failures == b.failures && a.rejections == b.rejections && a.active_pins == b.active_pins;
}
std::vector<float> Values(size_t count, int seed) {
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) values[i] = (int((i * 17 + seed) % 29) - 14) * 0.125f;
    return values;
}
NDArray Tensor(const Array<int64_t>& shape, const std::vector<float>& values, Device device = Device::CUDA()) {
    auto result = NDArray::Empty(shape, runtime::DataTypeFromString("float32"), device);
    Check(result.NBytes() == values.size() * sizeof(float), "input byte count mismatch");
    if (!values.empty()) result.CopyFromBytes(values.data(), result.NBytes());
    return result;
}
double Verify(const NDArray& result, const Array<int64_t>& shape, const std::vector<float>& expected,
              bool positive_zero = false) {
    const auto actual_shape = result.shape();
    Check(actual_shape.size() == shape.size() &&
          std::equal(actual_shape.begin(), actual_shape.end(), shape.begin()), "output is not compact actual shape");
    Check(result.device() == Device::CUDA() && result.NBytes() == expected.size() * sizeof(float),
          "output device or actual allocation byte count mismatch");
    std::vector<float> actual(expected.size());
    if (!actual.empty()) result.CopyToBytes(actual.data(), result.NBytes());
    double worst = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double error = std::abs(double(actual[i]) - expected[i]);
        Check(std::isfinite(actual[i]) && error <= 1e-6, "GPU output differs from independent host reference");
        if (positive_zero && expected[i] == 0.0f)
            Check(actual[i] == 0.0f && !std::signbit(actual[i]), "masked output must be exact positive zero");
        worst = std::max(worst, error);
    }
    return worst;
}
api::CompiledGraph Compile(const Function& function, std::vector<restricted::InputAxisSymbol> axes,
                           const profiling::ProfileOptions& options,
                           const std::shared_ptr<profiling::ProfileContext>& profile, size_t calls) {
    const profiling::ActivationScope activation(profile, "prepare");
    const auto config = api::CompileConfig::Create(BuildTarget(Device::CUDA()), 3, options);
    const auto prepared = Adapter::Prepare(function, config, std::move(axes));
    const auto graph = api::Compiler::CompileBounded(Adapter::MintBoundedCompileRequest(prepared));
    Check(graph.plan().mode() == runtime::ExecutablePlanMode::kDynamicFreshOutputV1 &&
          graph.plan().calls().size() == calls && graph.module().entry_count() == calls,
          "bounded compilation did not publish the expected complete graph");
    for (const auto& call : graph.plan().calls()) {
        const auto launch = graph.module().launch_metadata(call->symbol);
        Check(launch->backend == codegen::CodeGenBackend::kCUDA && launch->device == Device::CUDA(),
              "bounded graph changed backend or device");
        size_t scalars = 0;
        for (const auto& arg : graph.module().signature(call->symbol).arguments()) {
            if (arg->role == codegen::KernelArgRole::kRuntimeExtent) {
                ++scalars;
                Check(arg->device == Device::CUDA() && arg->dtype.code == kDLUInt && arg->dtype.bits == 64 &&
                      arg.shape().size() == 1 && arg.shape()[0] == 1,
                      "extent parameter lost the existing uint64[1] device ABI");
            }
        }
        Check(scalars > 0, "bounded test unexpectedly compiled a static exact kernel");
    }
    for (const auto& pin : graph.artifact_pins()) {
        const auto& key = pin.record().artifact_key.canonical_bytes();
        Check(key.find("cuda-nvrtc-driver-v8") != std::string::npos &&
              key.find("bounded-dynamic-serial-v3") != std::string::npos,
              "bounded CUDA artifact lost backend or schedule identity");
    }
    return graph;
}

void RunGraph(const std::filesystem::path& directory, bool matmul) {
    const std::string model = matmul ? "bounded_matmul" : "bounded_elementwise";
    profiling::ProfileOptions options;
    options.enabled = options.enable_cupti = true;
    options.ir_capture_mode = profiling::IRCaptureMode::kVerbose;
    options.bundle_dir = directory.string();
    const auto profile = profiling::ProfileContext::Create(options);
    ci::ClearPrimitiveCacheForTesting();
    const Var x("x", TensorType(matmul ? Array<int64_t>{3, 7} : Array<int64_t>{3, 4, 7}, "float32"));
    const Var y("y", TensorType(matmul ? Array<int64_t>{7, 5} : Array<int64_t>{1, 4, 1}, "float32"));
    const Expr product = matmul ? Expr(Call(relay::Op::Get("matmul"), {x, y})) :
        Expr(Call(relay::Op::Get("sqrt"), {Call(relay::Op::Get("nn_relu"), {Call(relay::Op::Get("add"), {x, y})})}));
    const Function function({x, y}, matmul ? product : Expr(Tuple({product, Call(relay::Op::Get("shape_of"), {x})})));
    const std::vector<restricted::InputAxisSymbol> axes = matmul
        ? std::vector<restricted::InputAxisSymbol>{{0, 0, "N", 0, 19, 1}, {0, 1, "K", 0, 23, 1},
                                                   {1, 0, "K", 0, 23, 1}, {1, 1, "M", 0, 17, 1}}
        : std::vector<restricted::InputAxisSymbol>{{0, 0, "B", 0, 19, 1}, {0, 1, "S", 0, 18, 2},
                                                   {1, 1, "S", 0, 18, 2}};
    auto graph = Compile(function, axes, options, profile, matmul ? 1 : 4);
    const auto after_compile = ci::GetPrimitiveCacheStats();
    const size_t call_count = matmul ? 1 : 4;
    Check(after_compile.entries == call_count && after_compile.misses == call_count,
          "test did not compile exactly one artifact per primitive");
    const std::vector<Array<int64_t>> cases = matmul
        ? std::vector<Array<int64_t>>{{3, 5, 7}, {19, 17, 23}, {17, 16, 5}, {1, 17, 23},
                                      {19, 1, 1}, {0, 17, 23}, {19, 0, 23}, {3, 5, 0}, {0, 0, 0}}
        : std::vector<Array<int64_t>>{{1, 2}, {3, 4}, {19, 18}, {17, 16}, {2, 6}, {0, 18}, {19, 0}, {0, 0}};
    const std::vector<DeviceStream> streams{DeviceStream::Create(Device::CUDA()), DeviceStream::Create(Device::CUDA())};
    std::vector<NDArray> retained;
    std::vector<float> retained_reference;
    Array<int64_t> retained_shape;
    std::set<const Object*> storage_ids;
    size_t checked = 0;
    double worst = 0;
    {
        const runtime::RuntimeSession first(graph.module(), graph.plan()), second(graph.module(), graph.plan());
        for (size_t shape_id = 0; shape_id < cases.size(); ++shape_id) {
            const auto& shape = cases[shape_id];
            std::vector<runtime::RunAsyncResult> pending;
            std::vector<std::vector<float>> references;
            const int64_t a = shape[0], b = shape[1], k = matmul ? shape[2] : 7;
            const Array<int64_t> x_shape = matmul ? Array<int64_t>{a, k} : Array<int64_t>{a, b, 7};
            const Array<int64_t> y_shape = matmul ? Array<int64_t>{k, b} : Array<int64_t>{1, b, 1};
            const Array<int64_t> output_shape = matmul ? Array<int64_t>{a, b} : x_shape;
            for (size_t repeat = 0; repeat < 2; ++repeat) {
                auto xv = Values(size_t(matmul ? a * k : a * b * 7), int(shape_id * 3 + repeat));
                auto yv = Values(size_t(matmul ? k * b : b), int(shape_id * 7 + repeat + 2));
                std::vector<float> expected(size_t(matmul ? a * b : a * b * 7), 0);
                for (size_t i = 0; i < expected.size(); ++i) {
                    if (matmul) {
                        double sum = 0;
                        for (int64_t r = 0; r < k; ++r) sum += double(xv[(i / b) * k + r]) * yv[r * b + i % b];
                        expected[i] = static_cast<float>(sum);
                    } else expected[i] = std::sqrt(std::max(0.0f, xv[i] + yv[(i / 7) % b]));
                }
                references.push_back(std::move(expected));
                if (shape_id == 0 && repeat == 0) {
                    retained_reference = references.back();
                    retained_shape = output_shape;
                }
                // Temporary input handles are gone before either completion is observed.
                pending.push_back((repeat ? second : first).RunAsync(
                    {Tensor(x_shape, xv), Tensor(y_shape, yv)}, streams[repeat],
                    {{"model", model}, {"stage", "bounded"}, {"repeat", std::to_string(repeat)},
                     {"sequence_length", std::to_string(b)}, {"shape_case", std::to_string(shape_id)}}));
            }
            for (size_t repeat = 0; repeat < pending.size(); ++repeat) {
                pending[repeat].completion.Wait();
                const auto& outputs = pending[repeat].outputs;
                Check(outputs.size() == (matmul ? 1 : 2), "output arity mismatch");
                worst = std::max(worst, Verify(outputs[0], output_shape, references[repeat]));
                checked += references[repeat].size();
                if (!matmul) {
                    std::vector<int64_t> actual(3);
                    Check(outputs[1].dtype().code == kDLInt && outputs[1].dtype().bits == 64 &&
                          outputs[1].NBytes() == 3 * sizeof(int64_t), "shape value ABI mismatch");
                    outputs[1].CopyToBytes(actual.data(), outputs[1].NBytes());
                    Check(actual == std::vector<int64_t>({a, b, 7}), "shape_of stored upper/representative dimensions");
                    checked += actual.size();
                }
                for (const auto& output : outputs) {
                    Check(storage_ids.insert(output.storage().get()).second, "fresh outputs reused a live allocation");
                    retained.push_back(output);
                }
            }
            Check(SameStats(after_compile, ci::GetPrimitiveCacheStats()), "runtime changed primitive cache statistics");
        }
        const auto bad = [&](const Array<int64_t>& xs, const Array<int64_t>& ys,
                             Device data_device = Device::CUDA(), DeviceStream stream = {}) {
            size_t xc = 1, yc = 1;
            for (int64_t dim : xs) xc *= static_cast<size_t>(dim);
            for (int64_t dim : ys) yc *= static_cast<size_t>(dim);
            const auto xt = Tensor(xs, Values(xc, 0), data_device), yt = Tensor(ys, Values(yc, 1));
            Reject([&] { first.RunAsync({xt, yt}, stream.defined() ? stream : streams[0],
                                       {{"model", model}, {"stage", "reject"}}); });
        };
        if (matmul) {
            bad({20, 7}, {7, 5}); bad({3, 24}, {24, 5}); bad({3, 7}, {7, 18});
            bad({3, 7}, {8, 5}); bad({3, 7, 1}, {7, 5});
        } else {
            bad({20, 4, 7}, {1, 4, 1}); bad({3, 20, 7}, {1, 20, 1});
            bad({3, 3, 7}, {1, 3, 1}); bad({3, 4, 7}, {1, 6, 1});
            bad({3, 4, 7, 1}, {1, 4, 1}); bad({3, 4, 8}, {1, 4, 1});
            bad({3, 4, 7}, {1, 4, 1}, Device::CPU());
            bad({3, 4, 7}, {1, 4, 1}, Device::CUDA(), DeviceStream::Default(Device::CPU()));
            Reject([&] { first.RunAsync({}, streams[0], {{"model", model}, {"stage", "reject"}}); });
        }
        Check(SameStats(after_compile, ci::GetPrimitiveCacheStats()), "rejected invocation reached compiler/cache");
    }
    // Drop sessions, the graph and cache pins before verifying retained data.
    graph = {};
    ci::ClearPrimitiveCacheForTesting();
    (void)Verify(retained[0], retained_shape, retained_reference);
    profile->Flush();
    std::cout << "[PASS] " << model << ": runs=" << cases.size() * 2 << " calls=" << call_count
              << " checked_values=" << checked << " max_abs_error=" << std::setprecision(12) << worst
              << " rejected=" << (matmul ? 5 : 9) << " no_runtime_compile=1\n";
}

void RejectOversizedScratch() {
    const auto config = api::CompileConfig::Create(BuildTarget(Device::CUDA()), 3);
    const auto rejected_by_proof = [&](const Function& function,
                                      std::vector<restricted::InputAxisSymbol> axes) {
        const auto prepared = Adapter::Prepare(function, config, std::move(axes));
        const auto request = Adapter::MintBoundedCompileRequest(prepared);
        try { (void)api::Compiler::CompileBounded(request); }
        catch (const std::exception& error) {
            Check(std::string(error.what()).find("BindCudaThreads") != std::string::npos,
                  "unsupported bounded graph failed outside the CUDA ownership proof");
            return;
        }
        throw std::runtime_error("oversized bounded scratch reached CUDA compilation");
    };
    const Var x("x", TensorType({3, 4}, "float32"));
    rejected_by_proof(Function({x}, Call(relay::Op::Get("softmax"), {x}, relay::SoftmaxAttrs::Create(-1))),
                      {{0, 0, "B", 1, 19, 1}, {0, 1, "S", 1, 20000, 1}});
    std::cout << "[PASS] bounded_cuda_rejects_oversized_scratch\n";
}

void RunReductions(const std::filesystem::path& directory) {
    profiling::ProfileOptions options;
    options.enabled = options.enable_cupti = true;
    options.ir_capture_mode = profiling::IRCaptureMode::kVerbose;
    options.bundle_dir = directory.string();
    const auto profile = profiling::ProfileContext::Create(options);
    ci::ClearPrimitiveCacheForTesting();
    const Var x("x", TensorType({3, 5, 7}, "float32")), mask("mask", TensorType({3, 5, 1}, "bool"));
    const auto op = [](const char* name, const Array<Expr>& args) { return Call(relay::Op::Get(name), args); };
    const Expr shifted = op("add", {x, Constant(Tensor({}, {1000.0f}, Device::CPU()))});
    const Expr columns = Call(relay::Op::Get("softmax"), {shifted}, relay::SoftmaxAttrs::Create(1));
    const Expr last = Call(relay::Op::Get("softmax"), {shifted}, relay::SoftmaxAttrs::Create(-1));
    const Expr masked = Call(relay::Op::Get("masked_softmax"), {shifted, mask}, relay::SoftmaxAttrs::Create(1));
    const Expr mean = Call(relay::Op::Get("reduce_mean"), {op("mul", {x, x})},
                          relay::ReduceMeanAttrs::Create({-1}, true));
    const Expr rms = op("divide", {x, op("sqrt", {op("add", {mean,
        Constant(Tensor({}, {0.0001f}, Device::CPU()))})})});
    auto graph = Compile(Function({x, mask}, Tuple({columns, last, masked, mean, rms})),
        {{0, 0, "B", 0, 19, 1}, {0, 1, "S", 1, 18, 1}, {1, 0, "B", 0, 19, 1}, {1, 1, "S", 1, 18, 1}},
        options, profile, 9);
    const auto frozen = ci::GetPrimitiveCacheStats();
    Check(frozen.entries == 9 && frozen.misses == 9, "reduction graph did not publish nine primitive artifacts");
    const std::vector<Array<int64_t>> cases{{1, 1}, {3, 5}, {19, 18}, {17, 16}, {2, 7}, {0, 18}, {19, 1}, {3, 5}};
    const std::vector<DeviceStream> streams{DeviceStream::Create(Device::CUDA()), DeviceStream::Create(Device::CUDA())};
    const auto mask_tensor = [](int64_t b, int64_t s, const std::vector<uint8_t>& values) {
        auto tensor = NDArray::Empty({b, s, 1}, runtime::DataTypeFromString("bool"), Device::CUDA());
        if (!values.empty()) tensor.CopyFromBytes(values.data(), values.size());
        return tensor;
    };
    std::vector<NDArray> retained;
    std::vector<float> retained_reference;
    std::set<const Object*> storage_ids;
    size_t checked = 0;
    double worst = 0;
    {
        const runtime::RuntimeSession first(graph.module(), graph.plan()), second(graph.module(), graph.plan());
        for (size_t shape_id = 0; shape_id < cases.size(); ++shape_id) {
            const int64_t b = cases[shape_id][0], s = cases[shape_id][1];
            std::vector<runtime::RunAsyncResult> pending;
            std::vector<std::vector<std::vector<float>>> references;
            for (int repeat = 0; repeat < 2; ++repeat) {
                auto values = Values(size_t(b * s * 7), int(shape_id * 3 + repeat));
                if (shape_id == 0 && repeat == 1) std::fill(values.begin(), values.end(), 0.0f);
                std::vector<uint8_t> masks(size_t(b * s));
                for (int64_t batch = 0; batch < b; ++batch) for (int64_t j = 0; j < s; ++j)
                    masks[batch * s + j] = batch % 3 != 0 && (j + repeat) % 3 != 0;
                std::vector<std::vector<float>> expected(5, std::vector<float>(values.size(), 0));
                expected[3].resize(size_t(b * s));
                for (int64_t batch = 0; batch < b; ++batch) {
                    for (int64_t k = 0; k < 7; ++k) {
                        double maximum = -INFINITY, masked_maximum = -INFINITY;
                        for (int64_t j = 0; j < s; ++j) {
                            const float value = values[(batch * s + j) * 7 + k] + 1000.0f;
                            maximum = std::max(maximum, double(value));
                            if (masks[batch * s + j]) masked_maximum = std::max(masked_maximum, double(value));
                        }
                        double total = 0, masked_total = 0;
                        for (int64_t j = 0; j < s; ++j) {
                            const float value = values[(batch * s + j) * 7 + k] + 1000.0f;
                            total += std::exp(double(value) - maximum);
                            if (masks[batch * s + j]) masked_total += std::exp(double(value) - masked_maximum);
                        }
                        for (int64_t j = 0; j < s; ++j) {
                            const size_t i = size_t((batch * s + j) * 7 + k);
                            const float value = values[i] + 1000.0f;
                            expected[0][i] = float(std::exp(double(value) - maximum) / total);
                            if (masks[batch * s + j]) expected[2][i] = float(std::exp(double(value) - masked_maximum) / masked_total);
                        }
                    }
                    for (int64_t j = 0; j < s; ++j) {
                        const size_t start = size_t((batch * s + j) * 7);
                        double maximum = -INFINITY, total = 0, squared = 0;
                        for (int64_t k = 0; k < 7; ++k) maximum = std::max(maximum, double(values[start + k] + 1000.0f));
                        for (int64_t k = 0; k < 7; ++k) {
                            total += std::exp(double(values[start + k] + 1000.0f) - maximum);
                            squared += double(values[start + k]) * values[start + k];
                        }
                        expected[3][batch * s + j] = float(squared / 7);
                        for (int64_t k = 0; k < 7; ++k) {
                            expected[1][start + k] = float(std::exp(double(values[start + k] + 1000.0f) - maximum) / total);
                            expected[4][start + k] = float(values[start + k] / std::sqrt(squared / 7 + double(0.0001f)));
                        }
                    }
                }
                references.push_back(std::move(expected));
                if (shape_id == 0 && repeat == 0) retained_reference = references.back()[0];
                if (shape_id == 7) {
                    // Only the masked output is checked for this case. Other
                    // tuple branches intentionally consume non-finite logits.
                    for (size_t i = 0; i < values.size(); ++i) if (!masks[i / 7])
                        values[i] = i % 3 == 0 ? std::numeric_limits<float>::quiet_NaN() :
                                    i % 3 == 1 ? std::numeric_limits<float>::infinity() :
                                                -std::numeric_limits<float>::infinity();
                }
                pending.push_back((repeat ? second : first).RunAsync(
                    {Tensor({b, s, 7}, values), mask_tensor(b, s, masks)}, streams[repeat],
                    {{"model", "bounded_reductions"}, {"stage", "bounded"}, {"shape_case", std::to_string(shape_id)},
                     {"repeat", std::to_string(repeat)}, {"sequence_length", std::to_string(s)},
                     {"masked_nonfinite", shape_id == 7 ? "1" : "0"}}));
            }
            for (size_t repeat = 0; repeat < pending.size(); ++repeat) {
                pending[repeat].completion.Wait();
                Check(pending[repeat].outputs.size() == 5, "reduction tuple arity changed");
                for (size_t i = 0; i < 5; ++i) {
                    const auto& output = pending[repeat].outputs[i];
                    if (shape_id != 7 || i == 2) {
                        worst = std::max(worst, Verify(output, {b, s, i == 3 ? 1 : 7}, references[repeat][i], i == 2));
                        checked += references[repeat][i].size();
                    }
                    Check(storage_ids.insert(output.storage().get()).second, "reduction outputs reused a live allocation");
                    retained.push_back(output);
                }
            }
            Check(SameStats(frozen, ci::GetPrimitiveCacheStats()), "reduction execution changed compiler cache");
        }
        for (const auto& shape : std::vector<Array<int64_t>>{{20, 5}, {3, 19}, {3, 0}}) {
            const int64_t b = shape[0], s = shape[1];
            const auto inputs = Array<NDArray>{Tensor({b, s, 7}, Values(size_t(b * s * 7), 0)),
                mask_tensor(b, s, std::vector<uint8_t>(size_t(b * s), 1))};
            Reject([&] { first.RunAsync(inputs, streams[0], {{"model", "bounded_reductions"}, {"stage", "reject"}}); });
        }
        Reject([&] { first.RunAsync({}, streams[0], {{"model", "bounded_reductions"}, {"stage", "reject"}}); });
        Check(SameStats(frozen, ci::GetPrimitiveCacheStats()), "invalid reductions reached compilation/cache");
    }
    graph = {};
    ci::ClearPrimitiveCacheForTesting();
    (void)Verify(retained[0], {1, 1, 7}, retained_reference);
    profile->Flush();
    std::cout << "[PASS] bounded_reductions: runs=16 calls=9 checked_values=" << checked
              << " max_abs_error=" << std::setprecision(12) << worst << " rejected=4 no_runtime_compile=1\n";
}

void RunAttention(const std::filesystem::path& directory) {
    profiling::ProfileOptions options;
    options.enabled = options.enable_cupti = true;
    options.ir_capture_mode = profiling::IRCaptureMode::kVerbose;
    options.bundle_dir = directory.string();
    const auto profile = profiling::ProfileContext::Create(options);
    ci::ClearPrimitiveCacheForTesting();
    const Var q("q", TensorType({2, 3, 3, 7}, "float32")), k("k", TensorType({2, 3, 5, 7}, "float32")),
              v("v", TensorType({2, 3, 5, 5}, "float32")), mask("mask", TensorType({2, 1, 3, 5}, "float32"));
    const Expr kt = Call(relay::Op::Get("transpose"), {k}, relay::TransposeAttrs::Create({0, 1, 3, 2}));
    const Expr scores = Call(relay::Op::Get("matmul"), {q, kt});
    const Expr weights = Call(relay::Op::Get("softmax"), {Call(relay::Op::Get("add"), {scores, mask})},
                              relay::SoftmaxAttrs::Create(-1));
    auto graph = Compile(Function({q, k, v, mask}, Call(relay::Op::Get("matmul"), {weights, v})),
        {{0, 0, "B", 0, 3, 1}, {0, 2, "Q", 0, 19, 1}, {1, 0, "B", 0, 3, 1}, {1, 2, "T", 1, 23, 1},
         {2, 0, "B", 0, 3, 1}, {2, 2, "T", 1, 23, 1}, {3, 0, "B", 0, 3, 1},
         {3, 2, "Q", 0, 19, 1}, {3, 3, "T", 1, 23, 1}}, options, profile, 5);
    const auto frozen = ci::GetPrimitiveCacheStats();
    Check(frozen.entries == 5 && frozen.misses == 5, "attention did not publish five bounded artifacts");
    const std::vector<Array<int64_t>> cases{{1, 1, 1}, {2, 3, 5}, {3, 19, 23}, {2, 17, 21}, {1, 0, 7}, {0, 19, 23}};
    const std::vector<DeviceStream> streams{DeviceStream::Create(Device::CUDA()), DeviceStream::Create(Device::CUDA())};
    std::vector<NDArray> retained;
    std::vector<float> retained_reference;
    size_t checked = 0;
    double worst = 0;
    {
        const runtime::RuntimeSession first(graph.module(), graph.plan()), second(graph.module(), graph.plan());
        for (size_t shape_id = 0; shape_id < cases.size(); ++shape_id) {
            const int64_t b = cases[shape_id][0], queries = cases[shape_id][1], total = cases[shape_id][2];
            std::vector<runtime::RunAsyncResult> pending;
            std::vector<std::vector<float>> references;
            for (int repeat = 0; repeat < 2; ++repeat) {
                auto qv = Values(size_t(b * 3 * queries * 7), int(shape_id + repeat));
                auto kv = Values(size_t(b * 3 * total * 7), int(shape_id * 3 + repeat));
                auto vv = Values(size_t(b * 3 * total * 5), int(shape_id * 5 + repeat));
                for (float& value : qv) value *= 0.125f;
                std::vector<float> mv(size_t(b * queries * total));
                for (int64_t batch = 0; batch < b; ++batch) for (int64_t query = 0; query < queries; ++query)
                    for (int64_t t = 0; t < total; ++t)
                        mv[(batch * queries + query) * total + t] = t < std::min(total, query + 1) ? 0.0f : -1000.0f;
                std::vector<float> expected(size_t(b * 3 * queries * 5), 0);
                for (int64_t batch = 0; batch < b; ++batch) for (int64_t head = 0; head < 3; ++head)
                    for (int64_t query = 0; query < queries; ++query) {
                        std::vector<double> row(size_t(total), 0);
                        for (int64_t t = 0; t < total; ++t) {
                            for (int64_t d = 0; d < 7; ++d)
                                row[t] += double(qv[((batch * 3 + head) * queries + query) * 7 + d]) *
                                          kv[((batch * 3 + head) * total + t) * 7 + d];
                            row[t] += mv[(batch * queries + query) * total + t];
                        }
                        const double maximum = *std::max_element(row.begin(), row.end());
                        double denominator = 0;
                        for (double& value : row) { value = std::exp(value - maximum); denominator += value; }
                        for (int64_t d = 0; d < 5; ++d) {
                            double value = 0;
                            for (int64_t t = 0; t < total; ++t)
                                value += row[t] / denominator * vv[((batch * 3 + head) * total + t) * 5 + d];
                            expected[((batch * 3 + head) * queries + query) * 5 + d] = float(value);
                        }
                    }
                if (shape_id == 0 && repeat == 0) retained_reference = expected;
                references.push_back(std::move(expected));
                pending.push_back((repeat ? second : first).RunAsync(
                    {Tensor({b, 3, queries, 7}, qv), Tensor({b, 3, total, 7}, kv),
                     Tensor({b, 3, total, 5}, vv), Tensor({b, 1, queries, total}, mv)}, streams[repeat],
                    {{"model", "bounded_attention"}, {"stage", "bounded"}, {"shape_case", std::to_string(shape_id)},
                     {"repeat", std::to_string(repeat)}, {"sequence_length", std::to_string(total)},
                     {"query_length", std::to_string(queries)}}));
            }
            for (size_t repeat = 0; repeat < pending.size(); ++repeat) {
                pending[repeat].completion.Wait();
                Check(pending[repeat].outputs.size() == 1, "attention output arity changed");
                worst = std::max(worst, Verify(pending[repeat].outputs[0], {b, 3, queries, 5}, references[repeat]));
                checked += references[repeat].size();
                retained.push_back(pending[repeat].outputs[0]);
            }
            Check(SameStats(frozen, ci::GetPrimitiveCacheStats()), "attention execution changed compiler cache");
        }
        for (const auto& shape : std::vector<Array<int64_t>>{{4, 3, 5}, {2, 20, 5}, {2, 3, 24}, {2, 3, 0}}) {
            const int64_t b = shape[0], queries = shape[1], total = shape[2];
            const Array<NDArray> inputs{
                Tensor({b, 3, queries, 7}, Values(size_t(b * 3 * queries * 7), 0)),
                Tensor({b, 3, total, 7}, Values(size_t(b * 3 * total * 7), 1)),
                Tensor({b, 3, total, 5}, Values(size_t(b * 3 * total * 5), 2)),
                Tensor({b, 1, queries, total}, std::vector<float>(size_t(b * queries * total), 0))};
            Reject([&] { first.RunAsync(inputs, streams[0], {{"model", "bounded_attention"}, {"stage", "reject"}}); });
        }
        Check(SameStats(frozen, ci::GetPrimitiveCacheStats()), "invalid attention reached compilation/cache");
    }
    graph = {};
    ci::ClearPrimitiveCacheForTesting();
    (void)Verify(retained[0], {1, 3, 1, 5}, retained_reference);
    profile->Flush();
    std::cout << "[PASS] bounded_attention: runs=12 calls=5 checked_values=" << checked
              << " max_abs_error=" << std::setprecision(12) << worst << " rejected=4 no_runtime_compile=1\n";
}

void TestUpperBoundsCacheIdentity() {
    ci::ClearPrimitiveCacheForTesting();
    const Var x("x", TensorType({4}, "float32"));
    const Function function({x}, Call(relay::Op::Get("nn_relu"), {x}));
    const auto config = api::CompileConfig::Create(BuildTarget(Device::CUDA()), 3);
    const auto compile = [&](int64_t upper) {
        return api::Compiler::CompileBounded(Adapter::MintBoundedCompileRequest(
            Adapter::Prepare(function, config, {{0, 0, "N", 1, upper, 1}})));
    };
    const auto small = compile(129), large = compile(513);
    Check(small.artifact_pins()[0].record().artifact_key != large.artifact_pins()[0].record().artifact_key,
          "different CUDA launch bounds reused the same artifact identity");
    const auto small_launch = small.module().launch_metadata(small.plan().calls()[0]->symbol);
    const auto large_launch = large.module().launch_metadata(large.plan().calls()[0]->symbol);
    Check(small_launch->grid.x == 1 && large_launch->grid.x == 3,
          "larger bound reused the smaller cached CUDA launch");
    const auto frozen = ci::GetPrimitiveCacheStats();
    Check(frozen.entries == 2 && frozen.misses == 2 && frozen.hits == 0,
          "bound-distinct CUDA compilation did not publish two artifacts");
    for (int mode = 0; mode < 2; ++mode) {
        const auto& graph = mode ? large : small;
        const int64_t size = mode ? 513 : 129;
        auto input = Values(size, 3), expected = input;
        for (auto& value : expected) value = std::max(0.0f, value);
        const auto result = runtime::RuntimeSession(graph.module(), graph.plan()).RunAsync(
            {Tensor({size}, input)}, DeviceStream::Create(Device::CUDA()));
        result.completion.Wait();
        (void)Verify(result.outputs[0], {size}, expected);
    }
    Check(SameStats(frozen, ci::GetPrimitiveCacheStats()), "identity regression runs changed compiler cache");
    std::cout << "[PASS] bounded_cuda_upper_bounds_cache_identity: 129+513 values, distinct 1/3-block artifacts\n";
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (!CollectDeviceAttributes(Device::CUDA()).exists) {
            std::cout << "[SKIP] no CUDA device\n";
            return 77;
        }
        Check(argc == 2, "provide an unused profiling evidence directory");
        RunGraph(std::filesystem::path(argv[1]) / "elementwise", false);
        RunGraph(std::filesystem::path(argv[1]) / "matmul", true);
        RunReductions(std::filesystem::path(argv[1]) / "reductions");
        RunAttention(std::filesystem::path(argv[1]) / "attention");
        RejectOversizedScratch();
        TestUpperBoundsCacheIdentity();
        std::cout << "[PASS] bounded_cuda_compact_shapes_and_runtime_extents\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] bounded_cuda: " << error.what() << '\n';
        return 1;
    }
}
