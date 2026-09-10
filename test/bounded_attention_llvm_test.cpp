// Compile one ordinary Relay attention graph, then execute changing B/Q/T
// through the production bounded compiler and RuntimeSession without cache work.
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../src/compiler/internal/dynamic_shape_contract.h"
#include "../src/compiler/internal/lowered_graph.h"
#include "../src/compiler/internal/primitive_cache.h"
#include "../src/compiler/internal/te_to_tir.h"
#include "kxc/compiler/compiler.h"
#include "kxc/compiler/restricted_symbolic_shape.h"
#include "kxc/relay/op.h"
#include "kxc/runtime/session.h"
#include "kxc/te/topi/broadcast.h"
#include "kxc/te/topi/reduction.h"
#include "kxc/tir/printer/print_ir.h"

namespace {
using namespace kxc;
namespace ci = api::internal;
namespace ri = relay::internal;
namespace restricted = api::experimental::restricted_symbolic_shape::v1;
using Adapter = restricted::RestrictedSymbolicShapeAdapter;
using runtime::NDArray;

void Check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Action>
void Rejects(Action action, const std::string& message) {
    try { action(); } catch (const std::exception&) { return; }
    throw std::runtime_error(message);
}

api::CompileConfig Config() {
    return api::CompileConfig::Create(BuildTarget(Device::CPU()), 2);
}

constexpr int64_t kHeads = 4, kDim = 96;

Function Attention() {
    // Q and T deliberately have equal representative sizes, but are distinct
    // expressions. Size-based overlay lookup used to conflate these inputs.
    const TensorType data({2, kHeads, 4, kDim}, "float32");
    const Var q("query", data), k("key", data), v("value", data);
    const Var mask("mask", TensorType({2, kHeads, 4, 4}, "float32"));
    const Expr kt = Call(relay::Op::Get("transpose"), {k},
                        relay::TransposeAttrs::Create({0, 1, 3, 2}));
    const Expr scores = Call(relay::Op::Get("matmul"), {q, kt});
    const Expr masked = Call(relay::Op::Get("add"), {scores, mask});
    const Expr weights = Call(relay::Op::Get("softmax"), {masked},
                             relay::SoftmaxAttrs::Create(-1));
    return Function({q, k, v, mask}, Call(relay::Op::Get("matmul"), {weights, v}));
}

std::vector<restricted::InputAxisSymbol> Axes(int64_t maximum = 8) {
    return {{0, 0, "B", 1, 3, 1}, {0, 2, "Q", 1, maximum, 1},
            {1, 0, "B", 1, 3, 1}, {1, 2, "T", 1, maximum, 1},
            {2, 0, "B", 1, 3, 1}, {2, 2, "T", 1, maximum, 1},
            {3, 0, "B", 1, 3, 1}, {3, 2, "Q", 1, maximum, 1},
            {3, 3, "T", 1, maximum, 1}};
}

NDArray Tensor(const std::vector<int64_t>& shape, int seed) {
    size_t count = 1;
    for (int64_t extent : shape) count *= static_cast<size_t>(extent);
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) {
        values[i] = static_cast<float>(std::sin(static_cast<double>(i + seed) * 0.17));
    }
    NDArray result = NDArray::Empty(shape, runtime::DataTypeFromString("float32"), Device::CPU());
    if (count) result.CopyFromBytes(values.data(), count * sizeof(float));
    return result;
}

std::vector<float> Read(const NDArray& array) {
    size_t count = 1;
    for (int64_t extent : array.shape()) count *= static_cast<size_t>(extent);
    std::vector<float> result(count);
    if (count) array.CopyToBytes(result.data(), count * sizeof(float));
    return result;
}

Array<NDArray> Inputs(int64_t batch, int64_t queries, int64_t total) {
    NDArray q = Tensor({batch, kHeads, queries, kDim}, 1);
    auto scaled = Read(q);
    for (float& value : scaled) value /= std::sqrt(static_cast<float>(kDim));
    q.CopyFromBytes(scaled.data(), scaled.size() * sizeof(float));
    NDArray mask = Tensor({batch, kHeads, queries, total}, 0);
    std::vector<float> mask_values(static_cast<size_t>(batch * kHeads * queries * total));
    for (int64_t row = 0; row < batch * kHeads * queries; ++row) {
        const int64_t visible = std::min(total, std::max<int64_t>(1, total - queries + row % queries + 1));
        for (int64_t t = 0; t < total; ++t) {
            mask_values[static_cast<size_t>(row * total + t)] = t < visible ? 0.0f : -10000.0f;
        }
    }
    if (!mask_values.empty()) mask.CopyFromBytes(mask_values.data(), mask_values.size() * sizeof(float));
    return {q, Tensor({batch, kHeads, total, kDim}, 5),
            Tensor({batch, kHeads, total, kDim}, 13), mask};
}

std::vector<double> Reference(const Array<NDArray>& inputs) {
    const int64_t batch = inputs[0].shape()[0];
    const int64_t queries = inputs[0].shape()[2], total = inputs[1].shape()[2];
    const auto q = Read(inputs[0]), k = Read(inputs[1]), v = Read(inputs[2]), mask = Read(inputs[3]);
    std::vector<double> output(q.size(), 0.0), scores(static_cast<size_t>(total));
    for (int64_t bh = 0; bh < batch * kHeads; ++bh) {
        for (int64_t row = 0; row < queries; ++row) {
            for (int64_t t = 0; t < total; ++t) {
                double dot = mask[(bh * queries + row) * total + t];
                for (int64_t d = 0; d < kDim; ++d) {
                    dot += static_cast<double>(q[(bh * queries + row) * kDim + d]) * k[(bh * total + t) * kDim + d];
                }
                scores[t] = dot;
            }
            const double maximum = *std::max_element(scores.begin(), scores.end());
            double denominator = 0;
            for (double& score : scores) denominator += (score = std::exp(score - maximum));
            for (int64_t d = 0; d < kDim; ++d) {
                for (int64_t t = 0; t < total; ++t) {
                    output[(bh * queries + row) * kDim + d] += scores[t] / denominator * v[(bh * total + t) * kDim + d];
                }
            }
        }
    }
    return output;
}

bool SameStats(const ci::PrimitiveCacheStats& a, const ci::PrimitiveCacheStats& b) {
    return a.hits == b.hits && a.misses == b.misses && a.entries == b.entries &&
           a.accounted_bytes == b.accounted_bytes && a.evictions == b.evictions &&
           a.in_flight == b.in_flight && a.merged_waiters == b.merged_waiters &&
           a.failures == b.failures && a.rejections == b.rejections && a.active_pins == b.active_pins;
}

void TestAttention() {
    ci::ClearPrimitiveCacheForTesting();
    profiling::ProfileOptions options;
    options.enabled = true;
    options.ir_capture_mode = profiling::IRCaptureMode::kDisabled;
    options.record_pass_ir = false;
    options.bundle_dir = (std::filesystem::current_path() / "out" / "bounded_attention_profile").string();
    // Preparation and compilation share one bundle owner. Independent contexts
    // targeting the same directory would overwrite one another on destruction.
    const auto context = profiling::ProfileContext::Create(options);
    const profiling::ActivationScope activation(context, "attention_validation");
    const auto config = api::CompileConfig::Create(BuildTarget(Device::CPU()), 2, options);
    const auto prepared = Adapter::Prepare(Attention(), config, Axes());
    const auto request = Adapter::MintBoundedCompileRequest(prepared);
    const auto compiled = api::Compiler::CompileBounded(request);
    Check(compiled.plan().calls().size() == 5 && compiled.module().entry_count() == 5,
          "attention must compile its five registered Relay calls");
    const auto stats = ci::GetPrimitiveCacheStats();
    const auto plan_abi = api::BuildPlanAbiFingerprint(compiled);
    const runtime::RuntimeSession session(compiled.module(), compiled.plan());
    const auto counts = [&] {
        context->Flush();
        std::ifstream events(std::filesystem::path(context->bundle_dir()) / "events.jsonl");
        Check(events.good(), "attention profile events are missing");
        std::pair<size_t, size_t> result{0, 0};
        std::string line;
        while (std::getline(events, line)) {
            if (line.find("\"event_type\":\"kernel_submit\"") != std::string::npos) ++result.first;
            if (line.find("\"event_type\":\"alloc\"") != std::string::npos) ++result.second;
        }
        return result;
    };
    double worst = 0;
    for (const auto& dimensions : std::vector<std::vector<int64_t>>{{1, 1, 3}, {2, 3, 5}, {3, 6, 6}, {1, 7, 8}}) {
        const auto inputs = Inputs(dimensions[0], dimensions[1], dimensions[2]);
        const auto reference = Reference(inputs);
        const auto result = session.Run(inputs, {{"stage", "bounded_attention"},
            {"plan_abi", plan_abi.digest()},
            {"batch", std::to_string(dimensions[0])}, {"query_length", std::to_string(dimensions[1])},
            {"total_length", std::to_string(dimensions[2])}});
        Check(result.size() == 1, "attention must have one output");
        const auto actual_shape = result[0].shape(), expected_shape = inputs[0].shape();
        Check(actual_shape.size() == expected_shape.size() &&
              std::equal(actual_shape.begin(), actual_shape.end(), expected_shape.begin()),
              "attention output shape changed incorrectly");
        const auto actual = Read(result[0]);
        for (size_t i = 0; i < actual.size(); ++i) {
            worst = std::max(worst, std::fabs(actual[i] - reference[i]));
            Check(std::isfinite(actual[i]) && std::fabs(actual[i] - reference[i]) < 1e-5,
                  "bounded attention differs from independent stable-softmax reference");
        }
        Check(SameStats(stats, ci::GetPrimitiveCacheStats()), "Run performed primitive cache or compilation work");
    }
    Check(counts().first == 20, "each valid run must submit five LLVM kernels");
    const auto rejects = [&](Array<NDArray> inputs) {
        const auto before = counts();
        Rejects([&] { (void)session.Run(inputs); }, "invalid attention input was accepted");
        Check(counts() == before,
              "invalid graph input reached a kernel or runtime allocation");
    };
    auto bad = Inputs(1, 3, 5);
    bad[2] = Tensor({1, kHeads, 4, kDim}, 0); rejects(bad); // downstream V/T mismatch
    bad = Inputs(1, 3, 5);
    bad[3] = Tensor({1, kHeads, 2, 5}, 0); rejects(bad); // downstream mask/Q mismatch
    rejects(Inputs(4, 3, 5));
    rejects(Inputs(1, 9, 5));
    bad = Inputs(1, 3, 5);
    bad[1] = Tensor({1, kHeads, 0, kDim}, 0); rejects(bad);
    bad = Inputs(1, 3, 5);
    bad[0] = Tensor({1, kHeads, 3, kDim - 1}, 0); rejects(bad);
    bad = Inputs(1, 3, 5);
    bad[0] = Tensor({1, 3, kDim}, 0); rejects(bad);
    bad = Inputs(1, 3, 5);
    bad[0] = NDArray::Zeros({1, kHeads, 3, kDim}, runtime::DataTypeFromString("int64"), Device::CPU());
    rejects(bad);
    rejects({Tensor({1, kHeads, 3, kDim}, 0)});
    Check(SameStats(stats, ci::GetPrimitiveCacheStats()), "failed calls performed cache work");
    std::cout << "attention: 4 shapes, 20 LLVM launches, worst_abs=" << worst << ", 9 invalid inputs zero-launch\n";

    const auto preparation = ci::PrepareBoundedCompile(request);
    const auto& partition = preparation.partitioned_graph();
    const auto softmax = ci::LowerPrimitiveUnit(partition.value_graph.values, partition.units[3],
                                               request.target(), preparation.unit_shape_contracts()[3]);
    Check(ri::GetTEScheduleContract(softmax->prim_func).find("bounded-dynamic-serial-v3") != std::string::npos,
          "new reduction/allocation policy is missing from schedule identity");
    const auto changed = Adapter::MintBoundedCompileRequest(Adapter::Prepare(Attention(), Config(), Axes(12)));
    const auto other = ci::PrepareBoundedCompile(changed);
    Check(preparation.unit_shape_contracts()[3].canonical_bytes() != other.unit_shape_contracts()[3].canonical_bytes(),
          "changing sequence/scratch bounds must change the unit contract identity");
}

void TestNonTrailingSoftmax() {
    const Var x("x", TensorType({2, 4, 3}, "float32"));
    const auto prepared = Adapter::Prepare(
        Function({x}, Call(relay::Op::Get("softmax"), {x}, relay::SoftmaxAttrs::Create(1))),
        Config(), {{0, 0, "B", 1, 3, 1}, {0, 1, "S", 1, 8, 1}});
    const auto compiled = api::Compiler::CompileBounded(Adapter::MintBoundedCompileRequest(prepared));
    const runtime::RuntimeSession session(compiled.module(), compiled.plan());
    const auto stats = ci::GetPrimitiveCacheStats();
    double worst = 0;
    for (const auto& dims : std::vector<std::vector<int64_t>>{{1, 1, 3}, {2, 3, 3}, {3, 8, 3}}) {
        NDArray input = Tensor(dims, 0);
        auto values = Read(input);
        for (int64_t b = 0; b < dims[0]; ++b) {
            for (int64_t s = 0; s < dims[1]; ++s) {
                for (int64_t d = 0; d < 3; ++d) {
                    values[(b * dims[1] + s) * 3 + d] = (s % 2 ? 10000.0f : -10000.0f) + b * 0.7f + d * 0.33f + s * 0.25f;
                }
            }
        }
        input.CopyFromBytes(values.data(), values.size() * sizeof(float));
        const auto actual = Read(session.Run({input})[0]);
        for (int64_t b = 0; b < dims[0]; ++b) {
            for (int64_t d = 0; d < 3; ++d) {
                double maximum = -std::numeric_limits<double>::infinity(), denominator = 0;
                for (int64_t s = 0; s < dims[1]; ++s) maximum = std::max(maximum, static_cast<double>(values[(b * dims[1] + s) * 3 + d]));
                for (int64_t s = 0; s < dims[1]; ++s) denominator += std::exp(values[(b * dims[1] + s) * 3 + d] - maximum);
                for (int64_t s = 0; s < dims[1]; ++s) {
                    const size_t i = static_cast<size_t>((b * dims[1] + s) * 3 + d);
                    const double expected = std::exp(values[i] - maximum) / denominator;
                    worst = std::max(worst, std::fabs(actual[i] - expected));
                    Check(std::isfinite(actual[i]) && std::fabs(actual[i] - expected) < 1e-6, "non-trailing dynamic softmax is unstable or indexed incorrectly");
                }
            }
        }
        Check(SameStats(stats, ci::GetPrimitiveCacheStats()), "non-trailing softmax recompiled");
    }
    std::cout << "softmax axis=1: 3 shapes, extreme logits +/-10000, worst_abs=" << worst << '\n';
}

void TestRejectionsAndScratchBounds() {
    const auto before = ci::GetPrimitiveCacheStats();
    const Var x("x", TensorType({2, 4}, "float32"));
    Rejects([&] { (void)Adapter::Prepare(Function({x}, Call(relay::Op::Get("softmax"), {x})), Config(),
                                       {{0, 1, "T", 0, 8, 1}}); }, "empty softmax reduction domain admitted");
    Rejects([&] { (void)Adapter::Prepare(Function({x}, Call(relay::Op::Get("transpose"), {x},
                                       relay::TransposeAttrs::Create({0, 0}))), Config(),
                                       {{0, 0, "B", 1, 4, 1}}); }, "duplicate transpose axis admitted");
    const Var a("a", TensorType({2, 3, 4}, "float32"));
    const Var b("b", TensorType({2, 4, 5}, "float32"));
    Rejects([&] { (void)Adapter::Prepare(Function({a, b}, Call(relay::Op::Get("matmul"), {a, b})), Config(),
                                       {{0, 0, "B", 1, 4, 1}, {1, 0, "C", 1, 4, 1}}); }, "unproved matmul batch broadcast admitted");
    const Var left("left", TensorType({2, 4}, "float32"));
    const Var right("right", TensorType({4, 3}, "float32"));
    Rejects([&] { (void)Adapter::Prepare(Function({left, right}, Call(relay::Op::Get("matmul"), {left, right})), Config(),
                                       {{0, 1, "K1", 1, 8, 1}, {1, 0, "K2", 1, 8, 1}}); },
            "equal sample sizes must not prove independent reduction symbols equal");
    Check(SameStats(before, ci::GetPrimitiveCacheStats()), "invalid preparation compiled kernels");

    const tir::Var extent("extent", tir::DataType::UInt(64));
    const auto n = ri::LoadRuntimeExtent(extent);
    const auto input = te::placeholder({n, n, n}, tir::DataType::Float(32), "input");
    const auto sum = te::topi::sum(input, {2}, true);
    const auto output = te::topi::add(input, sum);
    const auto schedule = ri::BuildBoundedDynamicTESchedule({output}, Config()->target, {extent});
    const auto lower = [&](const std::vector<int64_t>& bounds) {
        return ri::LowerTensorGraphToTIR({input}, {}, {output}, schedule, Config()->target,
                                       ri::PrimFuncIdentity{String("scratch")}, {extent}, {}, bounds);
    };
    Rejects([&] { (void)lower({}); }, "dynamic scratch without bounds admitted");
    Rejects([&] { (void)lower({std::numeric_limits<int32_t>::max()}); }, "scratch byte overflow admitted");
    Check(lower({8}).defined(), "bounded dynamic reduction scratch did not lower");
}
}  // namespace

int main() {
    try {
        TestAttention();
        TestNonTrailingSoftmax();
        TestRejectionsAndScratchBounds();
        std::cout << "All bounded attention tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
