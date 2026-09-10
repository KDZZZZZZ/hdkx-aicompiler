// Shared producers, repeated logical arguments, and explicit tuple outputs
// must survive bounded admission, exact materialization, and LLVM execution.
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "../src/compiler/internal/primitive_cache.h"
#include "kxc/compiler/compiler.h"
#include "kxc/compiler/restricted_symbolic_shape.h"
#include "kxc/relay/op.h"
#include "kxc/runtime/session.h"

namespace {
using namespace kxc;
namespace ci = api::internal;
namespace restricted = api::experimental::restricted_symbolic_shape::v1;
namespace shape = api::experimental::shape_specialization::v1;
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

Function Graph() {
    const TensorType type({4, 3}, "float32");
    const Var x("x", type), y("y", type);
    const Expr shared = Call(relay::Op::Get("add"), {x, y});
    const Expr square = Call(relay::Op::Get("mul"), {shared, shared});
    const Expr root = Call(relay::Op::Get("sqrt"), {square});
    const Expr transposed = Call(relay::Op::Get("transpose"), {shared},
                                relay::TransposeAttrs::Create({1, 0}));
    // shared is both a graph result and an input to later calls. x is a
    // passthrough result. The nested tuple's leaves have explicit field order.
    return Function({x, y}, Tuple({root, Tuple({shared, x}), transposed}));
}

std::vector<restricted::InputAxisSymbol> Axes() {
    return {{0, 0, "N", 1, 8, 1}, {1, 0, "N", 1, 8, 1}};
}

NDArray Tensor(std::vector<int64_t> dimensions, int seed = 0) {
    size_t count = 1;
    for (int64_t extent : dimensions) count *= static_cast<size_t>(extent);
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) {
        values[i] = static_cast<float>((static_cast<int>(i) + seed) % 13 - 6) * 0.25f;
    }
    NDArray result = NDArray::Empty(dimensions, runtime::DataTypeFromString("float32"), Device::CPU());
    if (count) result.CopyFromBytes(values.data(), count * sizeof(float));
    return result;
}

template <class T = float>
std::vector<T> Read(const NDArray& array) {
    std::vector<T> result(array.NBytes() / sizeof(T));
    if (!result.empty()) array.CopyToBytes(result.data(), array.NBytes());
    return result;
}

bool SameStats(const ci::PrimitiveCacheStats& a, const ci::PrimitiveCacheStats& b) {
    return a.hits == b.hits && a.misses == b.misses && a.entries == b.entries &&
           a.accounted_bytes == b.accounted_bytes && a.evictions == b.evictions &&
           a.in_flight == b.in_flight && a.merged_waiters == b.merged_waiters &&
           a.failures == b.failures && a.rejections == b.rejections && a.active_pins == b.active_pins;
}

void CheckResult(const Array<NDArray>& outputs, const Array<NDArray>& inputs) {
    Check(outputs.size() == 4, "nested tuple must flatten to four outputs");
    const int64_t n = inputs[0].shape()[0];
    for (size_t index = 0; index < outputs.size(); ++index) {
        const auto dimensions = outputs[index].shape();
        Check(dimensions.size() == 2 && dimensions[0] == (index == 3 ? 3 : n) &&
                  dimensions[1] == (index == 3 ? n : 3),
              "explicit output shape/order changed");
    }
    const auto x = Read(inputs[0]), y = Read(inputs[1]);
    const auto root = Read(outputs[0]), shared = Read(outputs[1]), passthrough = Read(outputs[2]), transposed = Read(outputs[3]);
    for (size_t i = 0; i < x.size(); ++i) {
        const float sum = x[i] + y[i];
        Check(root[i] == std::abs(sum) && shared[i] == sum && passthrough[i] == x[i] &&
                  transposed[(i % 3) * static_cast<size_t>(n) + i / 3] == sum,
              "DAG results differ from independent arithmetic/transpose reference");
    }
}

void TestSharedGraph() {
    ci::ClearPrimitiveCacheForTesting();
    profiling::ProfileOptions options;
    options.enabled = true;
    options.ir_capture_mode = profiling::IRCaptureMode::kDisabled;
    options.record_pass_ir = false;
    options.bundle_dir = (std::filesystem::current_path() / "out" / "bounded_graph_structure_profile").string();
    const auto context = profiling::ProfileContext::Create(options);
    const profiling::ActivationScope activation(context, "graph_structure_validation");
    const auto config = api::CompileConfig::Create(BuildTarget(Device::CPU()), 2, options);
    Function caller = Graph();
    const auto prepared = Adapter::Prepare(caller, config, Axes());
    // Mutating caller IR must not change the prepared snapshot.
    const auto* tuple = caller->body.As<TupleNode>();
    const auto* nested = tuple->fields[1].As<TupleNode>();
    const_cast<CallNode*>(nested->fields[0].As<CallNode>())->op = relay::Op::Get("mul");
    const auto counters = prepared.representative().counters();
    const auto request = Adapter::MintBoundedCompileRequest(prepared);
    const Function logical = request.logical_boundary_function();
    const auto* result_tuple = logical->body.As<TupleNode>();
    Check(result_tuple && result_tuple->fields.size() == 3, "result tuple structure was lost");
    const auto* result_nested = result_tuple->fields[1].As<TupleNode>();
    const auto* root = result_tuple->fields[0].As<CallNode>();
    const auto* square = root->args[0].As<CallNode>();
    Check(square->args.size() == 2 && square->args[0].get() == square->args[1].get() &&
              square->args[0].get() == result_nested->fields[0].get(),
          "materialization unfolded shared calls or deduplicated logical arguments");
    const auto compiled = api::Compiler::CompileBounded(request);
    Check(compiled.plan().calls().size() == 4 && compiled.module().entry_count() == 4 &&
              compiled.plan().calls()[1].input_value_ids().size() == 1,
          "shared producer or repeated operand created extra physical inputs/kernels");
    const auto stats = ci::GetPrimitiveCacheStats();
    const auto plan_abi = api::BuildPlanAbiFingerprint(compiled);
    const runtime::RuntimeSession session(compiled.module(), compiled.plan());
    const auto event_counts = [&] {
        context->Flush();
        std::ifstream events(std::filesystem::path(context->bundle_dir()) / "events.jsonl");
        Check(events.good(), "DAG profiling bundle is missing");
        std::pair<size_t, size_t> counts{0, 0};
        std::string line;
        while (std::getline(events, line)) {
            if (line.find("\"event_type\":\"kernel_submit\"") != std::string::npos) ++counts.first;
            if (line.find("\"event_type\":\"alloc\"") != std::string::npos) ++counts.second;
        }
        return counts;
    };
    Array<NDArray> retained, retained_inputs;
    for (int64_t n : {1, 3, 7, 8}) {
        const Array<NDArray> inputs{Tensor({n, 3}), Tensor({n, 3}, 5)};
        const auto outputs = session.Run(inputs, {{"stage", "bounded_graph_structure"},
            {"length", std::to_string(n)}, {"plan_abi", plan_abi.digest()}});
        CheckResult(outputs, inputs);
        if (retained.empty()) { retained = outputs; retained_inputs = inputs; }
        CheckResult(retained, retained_inputs);
        Check(SameStats(stats, ci::GetPrimitiveCacheStats()), "Run performed compile/cache work");
    }
    Check(event_counts().first == 16, "four graph runs must submit sixteen LLVM kernels");
    const auto reject_input = [&](Array<NDArray> inputs) {
        const auto before = event_counts();
        Rejects([&] { (void)session.Run(inputs); }, "invalid graph inputs were accepted");
        Check(event_counts() == before, "invalid graph inputs reached allocation or launch");
    };
    reject_input({Tensor({0, 3}), Tensor({0, 3})});
    reject_input({Tensor({9, 3}), Tensor({9, 3})});
    reject_input({Tensor({3, 3}), Tensor({4, 3})});
    reject_input({Tensor({3, 4}), Tensor({3, 4})});
    reject_input({Tensor({9}), Tensor({9})});
    reject_input({Tensor({3, 3}), NDArray::Zeros({3, 3}, runtime::DataTypeFromString("int64"), Device::CPU())});
    reject_input({Tensor({3, 3})});
    Check(SameStats(stats, ci::GetPrimitiveCacheStats()), "rejected Run performed compile/cache work");
    const auto after = prepared.representative().counters();
    Check(counters.execution_contract_resolutions == after.execution_contract_resolutions &&
              counters.relay_graph_pipelines == after.relay_graph_pipelines &&
              counters.capability_boundary_checks == after.capability_boundary_checks &&
              counters.value_graph_builds == after.value_graph_builds && counters.partitions == after.partitions,
          "bounded admission repeated the frozen preparation pipeline");
    std::cout << "shared DAG: 4 lengths, 4 explicit outputs, 16 LLVM launches, exact arithmetic, 7 zero-launch rejections\n";
}

void TestExactMaterialization() {
    const auto prepared = Adapter::Prepare(Graph(), Config(), Axes());
    for (int64_t n : {2, 6}) {
        const auto decision = Adapter::MintExact(prepared, shape::BindingSet({{"N", n}}));
        const Function function = Adapter::MaterializeExactFunction(prepared, decision);
        const auto compiled = api::Compiler::Compile(function, Config());
        Adapter::VerifyCompiledExactVariant(prepared, decision, compiled);
        const Array<NDArray> inputs{Tensor({n, 3}), Tensor({n, 3}, 5)};
        const runtime::RuntimeSession session(compiled.module(), compiled.plan());
        CheckResult(session.Run(inputs), inputs);

        // Swap same-shaped outputs and keep the same four operators. Boundary
        // shapes alone cannot bind this artifact to the authorized semantics.
        Function wrong = Adapter::MaterializeExactFunction(prepared, decision);
        auto* tuple = const_cast<TupleNode*>(wrong->body.As<TupleNode>());
        auto* nested = const_cast<TupleNode*>(tuple->fields[1].As<TupleNode>());
        const Expr saved = tuple->fields[0];
        tuple->fields[0] = nested->fields[0];
        nested->fields[0] = saved;
        const auto wrong_compiled = api::Compiler::Compile(wrong, Config());
        Rejects([&] { Adapter::VerifyCompiledExactVariant(prepared, decision, wrong_compiled); },
                "same-shaped reordered outputs escaped semantic binding");
    }
    std::cout << "exact profiles: lengths 2/6 executed; both same-shaped reordered artifacts rejected\n";
}

void TestSharedShapeValue() {
    const Var x("x", TensorType({4, 1, 3}, "float32"));
    const Expr shared = Call(relay::Op::Get("nn_relu"), {x});
    const Expr shape_value = Call(relay::Op::Get("shape_of"), {shared});
    NDArray indices = NDArray::Empty({2}, runtime::DataTypeFromString("int64"), Device::CPU());
    const int64_t raw_indices[] = {0, 2};
    indices.CopyFromBytes(raw_indices, sizeof(raw_indices));
    const Expr target = Call(relay::Op::Get("gather"), {shape_value, Constant(indices)},
                             relay::GatherAttrs::Create(0));
    const Expr reshaped = Call(relay::Op::Get("reshape_dynamic"), {shared, target});
    const Function graph({x}, Tuple({reshaped, shared, shape_value}));
    const auto compiled = api::Compiler::CompileBounded(Adapter::MintBoundedCompileRequest(
        Adapter::Prepare(graph, Config(), {{0, 0, "N", 1, 8, 1}})));
    Check(compiled.plan().calls().size() == 4, "shared data/shape graph lost the folded shape unit");
    const auto stats = ci::GetPrimitiveCacheStats();
    const runtime::RuntimeSession session(compiled.module(), compiled.plan());
    for (int64_t n : {1, 5, 8}) {
        const NDArray input = Tensor({n, 1, 3});
        const auto outputs = session.Run({input});
        Check(outputs.size() == 3 && outputs[0].shape().size() == 2 &&
                  outputs[0].shape()[0] == n && outputs[0].shape()[1] == 3 &&
                  outputs[1].shape().size() == 3 &&
                  Read<int64_t>(outputs[2]) == std::vector<int64_t>({n, 1, 3}),
              "mixed data/shape tuple result is incorrect");
        auto expected = Read(input);
        for (float& value : expected) value = std::max(0.0f, value);
        Check(Read(outputs[0]) == expected && Read(outputs[1]) == expected,
              "folded shape chain changed shared data values");
        Check(SameStats(stats, ci::GetPrimitiveCacheStats()), "shape DAG Run compiled kernels");
    }
    std::cout << "data/shape DAG: 3 lengths, folded target and int64 tuple result executed\n";
}

void TestUnsupportedResults() {
    const Var x("x", TensorType({4, 3}, "float32"));
    const Expr y = Call(relay::Op::Get("nn_relu"), {x});
    const auto before = ci::GetPrimitiveCacheStats();
    const auto reject_body = [&](Expr body) {
        Rejects([&] { (void)Adapter::Prepare(Function({x}, body), Config(), {{0, 0, "N", 1, 8, 1}}); },
                "unsupported result structure was admitted");
    };
    reject_body(Tuple({y, y}));
    reject_body(Tuple({y, Tuple(Array<Expr>())}));
    reject_body(Call(relay::Op::Get("nn_relu"), {Tuple({x, y})}));
    Check(SameStats(before, ci::GetPrimitiveCacheStats()), "invalid tuple preparation reached compiler cache");
}
}  // namespace

int main() {
    try {
        TestSharedGraph();
        TestExactMaterialization();
        TestSharedShapeValue();
        TestUnsupportedResults();
        std::cout << "All bounded graph structure tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
