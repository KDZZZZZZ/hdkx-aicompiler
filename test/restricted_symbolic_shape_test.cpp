#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "../src/compiler/internal/primitive_cache.h"
#include "kxc/compiler/compiler.h"
#include "kxc/compiler/restricted_symbolic_shape.h"
#include "kxc/relay/op.h"
#include "kxc/runtime/session.h"

namespace {
namespace restricted = kxc::api::experimental::restricted_symbolic_shape::v1;
namespace shape =
    kxc::api::experimental::shape_specialization::v1;

#define CHECK(x, m) do { if (!(x)) { std::cerr << "[FAIL] " << __FUNCTION__ << ": " << m << "\n"; return false; } } while (0)

bool Throws(const std::function<void()>& fn) {
    try { fn(); } catch (const std::exception&) { return true; }
    return false;
}

kxc::api::CompileConfig Config() {
    return kxc::api::CompileConfig::Create(kxc::BuildTarget(kxc::Device::CPU()));
}

kxc::Function UnaryGraph(const std::string& operation, int64_t extent = 4) {
    const kxc::TensorType type({extent}, "float32");
    const kxc::Var x("x", type);
    return kxc::Function({x}, kxc::Call(kxc::relay::Op::Get(operation), {x}));
}

kxc::Function MixedGraph() {
    const kxc::TensorType type({4}, "float32");
    const kxc::Var x("x", type), y("y", type);
    const kxc::Expr add = kxc::Call(kxc::relay::Op::Get("add"), {x, y});
    return kxc::Function({x, y}, kxc::Call(kxc::relay::Op::Get("nn_relu"), {
        kxc::Call(kxc::relay::Op::Get("sqrt"), {add})}));
}

kxc::Function UnusedBindingGraph() {
    const kxc::Var x("x", kxc::TensorType({4}, "float32"));
    const kxc::Var unused("unused", kxc::TensorType({5}, "float32"));
    return kxc::Function({x, unused}, kxc::Call(kxc::relay::Op::Get("nn_relu"), {x}));
}

bool SameCounters(const kxc::api::experimental::shape_exact::v1::ShapeExactPreparationCounters& a,
                  const kxc::api::experimental::shape_exact::v1::ShapeExactPreparationCounters& b) {
    return a.execution_contract_resolutions == b.execution_contract_resolutions &&
           a.relay_graph_pipelines == b.relay_graph_pipelines &&
           a.capability_boundary_checks == b.capability_boundary_checks &&
           a.value_graph_builds == b.value_graph_builds && a.partitions == b.partitions;
}

shape::BindingSet Bindings(int64_t n) { return shape::BindingSet({{"n", n}}); }

bool TestGateAndRestrictedExactSlice() {
#if !KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE
    CHECK(!restricted::RestrictedSymbolicShapeAdapter::IsEnabled(), "default gate must be off");
    CHECK(Throws([] { (void)restricted::RestrictedSymbolicShapeAdapter::Prepare(
              UnaryGraph("nn_relu"), Config(), {{0, 0, "n", 1, 8, 1}}); }),
          "off gate must fail closed");
    return true;
#elif !KXC_ENABLE_SHAPE_PRODUCTION_EXACT
    CHECK(restricted::RestrictedSymbolicShapeAdapter::IsEnabled(), "restricted gate must report enabled");
    CHECK(Throws([] { (void)restricted::RestrictedSymbolicShapeAdapter::Prepare(
              UnaryGraph("nn_relu"), Config(), {{0, 0, "n", 1, 8, 1}}); }),
          "restricted overlay must require its exact frozen representative gate");
    return true;
#else
    using restricted::RestrictedSymbolicShapeAdapter;
    const auto relu = RestrictedSymbolicShapeAdapter::Prepare(
        UnaryGraph("nn_relu"), Config(), {{0, 0, "n", 1, 8, 1}});
    const auto sqrt = RestrictedSymbolicShapeAdapter::Prepare(
        UnaryGraph("sqrt"), Config(), {{0, 0, "n", 1, 8, 1}});
    auto relay_to_mutate = MixedGraph();
    const auto mixed = RestrictedSymbolicShapeAdapter::Prepare(
        relay_to_mutate, Config(), {{0, 0, "n", 1, 8, 1}, {1, 0, "n", 1, 8, 1}});
    auto* mutated_call = const_cast<kxc::CallNode*>(relay_to_mutate->body.As<kxc::CallNode>());
    mutated_call->op = kxc::relay::Op::Get("mul");

    const auto frozen = mixed.representative().counters();
    const auto exact4 = RestrictedSymbolicShapeAdapter::MintExact(mixed, Bindings(4));
    const auto relu4 = RestrictedSymbolicShapeAdapter::MintExact(relu, Bindings(4));
    const auto sqrt4 = RestrictedSymbolicShapeAdapter::MintExact(sqrt, Bindings(4));
    CHECK(exact4.exact_requests().size() == 3,
          "mixed add/sqrt/relu chain must mint opaque exact snapshots");
    CHECK(Throws([&] { (void)RestrictedSymbolicShapeAdapter::ChangedUnitIndices(relu4, sqrt4); }),
          "different unit semantic contexts must be incomparable");

    auto detached = exact4.exact_requests();
    detached.clear();
    CHECK(exact4.exact_requests().size() == 3,
          "returned request snapshots cannot mutate decision authority");

    const auto unused = RestrictedSymbolicShapeAdapter::Prepare(UnusedBindingGraph(), Config(),
        {{0, 0, "n", 1, 8, 1}, {1, 0, "unused", 1, 8, 1}});
    const auto unused4 = RestrictedSymbolicShapeAdapter::MintExact(
        unused, shape::BindingSet({{"n", 4}, {"unused", 4}}));
    const auto unused5 = RestrictedSymbolicShapeAdapter::MintExact(
        unused, shape::BindingSet({{"n", 4}, {"unused", 5}}));
    CHECK(RestrictedSymbolicShapeAdapter::ChangedUnitIndices(unused4, unused5).empty(),
          "unrelated bindings must not invalidate an unchanged unit artifact boundary");

    const auto exact5 = RestrictedSymbolicShapeAdapter::MintExact(mixed, Bindings(5));
    CHECK(RestrictedSymbolicShapeAdapter::ChangedUnitIndices(exact4, exact5) ==
              std::vector<size_t>({0, 1, 2}) &&
          SameCounters(frozen, mixed.representative().counters()),
          "profiles must not rerun prepared pipeline or partition work");

    kxc::api::internal::ClearPrimitiveCacheForTesting();
    const auto before = kxc::api::internal::GetPrimitiveCacheStats();
    CHECK(Throws([&] { (void)RestrictedSymbolicShapeAdapter::MintExact(mixed, shape::BindingSet()); }) &&
          Throws([] { (void)RestrictedSymbolicShapeAdapter::Prepare(
              UnaryGraph("nn_relu", -1), Config(), {{0, 0, "n", 1, 8, 1}}); }),
          "unbound bindings and legacy dimensions must fail closed");
    const auto after = kxc::api::internal::GetPrimitiveCacheStats();
    CHECK(before.entries == after.entries && before.hits == after.hits && before.misses == after.misses &&
          before.failures == after.failures && SameCounters(frozen, mixed.representative().counters()),
          "invalid requests must not mutate compiler/cache or prepared counters");
    return true;
#endif
}

kxc::Function Rank2Unary(int64_t rows = 4, int64_t cols = 3) {
    const kxc::Var x("x", kxc::TensorType({rows, cols}, "float32"));
    return kxc::Function({x}, kxc::Call(kxc::relay::Op::Get("nn_relu"), {x}));
}

bool TestMaterializedVariantClosure() {
#if !KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE || !KXC_ENABLE_SHAPE_PRODUCTION_EXACT
    return true;
#else
    using restricted::RestrictedSymbolicShapeAdapter;
    const auto prep = RestrictedSymbolicShapeAdapter::Prepare(
        UnaryGraph("nn_relu"), Config(), {{0, 0, "n", 2, 8, 2}});
    const auto d6 = RestrictedSymbolicShapeAdapter::MintExact(prep, Bindings(6));
    const auto d4 = RestrictedSymbolicShapeAdapter::MintExact(prep, Bindings(4));

    // 不变量 5：不同 binding 不同 route，相同 binding 相同 route。
    CHECK(!(RestrictedSymbolicShapeAdapter::ExactDispatchKey(prep, d6) ==
            RestrictedSymbolicShapeAdapter::ExactDispatchKey(prep, d4)),
          "different bindings must mint different dispatch keys");
    CHECK(RestrictedSymbolicShapeAdapter::ExactDispatchKey(prep, d6) ==
              RestrictedSymbolicShapeAdapter::ExactDispatchKey(
                  prep, RestrictedSymbolicShapeAdapter::MintExact(prep, Bindings(6))),
          "identical bindings must mint identical dispatch keys");

    // 物化：参数类型必须携带决策求值后的 concrete shape 和原 dtype。
    const kxc::Function fn6 =
        RestrictedSymbolicShapeAdapter::MaterializeExactFunction(prep, d6);
    CHECK(fn6.defined() && fn6->params.size() == 1,
          "materialized function must keep the representative arity");
    const auto* materialized_type =
        fn6->params[0]->type_annotation.As<kxc::TensorTypeNode>();
    CHECK(materialized_type && materialized_type->shape.size() == 1 &&
              materialized_type->shape[0] == 6 &&
              materialized_type->dtype == "float32",
          "materialized parameter must carry the bound concrete extent");

    // 归属校验：别的模板铸出的决策必须被拒绝。
    const auto other = RestrictedSymbolicShapeAdapter::Prepare(
        UnaryGraph("sqrt"), Config(), {{0, 0, "n", 2, 8, 2}});
    const auto other6 = RestrictedSymbolicShapeAdapter::MintExact(other, Bindings(6));
    CHECK(Throws([&] { (void)RestrictedSymbolicShapeAdapter::MaterializeExactFunction(
              prep, other6); }),
          "foreign decision must not materialize against this template");
    CHECK(Throws([&] { (void)RestrictedSymbolicShapeAdapter::ExactDispatchKey(
              prep, other6); }),
          "foreign decision must not mint a route for this template");

    // 请求边界换算：输入 shape → bindings。
    CHECK(RestrictedSymbolicShapeAdapter::BindingsFromInputShapes(prep, {{6}}) ==
              Bindings(6),
          "input shapes must convert to the canonical binding set");
    CHECK(Throws([&] { (void)RestrictedSymbolicShapeAdapter::BindingsFromInputShapes(
              prep, {{6}, {6}}); }),
          "input count mismatch must fail closed");
    CHECK(Throws([&] { (void)RestrictedSymbolicShapeAdapter::BindingsFromInputShapes(
              prep, {{6, 2}}); }),
          "input rank mismatch must fail closed");

    // 非 overlay 静态轴必须与模板一致。
    const auto rank2 = RestrictedSymbolicShapeAdapter::Prepare(
        Rank2Unary(), Config(), {{0, 0, "n", 2, 8, 2}});
    CHECK(RestrictedSymbolicShapeAdapter::BindingsFromInputShapes(rank2, {{6, 3}}) ==
              Bindings(6),
          "static non-overlay axis must pass through unchanged");
    CHECK(Throws([&] { (void)RestrictedSymbolicShapeAdapter::BindingsFromInputShapes(
              rank2, {{6, 7}}); }),
          "static non-overlay axis mismatch must fail closed");

    // 共享 symbol 的输入必须一致。
    const auto mixed = RestrictedSymbolicShapeAdapter::Prepare(
        MixedGraph(), Config(), {{0, 0, "n", 1, 8, 1}, {1, 0, "n", 1, 8, 1}});
    CHECK(Throws([&] { (void)RestrictedSymbolicShapeAdapter::BindingsFromInputShapes(
              mixed, {{4}, {5}}); }),
          "conflicting shared-symbol extents must fail closed");
    CHECK(RestrictedSymbolicShapeAdapter::BindingsFromInputShapes(mixed, {{5}, {5}}) ==
              Bindings(5),
          "consistent shared-symbol extents must bind once");

#if KXC_USE_LLVM
    // 物化 variant 经用户显式 Compiler::Compile 编译并被验证函数绑定。
    const kxc::api::CompiledGraph g6 = kxc::api::Compiler::Compile(fn6, Config());
    RestrictedSymbolicShapeAdapter::VerifyCompiledExactVariant(prep, d6, g6);
    const kxc::Function fn4 =
        RestrictedSymbolicShapeAdapter::MaterializeExactFunction(prep, d4);
    const kxc::api::CompiledGraph g4 = kxc::api::Compiler::Compile(fn4, Config());
    CHECK(Throws([&] { RestrictedSymbolicShapeAdapter::VerifyCompiledExactVariant(
              prep, d6, g4); }),
          "verification must reject a variant compiled for other bindings");

    // 数值：relu(N=6) 输入含负值。
    kxc::runtime::NDArray input = kxc::runtime::NDArray::Zeros(
        {6}, kxc::runtime::DataTypeFromString("float32"), kxc::Device::CPU());
    const std::vector<float> raw = {-2, -1, 0, 1, 2, 3};
    input.CopyFromBytes(raw.data(), raw.size() * sizeof(float));
    kxc::runtime::RuntimeSession session(g6.module(), g6.plan());
    const auto outputs = session.Run({input});
    std::vector<float> actual(6);
    outputs[0].CopyToBytes(actual.data(), actual.size() * sizeof(float));
    CHECK(actual == std::vector<float>({0, 0, 0, 1, 2, 3}),
          "materialized relu variant must execute with the bound extent");
#endif
    return true;
#endif
}

}  // namespace

int main() {
    std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"gate_and_restricted_exact_slice", TestGateAndRestrictedExactSlice},
        {"materialized_variant_closure", TestMaterializedVariantClosure},
    };
    int failed = 0;
    for (const auto& test : tests) {
        try {
            if (test.second()) std::cout << "[PASS] " << test.first << "\n";
            else ++failed;
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << test.first << ": " << error.what() << "\n";
            ++failed;
        }
    }
    return failed == 0 ? 0 : 1;
}
