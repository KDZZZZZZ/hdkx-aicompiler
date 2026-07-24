#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "../src/compiler/internal/primitive_cache.h"
#include "kxc/compiler/restricted_symbolic_shape.h"
#include "kxc/relay/op.h"

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

}  // namespace

int main() {
    try { return TestGateAndRestrictedExactSlice() ? 0 : 1; }
    catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << "\n";
        return 1;
    }
}
