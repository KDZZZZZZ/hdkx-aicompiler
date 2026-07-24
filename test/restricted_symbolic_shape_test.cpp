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

static_assert(!std::is_aggregate_v<restricted::RestrictedDispatchDecision>);
static_assert(!std::is_default_constructible_v<restricted::RestrictedDispatchDecision>);

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

shape::ApplicabilityGuard Guard(const std::vector<std::string>& symbols = {"n"},
                                int64_t lower = 1, int64_t upper = 8) {
    std::vector<shape::Constraint> constraints;
    for (const auto& symbol : symbols) {
        constraints.push_back(shape::Constraint::Range(shape::DimExpr::Symbol(symbol), lower, upper));
        constraints.push_back(shape::Constraint::DivisibleBy(shape::DimExpr::Symbol(symbol), 1));
    }
    return shape::ApplicabilityGuard(symbols, std::move(constraints));
}

std::vector<std::string> ValueNames(const shape::GraphTemplate& graph) {
    std::vector<std::string> names;
    for (const auto& value : graph.shape_program().inputs()) names.push_back(value.name);
    for (const auto& value : graph.shape_program().outputs()) names.push_back(value.name);
    return names;
}

shape::BucketPolicy Bucket(const restricted::PreparedRestrictedSymbolicTemplate& prepared,
                           const restricted::RestrictedDispatchDecision& exact,
                           bool padded = true, size_t changed_tail = static_cast<size_t>(-1)) {
    std::vector<shape::BucketValueBoundary> boundaries;
    for (const std::string& name : ValueNames(prepared.graph_template())) {
        const auto& contract = exact.exact_oracle().profile().Value(name).contract;
        std::vector<int64_t> physical = contract.logical;
        if (padded) physical[0] = 8;
        boundaries.push_back({name, physical});
    }
    std::vector<shape::TailContract> tails;
    for (size_t index = 0; index < prepared.graph_template().ordered_units().size(); ++index) {
        const bool enabled = padded || index == changed_tail;
        tails.push_back({index, prepared.graph_template().ordered_units()[index].semantic_key,
                         enabled, enabled, enabled, enabled});
    }
    return shape::BucketPolicy(padded ? "n8" : "exact-capacity", 1, Guard(), std::move(boundaries),
                               std::move(tails), 0);
}

shape::PolymorphicPolicy Polymorphic(const restricted::PreparedRestrictedSymbolicTemplate& prepared,
                                     std::vector<std::string> proofs) {
    std::vector<shape::SymbolicBoundaryContract> boundaries;
    for (const auto& value : prepared.graph_template().shape_program().inputs()) {
        boundaries.push_back({value.name, value.contract.logical().dimensions(),
            value.contract.logical().axis_names()});
    }
    for (const auto& value : prepared.graph_template().shape_program().outputs()) {
        boundaries.push_back({value.name, value.contract.logical().dimensions(),
            value.contract.logical().axis_names()});
    }
    std::vector<shape::PolymorphicUnitProof> allowlist;
    for (size_t index = 0; index < proofs.size(); ++index) {
        allowlist.push_back({index, prepared.graph_template().ordered_units()[index].semantic_key, proofs[index]});
    }
    return shape::PolymorphicPolicy(1, Guard(), std::move(allowlist),
        {{0, "n_extent", "n", 1, 8, 1}}, std::move(boundaries), 0);
}

bool TestGateAndRestrictedSlice() {
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
    CHECK(exact4.kind() == restricted::DispatchKind::kExact && exact4.exact_requests().size() == 3 &&
          exact4.guarded_requests().empty(), "mixed add/sqrt/relu chain must mint opaque exact snapshots");
    CHECK(Throws([&] { (void)RestrictedSymbolicShapeAdapter::ChangedUnitIndices(relu4, sqrt4); }),
          "different unit semantic contexts must be incomparable");

    auto detached = exact4.exact_requests();
    detached.clear();
    CHECK(exact4.exact_requests().size() == 3, "returned request snapshots cannot mutate decision authority");
    const auto bucket4 = RestrictedSymbolicShapeAdapter::MintBucket(mixed, Bindings(4), Bucket(mixed, exact4));
    const auto poly4 = RestrictedSymbolicShapeAdapter::MintPolymorphic(mixed, Bindings(4),
        Polymorphic(mixed, {"restricted.add.equal-shape.v1", "restricted.sqrt.equal-shape.v1",
                            "restricted.relu.equal-shape.v1"}));
    CHECK(bucket4.guarded_requests().size() == 3 && poly4.guarded_requests().size() == 3,
          "guarded mint emits request snapshots only");
    CHECK(RestrictedSymbolicShapeAdapter::ChangedUnitIndices(exact4, bucket4) ==
              std::vector<size_t>({0, 1, 2}) &&
          RestrictedSymbolicShapeAdapter::ChangedUnitIndices(bucket4, bucket4).empty(),
          "complete artifact boundaries distinguish guarded kind without routing/profile identity");

    const auto unpadded = RestrictedSymbolicShapeAdapter::MintBucket(mixed, Bindings(4),
        Bucket(mixed, exact4, false));
    const auto first_tail_changed = RestrictedSymbolicShapeAdapter::MintBucket(mixed, Bindings(4),
        Bucket(mixed, exact4, false, 0));
    CHECK(RestrictedSymbolicShapeAdapter::ChangedUnitIndices(unpadded, first_tail_changed) ==
              std::vector<size_t>({0}),
          "direct unit comparator coverage: unrelated unit boundaries stay valid");

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
    CHECK(Throws([&] { (void)RestrictedSymbolicShapeAdapter::MintPolymorphic(
              relu, Bindings(4), Polymorphic(relu, {"arbitrary.proof"})); }),
          "only explicit versioned restricted operation proof tokens are authority");

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
    try { return TestGateAndRestrictedSlice() ? 0 : 1; }
    catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << "\n";
        return 1;
    }
}
