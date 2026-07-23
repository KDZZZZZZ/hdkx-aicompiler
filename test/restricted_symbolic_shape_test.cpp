#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "../src/compiler/internal/primitive_cache.h"
#include "kxc/compiler/restricted_symbolic_shape.h"
#include "kxc/relay/op.h"

namespace {
namespace restricted = kxc::api::experimental::restricted_symbolic_shape::v1;
namespace shape = kxc::shape::experimental::v1;

#define CHECK(x, m) do { if (!(x)) { std::cerr << "[FAIL] " << __FUNCTION__ << ": " << m << "\n"; return false; } } while (0)

bool Throws(const std::function<void()>& fn) {
    try { fn(); } catch (const std::exception&) { return true; }
    return false;
}

kxc::api::CompileConfig Config() {
    return kxc::api::CompileConfig::Create(kxc::BuildTarget(kxc::Device::CPU()));
}

kxc::Function AddGraph(int64_t extent = 4) {
    const kxc::TensorType type({extent}, "float32");
    const kxc::Var x("x", type), y("y", type);
    return kxc::Function({x, y}, kxc::Call(kxc::relay::Op::Get("add"), {x, y}));
}

kxc::Function UnsupportedGraph() {
    const kxc::TensorType type({4}, "float32");
    const kxc::Var x("x", type), y("y", type);
    return kxc::Function({x, y}, kxc::Call(kxc::relay::Op::Get("sub"), {x, y}));
}

bool SameCounters(const kxc::api::experimental::shape_exact::v1::ShapeExactPreparationCounters& a,
                  const kxc::api::experimental::shape_exact::v1::ShapeExactPreparationCounters& b) {
    return a.execution_contract_resolutions == b.execution_contract_resolutions &&
           a.relay_graph_pipelines == b.relay_graph_pipelines &&
           a.capability_boundary_checks == b.capability_boundary_checks &&
           a.value_graph_builds == b.value_graph_builds && a.partitions == b.partitions;
}

shape::BindingSet Bindings(int64_t n) { return shape::BindingSet({{"n", n}}); }

shape::ApplicabilityGuard Guard(int64_t lower = 1, int64_t upper = 8) {
    return shape::ApplicabilityGuard({"n"}, {
        shape::Constraint::Range(shape::DimExpr::Symbol("n"), lower, upper),
        shape::Constraint::DivisibleBy(shape::DimExpr::Symbol("n"), 1)});
}

std::vector<std::string> ValueNames(const shape::GraphTemplate& graph) {
    std::vector<std::string> names;
    for (const auto& value : graph.shape_program().inputs()) names.push_back(value.name);
    for (const auto& value : graph.shape_program().outputs()) names.push_back(value.name);
    return names;
}

shape::BucketPolicy Bucket(const restricted::PreparedRestrictedSymbolicTemplate& prepared,
                           const restricted::RestrictedDispatchDecision& exact,
                           bool valid_tail = true) {
    std::vector<shape::BucketValueBoundary> boundaries;
    for (const std::string& name : ValueNames(prepared.graph_template())) {
        const auto& contract = exact.exact_oracle.profile().Value(name).contract;
        std::vector<int64_t> physical = contract.logical;
        physical[0] = 8;
        boundaries.push_back({name, physical, {1}, contract.layout, contract.alignment,
                              contract.memory_scope, contract.abi});
    }
    const auto& unit = prepared.graph_template().ordered_units()[0];
    return shape::BucketPolicy("n8", 1, Guard(), std::move(boundaries),
        {{0, unit.semantic_key, valid_tail, valid_tail, valid_tail, valid_tail}}, 0);
}

shape::PolymorphicPolicy Polymorphic(const restricted::PreparedRestrictedSymbolicTemplate& prepared) {
    std::vector<shape::SymbolicBoundaryContract> boundaries;
    for (const auto& value : prepared.graph_template().shape_program().inputs()) {
        boundaries.push_back({value.name, value.contract.logical().dimensions(),
            value.contract.physical().layout(), value.contract.physical().alignment(),
            value.contract.physical().memory_scope(), value.contract.abi()});
    }
    for (const auto& value : prepared.graph_template().shape_program().outputs()) {
        boundaries.push_back({value.name, value.contract.logical().dimensions(),
            value.contract.physical().layout(), value.contract.physical().alignment(),
            value.contract.physical().memory_scope(), value.contract.abi()});
    }
    const auto& unit = prepared.graph_template().ordered_units()[0];
    return shape::PolymorphicPolicy(1, Guard(), {{0, unit.semantic_key, "restricted.add.equal-shape.v1"}},
        {{0, "n_extent", "n", 1, 8, 1}}, std::move(boundaries),
        prepared.graph_template().key().target_backend_abi(), 0);
}

bool TestGateAndRestrictedSlice() {
#if !KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE
    CHECK(!restricted::RestrictedSymbolicShapeAdapter::IsEnabled(), "default gate must be off");
    CHECK(Throws([] { (void)restricted::RestrictedSymbolicShapeAdapter::Prepare(
              AddGraph(), Config(), {{0, 0, "n", 1, 8, 1}, {1, 0, "n", 1, 8, 1}}); }),
          "off gate must fail closed");
    return true;
#elif !KXC_ENABLE_SHAPE_PRODUCTION_EXACT
    CHECK(restricted::RestrictedSymbolicShapeAdapter::IsEnabled(), "restricted gate must report enabled");
    CHECK(Throws([] { (void)restricted::RestrictedSymbolicShapeAdapter::Prepare(
              AddGraph(), Config(), {{0, 0, "n", 1, 8, 1}, {1, 0, "n", 1, 8, 1}}); }),
          "restricted overlay must require its exact frozen representative gate");
    return true;
#else
    using restricted::RestrictedSymbolicShapeAdapter;
    const auto prepared = RestrictedSymbolicShapeAdapter::Prepare(
        AddGraph(), Config(), {{0, 0, "n", 1, 8, 1}, {1, 0, "n", 1, 8, 1}});
    const auto frozen = prepared.representative().counters();
    const auto exact4 = RestrictedSymbolicShapeAdapter::MintExact(prepared, Bindings(4));
    CHECK(exact4.kind == restricted::DispatchKind::kExact && exact4.exact_requests.size() == 1 &&
          exact4.guarded_requests.empty(), "exact mint must contain one complete exact request");
    const auto bucket4 = RestrictedSymbolicShapeAdapter::MintBucket(prepared, Bindings(4), Bucket(prepared, exact4));
    const auto poly4 = RestrictedSymbolicShapeAdapter::MintPolymorphic(prepared, Bindings(4), Polymorphic(prepared));
    CHECK(bucket4.guarded_requests.size() == 1 && poly4.guarded_requests.size() == 1,
          "guard authority mint must emit requests only, never executable plans");
    CHECK(RestrictedSymbolicShapeAdapter::ChangedUnitIndices(exact4, bucket4) == std::vector<size_t>({0}) &&
          RestrictedSymbolicShapeAdapter::ChangedUnitIndices(bucket4, bucket4).empty(),
          "complete canonical request comparison must identify only changed units");
    const auto exact5 = RestrictedSymbolicShapeAdapter::MintExact(prepared, Bindings(5));
    CHECK(RestrictedSymbolicShapeAdapter::ChangedUnitIndices(exact4, exact5) == std::vector<size_t>({0}) &&
          SameCounters(frozen, prepared.representative().counters()),
          "profiles must not rerun prepared pipeline or partition work");

    kxc::api::internal::ClearPrimitiveCacheForTesting();
    const auto before = kxc::api::internal::GetPrimitiveCacheStats();
    CHECK(Throws([&] { (void)RestrictedSymbolicShapeAdapter::MintExact(prepared, shape::BindingSet()); }) &&
          Throws([&] { (void)RestrictedSymbolicShapeAdapter::MintBucket(prepared, Bindings(4), Bucket(prepared, exact4, false)); }) &&
          Throws([&] { (void)RestrictedSymbolicShapeAdapter::MintBucket(prepared, Bindings(4),
              shape::BucketPolicy("miss", 1, Guard(6, 8), Bucket(prepared, exact4).boundaries(),
                                  Bucket(prepared, exact4).tail_contracts(), 0)); }),
          "unbound, invalid-tail, and guard-miss requests must fail closed");
    const auto after = kxc::api::internal::GetPrimitiveCacheStats();
    CHECK(before.entries == after.entries && before.hits == after.hits && before.misses == after.misses &&
          before.failures == after.failures && SameCounters(frozen, prepared.representative().counters()),
          "invalid profile requests must not mutate compiler/cache or prepared counters");
    CHECK(Throws([] { (void)RestrictedSymbolicShapeAdapter::Prepare(
              AddGraph(-1), Config(), {{0, 0, "n", 1, 8, 1}, {1, 0, "n", 1, 8, 1}}); }) &&
          Throws([] { (void)RestrictedSymbolicShapeAdapter::Prepare(
              UnsupportedGraph(), Config(), {{0, 0, "n", 1, 8, 1}, {1, 0, "n", 1, 8, 1}}); }),
          "legacy dimensions and unsupported operations must fail before preparation authority");
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
