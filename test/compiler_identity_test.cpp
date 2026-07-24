/*! \file test/compiler_identity_test.cpp
 * \brief Verifies canonical compiler identity separation and full equality.
 */

#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "kxc/compiler/compiler.h"
#include "kxc/compiler/identity.h"
#include "kxc/relay/op.h"
#include "kxc/support/object_registration.h"

namespace {

#define TEST_CHECK(condition, message)                                           \
    do {                                                                          \
        if (!(condition)) {                                                        \
            std::cerr << "[FAIL] " << __FUNCTION__ << ": " << (message) << "\n"; \
            return false;                                                         \
        }                                                                         \
    } while (0)

bool Throws(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

class DerivedCallNode final : public kxc::CallNode {
public:
    std::string hidden_semantics;

    KXC_OBJECT_DECLARE
};

KXC_OBJECT_DEFINE_WITH_KEY(DerivedCallNode,
                           "kxc.test.compiler_identity.DerivedCallNode")

class UnregisteredDerivedCallNode final : public kxc::CallNode {
public:
    std::string hidden_semantics;
};

class UnregisteredDerivedFunctionNode final : public kxc::FunctionNode {
public:
    std::string hidden_semantics;
};

kxc::api::CompileConfig CpuConfig() {
    return kxc::api::CompileConfig::Create(
        kxc::BuildTarget(kxc::Device::CPU()), 1);
}

template <typename DerivedCall>
kxc::Function MakeDerivedCallFunction() {
    kxc::Var input("input", kxc::TensorType({2}, "float32"));
    auto* call = new DerivedCall();
    call->op = kxc::relay::Op::Get("nn_relu");
    call->args = {input};
    call->hidden_semantics = "must-not-be-omitted";
    return kxc::Function({input}, kxc::Expr(kxc::ObjectRef(call)));
}

bool TestGraphLocatorIsNotUnitSemantics() {
    using namespace kxc::api;
    const GraphValueLocator first{"graph-a", 7, 0};
    const GraphValueLocator second{"graph-a", 8, 0};
    TEST_CHECK(first.CanonicalBytes() != second.CanonicalBytes(),
               "graph routing locators must retain value ids");

    const UnitSemanticKey semantic("op=add;inputs=f32[4],f32[4]");
    const LinkSymbol symbol{"kxc_unit_7_add"};
    const StorageId storage{3};
    TEST_CHECK(semantic.defined() &&
                   semantic.canonical_bytes().find("graph-a") ==
                       std::string::npos &&
                   symbol.CanonicalBytes() != storage.CanonicalBytes() &&
                   symbol.CanonicalBytes() != first.CanonicalBytes(),
               "semantic, link, storage and routing identities stay disjoint");
    TEST_CHECK(Throws([] {
                   (void)GraphValueLocator{"", -1, 0}.CanonicalBytes();
               }) &&
                   Throws([] { (void)LinkSymbol{""}.CanonicalBytes(); }) &&
                   Throws([] { (void)StorageId{-1}.CanonicalBytes(); }),
               "invalid locator/link/storage identities must fail closed");
    return true;
}

bool TestDigestCollisionUsesCanonicalEquality() {
    using namespace kxc::api;
    const UnitSemanticKey first("canonical-unit-a", "forced-collision");
    const UnitSemanticKey second("canonical-unit-b", "forced-collision");
    TEST_CHECK(first.digest() == second.digest() && first != second,
               "digest is only an index; full canonical bytes decide equality");

    const PrimitiveArtifactKey first_artifact(
        first, "cpu-v1", "relay-tir-o2", 1, "schedule-v1", "llvm-v1",
        "artifact-collision");
    const PrimitiveArtifactKey second_artifact(
        second, "cpu-v1", "relay-tir-o2", 1, "schedule-v1", "llvm-v1",
        "artifact-collision");
    TEST_CHECK(first_artifact.digest() == second_artifact.digest() &&
                   first_artifact != second_artifact,
               "artifact lookup must compare complete canonical keys");
    return true;
}

bool TestEveryArtifactSemanticFieldCausesSafeMiss() {
    using namespace kxc::api;
    const UnitSemanticKey unit("unit-a");
    const PrimitiveArtifactKey baseline(
        unit, "cpu-v1", "pipeline-a", 1, "schedule-a", "backend-a");
    const std::vector<PrimitiveArtifactKey> changed = {
        PrimitiveArtifactKey(UnitSemanticKey("unit-b"), "cpu-v1",
                             "pipeline-a", 1, "schedule-a", "backend-a"),
        PrimitiveArtifactKey(unit, "cuda-v1", "pipeline-a", 1,
                             "schedule-a", "backend-a"),
        PrimitiveArtifactKey(unit, "cpu-v1", "pipeline-b", 1,
                             "schedule-a", "backend-a"),
        PrimitiveArtifactKey(unit, "cpu-v1", "pipeline-a", 2,
                             "schedule-a", "backend-a"),
        PrimitiveArtifactKey(unit, "cpu-v1", "pipeline-a", 1,
                             "schedule-b", "backend-a"),
        PrimitiveArtifactKey(unit, "cpu-v1", "pipeline-a", 1,
                             "schedule-a", "backend-b"),
    };
    for (const PrimitiveArtifactKey& candidate : changed) {
        TEST_CHECK(candidate != baseline,
                   "unit/target/pipeline/ABI/schedule/backend change must miss");
    }
    const PrimitiveArtifactKey same(
        unit, "cpu-v1", "pipeline-a", 1, "schedule-a", "backend-a");
    TEST_CHECK(same == baseline && same.digest() == baseline.digest(),
               "identical canonical artifact fields must be deterministic");
    return true;
}

bool TestDispatchAndPlanVariantRemainSeparate() {
    using namespace kxc::api;
    const kxc::Var input(
        "input", kxc::TensorType({4}, "float32"));
    const GraphSemanticKey graph = Compiler::BuildGraphSemanticKey(
        kxc::Function(
            {input}, kxc::Call(kxc::relay::Op::Get("nn_relu"), {input})));
    const ShapeProfileKey profile = BuildShapeProfileKey(
        graph, "logical=[4];physical=[4];valid=[4]", "bindings=[]",
        "exact-v1", 1);
    const DispatchKey exact("unit-a/cpu", "shape=[4];layout=contiguous",
                            "exact-v1");
    const DispatchKey bucket("unit-a/cpu", "shape=[1..8];valid=[4]",
                             "bucket-v1");
    TEST_CHECK(exact.defined() && bucket.defined() && !(exact == bucket),
               "dispatch applicability must not collapse into primitive semantics");

    const PrimitiveArtifactKey artifact(
        UnitSemanticKey("unit-a"), "cpu-v1", "pipeline-v1", 1,
        "schedule-v1", "backend-v1");
    const PlanVariantKey first = BuildPlanVariantKey(
        graph, profile, {{0, "unit_0", artifact, 0}}, "memory-plan-v1");
    const PlanVariantKey next_generation = BuildPlanVariantKey(
        graph, profile, {{0, "unit_0", artifact, 1}}, "memory-plan-v1");
    TEST_CHECK(first.defined() && next_generation.defined() &&
                   !(first == next_generation),
               "selected immutable generation is part of plan variant identity");
    TEST_CHECK(Throws([] {
                   const kxc::Var value(
                       "value", kxc::TensorType({4}, "float32"));
                   const GraphSemanticKey graph_key =
                       Compiler::BuildGraphSemanticKey(kxc::Function(
                           {value}, kxc::Call(
                               kxc::relay::Op::Get("nn_relu"), {value})));
                   const ShapeProfileKey shape_key = BuildShapeProfileKey(
                       graph_key, "shape=[4]", "bindings=[]", "exact", 1);
                   (void)BuildPlanVariantKey(
                       graph_key, shape_key, {}, "memory-v1");
               }),
               "a frozen plan variant requires selected artifacts");
    static_assert(!std::is_same_v<GraphSemanticKey, UnitSemanticKey>);
    static_assert(!std::is_same_v<PrimitiveArtifactKey, PlanVariantKey>);
    return true;
}

bool TestGraphSemanticIdentityRejectsUndefinedExprs() {
    using namespace kxc;
    const Expr leaf = relay::Op::Get("nn_relu");
    const Var binder("value", TensorType({2}, "float32"));
    const std::vector<std::pair<const char*, Function>> malformed = {
        {"function handle", Function()},
        {"function body", Function({}, Expr())},
        {"function parameter", Function({Var()}, leaf)},
        {"nested function body", Function({}, Function({}, Expr()))},
        {"call operator", Function({}, Call(Expr(), {}))},
        {"call argument", Function({}, Call(leaf, {Expr()}))},
        {"let binder", Function({}, Let(Var(), leaf, leaf))},
        {"let value", Function({}, Let(binder, Expr(), binder))},
        {"let body", Function({}, Let(binder, leaf, Expr()))},
        {"if condition", Function({}, If(Expr(), leaf, leaf))},
        {"if true branch", Function({}, If(leaf, Expr(), leaf))},
        {"if false branch", Function({}, If(leaf, leaf, Expr()))},
        {"tuple field", Function({}, Tuple({Expr()}))},
        {"tuple-get source", Function({}, TupleGetItem(Expr(), 0))},
    };
    for (const auto& [position, graph] : malformed) {
        TEST_CHECK(Throws([&] {
                       (void)api::Compiler::BuildGraphSemanticKey(graph);
                   }),
                   std::string("undefined ") + position +
                       " must not produce a graph artifact key");
    }
    return true;
}

bool TestGraphSemanticIdentityUsesExactNodeWhitelist() {
    using namespace kxc;
    Var input("input", TensorType({2}, "float32"));
    const Function normal(
        {input}, Call(relay::Op::Get("nn_relu"), {input}));
    TEST_CHECK(api::Compiler::BuildGraphSemanticKey(normal).defined(),
               "a supported exact Call node must produce an identity");
    TEST_CHECK(Throws([&] {
                   (void)api::Compiler::BuildGraphSemanticKey(
                       MakeDerivedCallFunction<DerivedCallNode>());
               }) &&
                   Throws([&] {
                       (void)api::Compiler::BuildGraphSemanticKey(
                           MakeDerivedCallFunction<
                               UnregisteredDerivedCallNode>());
                   }),
               "registered and unregistered derived Calls must fail closed");

    auto* disguised_call = new CallNode();
    disguised_call->op = relay::Op::Get("nn_relu");
    const Function disguised{ObjectRef(disguised_call)};
    auto* derived_function = new UnregisteredDerivedFunctionNode();
    derived_function->params = {input};
    derived_function->body = input;
    derived_function->hidden_semantics = "must-not-be-omitted";
    const Function derived_root{ObjectRef(derived_function)};
    TEST_CHECK(Throws([&] {
                   (void)api::Compiler::BuildGraphSemanticKey(disguised);
               }) &&
                   Throws([&] {
                       (void)api::Compiler::BuildGraphSemanticKey(
                           derived_root);
                   }),
               "the graph root must be an exact Function node");
    return true;
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"locator_is_not_semantics", TestGraphLocatorIsNotUnitSemantics},
        {"digest_collision_full_equality", TestDigestCollisionUsesCanonicalEquality},
        {"artifact_field_safe_miss", TestEveryArtifactSemanticFieldCausesSafeMiss},
        {"dispatch_and_plan_are_separate", TestDispatchAndPlanVariantRemainSeparate},
        {"graph_identity_rejects_undefined_exprs",
         TestGraphSemanticIdentityRejectsUndefinedExprs},
        {"graph_identity_exact_node_whitelist",
         TestGraphSemanticIdentityUsesExactNodeWhitelist},
    };
    int failures = 0;
    for (const auto& test : tests) {
        try {
            if (!test.second()) {
                ++failures;
                continue;
            }
            std::cout << "[PASS] " << test.first << "\n";
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << test.first << ": " << error.what() << "\n";
            ++failures;
        }
    }
    return failures == 0 ? 0 : 1;
}
