#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../src/compiler/internal/primitive_cache.h"
#include "../src/runtime/internal/memory_plan.h"
#include "kxc/compiler/compiler.h"
#include "kxc/compiler/restricted_symbolic_shape.h"
#include "kxc/compiler/shape_control.h"
#include "kxc/relay/op.h"
#include "kxc/runtime/session.h"

namespace {
namespace control =
    kxc::api::experimental::shape_control::v1;
namespace restricted =
    kxc::api::experimental::restricted_symbolic_shape::v1;
namespace spec =
    kxc::api::experimental::shape_specialization::v1;

#define CHECK(condition, message)                                               \
    do {                                                                         \
        if (!(condition)) {                                                      \
            std::cerr << "[FAIL] " << __FUNCTION__ << ": " << (message)       \
                      << "\n";                                                   \
            return false;                                                        \
        }                                                                        \
    } while (0)

bool Throws(const std::function<void()>& action) {
    try {
        action();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

kxc::api::CompileConfig Config() {
    return kxc::api::CompileConfig::Create(
        kxc::BuildTarget(kxc::Device::CPU()));
}

kxc::Function ReluAddGraph(int64_t extent = 4) {
    const kxc::TensorType type({extent}, "float32");
    const kxc::Var x("x", type);
    const kxc::Var y("y", type);
    return kxc::Function(
        {x, y},
        kxc::Call(kxc::relay::Op::Get("nn_relu"),
                  {kxc::Call(kxc::relay::Op::Get("add"), {x, y})}));
}

kxc::Function SqrtAddGraph(int64_t extent) {
    const kxc::TensorType type({extent}, "float32");
    const kxc::Var x("x", type);
    const kxc::Var y("y", type);
    return kxc::Function(
        {x, y},
        kxc::Call(kxc::relay::Op::Get("sqrt"),
                  {kxc::Call(kxc::relay::Op::Get("add"), {x, y})}));
}

spec::TensorShapeContract ExactContract(
    const std::vector<spec::DimExpr>& dimensions) {
    return spec::TensorShapeContract(
        spec::LogicalShape(dimensions),
        spec::PhysicalCapacity(dimensions), spec::ValidExtent(dimensions));
}

spec::GraphTemplate BindingTemplate(
    std::string locator = "binding.call") {
    const spec::DimExpr n = spec::DimExpr::Symbol("n");
    const auto left = ExactContract({n, spec::DimExpr::Const(3)});
    const auto right = ExactContract({spec::DimExpr::Const(2), n});
    const auto result = ExactContract({n, n});
    return spec::GraphTemplate(
        kxc::api::Compiler::BuildGraphSemanticKey(ReluAddGraph()),
        spec::ShapeProgram(
            {"n"}, {{"left", left}, {"right", right}},
            {{"result", result}},
            {spec::Constraint::Range(n, 2, 8),
             spec::Constraint::DivisibleBy(n, 2)}),
        {{spec::GraphLocalCallLocator(std::move(locator)),
          kxc::api::UnitSemanticKey("shape-control.binding-unit.v1"),
          {"left", "right"}, {"result"}}});
}

spec::GraphTemplate ArithmeticInputTemplate() {
    const spec::DimExpr n = spec::DimExpr::Symbol("n");
    const spec::DimExpr n_plus_one = spec::DimExpr::Add(
        {n, spec::DimExpr::Const(1)});
    const auto contract = ExactContract({n_plus_one});
    return spec::GraphTemplate(
        kxc::api::Compiler::BuildGraphSemanticKey(ReluAddGraph()),
        spec::ShapeProgram({"n"}, {{"source", contract}},
                           {{"result", contract}}),
        {{spec::GraphLocalCallLocator("arithmetic.call"),
          kxc::api::UnitSemanticKey("shape-control.arithmetic-unit.v1"),
          {"source"}, {"result"}}});
}

spec::GraphTemplate NonExactInputTemplate() {
    const spec::DimExpr n = spec::DimExpr::Symbol("n");
    const spec::TensorShapeContract input(
        spec::LogicalShape({n}),
        spec::PhysicalCapacity({spec::DimExpr::Const(8)}),
        spec::ValidExtent({n}));
    return spec::GraphTemplate(
        kxc::api::Compiler::BuildGraphSemanticKey(ReluAddGraph()),
        spec::ShapeProgram(
            {"n"}, {{"source", input}},
            {{"result", ExactContract({n})}},
            {spec::Constraint::Range(n, 0, 8)}),
        {{spec::GraphLocalCallLocator("nonexact.call"),
          kxc::api::UnitSemanticKey("shape-control.nonexact-unit.v1"),
          {"source"}, {"result"}}});
}

spec::GraphTemplate NonExactOutputTemplate() {
    const spec::DimExpr n = spec::DimExpr::Symbol("n");
    const spec::TensorShapeContract output(
        spec::LogicalShape({n}),
        spec::PhysicalCapacity({spec::DimExpr::Const(8)}),
        spec::ValidExtent({n}));
    return spec::GraphTemplate(
        kxc::api::Compiler::BuildGraphSemanticKey(ReluAddGraph()),
        spec::ShapeProgram(
            {"n"}, {{"source", ExactContract({n})}},
            {{"result", output}},
            {spec::Constraint::Range(n, 0, 8)}),
        {{spec::GraphLocalCallLocator("nonexact-output.call"),
          kxc::api::UnitSemanticKey("shape-control.nonexact-output-unit.v1"),
          {"source"}, {"result"}}});
}

spec::BindingSet BindFour(const spec::GraphTemplate& graph) {
    return control::BindExactInputShapes(
        graph, {{"right", {2, 4}}, {"left", {4, 3}}});
}

bool TestNamedExactBinding() {
    const spec::GraphTemplate graph = BindingTemplate();
    const spec::BindingSet bindings = BindFour(graph);
    CHECK(bindings.bindings().size() == 1 &&
              bindings.Find("n") == std::optional<int64_t>(4),
          "order-independent named inputs must bind the shared symbol once");

    const spec::ExactOracle oracle =
        spec::InstantiateExactProfile(graph, bindings);
    CHECK(oracle.profile().Value("left").contract.logical ==
              std::vector<int64_t>({4, 3}) &&
              oracle.profile().Value("right").contract.logical ==
                  std::vector<int64_t>({2, 4}) &&
              oracle.profile().Value("result").contract.logical ==
                  std::vector<int64_t>({4, 4}),
          "the binding must be evaluated by the complete ShapeProgram");
    return true;
}

bool TestBindingFailures() {
    const spec::GraphTemplate graph = BindingTemplate();
    CHECK(Throws([&] {
              (void)control::BindExactInputShapes(
                  graph, {{"left", {4, 3}}});
          }),
          "input count mismatch must fail closed");
    CHECK(Throws([&] {
              (void)control::BindExactInputShapes(
                  graph, {{"left", {4, 3}}, {"left", {4, 3}}});
          }),
          "duplicate names must fail closed");
    CHECK(Throws([&] {
              (void)control::BindExactInputShapes(
                  graph, {{"", {4, 3}}, {"right", {2, 4}}});
          }),
          "empty names must fail closed");
    CHECK(Throws([&] {
              (void)control::BindExactInputShapes(
                  graph, {{"left", {4, 3}}, {"unknown", {2, 4}}});
          }),
          "unknown or missing names must fail closed");
    CHECK(Throws([&] {
              (void)control::BindExactInputShapes(
                  graph, {{"left", {-1, 3}}, {"right", {2, -1}}});
          }),
          "negative extents must fail closed");
    CHECK(Throws([&] {
              (void)control::BindExactInputShapes(
                  graph, {{"left", {4}}, {"right", {2, 4}}});
          }),
          "rank mismatch must fail closed");
    CHECK(Throws([&] {
              (void)control::BindExactInputShapes(
                  graph, {{"left", {4, 5}}, {"right", {2, 4}}});
          }),
          "a static-axis mismatch must fail closed");
    CHECK(Throws([&] {
              (void)control::BindExactInputShapes(
                  graph, {{"left", {4, 3}}, {"right", {2, 6}}});
          }),
          "shared symbolic axes must agree");
    CHECK(Throws([&] {
              (void)control::BindExactInputShapes(
                  graph, {{"left", {3, 3}}, {"right", {2, 3}}});
          }),
          "divisibility constraints must be evaluated");
    CHECK(Throws([&] {
              (void)control::BindExactInputShapes(
                  graph, {{"left", {10, 3}}, {"right", {2, 10}}});
          }),
          "range constraints must be evaluated");

    const spec::GraphTemplate arithmetic = ArithmeticInputTemplate();
    CHECK(Throws([&] {
              (void)control::BindExactInputShapes(
                  arithmetic, {{"source", {5}}});
          }),
          "arithmetic input expressions must not be inverted");
    const spec::GraphTemplate nonexact = NonExactInputTemplate();
    const spec::GraphTemplate nonexact_output = NonExactOutputTemplate();
    CHECK(Throws([&] {
              (void)control::BindExactInputShapes(
                  nonexact, {{"source", {4}}});
          }) &&
              Throws([&] {
                  (void)control::BindExactInputShapes(
                      nonexact_output, {{"source", {4}}});
              }),
          "input or output logical/physical/valid disagreement must fail closed");
    return true;
}

bool TestPureRouteFailures() {
    const spec::GraphTemplate graph = BindingTemplate();
    const spec::ExactOracle oracle =
        spec::InstantiateExactProfile(graph, BindFour(graph));
    control::ExactProfileRouteTable routes(graph, "synthetic-cpu-target");
    CHECK(routes.size() == 0, "a route table must begin empty");
    CHECK(Throws([&] { (void)routes.Lookup(oracle); }),
          "an unpublished exact profile must miss without fallback");

    const spec::GraphTemplate changed = BindingTemplate("changed.call");
    const spec::ExactOracle foreign =
        spec::InstantiateExactProfile(changed, BindFour(changed));
    CHECK(graph.key() == changed.key() &&
              graph.CanonicalBytes() != changed.CanonicalBytes(),
          "the foreign-oracle fixture must preserve graph semantics only");
    CHECK(Throws([&] { (void)routes.Lookup(foreign); }),
          "an oracle from different template content must be rejected");
    CHECK(Throws([&] {
              routes.Publish(foreign, kxc::api::CompiledGraph(),
                             kxc::api::PlanAbiFingerprint());
          }),
          "foreign template content must fail before publication metadata");

    const spec::GraphTemplate wrong_graph(
        kxc::api::Compiler::BuildGraphSemanticKey(SqrtAddGraph(4)),
        graph.shape_program(), graph.ordered_units());
    const spec::ExactOracle wrong_graph_oracle =
        spec::InstantiateExactProfile(wrong_graph, BindFour(wrong_graph));
    CHECK(Throws([&] { (void)routes.Lookup(wrong_graph_oracle); }),
          "a profile from different graph semantics must be rejected");
    CHECK(Throws([&] {
              control::ExactProfileRouteTable invalid(graph, "");
              (void)invalid;
          }),
          "an empty target fingerprint must be rejected");
    return true;
}

#if KXC_USE_LLVM && KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE

struct CompiledProfile final {
    restricted::RestrictedDispatchDecision decision;
    kxc::api::CompiledGraph compiled;
};

std::vector<int64_t> ShapeVector(const kxc::Array<int64_t>& shape) {
    std::vector<int64_t> result;
    result.reserve(shape.size());
    for (const int64_t extent : shape) {
        result.push_back(extent);
    }
    return result;
}

std::vector<control::ConcreteInputShape> NamedInputShapes(
    const spec::GraphTemplate& graph,
    const std::vector<kxc::runtime::NDArray>& inputs) {
    if (graph.shape_program().inputs().size() != inputs.size()) {
        throw std::invalid_argument("test input arity differs from template");
    }
    std::vector<control::ConcreteInputShape> result;
    result.reserve(inputs.size());
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        result.push_back({graph.shape_program().inputs()[index].name,
                          ShapeVector(inputs[index].shape())});
    }
    std::reverse(result.begin(), result.end());
    return result;
}

spec::BindingSet BindExtent(
    const restricted::PreparedRestrictedSymbolicTemplate& prepared,
    int64_t extent) {
    const auto& inputs = prepared.graph_template().shape_program().inputs();
    if (inputs.size() != 2) {
        throw std::runtime_error("restricted fixture must have two inputs");
    }
    return control::BindExactInputShapes(
        prepared.graph_template(),
        {{inputs[1].name, {extent}}, {inputs[0].name, {extent}}});
}

CompiledProfile CompileProfile(
    const restricted::PreparedRestrictedSymbolicTemplate& prepared,
    int64_t extent) {
    restricted::RestrictedDispatchDecision decision =
        restricted::RestrictedSymbolicShapeAdapter::MintExact(
            prepared, BindExtent(prepared, extent));
    const kxc::Function materialized =
        restricted::RestrictedSymbolicShapeAdapter::MaterializeExactFunction(
            prepared, decision);
    kxc::api::CompiledGraph compiled =
        kxc::api::Compiler::Compile(materialized, Config());
    restricted::RestrictedSymbolicShapeAdapter::VerifyCompiledExactVariant(
        prepared, decision, compiled);
    return {std::move(decision), std::move(compiled)};
}

std::vector<kxc::api::OrderedArtifactIdentity> OrderedArtifacts(
    const kxc::api::CompiledGraph& compiled) {
    const auto calls = compiled.plan().calls();
    const auto& pins = compiled.artifact_pins();
    if (calls.empty() || calls.size() != pins.size()) {
        throw std::invalid_argument("compiled fixture has incomplete artifacts");
    }
    std::vector<kxc::api::OrderedArtifactIdentity> artifacts;
    artifacts.reserve(calls.size());
    for (std::size_t index = 0; index < calls.size(); ++index) {
        artifacts.push_back(
            {index, std::string(calls[index]->symbol),
             pins[index].record().artifact_key});
    }
    return artifacts;
}

kxc::api::PlanAbiFingerprint PlanAbi(
    const kxc::api::CompiledGraph& compiled) {
    return kxc::api::BuildPlanAbiFingerprint(
        compiled.module(), compiled.plan(), OrderedArtifacts(compiled));
}

kxc::api::PlanVariantKey ExpectedPlanVariant(
    const spec::GraphTemplate& graph, const spec::ExactOracle& oracle,
    const kxc::api::CompiledGraph& compiled) {
    const auto artifacts = OrderedArtifacts(compiled);
    std::vector<kxc::api::OrderedArtifactSelectionIdentity> selections;
    selections.reserve(artifacts.size());
    for (const auto& artifact : artifacts) {
        selections.push_back({artifact.call_index, artifact.link_symbol,
                              artifact.artifact_key, 0});
    }
    return kxc::api::BuildPlanVariantKey(
        graph.key(), oracle.profile().key(), selections,
        kxc::runtime::internal::kStaticMemoryPlanVersion);
}

std::string TargetFingerprint(const kxc::api::CompiledGraph& compiled) {
    const auto artifacts = OrderedArtifacts(compiled);
    const std::string fingerprint =
        artifacts.front().artifact_key.target_capability_fingerprint();
    for (const auto& artifact : artifacts) {
        if (artifact.artifact_key.target_capability_fingerprint() !=
            fingerprint) {
            throw std::invalid_argument(
                "compiled fixture spans target fingerprints");
        }
    }
    return fingerprint;
}

kxc::runtime::NDArray FloatArray(const std::vector<float>& values) {
    kxc::runtime::NDArray result = kxc::runtime::NDArray::Zeros(
        {static_cast<int64_t>(values.size())},
        kxc::runtime::DataTypeFromString("float32"), kxc::Device::CPU());
    result.CopyFromBytes(values.data(), values.size() * sizeof(float));
    return result;
}

std::vector<float> RunVariant(
    const control::PublishedExactVariant& variant,
    const std::vector<kxc::runtime::NDArray>& inputs,
    std::size_t output_size) {
    kxc::runtime::RuntimeSession session(variant.compiled_graph().module(),
                                         variant.compiled_graph().plan());
    kxc::Array<kxc::runtime::NDArray> arguments;
    for (const auto& input : inputs) {
        arguments.push_back(input);
    }
    const auto outputs = session.Run(arguments);
    if (outputs.size() != 1) {
        throw std::runtime_error("compiled fixture must produce one output");
    }
    std::vector<float> result(output_size);
    outputs[0].CopyToBytes(result.data(), result.size() * sizeof(float));
    return result;
}

std::vector<float> DispatchAndRun(
    const control::ExactProfileRouteTable& routes,
    const restricted::PreparedRestrictedSymbolicTemplate& prepared,
    const std::vector<kxc::runtime::NDArray>& inputs) {
    const spec::BindingSet bindings = control::BindExactInputShapes(
        prepared.graph_template(),
        NamedInputShapes(prepared.graph_template(), inputs));
    const restricted::RestrictedDispatchDecision decision =
        restricted::RestrictedSymbolicShapeAdapter::MintExact(prepared,
                                                               bindings);
    const control::PublishedExactVariant variant =
        routes.Lookup(decision.exact_oracle());
    return RunVariant(variant, inputs, inputs[0].shape()[0]);
}

bool SameControlPlaneCacheState(
    const kxc::api::internal::PrimitiveCacheStats& left,
    const kxc::api::internal::PrimitiveCacheStats& right) {
    return left.hits == right.hits && left.misses == right.misses &&
           left.entries == right.entries &&
           left.accounted_bytes == right.accounted_bytes &&
           left.evictions == right.evictions &&
           left.in_flight == right.in_flight &&
           left.merged_waiters == right.merged_waiters &&
           left.failures == right.failures &&
           left.rejections == right.rejections;
}

spec::GraphTemplate AddUnusedProfileSymbol(
    const spec::GraphTemplate& graph) {
    std::vector<std::string> symbols =
        graph.shape_program().declared_symbols();
    symbols.push_back("unused");
    std::vector<spec::Constraint> constraints =
        graph.shape_program().constraints();
    constraints.push_back(spec::Constraint::Range(
        spec::DimExpr::Symbol("unused"), 0, 1));
    return spec::GraphTemplate(
        graph.key(),
        spec::ShapeProgram(std::move(symbols),
                           graph.shape_program().inputs(),
                           graph.shape_program().outputs(),
                           std::move(constraints)),
        graph.ordered_units());
}

spec::ExactOracle OracleWithUnused(
    const spec::GraphTemplate& graph, const spec::ExactOracle& source,
    int64_t unused) {
    std::vector<spec::Binding> bindings =
        source.profile().bindings().bindings();
    bindings.push_back({"unused", unused});
    return spec::InstantiateExactProfile(
        graph, spec::BindingSet(std::move(bindings)));
}

bool TestCompiledFiniteConsumer() {
    CHECK(restricted::RestrictedSymbolicShapeAdapter::IsEnabled(),
          "the compiled consumer requires the restricted-shape gate");
    const auto prepared =
        restricted::RestrictedSymbolicShapeAdapter::Prepare(
            ReluAddGraph(), Config(),
            {{0, 0, "n", 2, 8, 2}, {1, 0, "n", 2, 8, 2}});

    kxc::api::internal::ClearPrimitiveCacheForTesting();
    CompiledProfile small = CompileProfile(prepared, 2);
    CompiledProfile large = CompileProfile(prepared, 4);
    kxc::api::CompiledGraph wrong_semantics =
        kxc::api::Compiler::Compile(SqrtAddGraph(2), Config());
    CHECK(Throws([&] {
              restricted::RestrictedSymbolicShapeAdapter::
                  VerifyCompiledExactVariant(prepared, small.decision,
                                             wrong_semantics);
          }),
          "the producer validator must reject same-boundary wrong semantics");

    const std::string fingerprint = TargetFingerprint(small.compiled);
    CHECK(fingerprint == TargetFingerprint(large.compiled),
          "finite variants must share one primitive target fingerprint");
    const kxc::api::PlanAbiFingerprint small_abi = PlanAbi(small.compiled);
    const kxc::api::PlanAbiFingerprint large_abi = PlanAbi(large.compiled);
    CHECK(small_abi != large_abi,
          "different exact shapes must have distinct plan ABIs");

    control::ExactProfileRouteTable routes(prepared.graph_template(),
                                            fingerprint);
    const auto cache_before_publish =
        kxc::api::internal::GetPrimitiveCacheStats();
    routes.Publish(small.decision.exact_oracle(), small.compiled, small_abi);
    routes.Publish(large.decision.exact_oracle(), large.compiled, large_abi);
    const auto cache_after_publish =
        kxc::api::internal::GetPrimitiveCacheStats();
    CHECK(SameControlPlaneCacheState(cache_before_publish,
                                     cache_after_publish),
          "publication must not compile or query the primitive cache");
    CHECK(routes.size() == 2,
          "two exact profiles must produce a finite two-route table");

    const control::PublishedExactVariant selected =
        routes.Lookup(small.decision.exact_oracle());
    CHECK(selected.compiled_graph().defined() &&
              selected.shape_profile_key() ==
                  small.decision.exact_oracle().profile().key() &&
              selected.dispatch_key() ==
                  restricted::RestrictedSymbolicShapeAdapter::
                      ExactDispatchKey(prepared, small.decision) &&
              selected.plan_abi() == small_abi &&
              selected.plan_variant_key() == ExpectedPlanVariant(
                  prepared.graph_template(),
                  small.decision.exact_oracle(), small.compiled),
          "lookup must expose only canonical existing identities");

    CHECK(Throws([&] {
              routes.Publish(small.decision.exact_oracle(), small.compiled,
                             small_abi);
          }),
          "duplicate exact routes must be rejected");

    const spec::GraphTemplate overlap_graph =
        AddUnusedProfileSymbol(prepared.graph_template());
    const spec::ExactOracle overlap_zero = OracleWithUnused(
        overlap_graph, small.decision.exact_oracle(), 0);
    const spec::ExactOracle overlap_one = OracleWithUnused(
        overlap_graph, small.decision.exact_oracle(), 1);
    CHECK(overlap_zero.profile().key() != overlap_one.profile().key(),
          "overlap fixtures must have distinct formal profile keys");
    control::ExactProfileRouteTable overlap_routes(overlap_graph,
                                                    fingerprint);
    overlap_routes.Publish(overlap_zero, small.compiled, small_abi);
    CHECK(Throws([&] {
              overlap_routes.Publish(overlap_one, small.compiled, small_abi);
          }),
          "different keys with identical concrete inputs must overlap");

    control::ExactProfileRouteTable wrong_target(
        prepared.graph_template(), fingerprint + ".other");
    CHECK(Throws([&] {
              wrong_target.Publish(small.decision.exact_oracle(),
                                   small.compiled, small_abi);
          }),
          "target-incompatible artifacts must be rejected");

    control::ExactProfileRouteTable wrong_abi(prepared.graph_template(),
                                               fingerprint);
    CHECK(Throws([&] {
              wrong_abi.Publish(small.decision.exact_oracle(),
                                small.compiled, large_abi);
          }),
          "publisher-supplied plan ABI disagreement must be rejected");
    CHECK(Throws([&] {
              control::ExactProfileRouteTable undefined(
                  prepared.graph_template(), fingerprint);
              undefined.Publish(small.decision.exact_oracle(),
                                kxc::api::CompiledGraph(), small_abi);
          }),
          "an undefined compiled graph must be rejected");
    CHECK(Throws([&] {
              control::ExactProfileRouteTable undefined_abi(
                  prepared.graph_template(), fingerprint);
              undefined_abi.Publish(small.decision.exact_oracle(),
                                    small.compiled,
                                    kxc::api::PlanAbiFingerprint());
          }),
          "an undefined expected plan ABI must be rejected");

    const control::PublishedExactVariant detached = [&] {
        control::ExactProfileRouteTable one(prepared.graph_template(),
                                             fingerprint);
        one.Publish(small.decision.exact_oracle(), small.compiled, small_abi);
        return one.Lookup(small.decision.exact_oracle());
    }();
    small.compiled = kxc::api::CompiledGraph();
    large.compiled = kxc::api::CompiledGraph();
    wrong_semantics = kxc::api::CompiledGraph();
    kxc::api::internal::ClearPrimitiveCacheForTesting();

    CHECK(RunVariant(detached,
                     {FloatArray({-1.0F, 2.0F}),
                      FloatArray({1.0F, 1.0F})},
                     2) == std::vector<float>({0.0F, 3.0F}),
          "a looked-up value must retain executable pins after table/cache loss");
    CHECK(DispatchAndRun(routes,
                         prepared,
                         {FloatArray({-4.0F, -1.0F, 2.0F, 3.0F}),
                          FloatArray({1.0F, 1.0F, 1.0F, -5.0F})}) ==
              std::vector<float>({0.0F, 0.0F, 3.0F, 0.0F}),
          "real named input shapes must route to and execute the N=4 variant");

    const auto cache_before_miss =
        kxc::api::internal::GetPrimitiveCacheStats();
    CHECK(Throws([&] {
              (void)DispatchAndRun(
                  routes, prepared,
                  {FloatArray({1, 1, 1, 1, 1, 1}),
                   FloatArray({1, 1, 1, 1, 1, 1})});
          }),
          "a valid but unpublished N=6 profile must miss without fallback");
    const auto cache_after_miss =
        kxc::api::internal::GetPrimitiveCacheStats();
    CHECK(SameControlPlaneCacheState(cache_before_miss, cache_after_miss),
          "a route miss must not compile or query the primitive cache");
    return true;
}

#endif

}  // namespace

int main() {
    std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"named_exact_binding", TestNamedExactBinding},
        {"binding_failures", TestBindingFailures},
        {"pure_route_failures", TestPureRouteFailures},
    };
#if KXC_USE_LLVM && KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE
    tests.push_back({"compiled_finite_consumer", TestCompiledFiniteConsumer});
#else
    std::cout << "[SKIP] compiled_finite_consumer requires LLVM and "
                 "KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE\n";
#endif

    int failures = 0;
    for (const auto& [name, test] : tests) {
        try {
            if (test()) {
                std::cout << "[PASS] " << name << "\n";
            } else {
                ++failures;
            }
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << name << ": " << error.what() << "\n";
            ++failures;
        }
    }
    return failures == 0 ? 0 : 1;
}
