// 请求边界 exact dispatch 闭环锁定测试。有限 route table 只接收调用方
// 显式编译并由 restricted exact adapter 验证过的产物；请求查找不编译。

#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "../src/compiler/internal/primitive_cache.h"
#include "kxc/compiler/compiler.h"
#include "kxc/compiler/restricted_symbolic_shape.h"
#include "kxc/compiler/shape_control.h"
#include "kxc/relay/op.h"
#include "kxc/runtime/session.h"

namespace {
namespace control = kxc::api::experimental::shape_control::v1;
namespace restricted = kxc::api::experimental::restricted_symbolic_shape::v1;
namespace shape = kxc::api::experimental::shape_specialization::v1;
using restricted::RestrictedSymbolicShapeAdapter;

#define CHECK(x, m) do { if (!(x)) { std::cerr << "[FAIL] " << __FUNCTION__ << ": " << m << "\n"; return false; } } while (0)

bool Throws(const std::function<void()>& fn) {
    try { fn(); } catch (const std::exception&) { return true; }
    return false;
}

kxc::api::CompileConfig Config() {
    return kxc::api::CompileConfig::Create(kxc::BuildTarget(kxc::Device::CPU()));
}

// representative N=4：relu(add(x, y))，两个输入共享 symbol n。
kxc::Function ReluAddGraph(int64_t extent = 4) {
    const kxc::TensorType type({extent}, "float32");
    const kxc::Var x("x", type), y("y", type);
    return kxc::Function({x, y}, kxc::Call(kxc::relay::Op::Get("nn_relu"), {
        kxc::Call(kxc::relay::Op::Get("add"), {x, y})}));
}

shape::BindingSet Bind(int64_t n) { return shape::BindingSet({{"n", n}}); }

kxc::api::PlanAbiFingerprint PlanAbi(
    const kxc::api::CompiledGraph& graph) {
    const auto calls = graph.plan().calls();
    std::vector<kxc::api::OrderedArtifactIdentity> artifacts;
    artifacts.reserve(calls.size());
    for (size_t index = 0; index < calls.size(); ++index) {
        artifacts.push_back(
            {index, std::string(calls[index]->symbol),
             graph.artifact_pins()[index].record().artifact_key});
    }
    return kxc::api::BuildPlanAbiFingerprint(
        graph.module(), graph.plan(), artifacts);
}

#if KXC_USE_LLVM && KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE
kxc::runtime::NDArray FloatArray(const std::vector<float>& values) {
    kxc::runtime::NDArray result = kxc::runtime::NDArray::Zeros(
        {static_cast<int64_t>(values.size())},
        kxc::runtime::DataTypeFromString("float32"), kxc::Device::CPU());
    result.CopyFromBytes(values.data(), values.size() * sizeof(float));
    return result;
}

std::vector<float> RunVariant(const kxc::api::CompiledGraph& graph,
                              const kxc::runtime::NDArray& x,
                              const kxc::runtime::NDArray& y,
                              size_t extent) {
    kxc::runtime::RuntimeSession session(graph.module(), graph.plan());
    const auto outputs = session.Run({x, y});
    std::vector<float> actual(extent);
    outputs[0].CopyToBytes(actual.data(), actual.size() * sizeof(float));
    return actual;
}
#endif

bool TestRequestBoundaryDispatchLoop() {
#if !KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE || !KXC_ENABLE_SHAPE_PRODUCTION_EXACT
    return true;
#else
    const kxc::api::CompileConfig config = Config();
    const auto prep = RestrictedSymbolicShapeAdapter::Prepare(
        ReluAddGraph(), config,
        {{0, 0, "n", 2, 8, 2}, {1, 0, "n", 2, 8, 2}});
    control::ExactProfileRouteTable route(
        prep.graph_template(),
        kxc::api::internal::BuildTargetCapabilityFingerprint(config->target));

    // fail closed：越界与不整除在 MintExact 处拒绝，换算处拒绝静态轴错配。
    CHECK(Throws([&] { (void)RestrictedSymbolicShapeAdapter::MintExact(prep, Bind(3)); }),
          "non-divisible extent must fail closed at minting");
    CHECK(Throws([&] { (void)RestrictedSymbolicShapeAdapter::MintExact(prep, Bind(10)); }),
          "above-upper extent must fail closed at minting");
    CHECK(Throws([&] { (void)RestrictedSymbolicShapeAdapter::BindingsFromInputShapes(
              prep, {{4}, {6}}); }),
          "conflicting shared-symbol inputs must fail closed at conversion");

    // 合法但未编译的 profile：查找 miss 必须明确失败且无隐式编译。
    kxc::api::internal::ClearPrimitiveCacheForTesting();
    const auto stats_before = kxc::api::internal::GetPrimitiveCacheStats();
    const auto d8 = RestrictedSymbolicShapeAdapter::MintExact(prep, Bind(8));
    CHECK(Throws([&] { (void)route.Lookup(d8.exact_oracle()); }),
          "uncompiled profile must miss the route table");
    const auto stats_after = kxc::api::internal::GetPrimitiveCacheStats();
    CHECK(stats_after.misses == stats_before.misses &&
              stats_after.in_flight == stats_before.in_flight &&
              stats_after.entries == stats_before.entries,
          "request-boundary minting and lookup must not compile anything");

#if KXC_USE_LLVM
    // 编译期：每个 profile 一次，编译意图显式来自用户 Compiler::Compile。
    for (const int64_t n : {int64_t{2}, int64_t{4}, int64_t{6}}) {
        const auto decision =
            RestrictedSymbolicShapeAdapter::MintExact(prep, Bind(n));
        const kxc::Function materialized =
            RestrictedSymbolicShapeAdapter::MaterializeExactFunction(prep, decision);
        const kxc::api::CompiledGraph compiled =
            kxc::api::Compiler::Compile(materialized, config);
        RestrictedSymbolicShapeAdapter::VerifyCompiledExactVariant(
            prep, decision, compiled);
        route.Publish(decision.exact_oracle(), compiled, PlanAbi(compiled));
    }
    CHECK(route.size() == 3, "three profiles must occupy three distinct routes");

    // 请求边界：named concrete shapes → binding → exact decision → lookup。
    const auto dispatch = [&](const kxc::runtime::NDArray& x,
                              const kxc::runtime::NDArray& y,
                              size_t extent) -> std::vector<float> {
        std::vector<control::ConcreteInputShape> inputs;
        for (const auto& input :
             prep.graph_template().shape_program().inputs()) {
            inputs.push_back(
                {input.name, {static_cast<int64_t>(extent)}});
        }
        const auto bindings = control::BindExactInputShapes(
            prep.graph_template(), inputs);
        const auto decision =
            RestrictedSymbolicShapeAdapter::MintExact(prep, bindings);
        const control::PublishedExactVariant found =
            route.Lookup(decision.exact_oracle());
        if (found.dispatch_key() !=
            RestrictedSymbolicShapeAdapter::ExactDispatchKey(prep, decision)) {
            throw std::runtime_error("route table returned a foreign dispatch");
        }
        return RunVariant(found.compiled_graph(), x, y, extent);
    };

    CHECK(dispatch(FloatArray({-3, -2, -1, 0, 1, 2}),
                   FloatArray({1, 1, 1, 1, 1, 1}), 6) ==
              std::vector<float>({0, 0, 0, 1, 2, 3}),
          "N=6 input must route to the N=6 variant and execute correctly");
    CHECK(dispatch(FloatArray({-1, 2}), FloatArray({1, 1}), 2) ==
              std::vector<float>({0, 3}),
          "N=2 input must route to the N=2 variant and execute correctly");

    // N=8 合法但未发布 → 明确失败，且仍无隐式编译。
    const auto stats_before_miss = kxc::api::internal::GetPrimitiveCacheStats();
    CHECK(Throws([&] { (void)dispatch(FloatArray({1, 1, 1, 1, 1, 1, 1, 1}),
                                      FloatArray({1, 1, 1, 1, 1, 1, 1, 1}), 8); }),
          "unpublished profile must fail closed at the request boundary");
    const auto stats_after_miss = kxc::api::internal::GetPrimitiveCacheStats();
    CHECK(stats_after_miss.misses == stats_before_miss.misses &&
              stats_after_miss.in_flight == stats_before_miss.in_flight,
          "a route miss must not trigger implicit compilation");
#endif
    return true;
#endif
}

}  // namespace

int main() {
    int failed = 0;
    try {
        if (TestRequestBoundaryDispatchLoop()) {
            std::cout << "[PASS] request_boundary_dispatch_loop\n";
        } else {
            ++failed;
        }
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] request_boundary_dispatch_loop: " << error.what() << "\n";
        ++failed;
    }
    return failed == 0 ? 0 : 1;
}
