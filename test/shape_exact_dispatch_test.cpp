// 请求边界 exact dispatch 闭环锁定测试（docs/plans/2026-07-26-shape-exact-
// dispatch-closure.md Task 3）。
//
// 控制面就是本测试里的一个普通 map——不存在生产 VariantTable/Router；
// 编译意图只来自显式 Compiler::Compile；请求边界永不隐式编译。

#include <exception>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "../src/compiler/internal/primitive_cache.h"
#include "kxc/compiler/compiler.h"
#include "kxc/compiler/restricted_symbolic_shape.h"
#include "kxc/relay/op.h"
#include "kxc/runtime/session.h"

namespace {
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
    const auto prep = RestrictedSymbolicShapeAdapter::Prepare(
        ReluAddGraph(), Config(),
        {{0, 0, "n", 2, 8, 2}, {1, 0, "n", 2, 8, 2}});

    // route 就是调用方持有的普通 map；key 是 dispatch key 的 canonical 字节。
    std::map<std::string, kxc::api::CompiledGraph> route;

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
    const auto key8 =
        RestrictedSymbolicShapeAdapter::ExactDispatchKey(prep, d8);
    CHECK(route.find(key8.canonical_bytes()) == route.end(),
          "uncompiled profile must miss the route map");
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
            kxc::api::Compiler::Compile(materialized, Config());
        RestrictedSymbolicShapeAdapter::VerifyCompiledExactVariant(
            prep, decision, compiled);
        route.emplace(RestrictedSymbolicShapeAdapter::ExactDispatchKey(
                          prep, decision).canonical_bytes(),
                      compiled);
    }
    CHECK(route.size() == 3, "three profiles must occupy three distinct routes");

    // 请求边界：输入 shape → binding → 决策 → route 查找 → 静态执行。
    const auto dispatch = [&](const kxc::runtime::NDArray& x,
                              const kxc::runtime::NDArray& y,
                              size_t extent) -> std::vector<float> {
        const auto bindings = RestrictedSymbolicShapeAdapter::BindingsFromInputShapes(
            prep, {{static_cast<int64_t>(extent)}, {static_cast<int64_t>(extent)}});
        const auto decision =
            RestrictedSymbolicShapeAdapter::MintExact(prep, bindings);
        const auto found = route.find(
            RestrictedSymbolicShapeAdapter::ExactDispatchKey(
                prep, decision).canonical_bytes());
        if (found == route.end()) {
            throw std::runtime_error("no published variant for this profile");
        }
        return RunVariant(found->second, x, y, extent);
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
