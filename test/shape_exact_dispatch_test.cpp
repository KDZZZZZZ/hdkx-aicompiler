// Exact-specializing controller E2E. Acquire is pure lookup; only the explicit
// CompileAndPublish operation may enter Compiler::Compile.

#include <algorithm>
#include <exception>
#include <functional>
#include <future>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "../src/compiler/internal/primitive_cache.h"
#include "kxc/compiler/shape_specialization_controller.h"
#include "kxc/relay/op.h"
#include "kxc/runtime/session.h"

namespace {
namespace control = kxc::api::experimental::shape_control::v1;
namespace restricted =
    kxc::api::experimental::restricted_symbolic_shape::v1;

#define CHECK(x, m) do { if (!(x)) { std::cerr << "[FAIL] " << __FUNCTION__ << ": " << m << "\n"; return false; } } while (0)

bool Throws(const std::function<void()>& fn) {
    try { fn(); } catch (const std::exception&) { return true; }
    return false;
}

kxc::api::CompileConfig Config() {
    return kxc::api::CompileConfig::Create(
        kxc::BuildTarget(kxc::Device::CPU()));
}

// representative N=4: relu(add(x, y)); both inputs share n.
kxc::Function ReluAddGraph() {
    const kxc::TensorType type({4}, "float32");
    const kxc::Var x("x", type), y("y", type);
    return kxc::Function(
        {x, y},
        kxc::Call(kxc::relay::Op::Get("nn_relu"),
                  {kxc::Call(kxc::relay::Op::Get("add"), {x, y})}));
}

std::vector<std::vector<int64_t>> Shapes(int64_t extent) {
    return {{extent}, {extent}};
}

bool SameCacheState(
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

#if KXC_USE_LLVM && KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE
kxc::runtime::NDArray FloatArray(const std::vector<float>& values) {
    kxc::runtime::NDArray result = kxc::runtime::NDArray::Zeros(
        {static_cast<int64_t>(values.size())},
        kxc::runtime::DataTypeFromString("float32"), kxc::Device::CPU());
    result.CopyFromBytes(values.data(), values.size() * sizeof(float));
    return result;
}

std::vector<float> RunVariant(
    const control::PublishedExactVariant& variant,
    const std::vector<float>& x, const std::vector<float>& y) {
    kxc::runtime::RuntimeSession session(variant.compiled_graph().module(),
                                         variant.compiled_graph().plan());
    const auto outputs = session.Run({FloatArray(x), FloatArray(y)});
    std::vector<float> actual(x.size());
    outputs[0].CopyToBytes(actual.data(), actual.size() * sizeof(float));
    return actual;
}
#endif

bool TestShapeSpecializationController() {
#if !KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE || !KXC_ENABLE_SHAPE_PRODUCTION_EXACT
    return true;
#else
    kxc::api::internal::ClearPrimitiveCacheForTesting();
    const kxc::api::CompileConfig config = Config();
    auto prepared = restricted::RestrictedSymbolicShapeAdapter::Prepare(
        ReluAddGraph(), config,
        {{0, 0, "n", 2, 8, 2}, {1, 0, "n", 2, 8, 2}});
    CHECK(Throws([&] {
              control::ShapeSpecializationController invalid(
                  prepared, config, 0);
              (void)invalid;
          }),
          "max_profiles must be positive");
    control::ShapeSpecializationController controller(
        std::move(prepared), config, 2);

    CHECK(Throws([&] { (void)controller.Acquire({{4}, {6}}); }),
          "shared-symbol disagreement must fail before lookup");
    CHECK(Throws([&] { (void)controller.Acquire({{4, 1}, {4}}); }),
          "invalid rank must fail before lookup");
    CHECK(Throws([&] { (void)controller.Acquire(Shapes(3)); }) &&
              Throws([&] { (void)controller.Acquire(Shapes(10)); }),
          "divisibility and bound violations must fail before lookup");

    const auto cache_before_miss =
        kxc::api::internal::GetPrimitiveCacheStats();
    CHECK(!controller.Acquire(Shapes(8)).has_value(),
          "a valid unpublished profile must be a pure miss");
    const auto cache_after_miss =
        kxc::api::internal::GetPrimitiveCacheStats();
    CHECK(SameCacheState(cache_before_miss, cache_after_miss),
          "Acquire miss must not touch the primitive cache");

#if KXC_USE_LLVM
    std::promise<void> start_signal;
    const std::shared_future<void> start =
        start_signal.get_future().share();
    std::vector<std::future<control::PublishedExactVariant>> futures;
    for (int index = 0; index < 8; ++index) {
        futures.push_back(std::async(
            std::launch::async, [&controller, start] {
                start.wait();
                return controller.CompileAndPublish(Shapes(2));
            }));
    }
    start_signal.set_value();

    std::vector<control::PublishedExactVariant> concurrent;
    for (auto& future : futures) concurrent.push_back(future.get());
    CHECK(concurrent.size() == 8 &&
              std::all_of(concurrent.begin(), concurrent.end(),
                          [&](const auto& variant) {
                              return variant.dispatch_key() ==
                                         concurrent.front().dispatch_key() &&
                                     variant.plan_abi() ==
                                         concurrent.front().plan_abi();
                          }),
          "concurrent callers for one profile must share one publication");
    const auto cache_after_concurrent =
        kxc::api::internal::GetPrimitiveCacheStats();
    CHECK(cache_after_concurrent.misses == cache_after_miss.misses + 2 &&
              cache_after_concurrent.hits == cache_after_miss.hits,
          "double-check locking must compile the two-unit profile only once");

    const control::PublishedExactVariant large =
        controller.CompileAndPublish(Shapes(6));
    const auto cache_after_profiles =
        kxc::api::internal::GetPrimitiveCacheStats();
    const auto small = controller.Acquire(Shapes(2));
    const auto large_hit = controller.Acquire(Shapes(6));
    const auto miss = controller.Acquire(Shapes(8));
    const control::PublishedExactVariant duplicate =
        controller.CompileAndPublish(Shapes(2));
    const auto cache_after_lookups =
        kxc::api::internal::GetPrimitiveCacheStats();
    CHECK(small.has_value() && large_hit.has_value() && !miss.has_value() &&
              duplicate.dispatch_key() == small->dispatch_key(),
          "Acquire must distinguish published hits from a valid miss");
    CHECK(SameCacheState(cache_after_profiles, cache_after_lookups),
          "hit, miss, and duplicate CompileAndPublish fast path must not touch cache");
    CHECK(Throws([&] {
              (void)controller.CompileAndPublish(Shapes(4));
          }) &&
              SameCacheState(cache_after_lookups,
                             kxc::api::internal::GetPrimitiveCacheStats()),
          "max_profiles must reject before a third profile is compiled");

    CHECK(RunVariant(*small, {-1.0F, 2.0F}, {1.0F, 1.0F}) ==
              std::vector<float>({0.0F, 3.0F}),
          "N=2 LLVM variant must execute numerically");
    CHECK(RunVariant(large,
                     {-3.0F, -2.0F, -1.0F, 0.0F, 1.0F, 2.0F},
                     {1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F}) ==
              std::vector<float>({0.0F, 0.0F, 0.0F, 1.0F, 2.0F, 3.0F}),
          "N=6 LLVM variant must execute numerically");
#endif
    return true;
#endif
}

}  // namespace

int main() {
    try {
        if (TestShapeSpecializationController()) {
            std::cout << "[PASS] shape_specialization_controller\n";
            return 0;
        }
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] shape_specialization_controller: "
                  << error.what() << "\n";
    }
    return 1;
}
