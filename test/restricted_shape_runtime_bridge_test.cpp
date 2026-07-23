#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "kxc/compiler/restricted_shape_runtime_bridge.h"
#include "kxc/relay/op.h"
#include "kxc/runtime/runtime_shape_session.h"

namespace {
namespace bridge = kxc::api::experimental::restricted_shape_runtime_bridge::v1;
namespace restricted = kxc::api::experimental::restricted_symbolic_shape::v1;
namespace shape = kxc::shape::experimental::v1;
namespace runtime = kxc::runtime;

#define CHECK(x, m) do { if (!(x)) { std::cerr << "[FAIL] " << __FUNCTION__ << ": " << m << "\n"; return false; } } while (0)

bool Throws(const std::function<void()>& body) {
    try { body(); } catch (const std::exception&) { return true; }
    return false;
}

kxc::api::CompileConfig Config() {
    return kxc::api::CompileConfig::Create(kxc::BuildTarget(kxc::Device::CPU()));
}

kxc::Function ReluGraph() {
    const kxc::Var x("x", kxc::TensorType({4}, "float32"));
    return kxc::Function({x}, kxc::Call(kxc::relay::Op::Get("nn_relu"), {x}));
}

kxc::Function SqrtGraph() {
    const kxc::Var x("x", kxc::TensorType({4}, "float32"));
    return kxc::Function({x}, kxc::Call(kxc::relay::Op::Get("sqrt"), {x}));
}

restricted::PreparedRestrictedSymbolicTemplate Prepared() {
    return restricted::RestrictedSymbolicShapeAdapter::Prepare(
        ReluGraph(), Config(), {{0, 0, "n", 0, 8, 1}});
}

shape::BindingSet Bind(int64_t n) { return shape::BindingSet({{"n", n}}); }

shape::ApplicabilityGuard Guard() {
    return shape::ApplicabilityGuard({"n"}, {
        shape::Constraint::Range(shape::DimExpr::Symbol("n"), 0, 8),
        shape::Constraint::DivisibleBy(shape::DimExpr::Symbol("n"), 1)});
}

shape::BucketPolicy Bucket(const restricted::PreparedRestrictedSymbolicTemplate& prepared,
                           const restricted::RestrictedDispatchDecision& exact,
                           int64_t capacity = 8) {
    std::vector<shape::BucketValueBoundary> boundaries;
    for (const auto& value : prepared.graph_template().shape_program().inputs()) {
        const auto& contract = exact.exact_oracle().profile().Value(value.name).contract;
        boundaries.push_back({value.name, {capacity}, {1}, contract.layout, contract.alignment,
                              contract.memory_scope, contract.abi});
    }
    for (const auto& value : prepared.graph_template().shape_program().outputs()) {
        const auto& contract = exact.exact_oracle().profile().Value(value.name).contract;
        boundaries.push_back({value.name, {capacity}, {1}, contract.layout, contract.alignment,
                              contract.memory_scope, contract.abi});
    }
    const auto& unit = prepared.graph_template().ordered_units().front();
    return shape::BucketPolicy("n8", 1, Guard(), std::move(boundaries),
                               {{0, unit.semantic_key, true, true, true, true}}, 0);
}

shape::PolymorphicPolicy Polymorphic(const restricted::PreparedRestrictedSymbolicTemplate& prepared) {
    std::vector<shape::SymbolicBoundaryContract> boundaries;
    const auto append = [&boundaries](const auto& values) {
        for (const auto& value : values) boundaries.push_back({
            value.name, value.contract.logical().dimensions(), value.contract.physical().layout(),
            value.contract.physical().alignment(), value.contract.physical().memory_scope(),
            value.contract.abi(), value.contract.logical().axis_names()});
    };
    append(prepared.graph_template().shape_program().inputs());
    append(prepared.graph_template().shape_program().outputs());
    const auto& unit = prepared.graph_template().ordered_units().front();
    return shape::PolymorphicPolicy(1, Guard(),
        {{0, unit.semantic_key, "restricted.relu.equal-shape.v1"}},
        {{0, "n_extent", "n", 0, 8, 1}}, std::move(boundaries),
        prepared.graph_template().key().target_backend_abi(), 0);
}

bridge::TrustedSynchronousLauncherDescriptor Launcher(
    const std::function<bool(const runtime::RuntimeShapeLaunchArgs&)>& body) {
    return {"trusted-local-cpu", "relu", [body](const runtime::RuntimeShapeLaunchArgs& args) {
        return body(args) ? runtime::RuntimeShapeLaunchResult{}
                          : runtime::RuntimeShapeLaunchResult{false, "numeric contract failed", {}};
    }, {}};
}

bool Has(const runtime::RuntimeShapeAsyncResult& result, runtime::RuntimeShapeEventKind kind) {
    for (const auto& event : result.events()) if (event.kind == kind) return true;
    return false;
}

bool TestExactZeroAndCanonicalGuards() {
    const auto prepared = Prepared();
    const auto exact = restricted::RestrictedSymbolicShapeAdapter::MintExact(prepared, Bind(4));
    int launches = 0;
    const auto plan = bridge::RestrictedShapeRuntimeBridge::Bind(prepared, exact, Launcher([&](const auto& args) {
        ++launches;
        auto* data = static_cast<float*>(args.outputs[0].data);
        for (std::size_t i = 0; i < args.outputs[0].physical[0]; ++i) data[i] = static_cast<float>(i + 1);
        return args.outputs[0].logical == std::vector<runtime::RuntimeShapeExtent>{4} &&
               args.outputs[0].bytes == 16;
    }));
    CHECK(plan.spec().inputs[0].axis_guards[0].exact == 4, "exact decision binds exact input guard");
    CHECK(plan.spec().entry.exact_abi_fingerprint ==
              runtime::RuntimeShapePlan::ExactAbiFingerprint(plan.spec().inputs, plan.spec().outputs),
          "axis guards are bound into exact ABI bytes");
    const auto result = runtime::RuntimeShapeSession(plan).Run({{{4}, "float32", "CPU:0", 1}});
    CHECK(result.ok() && launches == 1 && *static_cast<float*>(result.outputs()[0].data) == 1.0F,
          "exact numeric launcher fills final output");

    const auto zero = restricted::RestrictedSymbolicShapeAdapter::MintExact(prepared, Bind(0));
    const auto zero_plan = bridge::RestrictedShapeRuntimeBridge::Bind(prepared, zero, Launcher([](const auto& args) {
        return args.outputs[0].bytes == 0 && args.outputs[0].data == nullptr;
    }));
    const auto zero_result = runtime::RuntimeShapeSession(zero_plan).Run({{{0}, "float32", "CPU:0", 1}});
    CHECK(zero_result.ok() && zero_result.outputs()[0].bytes == 0, "zero exact output is a real zero-byte launch");
    return true;
}

bool TestBucketTailAndGuardMiss() {
    const auto prepared = Prepared();
    const auto exact = restricted::RestrictedSymbolicShapeAdapter::MintExact(prepared, Bind(4));
    const auto bucket = restricted::RestrictedSymbolicShapeAdapter::MintBucket(prepared, Bind(4), Bucket(prepared, exact));
    int launches = 0;
    const auto plan = bridge::RestrictedShapeRuntimeBridge::Bind(prepared, bucket, Launcher([&](const auto& args) {
        ++launches;
        const auto n = args.inputs[0].shape[0];
        auto* data = static_cast<float*>(args.outputs[0].data);
        for (std::size_t i = 0; i < args.outputs[0].physical[0]; ++i) data[i] = i < n ? 7.0F : -1.0F;
        return args.outputs[0].logical == std::vector<runtime::RuntimeShapeExtent>{n} &&
               args.outputs[0].valid == std::vector<runtime::RuntimeShapeExtent>{n} &&
               args.outputs[0].physical == std::vector<runtime::RuntimeShapeExtent>{8} &&
               args.outputs[0].bytes == 32;
    }));
    for (const auto n : {runtime::RuntimeShapeExtent{0}, runtime::RuntimeShapeExtent{3}, runtime::RuntimeShapeExtent{8}}) {
        const auto result = runtime::RuntimeShapeSession(plan).Run({{{n}, "float32", "CPU:0", 1}});
        CHECK(result.ok() && result.outputs()[0].physical[0] == 8, "bucket uses fixed, non-fuzzy capacity");
        if (n != 0) CHECK(static_cast<float*>(result.outputs()[0].data)[0] == 7.0F,
                           "bucket launcher wrote valid tail-aware output");
    }
    const auto missed = runtime::RuntimeShapeSession(plan).Run({{{9}, "float32", "CPU:0", 1}});
    CHECK(!missed.ok() && !Has(missed, runtime::RuntimeShapeEventKind::kShapeEval) &&
              !Has(missed, runtime::RuntimeShapeEventKind::kAllocate) && launches == 3,
          "guard miss happens before ShapeEval, allocation, and launch");
    return true;
}

bool TestPolymorphicAndFailures() {
    const auto prepared = Prepared();
    const auto exact = restricted::RestrictedSymbolicShapeAdapter::MintExact(prepared, Bind(4));
    const auto polymorphic = restricted::RestrictedSymbolicShapeAdapter::MintPolymorphic(
        prepared, Bind(4), Polymorphic(prepared));
    int launches = 0;
    const auto plan = bridge::RestrictedShapeRuntimeBridge::Bind(prepared, polymorphic, Launcher([&](const auto& args) {
        ++launches;
        auto* data = static_cast<float*>(args.outputs[0].data);
        for (std::size_t i = 0; i < args.outputs[0].logical[0]; ++i) data[i] = 3.0F;
        return args.outputs[0].logical == args.outputs[0].physical &&
               args.outputs[0].physical == args.outputs[0].valid;
    }));
    for (const auto n : {runtime::RuntimeShapeExtent{0}, runtime::RuntimeShapeExtent{2}, runtime::RuntimeShapeExtent{8}}) {
        const auto result = runtime::RuntimeShapeSession(plan).Run({{{n}, "float32", "CPU:0", 1}});
        CHECK(result.ok() && result.outputs()[0].bytes == n * sizeof(float),
              "polymorphic runtime extent controls exact capacity");
    }
    runtime::detail::FailNextRuntimeShapeOwnerTransferForTest();
    const auto oom = runtime::RuntimeShapeSession(plan).Run({{{2}, "float32", "CPU:0", 1}});
    CHECK(!oom.ok() && !Has(oom, runtime::RuntimeShapeEventKind::kKernel) && launches == 3,
          "OOM cleans up before trusted launcher");
    CHECK(Throws([&] { (void)bridge::RestrictedShapeRuntimeBridge::Bind(
              restricted::RestrictedSymbolicShapeAdapter::Prepare(
                  SqrtGraph(), Config(), {{0, 0, "n", 0, 8, 1}}),
              exact, Launcher([](const auto&) { return true; })); }),
          "decision and prepared template must match");
    CHECK(Throws([&] { (void)bridge::RestrictedShapeRuntimeBridge::Bind(prepared, exact, {}); }),
          "incomplete trusted launcher is rejected");
    CHECK(Throws([&] { (void)restricted::RestrictedSymbolicShapeAdapter::MintBucket(
              prepared, Bind(4), Bucket(prepared, exact, 3)); }),
          "invalid bucket decision is rejected before bridging");
    return true;
}

}  // namespace

int main() {
#if !KXC_ENABLE_RESTRICTED_SHAPE_RUNTIME_BRIDGE
    std::cerr << "restricted shape runtime bridge test must be built with gate on\n";
    return 1;
#else
    try {
        if (!bridge::RestrictedShapeRuntimeBridge::IsEnabled()) return 1;
        return TestExactZeroAndCanonicalGuards() && TestBucketTailAndGuardMiss() &&
                       TestPolymorphicAndFailures() ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << "\n";
        return 1;
    }
#endif
}
