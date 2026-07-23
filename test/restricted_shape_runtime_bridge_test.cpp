#include <algorithm>
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

kxc::Function AddGraph() {
    const kxc::Var x("x", kxc::TensorType({4}, "float32"));
    const kxc::Var y("y", kxc::TensorType({4}, "float32"));
    return kxc::Function({x, y}, kxc::Call(kxc::relay::Op::Get("add"), {x, y}));
}

restricted::PreparedRestrictedSymbolicTemplate PreparedRelu() {
    return restricted::RestrictedSymbolicShapeAdapter::Prepare(
        ReluGraph(), Config(), {{0, 0, "n", 0, 8, 1}});
}

restricted::PreparedRestrictedSymbolicTemplate PreparedAdd() {
    return restricted::RestrictedSymbolicShapeAdapter::Prepare(
        AddGraph(), Config(), {{0, 0, "n", 0, 8, 1}, {1, 0, "n", 0, 8, 1}});
}

shape::BindingSet Bind(int64_t n) { return shape::BindingSet({{"n", n}}); }

shape::ApplicabilityGuard Guard() {
    return shape::ApplicabilityGuard({"n"}, {
        shape::Constraint::Range(shape::DimExpr::Symbol("n"), 0, 8),
        shape::Constraint::DivisibleBy(shape::DimExpr::Symbol("n"), 1)});
}

shape::BucketPolicy Bucket(const restricted::PreparedRestrictedSymbolicTemplate& prepared,
                           const restricted::RestrictedDispatchDecision& exact) {
    std::vector<shape::BucketValueBoundary> boundaries;
    for (const auto& value : prepared.graph_template().shape_program().inputs()) {
        const auto& contract = exact.exact_oracle().profile().Value(value.name).contract;
        boundaries.push_back({value.name, {8}, {1}, contract.layout, contract.alignment,
                              contract.memory_scope, contract.abi});
    }
    for (const auto& value : prepared.graph_template().shape_program().outputs()) {
        const auto& contract = exact.exact_oracle().profile().Value(value.name).contract;
        boundaries.push_back({value.name, {8}, {1}, contract.layout, contract.alignment,
                              contract.memory_scope, contract.abi});
    }
    const auto& unit = prepared.graph_template().ordered_units().front();
    return shape::BucketPolicy("n8", 1, Guard(), std::move(boundaries),
                               {{0, unit.semantic_key, true, true, true, true}}, 0);
}

shape::PolymorphicPolicy Polymorphic(
    const restricted::PreparedRestrictedSymbolicTemplate& prepared, std::string scalar_name) {
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
        {{0, std::move(scalar_name), "n", 0, 8, 1}}, std::move(boundaries),
        prepared.graph_template().key().target_backend_abi(), 0);
}

std::string ArtifactIdentity(const restricted::RestrictedDispatchDecision& decision) {
    return decision.kind() == restricted::DispatchKind::kExact
        ? decision.exact_requests().back().artifact_key.CanonicalBytes()
        : decision.guarded_requests().back().artifact_key.CanonicalBytes();
}

std::string TailPolicyIdentity(const restricted::RestrictedDispatchDecision& decision) {
    const auto* policy = decision.bucket_policy();
    return policy ? policy->CanonicalString() : std::string{};
}

bridge::TrustedSynchronousLauncherDescriptor Launcher(
    const restricted::RestrictedDispatchDecision& decision, std::string entry_symbol,
    const std::function<bool(const runtime::RuntimeShapeLaunchArgs&)>& body) {
    bridge::TrustedSynchronousLauncherDescriptor descriptor;
    descriptor.module_label = "trusted-local-cpu";
    descriptor.entry_symbol = std::move(entry_symbol);
    descriptor.launcher = [body](const runtime::RuntimeShapeLaunchArgs& args) {
        return body(args) ? runtime::RuntimeShapeLaunchResult{}
                          : runtime::RuntimeShapeLaunchResult{false, "numeric contract failed", {}};
    };
    descriptor.artifact_identity = ArtifactIdentity(decision);
    descriptor.tail_policy_identity = TailPolicyIdentity(decision);
    return descriptor;
}

runtime::RuntimeShapeInput Input(const std::vector<float>& values) {
    return {{values.size()}, "float32", "CPU:0", 1, values.data(),
            values.size() * sizeof(float), {}};
}

bool Has(const runtime::RuntimeShapeAsyncResult& result, runtime::RuntimeShapeEventKind kind) {
    for (const auto& event : result.events()) if (event.kind == kind) return true;
    return false;
}

bool Equal(const float* actual, const std::vector<float>& expected) {
    return std::equal(expected.begin(), expected.end(), actual);
}

bool TestExactReluAndRequiredInputData() {
    const auto prepared = PreparedRelu();
    const auto exact = restricted::RestrictedSymbolicShapeAdapter::MintExact(prepared, Bind(4));
    int launches = 0;
    const auto plan = bridge::RestrictedShapeRuntimeBridge::Bind(prepared, exact, Launcher(
        exact, "relu", [&](const auto& args) {
            ++launches;
            const auto* input = static_cast<const float*>(args.inputs[0].data);
            auto* output = static_cast<float*>(args.outputs[0].data);
            for (std::size_t i = 0; i < args.outputs[0].logical[0]; ++i) {
                output[i] = std::max(input[i], 0.0F);
            }
            return args.runtime_extent_values.empty() &&
                   args.outputs[0].logical == std::vector<runtime::RuntimeShapeExtent>{4};
        }));
    CHECK(plan.spec().inputs[0].requires_data, "bridge plans require caller input data");
    const std::vector<float> input{-2.0F, 0.0F, 1.5F, 3.0F};
    const auto result = runtime::RuntimeShapeSession(plan).Run({Input(input)});
    CHECK(result.ok() && launches == 1 &&
              Equal(static_cast<const float*>(result.outputs()[0].data), {0.0F, 0.0F, 1.5F, 3.0F}),
          "exact relu executes CPU numeric work");
    const auto missing = runtime::RuntimeShapeSession(plan).Run({{{4}, "float32", "CPU:0", 1}});
    CHECK(!missing.ok() && !Has(missing, runtime::RuntimeShapeEventKind::kShapeEval) && launches == 1,
          "missing required input data fails before ShapeEval");
    return true;
}

bool TestBucketAddTailAndGuardMiss() {
    const auto prepared = PreparedAdd();
    const auto exact = restricted::RestrictedSymbolicShapeAdapter::MintExact(prepared, Bind(4));
    const auto bucket = restricted::RestrictedSymbolicShapeAdapter::MintBucket(prepared, Bind(4), Bucket(prepared, exact));
    int launches = 0;
    constexpr float kTail = -1234.0F;
    const auto plan = bridge::RestrictedShapeRuntimeBridge::Bind(prepared, bucket, Launcher(
        bucket, "add_bucket", [&](const auto& args) {
            ++launches;
            const auto n = args.inputs[0].shape[0];
            const auto* left = static_cast<const float*>(args.inputs[0].data);
            const auto* right = static_cast<const float*>(args.inputs[1].data);
            auto* output = static_cast<float*>(args.outputs[0].data);
            for (std::size_t i = 0; i < args.outputs[0].physical[0]; ++i) output[i] = kTail;
            for (std::size_t i = 0; i < n; ++i) output[i] = left[i] + right[i];
            return args.outputs[0].logical == std::vector<runtime::RuntimeShapeExtent>{n} &&
                   args.outputs[0].valid == std::vector<runtime::RuntimeShapeExtent>{n} &&
                   args.outputs[0].physical == std::vector<runtime::RuntimeShapeExtent>{8};
        }));
    const std::vector<float> left{1.0F, -2.0F, 3.5F};
    const std::vector<float> right{4.0F, 5.0F, -6.5F};
    const auto result = runtime::RuntimeShapeSession(plan).Run({Input(left), Input(right)});
    CHECK(result.ok() && launches == 1 &&
              Equal(static_cast<const float*>(result.outputs()[0].data), {5.0F, 3.0F, -3.0F}),
          "bucket add crops valid numeric output");
    const auto* output = static_cast<const float*>(result.outputs()[0].data);
    for (std::size_t i = 3; i < 8; ++i) CHECK(output[i] == kTail, "predicated bucket tail is untouched");

    const std::vector<float> miss(9, 1.0F);
    const auto missed = runtime::RuntimeShapeSession(plan).Run({Input(miss), Input(miss)});
    CHECK(!missed.ok() && !Has(missed, runtime::RuntimeShapeEventKind::kShapeEval) &&
              !Has(missed, runtime::RuntimeShapeEventKind::kAllocate) && launches == 1,
          "guard miss precedes ShapeEval, allocation, and launcher");

    auto wrong_tail = Launcher(bucket, "add_bucket", [](const auto&) { return true; });
    wrong_tail.tail_policy_identity = "wrong";
    CHECK(Throws([&] { (void)bridge::RestrictedShapeRuntimeBridge::Bind(prepared, bucket, std::move(wrong_tail)); }),
          "bridge rejects mismatched canonical bucket tail binding");
    return true;
}

bool TestPolymorphicScalarAbiAndRelu() {
    const auto prepared = PreparedRelu();
    const auto poly = restricted::RestrictedSymbolicShapeAdapter::MintPolymorphic(
        prepared, Bind(4), Polymorphic(prepared, "n_extent"));
    int launches = 0;
    const auto plan = bridge::RestrictedShapeRuntimeBridge::Bind(prepared, poly, Launcher(
        poly, "relu_poly", [&](const auto& args) {
            ++launches;
            const auto n = args.inputs[0].shape[0];
            const auto* input = static_cast<const float*>(args.inputs[0].data);
            auto* output = static_cast<float*>(args.outputs[0].data);
            for (std::size_t i = 0; i < n; ++i) output[i] = std::max(input[i], 0.0F);
            return args.runtime_extent_values == std::vector<runtime::RuntimeShapeExtent>{n} &&
                   args.outputs[0].logical == args.outputs[0].physical &&
                   args.outputs[0].physical == args.outputs[0].valid;
        }));
    CHECK(plan.spec().runtime_extent_abi.size() == 1 &&
              plan.spec().runtime_extent_abi[0].name == "n_extent" &&
              plan.spec().runtime_extent_abi[0].input_index == 0 &&
              plan.spec().runtime_extent_abi[0].axis == 0,
          "polymorphic scalar ABI carries ordered name/symbol/input-axis mapping");
    const std::vector<float> input{-1.0F, 2.0F};
    const auto result = runtime::RuntimeShapeSession(plan).Run({Input(input)});
    CHECK(result.ok() && launches == 1 &&
              Equal(static_cast<const float*>(result.outputs()[0].data), {0.0F, 2.0F}),
          "polymorphic relu consumes evaluated scalar and real CPU data");

    const auto renamed = restricted::RestrictedSymbolicShapeAdapter::MintPolymorphic(
        prepared, Bind(4), Polymorphic(prepared, "n_extent_renamed"));
    const auto renamed_plan = bridge::RestrictedShapeRuntimeBridge::Bind(prepared, renamed,
        Launcher(renamed, "relu_poly", [](const auto&) { return true; }));
    CHECK(plan.spec().entry.exact_abi_fingerprint != renamed_plan.spec().entry.exact_abi_fingerprint,
          "distinct scalar ABI produces distinct canonical entry ABI");
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
        return TestExactReluAndRequiredInputData() && TestBucketAddTailAndGuardMiss() &&
                       TestPolymorphicScalarAbiAndRelu() ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << "\n";
        return 1;
    }
#endif
}
