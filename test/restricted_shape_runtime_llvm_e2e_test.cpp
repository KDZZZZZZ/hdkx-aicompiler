#include <algorithm>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "kxc/compiler/compiler.h"
#include "kxc/compiler/restricted_shape_runtime_bridge.h"
#include "kxc/relay/op.h"
#include "kxc/runtime/ndarray.h"
#include "kxc/runtime/runtime_shape_session.h"
#include "kxc/runtime/session.h"

#ifndef KXC_USE_LLVM
#define KXC_USE_LLVM 0
#endif
#ifndef KXC_ENABLE_RESTRICTED_SHAPE_RUNTIME_BRIDGE
#define KXC_ENABLE_RESTRICTED_SHAPE_RUNTIME_BRIDGE 0
#endif
#ifndef KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE
#define KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE 0
#endif
#ifndef KXC_ENABLE_RUNTIME_SHAPE_TASKS
#define KXC_ENABLE_RUNTIME_SHAPE_TASKS 0
#endif
#ifndef KXC_ENABLE_SHAPE_PRODUCTION_EXACT
#define KXC_ENABLE_SHAPE_PRODUCTION_EXACT 0
#endif

#if !KXC_USE_LLVM || !KXC_ENABLE_RESTRICTED_SHAPE_RUNTIME_BRIDGE || \
    !KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE || !KXC_ENABLE_RUNTIME_SHAPE_TASKS || \
    !KXC_ENABLE_SHAPE_PRODUCTION_EXACT
#error "restricted Shape LLVM E2E must only be compiled with LLVM and bridge gates enabled"
#endif

namespace {
namespace bridge = kxc::api::experimental::restricted_shape_runtime_bridge::v1;
namespace restricted = kxc::api::experimental::restricted_symbolic_shape::v1;
namespace shape = kxc::shape::experimental::v1;
namespace runtime = kxc::runtime;

#define CHECK(x, m) do { if (!(x)) { std::cerr << "[FAIL] " << __FUNCTION__ << ": " << m << "\n"; return false; } } while (0)

constexpr int64_t kBucketCapacity = 8;

kxc::api::CompileConfig Config() {
    return kxc::api::CompileConfig::Create(kxc::BuildTarget(kxc::Device::CPU()));
}

kxc::Function ReluGraph(int64_t extent) {
    const kxc::Var x("x", kxc::TensorType({extent}, "float32"));
    return kxc::Function({x}, kxc::Call(kxc::relay::Op::Get("nn_relu"), {x}));
}

shape::BindingSet Bind(int64_t n) {
    return shape::BindingSet({{"n", n}});
}

restricted::PreparedRestrictedSymbolicTemplate PreparedRelu() {
    return restricted::RestrictedSymbolicShapeAdapter::Prepare(
        ReluGraph(kBucketCapacity), Config(), {{0, 0, "n", 1, kBucketCapacity, 1}});
}

shape::BucketPolicy Bucket(
    const restricted::PreparedRestrictedSymbolicTemplate& prepared,
    const restricted::RestrictedDispatchDecision& exact) {
    std::vector<shape::BucketValueBoundary> boundaries;
    const auto append = [&](const auto& values) {
        for (const auto& value : values) {
            const auto& contract = exact.exact_oracle().profile().Value(value.name).contract;
            boundaries.push_back({value.name, {kBucketCapacity}, {1}, contract.layout,
                                  contract.alignment, contract.memory_scope, contract.abi});
        }
    };
    append(prepared.graph_template().shape_program().inputs());
    append(prepared.graph_template().shape_program().outputs());
    const auto& unit = prepared.graph_template().ordered_units().front();
    return shape::BucketPolicy(
        "relu_n8", 1,
        shape::ApplicabilityGuard(
            {"n"}, {shape::Constraint::Range(shape::DimExpr::Symbol("n"), 1, kBucketCapacity),
                     shape::Constraint::DivisibleBy(shape::DimExpr::Symbol("n"), 1)}),
        std::move(boundaries), {{0, unit.semantic_key, true, true, true, true}}, 0);
}

std::string ArtifactIdentity(const restricted::RestrictedDispatchDecision& decision) {
    return decision.kind() == restricted::DispatchKind::kExact
        ? decision.exact_requests().back().artifact_key.CanonicalBytes()
        : decision.guarded_requests().back().artifact_key.CanonicalBytes();
}

struct StaticVariant final {
    const int64_t physical_extent;
    const kxc::api::CompiledGraph compiled;
    const std::shared_ptr<const runtime::RuntimeSession> session;
    const std::string plan_abi;

    explicit StaticVariant(int64_t extent)
        : physical_extent(extent),
          compiled(kxc::api::Compiler::Compile(ReluGraph(extent), Config())),
          session(std::make_shared<const runtime::RuntimeSession>(compiled.module, compiled.variant)),
          plan_abi(session->artifact_manifest()->plan_fingerprint) {}
};

runtime::RuntimeShapeLaunchResult Reject(std::string detail) {
    return runtime::RuntimeShapeLaunchResult{false, std::move(detail), {}};
}

bridge::TrustedSynchronousLauncherDescriptor Launcher(
    const restricted::RestrictedDispatchDecision& decision,
    std::shared_ptr<StaticVariant> state) {
    bridge::TrustedSynchronousLauncherDescriptor descriptor;
    descriptor.module_label = "llvm-static-relu";
    descriptor.entry_symbol = "relu_static_" + std::to_string(state->physical_extent);
    descriptor.module_lease = state;
    descriptor.artifact_identity = ArtifactIdentity(decision);
    descriptor.tail_policy_identity = decision.bucket_policy()
        ? decision.bucket_policy()->CanonicalString() : std::string{};
    descriptor.launcher = [state = std::move(state)](const runtime::RuntimeShapeLaunchArgs& args) {
        if (args.inputs.size() != 1 || args.outputs.size() != 1 ||
            args.inputs[0].shape.size() != 1 || args.outputs[0].logical.size() != 1 ||
            args.outputs[0].valid.size() != 1 || args.outputs[0].physical.size() != 1) {
            return Reject("unexpected restricted ReLU ABI");
        }
        const auto n = args.inputs[0].shape[0];
        const auto capacity = static_cast<std::size_t>(state->physical_extent);
        if (n != args.outputs[0].logical[0] || n != args.outputs[0].valid[0] ||
            args.outputs[0].physical[0] != capacity || n > capacity) {
            return Reject("logical/valid/physical ReLU contract mismatch");
        }

        // The static JIT variant receives an explicit zero-padded physical buffer.
        std::vector<float> padded(capacity, 0.0F);
        const auto* source = static_cast<const float*>(args.inputs[0].data);
        std::copy_n(source, static_cast<std::size_t>(n), padded.begin());
        const auto input = runtime::NDArray::Empty(
            {state->physical_extent}, runtime::DataTypeFromString("float32"), kxc::Device::CPU());
        input.CopyFromBytes(padded.data(), padded.size() * sizeof(float));
        const auto static_outputs = state->session->Run({input});
        if (static_outputs.size() != 1 ||
            static_outputs[0].NBytes() != capacity * sizeof(float)) {
            return Reject("static RuntimeSession output ABI mismatch");
        }
        std::vector<float> static_result(capacity);
        static_outputs[0].CopyToBytes(static_result.data(), static_result.size() * sizeof(float));

        // Crop only the valid prefix into a deterministically zeroed physical output.
        auto* destination = static_cast<float*>(args.outputs[0].data);
        std::fill_n(destination, capacity, 0.0F);
        std::copy_n(static_result.begin(), static_cast<std::size_t>(n), destination);
        return runtime::RuntimeShapeLaunchResult{};
    };
    return descriptor;
}

runtime::RuntimeShapeInput Input(const std::vector<float>& values) {
    return {{values.size()}, "float32", "CPU:0", 1, values.data(),
            values.size() * sizeof(float), {}};
}

bool Has(const runtime::RuntimeShapeAsyncResult& result, runtime::RuntimeShapeEventKind kind) {
    return std::any_of(result.events().begin(), result.events().end(), [kind](const auto& event) {
        return event.kind == kind;
    });
}

bool Equal(const float* actual, const std::vector<float>& expected) {
    return std::equal(expected.begin(), expected.end(), actual);
}

bool TestExactN2AndBucketN3ViaStaticLLVMVariants() {
    const auto prepared = PreparedRelu();
    const auto exact = restricted::RestrictedSymbolicShapeAdapter::MintExact(prepared, Bind(2));
    const auto bucket_seed = restricted::RestrictedSymbolicShapeAdapter::MintExact(prepared, Bind(3));
    const auto bucket = restricted::RestrictedSymbolicShapeAdapter::MintBucket(
        prepared, Bind(3), Bucket(prepared, bucket_seed));

    const auto exact_state = std::make_shared<StaticVariant>(2);
    const auto bucket_state = std::make_shared<StaticVariant>(kBucketCapacity);
    CHECK(exact_state->compiled.module.IsReady() && bucket_state->compiled.module.IsReady(),
          "Compiler::Compile must produce ready LLVM static variants");
    CHECK(!exact_state->plan_abi.empty() && !bucket_state->plan_abi.empty() &&
              exact_state->plan_abi != bucket_state->plan_abi,
          "physical n=2 and capacity=8 Compiler PlanVariants must carry distinct ABI evidence");

    const auto exact_plan = bridge::RestrictedShapeRuntimeBridge::Bind(
        prepared, exact, Launcher(exact, exact_state));
    const auto bucket_plan = bridge::RestrictedShapeRuntimeBridge::Bind(
        prepared, bucket, Launcher(bucket, bucket_state));
    CHECK(exact_plan.spec().entry.artifact_identity == ArtifactIdentity(exact) &&
              bucket_plan.spec().entry.artifact_identity == ArtifactIdentity(bucket),
          "bridge artifact identity is the restricted decision canonical bytes");
    CHECK(exact_plan.spec().outputs[0].physical[0].Evaluate({{2}}) == 2 &&
              bucket_plan.spec().outputs[0].physical[0].Evaluate({{3}}) == kBucketCapacity,
          "exact and bucket plans retain distinct physical contracts");

    const std::vector<float> exact_input{-2.0F, 1.5F};
    const auto exact_result = runtime::RuntimeShapeSession(exact_plan).Run({Input(exact_input)});
    CHECK(exact_result.ok() && exact_result.outputs().size() == 1 &&
              Equal(static_cast<const float*>(exact_result.outputs()[0].data), {0.0F, 1.5F}),
          "n=2 exact ReLU executes through the immutable static RuntimeSession launcher");

    const std::vector<float> bucket_input{-3.0F, 2.0F, 4.5F};
    const auto bucket_result = runtime::RuntimeShapeSession(bucket_plan).Run({Input(bucket_input)});
    CHECK(bucket_result.ok() && bucket_result.outputs().size() == 1 &&
              bucket_result.outputs()[0].logical == std::vector<runtime::RuntimeShapeExtent>{3} &&
              bucket_result.outputs()[0].valid == std::vector<runtime::RuntimeShapeExtent>{3} &&
              bucket_result.outputs()[0].physical == std::vector<runtime::RuntimeShapeExtent>{8} &&
              Equal(static_cast<const float*>(bucket_result.outputs()[0].data),
                    {0.0F, 2.0F, 4.5F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F}),
          "n=3 bucket ReLU zero-pads, runs physical capacity 8, and crops its valid prefix");

    const auto miss = runtime::RuntimeShapeSession(bucket_plan).Run(
        {Input(std::vector<float>(kBucketCapacity + 1, 1.0F))});
    CHECK(!miss.ok() && !Has(miss, runtime::RuntimeShapeEventKind::kShapeEval) &&
              !Has(miss, runtime::RuntimeShapeEventKind::kAllocate) &&
              !Has(miss, runtime::RuntimeShapeEventKind::kKernel),
          "n=9 guard miss must occur before shape evaluation, allocation, and LLVM launch");
    return true;
}

}  // namespace

int main() {
    try {
        if (!bridge::RestrictedShapeRuntimeBridge::IsEnabled()) return 1;
        return TestExactN2AndBucketN3ViaStaticLLVMVariants() ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << "\n";
        return 1;
    }
}
