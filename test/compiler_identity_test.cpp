/*! \file test/compiler_identity_test.cpp
 * \brief Verifies canonical compiler identity separation and full equality.
 */

#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "kxc/compiler/compiler.h"
#include "kxc/compiler/experimental_identity.h"
#include "kxc/compiler/identity.h"
#include "kxc/relay/op.h"
#include "kxc/runtime/compiled_module.h"
#include "kxc/runtime/executable_plan.h"
#include "kxc/runtime/kernel_abi.h"
#include "kxc/support/object_registration.h"
#include "../src/compiler/internal/identity_private.h"
#include "../src/runtime/internal/compiled_module_node.h"

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

kxc::Target TargetSnapshot(int device_id, std::string capability,
                           std::string kind = "llvm",
                           kxc::DeviceTypeCode device_type = kxc::kCPU) {
    auto* target = new kxc::TargetNode();
    target->kind = std::move(kind);
    target->device_type = device_type;
    target->device_id = device_id;
    target->attrs.exists = 1;
    target->attrs.arch = capability;
    target->attrs.compute_version = capability;
    target->attrs.driver_version = capability == "arch-a" ? 1 : 2;
    target->attrs.total_global_memory =
        capability == "arch-a" ? 16LL << 30 : 32LL << 30;
    target->attrs.max_clock_rate_khz =
        capability == "arch-a" ? 1000000 : 2000000;
    target->attrs.max_threads_per_block =
        capability == "arch-a" ? 512 : 1024;
    return kxc::Target(kxc::ObjectRef(target));
}

kxc::Function PlacementGraph(const kxc::VirtualDevice& virtual_device) {
    kxc::Var input("input", kxc::TensorType({2}, "float32"));
    kxc::Call body(kxc::relay::Op::Get("nn_relu"), {input});
    body.set_virtual_device(virtual_device);
    return kxc::Function({input}, body);
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

bool TestDigestCollisionUsesCanonicalEquality() {
    using namespace kxc::api;
    const UnitSemanticKey first = internal::IdentityAccess::UnitWithDigest(
        "canonical-unit-a", "forced-collision");
    const UnitSemanticKey second = internal::IdentityAccess::UnitWithDigest(
        "canonical-unit-b", "forced-collision");
    TEST_CHECK(first.digest() == second.digest() && first != second,
               "digest is only an index; full canonical bytes decide equality");

    const PrimitiveArtifactKey first_artifact =
        internal::IdentityAccess::ArtifactWithDigest(
            first, "cpu-v1", "relay-tir-o2", 1, "schedule-v1", "llvm-v1",
            "artifact-collision");
    const PrimitiveArtifactKey second_artifact =
        internal::IdentityAccess::ArtifactWithDigest(
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

class IdentityLauncher final : public kxc::codegen::KernelLauncher {
public:
    bool IsReady() const noexcept override { return true; }
    kxc::AsyncOperation Launch(
        const kxc::Array<kxc::runtime::NDArray>&,
        const kxc::DeviceStream&, const kxc::ObjectRef&) const override {
        return {};
    }
};

kxc::api::PlanAbiFingerprint PlanAbiForAlignment(
    uint64_t alignment, bool state_alias = false, int64_t valid_bytes = -1) {
    using namespace kxc;
    using namespace kxc::api;
    using namespace kxc::codegen;
    const DLDataType dtype{kDLFloat, 32, 1};
    const KernelSignature signature(
        "entry", {KernelArgSpec("input", KernelArgRole::kInput, dtype, {2},
                                 Device::CPU(), alignment),
                  KernelArgSpec("output", KernelArgRole::kOutput, dtype, {2},
                                Device::CPU(), alignment, true)});
    const KernelLaunchMetadata metadata(Device::CPU(), CodeGenBackend::kLLVM);
    const auto launcher = std::make_shared<IdentityLauncher>();
    const CompiledModule module = internal::BuildCompiledModule(
        BuildTarget(Device::CPU()),
        {internal::CompiledModuleEntry{signature, metadata,
                                       CompiledKernel(signature, metadata, launcher)}}, {});
    runtime::ExecutablePlan plan;
    if (state_alias) {
        plan = runtime::ExecutablePlan(
            {runtime::ValueSpec(0, 0, {2}, dtype, Device::CPU(), false,
                                false, false, false, false, true),
             runtime::ValueSpec(1, 0, {2}, dtype, Device::CPU(), false,
                                false, true, true, false, false, 0,
                                runtime::ValueWriteMode::kInPlace,
                                valid_bytes)},
            {runtime::KernelCall("entry", {0}, {1})}, {}, {}, {1}, {0});
    } else {
        plan = runtime::ExecutablePlan(
            {runtime::ValueSpec(0, 0, {2}, dtype, Device::CPU(), true),
             runtime::ValueSpec(1, 1, {2}, dtype, Device::CPU(), false,
                                false, true, false, false, false, -1,
                                runtime::ValueWriteMode::kAllocate,
                                valid_bytes)},
            {runtime::KernelCall("entry", {0}, {1})}, {0}, {}, {1});
    }
    const PrimitiveArtifactKey artifact(
        UnitSemanticKey("identity-plan-unit"), "cpu", "pipeline", 1,
        "schedule", "backend");
    return BuildPlanAbiFingerprint(module, plan, {{0, "entry", artifact}});
}

kxc::api::PlanAbiFingerprint DynamicPlanAbi(
    int64_t graph_upper, kxc::api::ModuleExtent invocation_upper) {
    using namespace kxc;
    using namespace kxc::api;
    using namespace kxc::codegen;
    const DLDataType dtype{kDLFloat, 32, 1};
    const KernelSignature signature(
        "dynamic_identity",
        {KernelArgSpec("input", KernelArgRole::kInput, dtype, {-1},
                       Device::CPU(), 8),
         KernelArgSpec("output", KernelArgRole::kOutput, dtype, {-1},
                       Device::CPU(), 16, true)});
    const KernelLaunchMetadata metadata(Device::CPU(),
                                        CodeGenBackend::kLLVM);
    ModuleInputContract input{{{0, 1, invocation_upper, 1, std::nullopt,
                                std::nullopt}}};
    const ModuleShapeExpr extent = ModuleShapeExpr::InputAxis(0, 0);
    ModuleTensorContract output;
    output.logical = {extent};
    output.physical = {extent};
    output.valid = {extent};
    output.max_bytes = 16 * sizeof(float);
    auto contract = std::make_shared<ModuleInvocationContract>(
        std::vector<ModuleInputContract>{input},
        std::vector<ModuleTensorContract>{output},
        std::vector<ModuleRuntimeExtentScalar>{});
    const auto launcher = std::make_shared<IdentityLauncher>();
    const CompiledModule module = internal::BuildCompiledModule(
        BuildTarget(Device::CPU()),
        {{signature, metadata,
          CompiledKernel(signature, metadata, launcher), std::move(contract)}},
        {});
    const runtime::ExecutablePlan plan(
        {runtime::ValueSpec(0, 0, {-1}, dtype, Device::CPU(), true),
         runtime::ValueSpec(1, 1, {-1}, dtype, Device::CPU(), false, false,
                            true)},
        {runtime::KernelCall("dynamic_identity", {0}, {1})}, {0}, {}, {1},
        {}, runtime::ExecutablePlanMode::kDynamicFreshOutputV1,
        {{0, 0, 1, graph_upper, 1, std::nullopt}});
    const PrimitiveArtifactKey artifact(
        UnitSemanticKey("dynamic-identity-unit"), "cpu", "pipeline", 1,
        "schedule", "backend");
    return BuildPlanAbiFingerprint(
        module, plan, {{0, "dynamic_identity", artifact}});
}

bool TestPlanAbiUsesKernelCanonicalBytes() {
    const kxc::api::PlanAbiFingerprint first = PlanAbiForAlignment(4);
    const kxc::api::PlanAbiFingerprint changed = PlanAbiForAlignment(8);
    TEST_CHECK(first.defined() && first != changed &&
                   first.canonical_bytes().find("kxc.kernel-signature.v3") !=
                       std::string::npos &&
                   first.canonical_bytes().find("kxc.kernel-launch-metadata.v1") !=
                       std::string::npos &&
                   first.canonical_bytes().find(
                       "executable-plan-abi-v7-dynamic-fresh-output") !=
                       std::string::npos &&
                   first.canonical_bytes().find(
                       "KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH") ==
                       std::string::npos &&
                   first.canonical_bytes().find("KernelSignature(") ==
                       std::string::npos,
               "Plan ABI must embed canonical kernel contracts rather than diagnostics");
    return true;
}

bool TestPlanAbiIncludesStateAliasAndExtent() {
    const kxc::api::PlanAbiFingerprint baseline = PlanAbiForAlignment(4);
    const kxc::api::PlanAbiFingerprint state_alias =
        PlanAbiForAlignment(4, true);
    const kxc::api::PlanAbiFingerprint bounded =
        PlanAbiForAlignment(4, false, 4);
    TEST_CHECK(baseline != state_alias && baseline != bounded &&
                   state_alias.canonical_bytes().find("alias_source_ordinal") !=
                       std::string::npos &&
                   state_alias.canonical_bytes().find("states_end") !=
                       std::string::npos,
               "Plan ABI identity must include state, alias topology, and valid bytes");
    return true;
}

bool TestPlanAbiIncludesDynamicModeGuardsAndInvocationContract() {
    const kxc::api::PlanAbiFingerprint baseline = DynamicPlanAbi(8, 8);
    const kxc::api::PlanAbiFingerprint graph_guard_changed =
        DynamicPlanAbi(7, 8);
    const kxc::api::PlanAbiFingerprint invocation_changed =
        DynamicPlanAbi(8, 7);
    TEST_CHECK(
        baseline != graph_guard_changed && baseline != invocation_changed &&
            baseline.canonical_bytes().find(
                "executable-plan-abi-v8-bounded-dynamic-graph") !=
                std::string::npos &&
            baseline.canonical_bytes().find("plan_mode") !=
                std::string::npos &&
            baseline.canonical_bytes().find(
                "KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH.v1") !=
                std::string::npos &&
            baseline.canonical_bytes().find("graph_guard_upper") !=
                std::string::npos &&
            baseline.canonical_bytes().find("KXC_MODULE_INVOKE_V2") !=
                std::string::npos,
        "Plan ABI must version dynamic mode, wildcard guards, and invocation bytes");
    return true;
}

bool TestGraphSemanticIdentityCanonicalizesLogicalPlacement() {
    using namespace kxc;
    using namespace kxc::api;
    const Device cpu = Device::CPU();
    const VirtualDevice first(cpu, TargetSnapshot(0, "arch-a"), "global", 7);
    const VirtualDevice same(cpu, TargetSnapshot(0, "arch-b"), "global", 7);
    const GraphSemanticKey first_key =
        Compiler::BuildGraphSemanticKey(PlacementGraph(first));
    const GraphSemanticKey same_key =
        Compiler::BuildGraphSemanticKey(PlacementGraph(same));
    TEST_CHECK(first_key == same_key,
               "Relay placement must exclude Target capability snapshots");

    const GraphSemanticKey different_device = Compiler::BuildGraphSemanticKey(
        PlacementGraph(VirtualDevice(
            Device::CUDA(1), TargetSnapshot(1, "arch-b", "cuda", kCUDA),
            "global", 7)));
    const GraphSemanticKey different_scope = Compiler::BuildGraphSemanticKey(
        PlacementGraph(VirtualDevice(cpu, TargetSnapshot(0, "arch-b"), "shared", 7)));
    const GraphSemanticKey different_id = Compiler::BuildGraphSemanticKey(
        PlacementGraph(VirtualDevice(cpu, TargetSnapshot(0, "arch-b"), "global", 8)));
    TEST_CHECK(first_key != different_device && first_key != different_scope &&
                   first_key != different_id,
               "logical device, memory scope, and virtual-device id must miss");

    const GraphSemanticKey target_only_first = Compiler::BuildGraphSemanticKey(
        PlacementGraph(VirtualDevice(TargetSnapshot(0, "arch-a"), "global", 7)));
    const GraphSemanticKey target_only_same = Compiler::BuildGraphSemanticKey(
        PlacementGraph(VirtualDevice(TargetSnapshot(0, "arch-b"), "global", 7)));
    const GraphSemanticKey target_only_different = Compiler::BuildGraphSemanticKey(
        PlacementGraph(VirtualDevice(TargetSnapshot(0, "arch-b", "c"), "global", 7)));
    const GraphSemanticKey target_only_different_id =
        Compiler::BuildGraphSemanticKey(
            PlacementGraph(VirtualDevice(TargetSnapshot(1, "arch-b"), "global", 7)));
    TEST_CHECK(target_only_first == target_only_same &&
                   target_only_first != target_only_different &&
                   target_only_first != target_only_different_id,
               "target-only placement retains kind/id while excluding capabilities");

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
        {"digest_collision_full_equality", TestDigestCollisionUsesCanonicalEquality},
        {"artifact_field_safe_miss", TestEveryArtifactSemanticFieldCausesSafeMiss},
        {"dispatch_and_plan_are_separate", TestDispatchAndPlanVariantRemainSeparate},
        {"plan_abi_kernel_canonical", TestPlanAbiUsesKernelCanonicalBytes},
        {"plan_abi_state_alias_extent", TestPlanAbiIncludesStateAliasAndExtent},
        {"plan_abi_dynamic_mode_guards_invocation",
         TestPlanAbiIncludesDynamicModeGuardsAndInvocationContract},
        {"graph_identity_logical_placement",
         TestGraphSemanticIdentityCanonicalizesLogicalPlacement},
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
