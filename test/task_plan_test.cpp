/*! \file test/task_plan_test.cpp
 * \brief Verifies frozen per-call Region/task-DAG DTO and dependency validation.
 */

#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "kxc/runtime/task_plan.h"
#include "../src/runtime/internal/compiled_module_node.h"

namespace {

#define TEST_CHECK(condition, message)                                           \
    do {                                                                          \
        if (!(condition)) {                                                        \
            std::cerr << "[FAIL] " << __FUNCTION__ << ": " << (message) << '\n'; \
            return false;                                                         \
        }                                                                         \
    } while (false)

bool Throws(const std::function<void()>& function) {
    try {
        function();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

DLDataType Float32() { return DLDataType{kDLFloat, 32, 1}; }

class TaskPlanLauncher final : public kxc::codegen::KernelLauncher {
public:
    bool IsReady() const noexcept override { return true; }
    kxc::AsyncOperation Launch(const kxc::Array<kxc::runtime::NDArray>&,
                               const kxc::DeviceStream&,
                               const kxc::ObjectRef&) const override {
        return {};
    }
};

kxc::api::CompiledModule ModuleForExactAbi(uint64_t alignment) {
    using namespace kxc;
    using namespace kxc::api;
    using namespace kxc::codegen;
    const KernelSignature signature(
        "entry", {KernelArgSpec("input", KernelArgRole::kInput, Float32(),
                                 {4}, Device::CPU(), alignment),
                  KernelArgSpec("output", KernelArgRole::kOutput, Float32(),
                                {4}, Device::CPU(), alignment, true)});
    const KernelLaunchMetadata launch(Device::CPU(), CodeGenBackend::kLLVM);
    const auto launcher = std::make_shared<TaskPlanLauncher>();
    return internal::BuildCompiledModule(
        BuildTarget(Device::CPU()),
        {internal::CompiledModuleEntry{tir::PrimFunc(), signature, launch,
                                       CompiledKernel(signature, launch, launcher)}}, {});
}

kxc::runtime::ExecutablePlan PlanForExactAbi() {
    using namespace kxc;
    using namespace kxc::runtime;
    return ExecutablePlan(
        {ValueSpec(0, 0, {4}, Float32(), Device::CPU(), true),
         ValueSpec(1, 1, {4}, Float32(), Device::CPU(), false, false, true)},
        {KernelCall("entry", {0}, {1})}, {0}, {}, {1});
}

kxc::runtime::FrozenTaskPlan MakeValidPlan() {
    using namespace kxc;
    using namespace kxc::runtime;
    const Device cpu = Device::CPU();
    return FrozenTaskPlan(
        kFrozenTaskPlanVersion,
        {ValueSpec(0, 0, {4}, Float32(), cpu, true),
         ValueSpec(1, 1, {4}, Float32(), cpu, false, true),
         ValueSpec(2, 2, {4}, Float32(), cpu, false, false, true)},
        {TaskSpec(11, TaskKind::kKernel, cpu, {0, 1, 0}, {2}, {10},
                  "kernel_a", 7),
         TaskSpec(10, TaskKind::kAllocate, cpu, {}, {2}, {}, String(), 0, 0,
                  32)},
        {RegionSpec(0, RegionKind::kPerCall, "", {10, 11}, {0, 1}, {2}, {1})},
        {0}, {1}, {2});
}

kxc::runtime::FrozenTaskPlan WithDeclaredManifest(
    const kxc::runtime::FrozenTaskPlan& plan, uint64_t generation = 7) {
    using namespace kxc;
    using namespace kxc::runtime;
    const String identity("artifact-key-canonical-v1");
    const String abi("exact-abi-v1");
    Array<SelectedArtifactBinding> bindings{SelectedArtifactBinding(
        ArtifactBindingKind::kTask, 11, identity, generation, abi, "kernel_a",
        ComputeEntryBindingFingerprint(ArtifactBindingKind::kTask, 11, identity,
                                       generation, abi, "kernel_a"))};
    return plan.WithManifest(SelectedArtifactManifest(
        ComputeFrozenTaskPlanFingerprint(plan, bindings), bindings));
}

bool TestFrozenDtoAndDeterministicTopology() {
    using namespace kxc;
    using namespace kxc::runtime;
    FrozenTaskPlan plan = MakeValidPlan();
    const Array<int64_t> topology = plan.topological_task_ids();
    TEST_CHECK(std::vector<int64_t>(topology.begin(), topology.end()) ==
                   std::vector<int64_t>({10, 11}),
               "topological order must use task id as deterministic tie-breaker");
    TEST_CHECK(plan.tasks()[0].input_value_ids().size() == 3 &&
                   plan.regions()[0]->semantic_key == "" &&
                   plan.get()->GetTypeKey() == "kxc.runtime.FrozenTaskPlanNode",
               "frozen task metadata must remain stable");
    Array<TaskSpec> copied_tasks = plan.tasks();
    copied_tasks.push_back(
        TaskSpec(99, TaskKind::kKernel, Device::CPU(), {0}, {2}, {}, "kernel"));
    TEST_CHECK(plan.tasks().size() == 2,
               "public arrays must not mutate the frozen plan");
    return true;
}

bool TestArtifactDeclarationContracts() {
    using namespace kxc;
    using namespace kxc::runtime;
    const FrozenTaskPlan raw = MakeValidPlan();
    const FrozenTaskPlan frozen = WithDeclaredManifest(raw);
    TEST_CHECK(frozen.manifest().defined() &&
                   frozen.manifest().bindings().size() == 1 &&
                   frozen.manifest().bindings()[0]->generation == 7 &&
                   frozen.manifest().bindings()[0]->artifact_identity ==
                       "artifact-key-canonical-v1",
               "frozen plans must retain the trusted task artifact declaration");

    const String identity("artifact-key-canonical-v1");
    const String abi("exact-abi-v1");
    const SelectedArtifactBinding wrong_generation(
        ArtifactBindingKind::kTask, 11, identity, 0, abi, "kernel_a",
        ComputeEntryBindingFingerprint(ArtifactBindingKind::kTask, 11, identity,
                                       0, abi, "kernel_a"));
    TEST_CHECK(Throws([&] {
                   (void)raw.WithManifest(SelectedArtifactManifest(
                       ComputeFrozenTaskPlanFingerprint(raw, {wrong_generation}),
                       {wrong_generation}));
               }) &&
                   Throws([&] {
                       (void)raw.WithManifest(SelectedArtifactManifest(
                           "wrong-plan-fingerprint",
                           frozen.manifest().bindings()));
                   }),
               "manifest generation and fingerprint drift must fail closed");

    const Device cpu = Device::CPU();
    const ExecutablePlan ordered(
        {ValueSpec(0, 0, {1}, Float32(), cpu, true),
         ValueSpec(1, 1, {1}, Float32(), cpu, false, false, true)},
        {KernelCall("entry", {0}, {1})}, {0}, {}, {1});
    const String call_identity("artifact-call-canonical-v1");
    const String call_abi("call-exact-abi-v1");
    const SelectedArtifactBinding call_binding(
        ArtifactBindingKind::kCall, 0, call_identity, 0, call_abi, "entry",
        ComputeEntryBindingFingerprint(ArtifactBindingKind::kCall, 0,
                                       call_identity, 0, call_abi, "entry"));
    const PlanVariant variant(
        ordered, SelectedArtifactManifest(
                     ComputePlanVariantFingerprint(ordered, {call_binding}),
                     {call_binding}));
    TEST_CHECK(variant.manifest().bindings()[0]->entry_symbol == "entry" &&
                   !std::string(variant.manifest()->plan_fingerprint).empty(),
               "PlanVariant must retain a complete call declaration");
    return true;
}

bool TestExactAbiFingerprintUsesKernelCanonicalBytes() {
    using namespace kxc;
    using namespace kxc::runtime;
    const ExecutablePlan plan = PlanForExactAbi();
    const String first = ComputeCallExactAbiFingerprint(ModuleForExactAbi(4), plan, 0);
    const String changed = ComputeCallExactAbiFingerprint(ModuleForExactAbi(8), plan, 0);
    TEST_CHECK(std::string(first) != std::string(changed) &&
                   std::string(first).find(
                       "runtime-call-exact-abi-v2-canonical-kernel-abi:") == 0,
               "module-entry exact ABI must change with canonical KernelSignature fields");
    return true;
}

bool TestTaskKindsAndInvalidEnums() {
    using namespace kxc;
    using namespace kxc::runtime;
    const Device cpu = Device::CPU();
    TEST_CHECK(Throws([&] {
                   TaskSpec task(0, TaskKind::kKernel, cpu, {}, {1}, {});
               }) &&
                   Throws([&] {
                       TaskSpec task(0, TaskKind::kAllocate, cpu, {}, {1}, {},
                                     String(), 0, 0, 0);
                   }) &&
                   Throws([&] {
                       TaskSpec task(0, TaskKind::kAllocate, cpu, {}, {1}, {},
                                     String(), 0, 0, 3);
                   }) &&
                   Throws([&] {
                       TaskSpec task(0, static_cast<TaskKind>(1), cpu, {}, {}, {});
                   }) &&
                   Throws([&] {
                       RegionSpec region(0, static_cast<RegionKind>(1), "", {0},
                                         {}, {}, {});
                   }),
               "only kernel, allocate, and per-call enum values are accepted");
    return true;
}

bool TestStaticCyclesAndMissingDependencies() {
    using namespace kxc;
    using namespace kxc::runtime;
    const Device cpu = Device::CPU();
    TEST_CHECK(Throws([&] {
                   FrozenTaskPlan plan(
                       kFrozenTaskPlanVersion,
                       {ValueSpec(0, 0, {-1, 4}, Float32(), cpu, true),
                        ValueSpec(1, 1, {3, 4}, Float32(), cpu, false, false,
                                  true)},
                       {TaskSpec(0, TaskKind::kAllocate, cpu, {}, {1}, {},
                                 String(), 0, 0, 16),
                        TaskSpec(1, TaskKind::kKernel, cpu, {0}, {1}, {0},
                                 "dynamic")},
                       {RegionSpec(0, RegionKind::kPerCall, "", {0, 1}, {0},
                                   {1}, {})},
                       {0}, {}, {1});
               }),
               "frozen plans must reject dynamic dimensions");

    const Array<ValueSpec> source{
        ValueSpec(0, 0, {1}, Float32(), cpu, true, false, true),
    };
    TEST_CHECK(Throws([&] {
                   FrozenTaskPlan plan(
                       kFrozenTaskPlanVersion, source,
                       {TaskSpec(0, TaskKind::kKernel, cpu, {}, {0}, {1}, "a"),
                        TaskSpec(1, TaskKind::kKernel, cpu, {}, {0}, {0}, "b")},
                       {RegionSpec(0, RegionKind::kPerCall, "", {0, 1}, {0},
                                   {}, {})},
                       {0}, {}, {0});
               }) &&
                   Throws([&] {
                       FrozenTaskPlan plan(
                           kFrozenTaskPlanVersion, source,
                           {TaskSpec(0, TaskKind::kKernel, cpu, {}, {0}, {99},
                                     "a")},
                           {RegionSpec(0, RegionKind::kPerCall, "", {0}, {0},
                                       {}, {})},
                           {0}, {}, {0});
                   }),
               "cycles and unknown task dependencies must fail closed");
    return true;
}

bool TestProducerAllocationAndBoundaryValidation() {
    using namespace kxc;
    using namespace kxc::runtime;
    const Device cpu = Device::CPU();
    const Array<ValueSpec> values{
        ValueSpec(0, 0, {1}, Float32(), cpu, true),
        ValueSpec(1, 1, {1}, Float32(), cpu, false, false, true),
    };
    TEST_CHECK(Throws([&] {
                   FrozenTaskPlan plan(
                       kFrozenTaskPlanVersion, values,
                       {TaskSpec(1, TaskKind::kKernel, cpu, {0}, {1}, {}, "k")},
                       {RegionSpec(0, RegionKind::kPerCall, "", {1}, {0}, {1},
                                   {})},
                       {0}, {}, {1});
               }) &&
                   Throws([&] {
                       FrozenTaskPlan plan(
                           kFrozenTaskPlanVersion, values,
                           {TaskSpec(0, TaskKind::kAllocate, cpu, {}, {1}, {},
                                     String(), 0, 0, 1),
                            TaskSpec(1, TaskKind::kKernel, cpu, {0}, {1}, {},
                                     "k")},
                           {RegionSpec(0, RegionKind::kPerCall, "", {0, 1}, {0},
                                       {1}, {})},
                           {0}, {}, {1});
                   }) &&
                   Throws([&] {
                       FrozenTaskPlan plan(
                           kFrozenTaskPlanVersion, values,
                           {TaskSpec(0, TaskKind::kAllocate, cpu, {}, {1}, {},
                                     String(), 0, 0, 1),
                            TaskSpec(1, TaskKind::kKernel, cpu, {0}, {1}, {0},
                                     "k")},
                           {RegionSpec(0, RegionKind::kPerCall, "", {0, 1}, {},
                                       {1}, {})},
                           {0}, {}, {1});
                   }),
               "producer, allocation dependency, and region boundaries must validate");
    return true;
}

bool TestDataOrderAndStorageSharingGuards() {
    using namespace kxc;
    using namespace kxc::runtime;
    const Device cpu = Device::CPU();
    TEST_CHECK(Throws([&] {
                   FrozenTaskPlan plan(
                       kFrozenTaskPlanVersion,
                       {ValueSpec(0, 0, {1}, Float32(), cpu, true),
                        ValueSpec(1, 1, {1}, Float32(), cpu),
                        ValueSpec(2, 2, {1}, Float32(), cpu, false, false,
                                  true)},
                       {TaskSpec(0, TaskKind::kAllocate, cpu, {}, {1}, {},
                                 String(), 0, 0, 1),
                        TaskSpec(1, TaskKind::kKernel, cpu, {0}, {1}, {0}, "a"),
                        TaskSpec(2, TaskKind::kAllocate, cpu, {}, {2}, {},
                                 String(), 0, 0, 1),
                        TaskSpec(3, TaskKind::kKernel, cpu, {1}, {2}, {2}, "b")},
                       {RegionSpec(0, RegionKind::kPerCall, "", {0, 1}, {0}, {1},
                                   {}),
                        RegionSpec(1, RegionKind::kPerCall, "", {2, 3}, {1}, {2},
                                   {})},
                       {0}, {}, {2});
               }) &&
                   Throws([&] {
                       FrozenTaskPlan plan(
                           kFrozenTaskPlanVersion,
                           {ValueSpec(0, 0, {1}, Float32(), cpu, true),
                            ValueSpec(1, 7, {1}, Float32(), cpu),
                            ValueSpec(2, 7, {1}, Float32(), cpu),
                            ValueSpec(3, 3, {1}, Float32(), cpu, false, false,
                                      true)},
                           {TaskSpec(10, TaskKind::kAllocate, cpu, {}, {1}, {},
                                     String(), 0, 0, 1),
                            TaskSpec(11, TaskKind::kKernel, cpu, {0}, {1}, {10},
                                     "a"),
                            TaskSpec(20, TaskKind::kAllocate, cpu, {}, {2}, {},
                                     String(), 0, 0, 1),
                            TaskSpec(21, TaskKind::kKernel, cpu, {0}, {2}, {20},
                                     "b"),
                            TaskSpec(30, TaskKind::kAllocate, cpu, {}, {3}, {11, 21},
                                     String(), 0, 0, 1),
                            TaskSpec(31, TaskKind::kKernel, cpu, {1, 2}, {3},
                                     {30, 11, 21}, "join")},
                           {RegionSpec(0, RegionKind::kPerCall, "",
                                       {10, 11, 20, 21, 30, 31}, {0}, {3}, {})},
                           {0}, {}, {3});
                   }),
               "data order and overlapping storage sharing must fail");
    return true;
}

bool TestStreamAndOrderedEffectGuards() {
    using namespace kxc;
    using namespace kxc::runtime;
    const Device cpu = Device::CPU();
    TEST_CHECK(Throws([&] {
                   FrozenTaskPlan plan(
                       kFrozenTaskPlanVersion,
                       {ValueSpec(0, 0, {1}, Float32(), cpu, true),
                        ValueSpec(1, 1, {1}, Float32(), cpu, false, false,
                                  true)},
                       {TaskSpec(0, TaskKind::kAllocate, cpu, {}, {1}, {},
                                 String(), 0, 1, 1)},
                       {RegionSpec(0, RegionKind::kPerCall, "", {0}, {}, {1},
                                   {})},
                       {0}, {}, {1});
               }),
               "v1 must reject non-zero streams before lifecycle support exists");

    const Array<ValueSpec> values{
        ValueSpec(0, 0, {1}, Float32(), cpu, true),
        ValueSpec(1, 1, {1}, Float32(), cpu, false, false, true),
        ValueSpec(2, 2, {1}, Float32(), cpu, false, false, true),
    };
    const Array<TaskSpec> tasks{
        TaskSpec(10, TaskKind::kAllocate, cpu, {}, {1}, {}, String(), 0, 0, 1),
        TaskSpec(11, TaskKind::kKernel, cpu, {0}, {1}, {10}, "left"),
        TaskSpec(20, TaskKind::kAllocate, cpu, {}, {2}, {}, String(), 0, 0, 1),
        TaskSpec(21, TaskKind::kKernel, cpu, {0}, {2}, {20}, "right"),
    };
    TEST_CHECK(Throws([&] {
                   FrozenTaskPlan plan(
                       kFrozenTaskPlanVersion, values, tasks,
                       {RegionSpec(0, RegionKind::kPerCall, "", {10, 11}, {0},
                                   {1}, {}, RegionEffect::kOrdered),
                        RegionSpec(1, RegionKind::kPerCall, "", {20, 21}, {0},
                                   {2}, {}, RegionEffect::kOrdered)},
                       {0}, {}, {1, 2});
               }) &&
                   Throws([&] {
                       FrozenTaskPlan plan(
                           kFrozenTaskPlanVersion, values, tasks,
                           {RegionSpec(0, RegionKind::kPerCall, "",
                                       {10, 11, 20, 21}, {0}, {1, 2}, {},
                                       RegionEffect::kOrdered)},
                           {0}, {}, {1, 2});
                   }),
               "ordered regions and ordered actions require total dependencies");
    return true;
}

bool TestStorageContractAndObjectGuards() {
    using namespace kxc;
    using namespace kxc::runtime;
    const Device cpu = Device::CPU();
    TEST_CHECK(Throws([&] {
                   FrozenTaskPlan plan(
                       kFrozenTaskPlanVersion,
                       {ValueSpec(0, 0, {1}, Float32(), cpu, true),
                        ValueSpec(1, 7, {1}, Float32(), cpu),
                        ValueSpec(2, 7, {2}, Float32(), cpu),
                        ValueSpec(3, 3, {2}, Float32(), cpu, false, false,
                                  true)},
                       {TaskSpec(10, TaskKind::kAllocate, cpu, {}, {1}, {},
                                 String(), 0, 0, 1),
                        TaskSpec(11, TaskKind::kKernel, cpu, {0}, {1}, {10}, "a"),
                        TaskSpec(20, TaskKind::kAllocate, cpu, {}, {2}, {11},
                                 String(), 0, 0, 1),
                        TaskSpec(21, TaskKind::kKernel, cpu, {0}, {2}, {20, 11},
                                 "b"),
                        TaskSpec(30, TaskKind::kAllocate, cpu, {}, {3}, {21},
                                 String(), 0, 0, 1),
                        TaskSpec(31, TaskKind::kKernel, cpu, {2}, {3}, {30, 21},
                                 "c")},
                       {RegionSpec(0, RegionKind::kPerCall, "",
                                   {10, 11, 20, 21, 30, 31}, {0}, {3}, {})},
                       {0}, {}, {3});
               }) &&
                   Throws([&] { FrozenTaskPlan wrong(ObjectRef(Device::CPU())); }) &&
                   Throws([&] {
                       FrozenTaskPlan undefined{ObjectRef()};
                       (void)undefined.tasks();
                   }),
               "storage contracts and invalid object access must fail closed");
    return true;
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"frozen_dto_and_deterministic_topology", TestFrozenDtoAndDeterministicTopology},
        {"artifact_declaration_contracts", TestArtifactDeclarationContracts},
        {"exact_abi_kernel_canonical", TestExactAbiFingerprintUsesKernelCanonicalBytes},
        {"task_kinds_and_invalid_enums", TestTaskKindsAndInvalidEnums},
        {"static_cycles_and_missing_dependencies", TestStaticCyclesAndMissingDependencies},
        {"producer_allocation_and_boundary_validation", TestProducerAllocationAndBoundaryValidation},
        {"data_order_and_storage_sharing_guards", TestDataOrderAndStorageSharingGuards},
        {"stream_and_ordered_effect_guards", TestStreamAndOrderedEffectGuards},
        {"storage_contract_and_object_guards", TestStorageContractAndObjectGuards},
    };
    int failures = 0;
    for (const auto& test : tests) {
        try {
            if (!test.second()) {
                ++failures;
                continue;
            }
            std::cout << "[PASS] " << test.first << '\n';
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << test.first << ": " << error.what() << '\n';
            ++failures;
        }
    }
    return failures == 0 ? 0 : 1;
}
