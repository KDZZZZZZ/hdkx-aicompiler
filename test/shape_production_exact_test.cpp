#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../src/compiler/internal/lowered_graph.h"
#include "../src/compiler/internal/primitive_cache.h"
#include "kxc/compiler/experimental_identity.h"
#include "kxc/compiler/shape_exact.h"
#include "kxc/relay/op.h"
#include "kxc/relay/transforms/infer_type.h"
#include "kxc/runtime/session.h"

namespace {
namespace shape_exact = kxc::api::experimental::shape_exact::v1;

#define CHECK(x, m) do { if (!(x)) { std::cerr << "[FAIL] " << __FUNCTION__ << ": " << m << "\n"; return false; } } while (0)

bool Throws(const std::function<void()>& fn) {
    try { fn(); } catch (const std::exception&) { return true; }
    return false;
}

kxc::Function TwoUnitGraph(const std::string& dtype = "float32", int64_t extent = 4) {
    kxc::TensorType type({extent}, dtype);
    kxc::Var x("x", type), y("y", type), z("z", type);
    return kxc::Function({x, y, z},
        kxc::Call(kxc::relay::Op::Get("mul"),
                  {kxc::Call(kxc::relay::Op::Get("add"), {x, y}), z}));
}

kxc::Function RenumberedGraph() {
    const kxc::TensorType type({4}, "float32");
    const kxc::Var x("x", type), y("y", type), z("z", type);
    return kxc::Function(
        {x, y, z},
        kxc::Tuple({kxc::Call(kxc::relay::Op::Get("mul"), {x, z}),
                    kxc::Call(kxc::relay::Op::Get("add"), {x, y})}));
}

kxc::api::CompileConfig Config() {
    return kxc::api::CompileConfig::Create(kxc::BuildTarget(kxc::Device::CPU()));
}

std::pair<kxc::TargetNode*, kxc::Target> MutableTargetSource() {
    const kxc::Target source = kxc::BuildTarget(kxc::Device::CPU());
    auto* node = new kxc::TargetNode();
    node->kind = source->kind;
    node->device_type = source->device_type;
    node->device_id = source->device_id;
    node->attrs = source->attrs;
    return {node, kxc::Target(kxc::ObjectRef(node))};
}

bool SameCounters(const shape_exact::ShapeExactPreparationCounters& left,
                  const shape_exact::ShapeExactPreparationCounters& right) {
    return left.execution_contract_resolutions ==
               right.execution_contract_resolutions &&
           left.relay_graph_pipelines == right.relay_graph_pipelines &&
           left.capability_boundary_checks ==
               right.capability_boundary_checks &&
           left.value_graph_builds == right.value_graph_builds &&
           left.partitions == right.partitions;
}

kxc::api::PlanAbiFingerprint PlanAbi(
    const kxc::api::CompiledModule& module,
    const kxc::runtime::ExecutablePlan& plan,
    const std::vector<kxc::api::ArtifactPin>& pins) {
    std::vector<kxc::api::OrderedArtifactIdentity> artifacts;
    const kxc::Array<kxc::runtime::KernelCall> calls = plan.calls();
    artifacts.reserve(calls.size());
    for (size_t index = 0; index < calls.size(); ++index) {
        artifacts.push_back({index, std::string(calls[index]->symbol),
                             pins[index].record().artifact_key});
    }
    return kxc::api::BuildPlanAbiFingerprint(module, plan, artifacts);
}

kxc::runtime::NDArray FloatArray(const std::vector<float>& values) {
    kxc::runtime::NDArray result = kxc::runtime::NDArray::Zeros(
        {static_cast<int64_t>(values.size())},
        kxc::runtime::DataTypeFromString("float32"), kxc::Device::CPU());
    result.CopyFromBytes(values.data(), values.size() * sizeof(float));
    return result;
}

kxc::Function ConstantGraphWithPayload(kxc::runtime::NDArray payload) {
    const kxc::TensorType type({4}, "float32");
    const kxc::Var input("input", type);
    const kxc::Constant constant(std::move(payload));
    return kxc::Function(
        {input}, kxc::Call(kxc::relay::Op::Get("add"), {input, constant}));
}

kxc::Function ConstantGraph(float value) {
    return ConstantGraphWithPayload(
        FloatArray({value, value, value, value}));
}

kxc::Function PassThroughGraph() {
    const kxc::Var input("input", kxc::TensorType({4}, "float32"));
    return kxc::Function({input}, input);
}

struct MutableAddGraph final {
    kxc::Var x;
    kxc::Var y;
    kxc::Call call;
    kxc::Function function;
};

MutableAddGraph AddGraphWithHandles() {
    const kxc::TensorType type({4}, "float32");
    kxc::Var x("x", type);
    kxc::Var y("y", type);
    kxc::Call call(kxc::relay::Op::Get("add"), {x, y});
    return {x, y, call, kxc::Function({x, y}, call)};
}

struct MutableTransposeGraph final {
    kxc::TensorType type;
    kxc::Var input;
    kxc::relay::TransposeAttrs attrs;
    kxc::Call call;
    kxc::Function function;
};

MutableTransposeGraph TransposeGraphWithHandles() {
    kxc::TensorType type({2, 3}, "float32");
    kxc::Var input("input", type);
    kxc::relay::TransposeAttrs attrs =
        kxc::relay::TransposeAttrs::Create({1, 0});
    kxc::Call call(kxc::relay::Op::Get("transpose"), {input}, attrs);
    return {type, input, attrs, call, kxc::Function({input}, call)};
}

kxc::runtime::NDArray FloatMatrix(
    const std::vector<float>& values, kxc::Array<int64_t> shape) {
    kxc::runtime::NDArray result = kxc::runtime::NDArray::Empty(
        std::move(shape), kxc::runtime::DataTypeFromString("float32"),
        kxc::Device::CPU());
    result.CopyFromBytes(values.data(), values.size() * sizeof(float));
    return result;
}

bool TestGateAndPreparation() {
    using shape_exact::ProductionExactShapeAdapter;
#if !KXC_ENABLE_SHAPE_PRODUCTION_EXACT
    CHECK(!ProductionExactShapeAdapter::IsEnabled(), "default gate must report disabled");
    CHECK(Throws([] { (void)ProductionExactShapeAdapter::PrepareGraphTemplate(TwoUnitGraph(), Config()); }),
          "disabled production adapter must fail closed");
    return true;
#else
    CHECK(ProductionExactShapeAdapter::IsEnabled(), "enabled gate must report enabled");
    const auto prepared = ProductionExactShapeAdapter::PrepareGraphTemplate(TwoUnitGraph(), Config());
    const auto counters = prepared.counters();
    CHECK(counters.execution_contract_resolutions == 1 && counters.relay_graph_pipelines == 1 &&
          counters.capability_boundary_checks == 2 && counters.value_graph_builds == 1 &&
          counters.partitions == 1, "preparation must execute each graph step exactly once");
    CHECK(prepared.unit_count() == 2 && !prepared.multi_profile_supported(),
          "concrete Relay template must retain two units and hard-gate multi-profile");
    const auto oracle = ProductionExactShapeAdapter::InstantiateExactProfile(
        prepared, kxc::shape::experimental::v1::BindingSet());
    CHECK(oracle.profile().bindings().bindings().empty(), "only empty concrete profile is accepted");
    CHECK(Throws([&] { (void)ProductionExactShapeAdapter::InstantiateExactProfile(
              prepared, kxc::shape::experimental::v1::BindingSet({{"s", 4}})); }),
          "non-empty symbolic binding must fail before compilation/cache work");
#if !KXC_USE_LLVM
    kxc::api::internal::ClearPrimitiveCacheForTesting();
    const auto before = kxc::api::internal::GetPrimitiveCacheStats();
    CHECK(Throws([&] { (void)ProductionExactShapeAdapter::AssembleExactPlan(prepared, oracle); }),
          "LLVM-off backend assembly must fail clearly after true preparation");
    const auto after = kxc::api::internal::GetPrimitiveCacheStats();
    CHECK(before.entries == after.entries && before.misses == after.misses &&
              before.in_flight == after.in_flight &&
              before.failures == after.failures,
          "known backend unavailability must fail before cache acquisition");
#endif
    return true;
#endif
}

bool TestCallerRelayIsolation() {
#if !KXC_ENABLE_SHAPE_PRODUCTION_EXACT
    return true;
#else
    using shape_exact::ProductionExactShapeAdapter;
    MutableTransposeGraph source = TransposeGraphWithHandles();
    CHECK(!source.function.checked_type().defined() &&
              !source.call.checked_type().defined() &&
              !source.input.checked_type().defined(),
          "caller graph must begin untyped for the isolation attack");
    const auto prepared = ProductionExactShapeAdapter::PrepareGraphTemplate(
        source.function, Config());
    CHECK(!source.function.checked_type().defined() &&
              !source.call.checked_type().defined() &&
              !source.input.checked_type().defined(),
          "PrepareGraphTemplate must not type-mutate caller-owned Relay nodes");
    const std::string frozen_key =
        prepared.graph_template().key().canonical_bytes();

    auto* call = const_cast<kxc::CallNode*>(source.call.operator->());
    call->op = kxc::relay::Op::Get("sqrt");
    auto* attrs = const_cast<kxc::relay::TransposeAttrsNode*>(
        source.attrs.operator->());
    attrs->perm[0] = 0;
    attrs->perm[1] = 1;
    auto* variable = const_cast<kxc::VarNode*>(source.input.operator->());
    variable->type_annotation = kxc::TensorType({7}, "int32");
    kxc::SetCheckedType(source.input, kxc::TensorType({7}, "int32"));
    auto* function = const_cast<kxc::FunctionNode*>(
        source.function.operator->());
    function->body = source.input;
    kxc::SetCheckedType(source.function,
                        kxc::TensorType({7}, "int32"));
    CHECK(prepared.graph_template().key().canonical_bytes() == frozen_key,
          "caller mutation must not alter prepared template identity");

    const auto oracle = ProductionExactShapeAdapter::InstantiateExactProfile(
        prepared, kxc::shape::experimental::v1::BindingSet());
#if !KXC_USE_LLVM
    kxc::api::internal::ClearPrimitiveCacheForTesting();
    const auto before = kxc::api::internal::GetPrimitiveCacheStats();
    CHECK(Throws([&] {
              (void)ProductionExactShapeAdapter::AssembleExactPlan(
                  prepared, oracle);
          }),
          "LLVM-off assembly must still fail at the backend boundary");
    const auto after = kxc::api::internal::GetPrimitiveCacheStats();
    CHECK(before.entries == after.entries && before.misses == after.misses &&
              before.failures == after.failures,
          "post-prepare caller mutation must not reach or poison the cache");
#endif
    return true;
#endif
}

bool TestPreparedConstantSnapshot() {
    const auto payload = FloatArray({1, 1, 1, 1});
    const kxc::Function typed = kxc::relay::InferTypePass(
        ConstantGraphWithPayload(payload));
    const kxc::api::internal::PreparedStaticGraph prepared =
        kxc::api::internal::PrepareStaticGraph(
            typed, kxc::Device::CPU(),
            kxc::BuildTarget(kxc::Device::CPU()),
            kxc::String("prepared-constant-freeze-test"));
    const std::vector<float> mutated{9, 9, 9, 9};
    payload.CopyFromBytes(mutated.data(), payload.NBytes());
    for (const kxc::api::internal::PrimitiveUnit& unit :
         prepared.partitioned.units) {
        (void)kxc::api::internal::LowerPrimitiveUnit(
            prepared.partitioned.value_graph.values, unit);
    }
    const kxc::api::internal::ValueInfo& constant =
        prepared.partitioned.value_graph.values[
            prepared.partitioned.constant_value_ids[0]];
    const auto* frozen = constant.source.As<kxc::ConstantNode>();
    CHECK(frozen && frozen->data.defined(),
          "prepared constant snapshot must retain one constant");
    std::vector<float> actual(4);
    frozen->data.CopyToBytes(actual.data(), actual.size() * sizeof(float));
    CHECK(actual == std::vector<float>({1, 1, 1, 1}),
          "prepared graph must deep-freeze constant payload bytes");
    return true;
}

bool TestNegativesBeforeCache() {
#if !KXC_ENABLE_SHAPE_PRODUCTION_EXACT
    return true;
#else
    using shape_exact::ProductionExactShapeAdapter;
    kxc::api::internal::ClearPrimitiveCacheForTesting();
    const auto before = kxc::api::internal::GetPrimitiveCacheStats();
    CHECK(Throws([] { (void)ProductionExactShapeAdapter::PrepareGraphTemplate(TwoUnitGraph("float32", -1), Config()); }),
          "legacy -1 must be rejected");
    const auto integer_graph =
        ProductionExactShapeAdapter::PrepareGraphTemplate(
            TwoUnitGraph("int32"), Config());
    CHECK(integer_graph.unit_count() == 2,
          "Shape preparation must not own or restrict the compiler dtype contract");
    CHECK(Throws([] { (void)ProductionExactShapeAdapter::PrepareGraphTemplate(PassThroughGraph(), Config()); }),
          "a no-compute graph must be rejected during preparation");
    const auto left = ProductionExactShapeAdapter::PrepareGraphTemplate(
        TwoUnitGraph(), Config());
    const auto right = ProductionExactShapeAdapter::PrepareGraphTemplate(
        TwoUnitGraph("float32", 5), Config());
    const auto foreign = ProductionExactShapeAdapter::InstantiateExactProfile(
        right, kxc::shape::experimental::v1::BindingSet());
    CHECK(Throws([&] { (void)ProductionExactShapeAdapter::AssembleExactPlan(left, foreign); }),
          "foreign exact oracle must fail before compiler/cache work");

    auto [target_node, target_source] = MutableTargetSource();
    const kxc::api::CompileConfig config =
        kxc::api::CompileConfig::Create(target_source);
    const auto frozen = ProductionExactShapeAdapter::PrepareGraphTemplate(
        TwoUnitGraph(), config);
    const std::string frozen_key =
        frozen.graph_template().key().canonical_bytes();
    const kxc::api::UnitSemanticKey target_test_semantic(
        "shape-production-target-test");
    const kxc::api::PrimitiveArtifactKey artifact_before =
        kxc::api::internal::BuildPrimitiveArtifactKey(
            target_test_semantic, config->target, "pipeline-test",
            "schedule-test", "backend-test");
    ++target_node->attrs.max_shared_memory_per_block;
    const kxc::api::PrimitiveArtifactKey artifact_after_source_mutation =
        kxc::api::internal::BuildPrimitiveArtifactKey(
            target_test_semantic, config->target, "pipeline-test",
            "schedule-test", "backend-test");
    CHECK(frozen.graph_template().key().canonical_bytes() == frozen_key &&
              artifact_before == artifact_after_source_mutation,
          "CompileConfig target snapshot must isolate prepared identity and artifacts");
    const kxc::api::CompileConfig changed_config =
        kxc::api::CompileConfig::Create(target_source);
    const kxc::api::PrimitiveArtifactKey artifact_changed =
        kxc::api::internal::BuildPrimitiveArtifactKey(
            target_test_semantic, changed_config->target, "pipeline-test",
            "schedule-test", "backend-test");
    const auto changed = ProductionExactShapeAdapter::PrepareGraphTemplate(
        TwoUnitGraph(), changed_config);
    CHECK(artifact_before != artifact_changed &&
              frozen.graph_template().key() == changed.graph_template().key(),
          "a new config must capture scheduling target changes without changing graph semantics");
    const std::string changed_key =
        changed.graph_template().key().canonical_bytes();
    target_node->attrs.available_global_memory ^= 1;
    const kxc::api::CompileConfig volatile_config =
        kxc::api::CompileConfig::Create(target_source);
    const kxc::api::PrimitiveArtifactKey artifact_volatile_only =
        kxc::api::internal::BuildPrimitiveArtifactKey(
            target_test_semantic, volatile_config->target, "pipeline-test",
            "schedule-test", "backend-test");
    const auto volatile_only =
        ProductionExactShapeAdapter::PrepareGraphTemplate(
            TwoUnitGraph(), volatile_config);
    CHECK(volatile_only.graph_template().key().canonical_bytes() == changed_key &&
              artifact_changed == artifact_volatile_only,
          "volatile available memory must not split codegen identity");
    const auto changed_profile =
        ProductionExactShapeAdapter::InstantiateExactProfile(
            changed, kxc::shape::experimental::v1::BindingSet());
    const auto frozen_profile =
        ProductionExactShapeAdapter::InstantiateExactProfile(
            frozen, kxc::shape::experimental::v1::BindingSet());
    CHECK(changed_profile.profile().key() == frozen_profile.profile().key(),
          "ShapeProfileKey must remain independent of target capability");
    const auto constant_one =
        ProductionExactShapeAdapter::PrepareGraphTemplate(
            ConstantGraph(1.0F), Config());
    const auto constant_two =
        ProductionExactShapeAdapter::PrepareGraphTemplate(
            ConstantGraph(2.0F), Config());
    CHECK(!(constant_one.graph_template().key() ==
            constant_two.graph_template().key()) &&
              constant_one.graph_template().ordered_units()[0].semantic_key ==
                  constant_two.graph_template().ordered_units()[0].semantic_key,
          "constant payload must bind graph/profile identity but not reusable unit semantics");
    const auto constant_one_profile =
        ProductionExactShapeAdapter::InstantiateExactProfile(
            constant_one, kxc::shape::experimental::v1::BindingSet());
    const auto constant_two_profile =
        ProductionExactShapeAdapter::InstantiateExactProfile(
            constant_two, kxc::shape::experimental::v1::BindingSet());
    CHECK(!(constant_one_profile.profile().key() ==
            constant_two_profile.profile().key()),
          "different frozen constant pools must produce different exact profiles");
    const auto after = kxc::api::internal::GetPrimitiveCacheStats();
    CHECK(before.entries == after.entries && before.misses == after.misses,
          "negative/preparation-only exact requests must not mutate cache");
    return true;
#endif
}

#if KXC_USE_LLVM && KXC_ENABLE_SHAPE_PRODUCTION_EXACT
bool TestLLVMRelayMutationCannotPoisonCache() {
    using shape_exact::ProductionExactShapeAdapter;
    MutableAddGraph attacked_source = AddGraphWithHandles();
    kxc::api::internal::ClearPrimitiveCacheForTesting();
    const auto attacked_prepared =
        ProductionExactShapeAdapter::PrepareGraphTemplate(
            attacked_source.function, Config());
    const auto attacked_oracle =
        ProductionExactShapeAdapter::InstantiateExactProfile(
            attacked_prepared, kxc::shape::experimental::v1::BindingSet());
    auto* attacked_call = const_cast<kxc::CallNode*>(
        attacked_source.call.operator->());
    attacked_call->op = kxc::relay::Op::Get("mul");
    const kxc::relay::Op& registered_add = kxc::relay::Op::Get("add");
    const kxc::VirtualDevice saved_op_device =
        registered_add.virtual_device();
    registered_add.set_virtual_device(
        kxc::VirtualDevice::ForDevice(kxc::Device::CUDA()));
    shape_exact::ExactPlanVariant attacked;
    try {
        attacked = ProductionExactShapeAdapter::AssembleExactPlan(
            attacked_prepared, attacked_oracle);
    } catch (...) {
        registered_add.set_virtual_device(saved_op_device);
        throw;
    }
    registered_add.set_virtual_device(saved_op_device);
    const auto after_attacked =
        kxc::api::internal::GetPrimitiveCacheStats();

    MutableAddGraph clean_source = AddGraphWithHandles();
    const auto clean_prepared =
        ProductionExactShapeAdapter::PrepareGraphTemplate(
            clean_source.function, Config());
    const auto clean_oracle =
        ProductionExactShapeAdapter::InstantiateExactProfile(
            clean_prepared, kxc::shape::experimental::v1::BindingSet());
    const auto clean = ProductionExactShapeAdapter::AssembleExactPlan(
        clean_prepared, clean_oracle);
    const auto after_clean = kxc::api::internal::GetPrimitiveCacheStats();
    CHECK(after_attacked.misses == 1 && after_clean.misses == 1 &&
              after_clean.hits >= after_attacked.hits + 1 &&
              attacked.artifact_pins()[0].record().artifact_key ==
                  clean.artifact_pins()[0].record().artifact_key,
          "mutated caller Call must neither change code nor publish new code under the old key");

    const kxc::Array<kxc::runtime::NDArray> inputs{
        FloatArray({2, 3, 4, 5}), FloatArray({3, 4, 5, 6})};
    const shape_exact::ExactPlanVariant* variants[]{&attacked, &clean};
    for (const auto* variant : variants) {
        kxc::runtime::RuntimeSession session(variant->module(),
                                             variant->plan());
        const auto outputs = session.Run(inputs);
        std::vector<float> actual(4);
        outputs[0].CopyToBytes(actual.data(), actual.size() * sizeof(float));
        CHECK(actual == std::vector<float>({5, 7, 9, 11}),
              "old add semantic key must always execute add, never caller-mutated mul");
    }
    return true;
}

bool TestLLVMRelayAttrsTypesAndBodySnapshot() {
    using shape_exact::ProductionExactShapeAdapter;
    MutableTransposeGraph source = TransposeGraphWithHandles();
    const auto prepared = ProductionExactShapeAdapter::PrepareGraphTemplate(
        source.function, Config());
    const auto oracle = ProductionExactShapeAdapter::InstantiateExactProfile(
        prepared, kxc::shape::experimental::v1::BindingSet());

    auto* attrs = const_cast<kxc::relay::TransposeAttrsNode*>(
        source.attrs.operator->());
    attrs->perm[0] = 0;
    attrs->perm[1] = 1;
    auto* variable = const_cast<kxc::VarNode*>(source.input.operator->());
    variable->type_annotation = kxc::TensorType({7}, "int32");
    kxc::SetCheckedType(source.input, kxc::TensorType({7}, "int32"));
    auto* function = const_cast<kxc::FunctionNode*>(
        source.function.operator->());
    function->body = source.input;
    kxc::SetCheckedType(source.function,
                        kxc::TensorType({7}, "int32"));

    const auto variant = ProductionExactShapeAdapter::AssembleExactPlan(
        prepared, oracle);
    kxc::runtime::RuntimeSession session(variant.module(), variant.plan());
    const auto outputs = session.Run(
        {FloatMatrix({1, 2, 3, 4, 5, 6}, {2, 3})});
    std::vector<float> actual(6);
    outputs[0].CopyToBytes(actual.data(), actual.size() * sizeof(float));
    CHECK(actual == std::vector<float>({1, 4, 2, 5, 3, 6}),
          "assembly must lower the frozen transpose attrs/types/body snapshot");
    return true;
}

bool TestLLVMNormalAndExactShareStaticAssembly() {
    using shape_exact::ProductionExactShapeAdapter;
    kxc::api::internal::ClearPrimitiveCacheForTesting();
    const kxc::api::CompiledGraph normal =
        kxc::api::Compiler::Compile(TwoUnitGraph(), Config());
    const auto prepared = ProductionExactShapeAdapter::PrepareGraphTemplate(
        TwoUnitGraph(), Config());
    const auto oracle = ProductionExactShapeAdapter::InstantiateExactProfile(
        prepared, kxc::shape::experimental::v1::BindingSet());
    const auto exact = ProductionExactShapeAdapter::AssembleExactPlan(
        prepared, oracle);
    bool same_ordered_keys =
        normal.artifact_pins().size() == exact.artifact_pins().size();
    for (size_t index = 0; index < normal.artifact_pins().size(); ++index) {
        same_ordered_keys = same_ordered_keys &&
            normal.artifact_pins()[index].record().artifact_key ==
                exact.artifact_pins()[index].record().artifact_key;
    }
    const bool same_plan_abi = PlanAbi(
        normal.module(), normal.plan(), normal.artifact_pins()) == PlanAbi(
            exact.module(), exact.plan(), exact.artifact_pins());
    kxc::api::internal::ClearPrimitiveCacheForTesting();
    CHECK(normal.graph_semantic_key() ==
              exact.plan_variant_key().graph_semantic_key() &&
              same_ordered_keys && same_plan_abi,
          "normal and exact paths must publish the same static graph ABI");
    return true;
}

bool TestLLVMExactCacheRuntimeAndLifecycle() {
    using shape_exact::ProductionExactShapeAdapter;
    const auto inputs = kxc::Array<kxc::runtime::NDArray>{
        FloatArray({1, 2, 3, 4}), FloatArray({5, 6, 7, 8}),
        FloatArray({2, 3, 4, 5})};
    std::vector<float> actual(4);

    kxc::api::internal::ClearPrimitiveCacheForTesting();
    {
        const auto prepared =
            ProductionExactShapeAdapter::PrepareGraphTemplate(
                TwoUnitGraph(), Config());
        const auto frozen_counters = prepared.counters();
        const auto oracle =
            ProductionExactShapeAdapter::InstantiateExactProfile(
                prepared, kxc::shape::experimental::v1::BindingSet());
        const auto first = ProductionExactShapeAdapter::AssembleExactPlan(
            prepared, oracle);
        const auto after_first =
            kxc::api::internal::GetPrimitiveCacheStats();
        CHECK(first.artifact_pins().size() == 2 &&
                  first.plan().calls().size() == 2 &&
                  first.module().entry_count() == 2 &&
                  after_first.misses == 2,
              "first exact assembly must publish two real primitive artifacts");
        for (const auto& pin : first.artifact_pins()) {
            CHECK(kxc::api::internal::LookupPrimitiveCache(
                          pin.record().artifact_key).defined(),
                  "every retained production pin must be discoverable by its full key");
        }
        const auto second = ProductionExactShapeAdapter::AssembleExactPlan(
            prepared, oracle);
        const auto after_second =
            kxc::api::internal::GetPrimitiveCacheStats();
        CHECK(second.artifact_pins().size() == 2 &&
                  after_second.hits >= after_first.hits + 2 &&
                  SameCounters(prepared.counters(), frozen_counters),
              "second exact assembly must hit cache without rerunning graph preparation");
        const auto renumbered_prepared =
            ProductionExactShapeAdapter::PrepareGraphTemplate(
                RenumberedGraph(), Config());
        const auto renumbered_oracle =
            ProductionExactShapeAdapter::InstantiateExactProfile(
                renumbered_prepared,
                kxc::shape::experimental::v1::BindingSet());
        const auto renumbered =
            ProductionExactShapeAdapter::AssembleExactPlan(
                renumbered_prepared, renumbered_oracle);
        const auto after_renumbered =
            kxc::api::internal::GetPrimitiveCacheStats();
        CHECK(renumbered.artifact_pins().size() == 2 &&
                  after_renumbered.hits >= after_second.hits + 2 &&
                  after_renumbered.misses == after_second.misses,
              "graph-local unit renumbering must reuse real semantic artifacts and relocate signatures");
        const auto other_prepared =
            ProductionExactShapeAdapter::PrepareGraphTemplate(
                TwoUnitGraph("float32", 5), Config());
        const auto other_oracle =
            ProductionExactShapeAdapter::InstantiateExactProfile(
                other_prepared,
                kxc::shape::experimental::v1::BindingSet());
        const auto other = ProductionExactShapeAdapter::AssembleExactPlan(
            other_prepared, other_oracle);
        const auto after_other =
            kxc::api::internal::GetPrimitiveCacheStats();
        CHECK(other.artifact_pins().size() == 2 &&
                  after_other.misses >= after_renumbered.misses + 2,
              "different concrete shapes must compile exact artifacts, never fuzzy-reuse a larger or smaller profile");

        const auto mutable_constant = FloatArray({1, 1, 1, 1});
        const auto constant_one_prepared =
            ProductionExactShapeAdapter::PrepareGraphTemplate(
                ConstantGraphWithPayload(mutable_constant), Config());
        const std::vector<float> mutated_constant{9, 9, 9, 9};
        mutable_constant.CopyFromBytes(mutated_constant.data(),
                                       mutable_constant.NBytes());
        const auto constant_one_oracle =
            ProductionExactShapeAdapter::InstantiateExactProfile(
                constant_one_prepared,
                kxc::shape::experimental::v1::BindingSet());
        const auto constant_one = ProductionExactShapeAdapter::AssembleExactPlan(
            constant_one_prepared, constant_one_oracle);
        const auto after_constant_one =
            kxc::api::internal::GetPrimitiveCacheStats();
        const auto constant_two_prepared =
            ProductionExactShapeAdapter::PrepareGraphTemplate(
                ConstantGraph(2.0F), Config());
        const auto constant_two_oracle =
            ProductionExactShapeAdapter::InstantiateExactProfile(
                constant_two_prepared,
                kxc::shape::experimental::v1::BindingSet());
        const auto constant_two = ProductionExactShapeAdapter::AssembleExactPlan(
            constant_two_prepared, constant_two_oracle);
        const auto after_constant_two =
            kxc::api::internal::GetPrimitiveCacheStats();
        CHECK(after_constant_one.misses >= after_other.misses + 1 &&
                  after_constant_two.misses == after_constant_one.misses &&
                  after_constant_two.hits >= after_constant_one.hits + 1 &&
                  constant_one.artifact_pins()[0].record().artifact_key ==
                      constant_two.artifact_pins()[0].record().artifact_key &&
                  !(constant_one.plan_variant_key() ==
                    constant_two.plan_variant_key()),
              "constant payloads must reuse primitive code but split frozen plan identity");
        auto exposed_constants = constant_one.module().constants();
        const std::vector<float> exposed_mutation{7, 7, 7, 7};
        exposed_constants.begin()->second.CopyFromBytes(
            exposed_mutation.data(),
            exposed_mutation.size() * sizeof(float));
        const auto constant_one_reassembled =
            ProductionExactShapeAdapter::AssembleExactPlan(
                constant_one_prepared, constant_one_oracle);
        CHECK(constant_one_reassembled.plan_variant_key() ==
                  constant_one.plan_variant_key(),
              "public constant snapshots must not mutate a variant or its prepared template");
        kxc::runtime::RuntimeSession constant_one_session(
            constant_one.module(), constant_one.plan());
        kxc::runtime::RuntimeSession constant_two_session(
            constant_two.module(), constant_two.plan());
        const auto constant_input = FloatArray({1, 2, 3, 4});
        const auto constant_one_outputs =
            constant_one_session.Run({constant_input});
        constant_one_outputs[0].CopyToBytes(
            actual.data(), actual.size() * sizeof(float));
        CHECK(actual == std::vector<float>({2, 3, 4, 5}),
              "first constant pool must remain bound to its frozen module");
        const auto constant_two_outputs =
            constant_two_session.Run({constant_input});
        constant_two_outputs[0].CopyToBytes(
            actual.data(), actual.size() * sizeof(float));
        CHECK(actual == std::vector<float>({3, 4, 5, 6}),
              "second constant pool must remain bound to its frozen module");

        kxc::runtime::RuntimeSession session(first.module(), first.plan());
        const auto outputs = session.Run(inputs);
        outputs[0].CopyToBytes(actual.data(), actual.size() * sizeof(float));
        CHECK(actual == std::vector<float>({12, 24, 40, 60}),
              "static RuntimeSession must execute exact module numerically");
        CHECK(Throws([&] {
                  (void)session.Run(
                      {FloatArray({1, 2, 3}), inputs[1], inputs[2]});
              }),
              "wrong runtime shape must fail closed");
        const auto wrong_dtype = kxc::runtime::NDArray::Zeros(
            {4}, kxc::runtime::DataTypeFromString("int32"),
            kxc::Device::CPU());
        CHECK(Throws([&] {
                  (void)session.Run({wrong_dtype, inputs[1], inputs[2]});
              }),
              "wrong runtime dtype must fail closed");
    }

    std::unique_ptr<kxc::runtime::RuntimeSession> retained;
    {
        auto prepared = ProductionExactShapeAdapter::PrepareGraphTemplate(
            TwoUnitGraph(), Config());
        auto oracle = ProductionExactShapeAdapter::InstantiateExactProfile(
            prepared, kxc::shape::experimental::v1::BindingSet());
        auto variant = ProductionExactShapeAdapter::AssembleExactPlan(
            prepared, oracle);
        retained = std::make_unique<kxc::runtime::RuntimeSession>(
            variant.module(), variant.plan());
    }
    kxc::api::internal::ClearPrimitiveCacheForTesting();
    const auto retained_outputs = retained->Run(inputs);
    retained_outputs[0].CopyToBytes(actual.data(),
                                    actual.size() * sizeof(float));
    CHECK(actual == std::vector<float>({12, 24, 40, 60}),
          "RuntimeSession must execute after all Shape/compiler variant and cache state is gone");
    return true;
}
#endif

}  // namespace

int main() {
    std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"gate_and_preparation", TestGateAndPreparation},
        {"caller_relay_isolation", TestCallerRelayIsolation},
        {"prepared_constant_snapshot", TestPreparedConstantSnapshot},
        {"negatives_before_cache", TestNegativesBeforeCache},
    };
#if KXC_USE_LLVM && KXC_ENABLE_SHAPE_PRODUCTION_EXACT
    tests.push_back({"llvm_relay_mutation_cache_safety", TestLLVMRelayMutationCannotPoisonCache});
    tests.push_back({"llvm_relay_attrs_types_body_snapshot", TestLLVMRelayAttrsTypesAndBodySnapshot});
    tests.push_back({"llvm_normal_exact_static_assembly", TestLLVMNormalAndExactShareStaticAssembly});
    tests.push_back({"llvm_exact_cache_runtime_lifecycle", TestLLVMExactCacheRuntimeAndLifecycle});
#endif
    int failed = 0;
    for (const auto& test : tests) {
        try {
            if (test.second()) std::cout << "[PASS] " << test.first << "\n";
            else ++failed;
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << test.first << ": " << error.what() << "\n";
            ++failed;
        }
    }
    return failed == 0 ? 0 : 1;
}
