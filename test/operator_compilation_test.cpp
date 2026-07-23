/*! \file test/operator_compilation_test.cpp
 * \brief Characterizes whole-graph lowering and locks per-operator unit counts.
 */

#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../src/compiler/internal/lowered_graph.h"
#include "../src/compiler/internal/primitive_cache.h"
#include "kxc/compiler/compiler.h"
#include "kxc/compiler/lowering/relay_to_tir.h"
#include "kxc/relay/op.h"
#include "kxc/relay/op_attr_types.h"
#include "kxc/relay/op_macros.h"
#include "kxc/relay/transforms/infer_type.h"
#include "kxc/relay/relay.h"
#include "kxc/runtime/session.h"
#include "kxc/te/te.h"

namespace {

#define TEST_CHECK(condition, message)                                           \
    do {                                                                          \
        if (!(condition)) {                                                        \
            std::cerr << "[FAIL] " << __FUNCTION__ << ": " << (message) << "\n"; \
            return false;                                                         \
        }                                                                         \
    } while (0)

struct GraphFixture {
    const char* name;
    kxc::Function function;
    size_t expected_compute_calls;
    int64_t expected_graph_outputs;
};

kxc::Call Add(const kxc::Expr& lhs, const kxc::Expr& rhs) {
    return kxc::Call(kxc::relay::Op::Get("add"), {lhs, rhs});
}

kxc::Call Multiply(const kxc::Expr& lhs, const kxc::Expr& rhs) {
    return kxc::Call(kxc::relay::Op::Get("mul"), {lhs, rhs});
}

void CollectUniqueCalls(const kxc::Expr& expr,
                        std::unordered_set<const kxc::Object*>* calls) {
    if (!expr.defined()) return;
    if (const auto* call = expr.As<kxc::CallNode>()) {
        if (!calls->insert(expr.get()).second) return;
        for (const auto& arg : call->args) CollectUniqueCalls(arg, calls);
        return;
    }
    if (const auto* function = expr.As<kxc::FunctionNode>()) {
        CollectUniqueCalls(function->body, calls);
        return;
    }
    if (const auto* tuple = expr.As<kxc::TupleNode>()) {
        for (const auto& field : tuple->fields) CollectUniqueCalls(field, calls);
        return;
    }
    if (const auto* get_item = expr.As<kxc::TupleGetItemNode>()) {
        CollectUniqueCalls(get_item->tuple, calls);
        return;
    }
    if (const auto* let = expr.As<kxc::LetNode>()) {
        CollectUniqueCalls(let->value, calls);
        CollectUniqueCalls(let->body, calls);
        return;
    }
    if (const auto* branch = expr.As<kxc::IfNode>()) {
        CollectUniqueCalls(branch->cond, calls);
        CollectUniqueCalls(branch->true_branch, calls);
        CollectUniqueCalls(branch->false_branch, calls);
    }
}

size_t CountUniqueCalls(const kxc::Function& function) {
    std::unordered_set<const kxc::Object*> calls;
    CollectUniqueCalls(function, &calls);
    return calls.size();
}

kxc::api::internal::LoweredGraph LowerForTest(
    const kxc::Function& function) {
    return kxc::api::internal::LowerGraph(
        kxc::relay::InferTypePass(function));
}

bool ReadIntAttr(const kxc::tir::PrimFunc& function, const char* key,
                 int64_t* value) {
    const kxc::String attr_key(key);
    if (!function.defined() || !function->attrs.count(attr_key)) return false;
    const auto* integer = function->attrs.at(attr_key).As<kxc::tir::IntImmNode>();
    if (!integer) return false;
    *value = integer->value;
    return true;
}

bool HasGlobalSymbol(const kxc::tir::PrimFunc& function, const char* expected) {
    const kxc::String key("global_symbol");
    if (!function.defined() || !function->attrs.count(key)) return false;
    try {
        return kxc::String(function->attrs.at(key)) == expected;
    } catch (const std::exception&) {
        return false;
    }
}

bool ReadStringAttr(const kxc::tir::PrimFunc& function, const char* key,
                    std::string* value) {
    const kxc::String attr_key(key);
    if (!function.defined() || !function->attrs.count(attr_key)) return false;
    try {
        *value = std::string(kxc::String(function->attrs.at(attr_key)));
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

std::vector<GraphFixture> MakeFixtures() {
    using namespace kxc;
    const TensorType tensor_type({4}, "float32");

    Var chain_x("chain_x", tensor_type);
    Var chain_y("chain_y", tensor_type);
    Var chain_z("chain_z", tensor_type);
    Call chain_first = Add(chain_x, chain_y);
    Call chain_output = Multiply(chain_first, chain_z);

    Var branch_x("branch_x", tensor_type);
    Var branch_y("branch_y", tensor_type);
    Var branch_z("branch_z", tensor_type);
    Call branch_root = Add(branch_x, branch_y);
    Call branch_left = Multiply(branch_root, branch_z);
    Call branch_right = Add(branch_root, branch_z);

    Var diamond_x("diamond_x", tensor_type);
    Var diamond_y("diamond_y", tensor_type);
    Var diamond_z("diamond_z", tensor_type);
    Call diamond_root = Add(diamond_x, diamond_y);
    Call diamond_left = Multiply(diamond_root, diamond_z);
    Call diamond_right = Add(diamond_root, diamond_z);
    Call diamond_output = Add(diamond_left, diamond_right);

    Var tuple_x("tuple_x", tensor_type);
    Var tuple_y("tuple_y", tensor_type);
    Call tuple_left = Add(tuple_x, tuple_y);
    Call tuple_right = Multiply(tuple_x, tuple_y);

    return {
        {"chain", Function({chain_x, chain_y, chain_z}, chain_output), 2, 1},
        {"branch",
         Function({branch_x, branch_y, branch_z},
                  Tuple({branch_left, branch_right})),
         3, 2},
        {"diamond",
         Function({diamond_x, diamond_y, diamond_z}, diamond_output), 4, 1},
        {"tuple_output", Function({tuple_x, tuple_y}, Tuple({tuple_left, tuple_right})),
         2, 2},
    };
}

bool TestWholeGraphCompatibilityBaseline() {
    for (const auto& fixture : MakeFixtures()) {
        const kxc::relay::LoweredFunction lowered =
            kxc::relay::LowerToTIR(fixture.function);
        TEST_CHECK(lowered.defined() && lowered->prim_func.defined(),
                   std::string(fixture.name) + " did not produce a PrimFunc");
        TEST_CHECK(CountUniqueCalls(fixture.function) == fixture.expected_compute_calls,
                   std::string(fixture.name) + " fixture call count changed");
        TEST_CHECK(HasGlobalSymbol(lowered->prim_func, "main"),
                   std::string(fixture.name) +
                       " no longer uses the current whole-graph main symbol");
        int64_t output_count = -1;
        TEST_CHECK(ReadIntAttr(lowered->prim_func, "kxc.output_count", &output_count) &&
                       output_count == fixture.expected_graph_outputs,
                   std::string(fixture.name) + " graph output ABI changed");
    }
    return true;
}

bool TestPerOperatorTargetCardinality() {
    for (const auto& fixture : MakeFixtures()) {
        const size_t expected_units = CountUniqueCalls(fixture.function);
        const kxc::api::internal::LoweredGraph lowered =
            LowerForTest(fixture.function);
        TEST_CHECK(expected_units == fixture.expected_compute_calls,
                   std::string(fixture.name) +
                       " must allocate exactly one target unit per compute Call");
        TEST_CHECK(lowered.primitives.size() == expected_units &&
                       lowered.plan.calls().size() == expected_units,
                   std::string(fixture.name) +
                       " must produce exactly one PrimFunc and KernelCall per compute Call");
        TEST_CHECK(lowered.plan.output_value_ids().size() ==
                       static_cast<size_t>(fixture.expected_graph_outputs),
                   std::string(fixture.name) + " graph output value count changed");

        std::unordered_set<std::string> symbols;
        for (size_t index = 0; index < lowered.primitives.size(); ++index) {
            const auto& primitive = lowered.primitives[index];
            int64_t unit_id = -1;
            std::string identity;
            TEST_CHECK(primitive.lowered.defined() &&
                           primitive.lowered->prim_func.defined() &&
                           ReadIntAttr(primitive.lowered->prim_func, "kxc.unit_id",
                                       &unit_id) &&
                           unit_id == static_cast<int64_t>(index),
                       std::string(fixture.name) + " PrimFunc unit id mismatch");
            TEST_CHECK(ReadStringAttr(primitive.lowered->prim_func,
                                      "kxc.operator_identity", &identity) &&
                           identity == std::string(primitive.operator_identity),
                       std::string(fixture.name) +
                           " PrimFunc operator identity mismatch");
            TEST_CHECK(symbols.insert(std::string(primitive.symbol)).second,
                       std::string(fixture.name) + " PrimFunc symbols must be unique");
        }

        const kxc::Array<kxc::relay::LoweredFunction> public_results =
            kxc::relay::LowerOperatorCallsToTIR(fixture.function);
        TEST_CHECK(public_results.size() == expected_units,
                   std::string(fixture.name) +
                       " public per-operator lowering cardinality mismatch");
    }
    return true;
}

bool TestProducerCallsRemainOutsideConsumerPrimFunc() {
    const GraphFixture chain = MakeFixtures()[0];
    const kxc::api::internal::LoweredGraph lowered =
        LowerForTest(chain.function);
    TEST_CHECK(lowered.primitives.size() == 2,
               "chain must lower to two independent primitives");
    int64_t first_inputs = -1;
    int64_t second_inputs = -1;
    TEST_CHECK(ReadIntAttr(lowered.primitives[0].lowered->prim_func,
                           "kxc.input_count", &first_inputs) &&
                   ReadIntAttr(lowered.primitives[1].lowered->prim_func,
                               "kxc.input_count", &second_inputs) &&
                   first_inputs == 2 && second_inputs == 2,
               "each chain unit must expose only its two boundary values");
    TEST_CHECK(lowered.primitives[1].lowered->prim_func->params.size() == 3,
               "consumer PrimFunc ABI must be two inputs plus one output");
    return true;
}

bool TestProductionLoweringRejectsStaticSizeOverflow() {
    using namespace kxc;
    const auto make_function = [](Array<int64_t> shape) {
        Var lhs("lhs", TensorType(shape, "float32"));
        Var rhs("rhs", TensorType(shape, "float32"));
        return Function({lhs, rhs}, Add(lhs, rhs));
    };
    const auto rejects = [&](Array<int64_t> shape) {
        try {
            (void)LowerForTest(make_function(std::move(shape)));
        } catch (const std::exception&) {
            return true;
        }
        return false;
    };
    constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();
    TEST_CHECK(rejects({kInt32Max + 1}),
               "production per-unit lowering must reject extents above INT32_MAX");
    TEST_CHECK(rejects({kInt32Max, kInt32Max, 3}),
               "production per-unit lowering must reject row-major product overflow");
    TEST_CHECK(LowerForTest(make_function({0, kInt32Max, kInt32Max})).primitives.size() == 1,
               "production per-unit lowering must retain legal zero-element tensors");
    return true;
}

bool TestSharedConstantUsesStableGraphValueKey() {
    using namespace kxc;
    TensorType type({4}, "float32");
    Var input("input", type);
    runtime::NDArray payload = runtime::NDArray::Zeros(
        {4}, DLDataType{kDLFloat, 32, 1}, Device::CPU());
    Constant constant(payload);
    Call first = Add(input, constant);
    Function function({input}, Multiply(first, constant));

    const api::internal::LoweredGraph lowered = LowerForTest(function);
    TEST_CHECK(lowered.primitives.size() == 2 && lowered.constants.size() == 1,
               "shared constant must be deduplicated graph-wide");
    TEST_CHECK(lowered.plan.constant_value_ids().size() == 1 &&
                   lowered.plan.constant_value_ids()[0] == 1,
               "ExecutablePlan must preserve ordered graph constant value ids");
    for (const auto& primitive : lowered.primitives) {
        const Array<relay::ConstantBinding> constants =
            primitive.lowered.constants();
        TEST_CHECK(constants.size() == 1 &&
                       constants[0]->key == "relay.constant.v1",
                   "each user unit must bind only the stable constant value key it uses");
    }
    return true;
}

const kxc::relay::Op& MultiOutputTestOp() {
    using namespace kxc;
    using namespace kxc::relay;
    static bool registered = false;
    if (!registered) {
        OperatorSpec spec;
        spec.name = "test_multi_output";
        spec.category = "test";
        spec.input_arity.num_inputs = 1;
        spec.output_arity = 2;
        spec.type_relation_key = "FInferType";
        spec.lowering_kind = OperatorLoweringKind::kMultiTE;
        spec.lowering_key = "FRelayToTEMulti";
        Op op = Op::Register(spec);
        OpRegEntry(op)
            .set_attr<FInferType>(
                "FInferType",
                FInferType([](const Attrs&, const Array<Type>& inputs) {
                    if (inputs.size() != 1 || !inputs[0].As<TensorTypeNode>()) {
                        throw std::runtime_error(
                            "test_multi_output expects one tensor input");
                    }
                    return TupleType({inputs[0], inputs[0]});
                }))
            .set_attr<FRelayToTEMulti>(
                "FRelayToTEMulti",
                FRelayToTEMulti([](const Attrs&,
                                   const Array<te::Tensor>& inputs,
                                   const Type&) {
                    if (inputs.size() != 1) {
                        throw std::runtime_error(
                            "test_multi_output lowering expects one tensor");
                    }
                    te::Tensor first = te::compute(
                        inputs[0]->shape,
                        [&](const Array<tir::Var>& axes) {
                            return inputs[0](axes);
                        },
                        "multi_first");
                    te::Tensor second = te::compute(
                        inputs[0]->shape,
                        [&](const Array<tir::Var>& axes) {
                            return inputs[0](axes) +
                                   tir::FloatImm(1.0f, inputs[0]->dtype);
                        },
                        "multi_second");
                    return Array<te::Tensor>{first, second};
                }));
        registered = true;
    }
    return Op::Get("test_multi_output");
}

bool TestSingleUnitSupportsMultipleOutputs() {
    using namespace kxc;
    TensorType type({4}, "float32");
    Var input("input", type);
    Call call(MultiOutputTestOp(), {input});
    Function function({input}, call);
    const api::internal::LoweredGraph lowered = LowerForTest(function);
    TEST_CHECK(lowered.primitives.size() == 1 &&
                   lowered.plan.output_value_ids().size() == 2,
               "one multi-output Call must remain one unit with two stable values");
    int64_t output_count = -1;
    TEST_CHECK(ReadIntAttr(lowered.primitives[0].lowered->prim_func,
                           "kxc.output_count", &output_count) &&
                   output_count == 2 &&
                   lowered.primitives[0].lowered->prim_func->params.size() == 3,
               "multi-output PrimFunc ABI must contain one input and two outputs");
    return true;
}

#if KXC_USE_LLVM

kxc::runtime::NDArray FilledTensor(float value) {
    using namespace kxc;
    runtime::NDArray result = runtime::NDArray::Empty(
        {4}, runtime::DataTypeFromString("float32"), Device::CPU());
    std::vector<float> payload(4, value);
    result.CopyFromBytes(payload.data(), payload.size() * sizeof(float));
    return result;
}

bool TensorEquals(const kxc::runtime::NDArray& value, float expected) {
    std::vector<float> payload(4);
    value.CopyToBytes(payload.data(), payload.size() * sizeof(float));
    for (float actual : payload) {
        if (actual != expected) return false;
    }
    return true;
}

bool TestOperatorGraphsExecuteNumerically() {
    using namespace kxc;
    const auto fixtures = MakeFixtures();
    const std::vector<std::vector<float>> expected{
        {9.0f}, {9.0f, 6.0f}, {15.0f}, {3.0f, 2.0f}};
    for (size_t fixture_index = 0; fixture_index < fixtures.size();
         ++fixture_index) {
        const auto artifacts = api::Compiler::Compile(
            fixtures[fixture_index].function,
            api::CompileConfig::Create(BuildTarget(Device::CPU()), 2));
        TEST_CHECK(artifacts.module.entry_count() ==
                           fixtures[fixture_index].expected_compute_calls &&
                       artifacts.plan.calls().size() ==
                           fixtures[fixture_index].expected_compute_calls,
                   std::string(fixtures[fixture_index].name) +
                       " compiled entry/call cardinality mismatch");
        runtime::RuntimeSession session(artifacts.module, artifacts.plan);
        Array<runtime::NDArray> inputs;
        inputs.push_back(FilledTensor(1.0f));
        inputs.push_back(FilledTensor(2.0f));
        if (artifacts.plan.input_value_ids().size() == 3) {
            inputs.push_back(FilledTensor(3.0f));
        }
        const Array<runtime::NDArray> outputs = session.Run(inputs);
        TEST_CHECK(outputs.size() == expected[fixture_index].size(),
                   std::string(fixtures[fixture_index].name) +
                       " runtime output count mismatch");
        for (size_t output_index = 0; output_index < outputs.size();
             ++output_index) {
            TEST_CHECK(TensorEquals(outputs[output_index],
                                    expected[fixture_index][output_index]),
                       std::string(fixtures[fixture_index].name) +
                           " runtime numeric result mismatch");
        }
    }
    return true;
}

bool TestSharedConstantExecutesNumerically() {
    using namespace kxc;
    TensorType type({4}, "float32");
    Var input("input", type);
    runtime::NDArray payload = FilledTensor(3.0f);
    Constant constant(payload);
    Call first = Add(input, constant);
    Function function({input}, Multiply(first, constant));
    const auto artifacts = api::Compiler::Compile(
        function, api::CompileConfig::Create(BuildTarget(Device::CPU()), 2));
    runtime::RuntimeSession session(artifacts.module, artifacts.plan);
    const Array<runtime::NDArray> outputs = session.Run({FilledTensor(2.0f)});
    TEST_CHECK(artifacts.module.constants().size() == 1 &&
                   artifacts.plan.constant_value_ids().size() == 1 &&
                   outputs.size() == 1 && TensorEquals(outputs[0], 15.0f),
               "shared constant must remain one value and feed both kernels");
    return true;
}

bool TestMultiOutputExecutesNumerically() {
    using namespace kxc;
    TensorType type({4}, "float32");
    Var input("input", type);
    Function function({input}, Call(MultiOutputTestOp(), {input}));
    const auto artifacts = api::Compiler::Compile(
        function, api::CompileConfig::Create(BuildTarget(Device::CPU()), 2));
    runtime::RuntimeSession session(artifacts.module, artifacts.plan);
    const Array<runtime::NDArray> outputs = session.Run({FilledTensor(5.0f)});
    TEST_CHECK(artifacts.module.entry_count() == 1 && outputs.size() == 2 &&
                   TensorEquals(outputs[0], 5.0f) &&
                   TensorEquals(outputs[1], 6.0f),
               "one primitive with two outputs must preserve numeric output order");
    return true;
}

bool TestPrimitiveCacheUsesFullStableIdentity() {
    using namespace kxc;
    api::internal::ClearPrimitiveCacheForTesting();
    TensorType type({4}, "float32");
    Var lhs("lhs", type);
    Var rhs("rhs", type);
    Function function({lhs, rhs}, Multiply(Add(lhs, rhs), rhs));
    const api::CompileConfig config =
        api::CompileConfig::Create(BuildTarget(Device::CPU()), 2);

    const auto first = api::Compiler::Compile(function, config);
    const api::internal::PrimitiveCacheStats after_first =
        api::internal::GetPrimitiveCacheStats();
    const auto second = api::Compiler::Compile(function, config);
    const api::internal::PrimitiveCacheStats after_second =
        api::internal::GetPrimitiveCacheStats();
    api::ProductionArtifactCacheAdapter adapter;
    bool public_pins_are_production_backed =
        first.artifact_pins.size() == 2 && second.artifact_pins.size() == 2;
    for (const api::ArtifactPin& pin : second.artifact_pins) {
        public_pins_are_production_backed =
            public_pins_are_production_backed && pin.defined() &&
            adapter.Lookup(pin.handle().record().artifact_key).kind ==
                api::ArtifactLookupKind::kHit;
    }
    bool compiler_declaration_matches_pins = second.variant.defined() &&
        second.variant.manifest().retention_lease() != nullptr &&
        second.variant.manifest().bindings().size() ==
            second.artifact_pins.size();
    if (compiler_declaration_matches_pins) {
        const auto bindings = second.variant.manifest().bindings();
        for (size_t index = 0; index < bindings.size(); ++index) {
            compiler_declaration_matches_pins =
                compiler_declaration_matches_pins &&
                bindings[index]->generation == 0 &&
                bindings[index]->invocation_id ==
                    static_cast<int64_t>(index) &&
                bindings[index]->artifact_identity ==
                    second.artifact_pins[index]
                        .handle()
                        .record()
                        .artifact_key.canonical_bytes();
        }
    }
    TEST_CHECK(first.module.entry_count() == 2 &&
                   second.module.entry_count() == 2 &&
                   public_pins_are_production_backed &&
                   compiler_declaration_matches_pins &&
                   after_first.misses == 2 && after_first.hits == 0 &&
                   after_first.entries == 2 && after_second.misses == 2 &&
                   after_second.hits == 2,
               "repeat compilation should return public pins from the production cache");
    return true;
}

bool TestProductionCompileVariantRuntimeE2E() {
    using namespace kxc;
    api::internal::ClearPrimitiveCacheForTesting();
    const api::ProductionArtifactCacheAdapter adapter;
    std::optional<runtime::RuntimeSession> session;
    std::optional<runtime::RunAsyncResult> run_result;
    std::weak_ptr<const void> retained_production_lease;
    std::vector<std::string> expected_identities;
    size_t pin_count = 0;

    {
        const GraphFixture chain = MakeFixtures()[0];
        api::CompiledGraph compiled = api::Compiler::Compile(
            chain.function,
            api::CompileConfig::Create(BuildTarget(Device::CPU()), 2));
        pin_count = compiled.artifact_pins.size();
        const runtime::SelectedArtifactManifest declaration =
            compiled.variant.manifest();
        retained_production_lease = declaration.retention_lease();
        const Array<runtime::SelectedArtifactBinding> bindings =
            declaration.bindings();
        TEST_CHECK(pin_count == chain.expected_compute_calls &&
                       bindings.size() == pin_count &&
                       declaration.retention_lease() != nullptr &&
                       adapter.stats().active_pins == pin_count,
                   "Compiler must assemble a production-backed declaration and lease");
        for (size_t index = 0; index < pin_count; ++index) {
            const std::string identity =
                compiled.artifact_pins[index]
                    .handle()
                    .record()
                    .artifact_key.canonical_bytes();
            TEST_CHECK(bindings[index]->invocation_id ==
                               static_cast<int64_t>(index) &&
                           bindings[index]->generation == 0 &&
                           std::string(bindings[index]->artifact_identity) ==
                               identity,
                       "Compiler declaration must match each production pin");
            expected_identities.push_back(identity);
        }

        session.emplace(compiled.module, compiled.variant,
                        runtime::RuntimeExecutionMode::kTaskDAG);
#if KXC_ENABLE_REGION_TASK_DAG
        const bool selection_ok =
            session->UsesTaskDAG() &&
            session->TaskDAGSelection().fallback_reason ==
                runtime::FallbackReason::kNone;
#else
        const bool selection_ok =
            !session->UsesTaskDAG() &&
            session->TaskDAGSelection().fallback_reason ==
                runtime::FallbackReason::kFeatureDisabled;
#endif
        const Array<runtime::SelectedArtifactBinding> session_bindings =
            session->artifact_manifest().bindings();
        bool session_declaration_matches =
            session_bindings.size() == expected_identities.size();
        for (size_t index = 0;
             session_declaration_matches && index < session_bindings.size();
             ++index) {
            session_declaration_matches =
                std::string(session_bindings[index]->artifact_identity) ==
                expected_identities[index];
        }
        TEST_CHECK(selection_ok && session_declaration_matches,
                   "RuntimeSession must preserve the production declaration");

        run_result.emplace(session->RunAsync(
            {FilledTensor(1.0f), FilledTensor(2.0f), FilledTensor(3.0f)},
            DeviceStream::Default(Device::CPU())));
    }

    session.reset();
    TEST_CHECK(run_result && run_result->outputs.size() == 1 &&
                   TensorEquals(run_result->outputs[0], 9.0f) &&
                   !retained_production_lease.expired() &&
                   adapter.stats().active_pins == pin_count,
               "completion must retain production pins and the numeric result");
    run_result->completion.Wait();
    run_result->completion = AsyncOperation();
    TEST_CHECK(retained_production_lease.expired() &&
                   adapter.stats().active_pins == 0,
               "production pin lease must release with completion ownership");
    api::internal::ClearPrimitiveCacheForTesting();
    return true;
}

bool TestPrimitiveCacheReusesRenumberedUnit() {
    using namespace kxc;
    api::internal::ClearPrimitiveCacheForTesting();
    TensorType type({4}, "float32");
    Var lhs("lhs", type);
    Var rhs("rhs", type);
    const api::CompileConfig config =
        api::CompileConfig::Create(BuildTarget(Device::CPU()), 2);
    const auto direct = api::Compiler::Compile(
        Function({lhs, rhs}, Add(lhs, rhs)), config);
    const api::internal::PrimitiveCacheStats after_direct =
        api::internal::GetPrimitiveCacheStats();

    Var shifted_lhs("lhs", type);
    Var shifted_rhs("rhs", type);
    Var unrelated("unrelated", type);
    Function shifted(
        {shifted_lhs, shifted_rhs, unrelated},
        Tuple({Multiply(shifted_lhs, unrelated),
               Add(shifted_lhs, shifted_rhs)}));
    const auto reused = api::Compiler::Compile(shifted, config);
    const api::internal::PrimitiveCacheStats after_reused =
        api::internal::GetPrimitiveCacheStats();
    runtime::RuntimeSession session(reused.module, reused.plan);
    const Array<runtime::NDArray> outputs = session.Run(
        {FilledTensor(2.0f), FilledTensor(3.0f), FilledTensor(4.0f)});
    TEST_CHECK(direct.module.entry_count() == 1 &&
                   reused.module.entry_count() == 2 && outputs.size() == 2 &&
                   TensorEquals(outputs[0], 8.0f) &&
                   TensorEquals(outputs[1], 5.0f) &&
                   after_direct.misses == 1 && after_direct.hits == 0 &&
                   after_reused.misses == 2 && after_reused.hits == 1,
               "renumbered add must reuse a validated symbol alias numerically");
    return true;
}

#endif

}  // namespace

int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"whole_graph_compatibility_baseline", TestWholeGraphCompatibilityBaseline},
        {"per_operator_target_cardinality", TestPerOperatorTargetCardinality},
        {"producer_calls_remain_outside_consumer",
         TestProducerCallsRemainOutsideConsumerPrimFunc},
        {"production_lowering_rejects_static_size_overflow",
         TestProductionLoweringRejectsStaticSizeOverflow},
        {"shared_constant_uses_stable_key", TestSharedConstantUsesStableGraphValueKey},
        {"single_unit_supports_multiple_outputs", TestSingleUnitSupportsMultipleOutputs},
#if KXC_USE_LLVM
        {"operator_graphs_execute_numerically",
         TestOperatorGraphsExecuteNumerically},
        {"shared_constant_executes_numerically",
         TestSharedConstantExecutesNumerically},
        {"multi_output_executes_numerically",
         TestMultiOutputExecutesNumerically},
        {"primitive_cache_uses_full_stable_identity",
         TestPrimitiveCacheUsesFullStableIdentity},
        {"production_compile_variant_runtime_e2e",
         TestProductionCompileVariantRuntimeE2E},
        {"primitive_cache_reuses_renumbered_unit",
         TestPrimitiveCacheReusesRenumberedUnit},
#endif
    };
    bool ok = true;
    for (const auto& test : tests) {
        const bool passed = test.second();
        std::cout << (passed ? "[PASS] " : "[FAIL] ") << test.first << "\n";
        ok = passed && ok;
    }
    return ok ? 0 : 1;
}
