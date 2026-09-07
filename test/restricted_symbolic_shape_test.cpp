#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "../src/compiler/internal/dynamic_shape_contract.h"
#include "../src/compiler/internal/primitive_cache.h"
#include "kxc/compiler/compiler.h"
#include "kxc/compiler/restricted_symbolic_shape.h"
#include "kxc/relay/op.h"
#include "kxc/runtime/session.h"

namespace {
namespace restricted = kxc::api::experimental::restricted_symbolic_shape::v1;
namespace shape =
    kxc::api::experimental::shape_specialization::v1;
namespace compiler_internal = kxc::api::internal;

static_assert(!std::is_default_constructible_v<restricted::BoundedCompileRequest>);
static_assert(std::is_copy_constructible_v<restricted::BoundedCompileRequest>);
static_assert(!std::is_constructible_v<restricted::BoundedCompileRequest,
                                       kxc::Function, shape::BindingSet>);
static_assert(!std::is_default_constructible_v<
              compiler_internal::BoundedCompilePreparation>);
static_assert(std::is_copy_constructible_v<
              compiler_internal::BoundedCompilePreparation>);

#define CHECK(x, m) do { if (!(x)) { std::cerr << "[FAIL] " << __FUNCTION__ << ": " << m << "\n"; return false; } } while (0)

bool Throws(const std::function<void()>& fn) {
    try { fn(); } catch (const std::exception&) { return true; }
    return false;
}

kxc::api::CompileConfig Config() {
    return kxc::api::CompileConfig::Create(kxc::BuildTarget(kxc::Device::CPU()));
}

kxc::api::CompileConfig SyntheticCudaConfig() {
    auto* node = new kxc::TargetNode();
    node->kind = "cuda";
    node->device_type = kxc::kCUDA;
    node->device_id = 0;
    node->attrs.exists = 1;
    node->attrs.device_name = "synthetic-cuda";
    node->attrs.arch = "sm_80";
    node->attrs.compute_version = "8.0";
    node->attrs.compute_version_major = 8;
    node->attrs.compute_version_minor = 0;
    node->attrs.max_threads_per_block = 1024;
    node->attrs.warp_size = 32;
    node->attrs.multi_processor_count = 1;
    return kxc::api::CompileConfig::Create(
        kxc::Target(kxc::ObjectRef(node)));
}

kxc::Function UnaryGraph(const std::string& operation, int64_t extent = 4) {
    const kxc::TensorType type({extent}, "float32");
    const kxc::Var x("x", type);
    return kxc::Function({x}, kxc::Call(kxc::relay::Op::Get(operation), {x}));
}

kxc::Function MixedGraph() {
    const kxc::TensorType type({4}, "float32");
    const kxc::Var x("x", type), y("y", type);
    const kxc::Expr add = kxc::Call(kxc::relay::Op::Get("add"), {x, y});
    return kxc::Function({x, y}, kxc::Call(kxc::relay::Op::Get("nn_relu"), {
        kxc::Call(kxc::relay::Op::Get("sqrt"), {add})}));
}

kxc::Function BoundedChain() {
    const kxc::TensorType type({4, 4}, "float32");
    const kxc::Var x("x", type), y("y", type);
    const kxc::Expr add = kxc::Call(kxc::relay::Op::Get("add"), {x, y});
    const kxc::Expr relu =
        kxc::Call(kxc::relay::Op::Get("nn_relu"), {add});
    return kxc::Function(
        {x, y}, kxc::Call(kxc::relay::Op::Get("sqrt"), {relu}));
}

kxc::Function UnusedBindingGraph() {
    const kxc::Var x("x", kxc::TensorType({4}, "float32"));
    const kxc::Var unused("unused", kxc::TensorType({5}, "float32"));
    return kxc::Function({x, unused}, kxc::Call(kxc::relay::Op::Get("nn_relu"), {x}));
}

bool SameCounters(const kxc::api::experimental::shape_exact::v1::ShapeExactPreparationCounters& a,
                  const kxc::api::experimental::shape_exact::v1::ShapeExactPreparationCounters& b) {
    return a.execution_contract_resolutions == b.execution_contract_resolutions &&
           a.relay_graph_pipelines == b.relay_graph_pipelines &&
           a.capability_boundary_checks == b.capability_boundary_checks &&
           a.value_graph_builds == b.value_graph_builds && a.partitions == b.partitions;
}

shape::BindingSet Bindings(int64_t n) { return shape::BindingSet({{"n", n}}); }

bool TestGateAndRestrictedExactSlice() {
#if !KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE
    CHECK(!restricted::RestrictedSymbolicShapeAdapter::IsEnabled(), "default gate must be off");
    CHECK(Throws([] { (void)restricted::RestrictedSymbolicShapeAdapter::Prepare(
              UnaryGraph("nn_relu"), Config(), {{0, 0, "n", 1, 8, 1}}); }),
          "off gate must fail closed");
    return true;
#elif !KXC_ENABLE_SHAPE_PRODUCTION_EXACT
    CHECK(restricted::RestrictedSymbolicShapeAdapter::IsEnabled(), "restricted gate must report enabled");
    CHECK(Throws([] { (void)restricted::RestrictedSymbolicShapeAdapter::Prepare(
              UnaryGraph("nn_relu"), Config(), {{0, 0, "n", 1, 8, 1}}); }),
          "restricted overlay must require its exact frozen representative gate");
    return true;
#else
    using restricted::RestrictedSymbolicShapeAdapter;
    const auto relu = RestrictedSymbolicShapeAdapter::Prepare(
        UnaryGraph("nn_relu"), Config(), {{0, 0, "n", 1, 8, 1}});
    const auto sqrt = RestrictedSymbolicShapeAdapter::Prepare(
        UnaryGraph("sqrt"), Config(), {{0, 0, "n", 1, 8, 1}});
    auto relay_to_mutate = MixedGraph();
    const auto mixed = RestrictedSymbolicShapeAdapter::Prepare(
        relay_to_mutate, Config(), {{0, 0, "n", 1, 8, 1}, {1, 0, "n", 1, 8, 1}});
    auto* mutated_call = const_cast<kxc::CallNode*>(relay_to_mutate->body.As<kxc::CallNode>());
    mutated_call->op = kxc::relay::Op::Get("mul");

    const auto frozen = mixed.representative().counters();
    const auto exact4 = RestrictedSymbolicShapeAdapter::MintExact(mixed, Bindings(4));
    const auto relu4 = RestrictedSymbolicShapeAdapter::MintExact(relu, Bindings(4));
    const auto sqrt4 = RestrictedSymbolicShapeAdapter::MintExact(sqrt, Bindings(4));
    CHECK(exact4.exact_requests().size() == 3,
          "mixed add/sqrt/relu chain must mint opaque exact snapshots");
    CHECK(Throws([&] { (void)RestrictedSymbolicShapeAdapter::ChangedUnitIndices(relu4, sqrt4); }),
          "different unit semantic contexts must be incomparable");

    auto detached = exact4.exact_requests();
    detached.clear();
    CHECK(exact4.exact_requests().size() == 3,
          "returned request snapshots cannot mutate decision authority");

    const auto unused = RestrictedSymbolicShapeAdapter::Prepare(UnusedBindingGraph(), Config(),
        {{0, 0, "n", 1, 8, 1}, {1, 0, "unused", 1, 8, 1}});
    const auto unused4 = RestrictedSymbolicShapeAdapter::MintExact(
        unused, shape::BindingSet({{"n", 4}, {"unused", 4}}));
    const auto unused5 = RestrictedSymbolicShapeAdapter::MintExact(
        unused, shape::BindingSet({{"n", 4}, {"unused", 5}}));
    CHECK(RestrictedSymbolicShapeAdapter::ChangedUnitIndices(unused4, unused5).empty(),
          "unrelated bindings must not invalidate an unchanged unit artifact boundary");

    const auto exact5 = RestrictedSymbolicShapeAdapter::MintExact(mixed, Bindings(5));
    CHECK(RestrictedSymbolicShapeAdapter::ChangedUnitIndices(exact4, exact5) ==
              std::vector<size_t>({0, 1, 2}) &&
          SameCounters(frozen, mixed.representative().counters()),
          "profiles must not rerun prepared pipeline or partition work");

    kxc::api::internal::ClearPrimitiveCacheForTesting();
    const auto before = kxc::api::internal::GetPrimitiveCacheStats();
    CHECK(Throws([&] { (void)RestrictedSymbolicShapeAdapter::MintExact(mixed, shape::BindingSet()); }) &&
          Throws([] { (void)RestrictedSymbolicShapeAdapter::Prepare(
              UnaryGraph("nn_relu", -1), Config(), {{0, 0, "n", 1, 8, 1}}); }),
          "unbound bindings and legacy dimensions must fail closed");
    const auto after = kxc::api::internal::GetPrimitiveCacheStats();
    CHECK(before.entries == after.entries && before.hits == after.hits && before.misses == after.misses &&
          before.failures == after.failures && SameCounters(frozen, mixed.representative().counters()),
          "invalid requests must not mutate compiler/cache or prepared counters");
    return true;
#endif
}

kxc::Function Rank2Unary(int64_t rows = 4, int64_t cols = 3) {
    const kxc::Var x("x", kxc::TensorType({rows, cols}, "float32"));
    return kxc::Function({x}, kxc::Call(kxc::relay::Op::Get("nn_relu"), {x}));
}

bool TestMaterializedVariantClosure() {
#if !KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE || !KXC_ENABLE_SHAPE_PRODUCTION_EXACT
    return true;
#else
    using restricted::RestrictedSymbolicShapeAdapter;
    const auto prep = RestrictedSymbolicShapeAdapter::Prepare(
        UnaryGraph("nn_relu"), Config(), {{0, 0, "n", 2, 8, 2}});
    const auto d6 = RestrictedSymbolicShapeAdapter::MintExact(prep, Bindings(6));
    const auto d4 = RestrictedSymbolicShapeAdapter::MintExact(prep, Bindings(4));

    // 不变量 5：不同 binding 不同 route，相同 binding 相同 route。
    CHECK(!(RestrictedSymbolicShapeAdapter::ExactDispatchKey(prep, d6) ==
            RestrictedSymbolicShapeAdapter::ExactDispatchKey(prep, d4)),
          "different bindings must mint different dispatch keys");
    CHECK(RestrictedSymbolicShapeAdapter::ExactDispatchKey(prep, d6) ==
              RestrictedSymbolicShapeAdapter::ExactDispatchKey(
                  prep, RestrictedSymbolicShapeAdapter::MintExact(prep, Bindings(6))),
          "identical bindings must mint identical dispatch keys");

    // 物化：参数类型必须携带决策求值后的 concrete shape 和原 dtype。
    const kxc::Function fn6 =
        RestrictedSymbolicShapeAdapter::MaterializeExactFunction(prep, d6);
    CHECK(fn6.defined() && fn6->params.size() == 1,
          "materialized function must keep the representative arity");
    const auto* materialized_type =
        fn6->params[0]->type_annotation.As<kxc::TensorTypeNode>();
    CHECK(materialized_type && materialized_type->shape.size() == 1 &&
              materialized_type->shape[0] == 6 &&
              materialized_type->dtype == "float32",
          "materialized parameter must carry the bound concrete extent");

    // 归属校验：别的模板铸出的决策必须被拒绝。
    const auto other = RestrictedSymbolicShapeAdapter::Prepare(
        UnaryGraph("sqrt"), Config(), {{0, 0, "n", 2, 8, 2}});
    const auto other6 = RestrictedSymbolicShapeAdapter::MintExact(other, Bindings(6));
    CHECK(Throws([&] { (void)RestrictedSymbolicShapeAdapter::MaterializeExactFunction(
              prep, other6); }),
          "foreign decision must not materialize against this template");
    CHECK(Throws([&] { (void)RestrictedSymbolicShapeAdapter::ExactDispatchKey(
              prep, other6); }),
          "foreign decision must not mint a route for this template");

    // 请求边界换算：输入 shape → bindings。
    CHECK(RestrictedSymbolicShapeAdapter::BindingsFromInputShapes(prep, {{6}}) ==
              Bindings(6),
          "input shapes must convert to the canonical binding set");
    CHECK(Throws([&] { (void)RestrictedSymbolicShapeAdapter::BindingsFromInputShapes(
              prep, {{6}, {6}}); }),
          "input count mismatch must fail closed");
    CHECK(Throws([&] { (void)RestrictedSymbolicShapeAdapter::BindingsFromInputShapes(
              prep, {{6, 2}}); }),
          "input rank mismatch must fail closed");

    // 非 overlay 静态轴必须与模板一致。
    const auto rank2 = RestrictedSymbolicShapeAdapter::Prepare(
        Rank2Unary(), Config(), {{0, 0, "n", 2, 8, 2}});
    CHECK(RestrictedSymbolicShapeAdapter::BindingsFromInputShapes(rank2, {{6, 3}}) ==
              Bindings(6),
          "static non-overlay axis must pass through unchanged");
    CHECK(Throws([&] { (void)RestrictedSymbolicShapeAdapter::BindingsFromInputShapes(
              rank2, {{6, 7}}); }),
          "static non-overlay axis mismatch must fail closed");

    // 共享 symbol 的输入必须一致。
    const auto mixed = RestrictedSymbolicShapeAdapter::Prepare(
        MixedGraph(), Config(), {{0, 0, "n", 1, 8, 1}, {1, 0, "n", 1, 8, 1}});
    CHECK(Throws([&] { (void)RestrictedSymbolicShapeAdapter::BindingsFromInputShapes(
              mixed, {{4}, {5}}); }),
          "conflicting shared-symbol extents must fail closed");
    CHECK(RestrictedSymbolicShapeAdapter::BindingsFromInputShapes(mixed, {{5}, {5}}) ==
              Bindings(5),
          "consistent shared-symbol extents must bind once");

#if KXC_USE_LLVM
    // 物化 variant 经用户显式 Compiler::Compile 编译并被验证函数绑定。
    const kxc::api::CompiledGraph g6 = kxc::api::Compiler::Compile(fn6, Config());
    RestrictedSymbolicShapeAdapter::VerifyCompiledExactVariant(prep, d6, g6);
    const kxc::Function fn4 =
        RestrictedSymbolicShapeAdapter::MaterializeExactFunction(prep, d4);
    const kxc::api::CompiledGraph g4 = kxc::api::Compiler::Compile(fn4, Config());
    CHECK(Throws([&] { RestrictedSymbolicShapeAdapter::VerifyCompiledExactVariant(
              prep, d6, g4); }),
          "verification must reject a variant compiled for other bindings");

    // 语义绑定回归：同边界（shape/dtype/call 数）但算子不同的产物必须被
    // 拒绝——sqrt(N=6) 与授权的 relu(N=6) 边界完全一致，只有语义身份不同。
    const kxc::Function fn6_sqrt =
        RestrictedSymbolicShapeAdapter::MaterializeExactFunction(other, other6);
    const kxc::api::CompiledGraph g6_sqrt =
        kxc::api::Compiler::Compile(fn6_sqrt, Config());
    RestrictedSymbolicShapeAdapter::VerifyCompiledExactVariant(
        other, other6, g6_sqrt);
    CHECK(Throws([&] { RestrictedSymbolicShapeAdapter::VerifyCompiledExactVariant(
              prep, d6, g6_sqrt); }),
          "same-boundary wrong-operator artifact must fail semantic binding");

    // 数值：relu(N=6) 输入含负值。
    kxc::runtime::NDArray input = kxc::runtime::NDArray::Zeros(
        {6}, kxc::runtime::DataTypeFromString("float32"), kxc::Device::CPU());
    const std::vector<float> raw = {-2, -1, 0, 1, 2, 3};
    input.CopyFromBytes(raw.data(), raw.size() * sizeof(float));
    kxc::runtime::RuntimeSession session(g6.module(), g6.plan());
    const auto outputs = session.Run({input});
    std::vector<float> actual(6);
    outputs[0].CopyToBytes(actual.data(), actual.size() * sizeof(float));
    CHECK(actual == std::vector<float>({0, 0, 0, 1, 2, 3}),
          "materialized relu variant must execute with the bound extent");
#endif
    return true;
#endif
}

shape::TensorShapeContract SymbolicContract(
    const std::vector<shape::DimExpr>& dimensions) {
    return shape::TensorShapeContract(
        shape::LogicalShape(dimensions), shape::PhysicalCapacity(dimensions),
        shape::ValidExtent(dimensions));
}

bool TestBoundedCompileAdmissionAndUnitContracts() {
#if !KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE || !KXC_ENABLE_SHAPE_PRODUCTION_EXACT
    return true;
#else
    using restricted::RestrictedSymbolicShapeAdapter;
    kxc::Function caller = BoundedChain();
    const auto prepared = RestrictedSymbolicShapeAdapter::Prepare(
        caller, Config(),
        {{0, 0, "n", 2, 8, 2}, {1, 0, "n", 2, 8, 2}});
    // Caller mutation after preparation cannot alter the request snapshot.
    auto* root = const_cast<kxc::CallNode*>(caller->body.As<kxc::CallNode>());
    root->op = kxc::relay::Op::Get("mul");

    const auto cache_before =
        compiler_internal::GetPrimitiveCacheStats();
    const restricted::BoundedCompileRequest request =
        RestrictedSymbolicShapeAdapter::MintBoundedCompileRequest(prepared);
    CHECK(request.applicability_version() ==
              restricted::kBoundedCompileApplicabilityVersion &&
              request.graph_template().CanonicalBytes() ==
                  prepared.graph_template().CanonicalBytes() &&
              request.target()->kind == "llvm" &&
              request.target()->device_type == kxc::kCPU,
          "bounded request must bind template, applicability, config, and target");
    CHECK(request.representative_oracle().profile().Value("value.0")
                  .contract.logical == std::vector<int64_t>({4, 4}),
          "bounded request must carry the representative proof");
    kxc::Function detached_representative = request.representative();
    const_cast<kxc::CallNode*>(
        detached_representative->body.As<kxc::CallNode>())
        ->op = kxc::relay::Op::Get("nn_relu");
    CHECK(kxc::api::Compiler::BuildGraphSemanticKey(request.representative()) ==
              request.graph_template().key(),
          "mutating a returned Relay copy cannot mutate request authority");

    const kxc::Function logical_boundary =
        request.logical_boundary_function();
    const auto* logical_x = logical_boundary->params[0]
                                ->type_annotation.As<kxc::TensorTypeNode>();
    const auto* logical_y = logical_boundary->params[1]
                                ->type_annotation.As<kxc::TensorTypeNode>();
    CHECK(logical_x && logical_y && logical_x->dtype == "float32" &&
              logical_y->dtype == "float32" &&
              logical_x->shape.size() == 2 && logical_x->shape[0] == -1 &&
              logical_x->shape[1] == 4 && logical_y->shape[0] == -1 &&
              logical_y->shape[1] == 4,
          "only direct symbolic axes may become fixed-rank -1 boundaries");
    CHECK(Throws([&] {
              (void)compiler_internal::BuildValueGraph(
                  logical_boundary, kxc::Device::CPU());
          }) &&
              Throws([&] {
                  (void)kxc::api::Compiler::Compile(
                      logical_boundary, Config());
              }),
          "bare dynamic Functions must still fail static ValueGraph and Compiler admission");

    const compiler_internal::BoundedCompilePreparation result =
        compiler_internal::PrepareBoundedCompile(request);
    CHECK(result.version() ==
              compiler_internal::kBoundedCompilePreparationVersion &&
              result.partitioned_graph().units.size() == 3 &&
              result.unit_shape_contracts().size() == 3,
          "add/relu/sqrt must produce three ordered bounded units");
    for (const auto& value : result.partitioned_graph().value_graph.values) {
        const auto* type = value.checked_type.As<kxc::TensorTypeNode>();
        CHECK(type && type->dtype == "float32" && type->shape.size() == 2 &&
                  type->shape[0] == -1 && type->shape[1] == 4,
              "bounded logical values must preserve dtype/rank and direct axes");
    }

    const auto& add = result.unit_shape_contracts()[0];
    CHECK(add.version() ==
              compiler_internal::kDynamicUnitShapeContractVersion &&
              add.local_input_guards().size() == 2 &&
              add.local_input_guards()[0].size() == 2 &&
              add.local_input_guards()[1].size() == 2,
          "add must expose one local guard for every input axis");
    const auto& anchor = add.local_input_guards()[0][0];
    const auto& shared = add.local_input_guards()[1][0];
    CHECK(anchor.lower == 2 && anchor.upper == 8 &&
              anchor.divisible_by == 2 && !anchor.exact && !anchor.equal_to &&
              shared.equal_to && shared.equal_to->input_index == 0 &&
              shared.equal_to->axis == 0 &&
              add.local_input_guards()[0][1].exact ==
                  std::optional<std::uint64_t>(4),
          "bounds, divisibility, constants, and shared-axis equality must be canonical");
    const auto& output = add.output_shape_expressions()[0];
    CHECK(output.size() == 2 &&
              output[0].kind() ==
                  compiler_internal::DynamicShapeExpr::Kind::kInputAxis &&
              output[0].input_index() == 0 && output[0].axis() == 0 &&
              output[1].kind() ==
                  compiler_internal::DynamicShapeExpr::Kind::kConst &&
              output[1].constant() == 4 &&
              add.runtime_extent_expressions().size() == 1 &&
              add.runtime_extent_expressions()[0] == output[0],
          "only dynamic InputAxis expressions may enter runtime extent order");
    CHECK(result.graph_input_guards().size() == 2 &&
              result.graph_input_guards()[0].input_index == 0 &&
              result.graph_input_guards()[0].axis == 0 &&
              !result.graph_input_guards()[0].equal_to &&
              result.graph_input_guards()[1].input_index == 1 &&
              result.graph_input_guards()[1].axis == 0 &&
              result.graph_input_guards()[1].equal_to &&
              result.graph_input_guards()[1].equal_to->input_index == 0 &&
              result.graph_input_guards()[1].equal_to->axis == 0,
          "graph preflight guards must follow wildcard input-axis order");

    const auto rebuilt = compiler_internal::BuildDynamicUnitShapeContracts(
        request.graph_template(), {"add", "nn_relu", "sqrt"});
    CHECK(rebuilt.size() == result.unit_shape_contracts().size(),
          "contract producer cardinality must be deterministic");
    for (size_t i = 0; i < rebuilt.size(); ++i) {
        CHECK(!rebuilt[i].canonical_bytes().empty() &&
                  rebuilt[i].canonical_bytes() ==
                      result.unit_shape_contracts()[i].canonical_bytes(),
              "GraphTemplate + ordered UnitSkeleton must be the sole canonical producer");
    }
    const auto cache_after = compiler_internal::GetPrimitiveCacheStats();
    CHECK(cache_before.entries == cache_after.entries &&
              cache_before.hits == cache_after.hits &&
              cache_before.misses == cache_after.misses &&
              cache_before.in_flight == cache_after.in_flight,
          "bounded structural preparation must not compile or touch primitive cache");
    return true;
#endif
}

bool TestBoundedCompileFailsClosedStructurally() {
#if !KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE || !KXC_ENABLE_SHAPE_PRODUCTION_EXACT
    return true;
#else
    using restricted::RestrictedSymbolicShapeAdapter;
    const kxc::TensorType vector4({4}, "float32");
    const kxc::Var x("x", vector4), y("y", vector4);
    CHECK(Throws([&] {
              (void)RestrictedSymbolicShapeAdapter::Prepare(
                  kxc::Function({x}, kxc::Call(kxc::relay::Op::Get("softmax"), {x})),
                  Config(), {{0, 0, "n", 1, 8, 1}});
          }),
          "unknown bounded operators must fail before request minting");
    const kxc::Var wrong_dtype("wrong_dtype",
                               kxc::TensorType({4}, "int32"));
    CHECK(Throws([&] {
              (void)RestrictedSymbolicShapeAdapter::Prepare(
                  kxc::Function(
                      {x, wrong_dtype},
                      kxc::Call(kxc::relay::Op::Get("add"),
                                {x, wrong_dtype})),
                  Config(), {{0, 0, "n", 1, 8, 1},
                             {1, 0, "n", 1, 8, 1}});
          }),
          "equal-rank values with different dtypes must fail admission");

    const kxc::runtime::NDArray data = kxc::runtime::NDArray::Zeros(
        {4}, kxc::runtime::DataTypeFromString("float32"), kxc::Device::CPU());
    CHECK(Throws([&] {
              (void)RestrictedSymbolicShapeAdapter::Prepare(
                  kxc::Function({x}, kxc::Call(kxc::relay::Op::Get("add"),
                                               {x, kxc::Constant(data)})),
                  Config(), {{0, 0, "n", 1, 8, 1}});
          }),
          "constants must fail before bounded request minting");
    CHECK(Throws([&] {
              (void)RestrictedSymbolicShapeAdapter::Prepare(
                  kxc::Function({x}, kxc::If(x, x, x)), Config(),
                  {{0, 0, "n", 1, 8, 1}});
          }),
          "control flow must fail before bounded request minting");

    const kxc::Var bx("bx", kxc::TensorType({4, 1}, "float32"));
    const kxc::Var by("by", kxc::TensorType({4, 3}, "float32"));
    CHECK(Throws([&] {
              (void)RestrictedSymbolicShapeAdapter::Prepare(
                  kxc::Function({bx, by},
                      kxc::Call(kxc::relay::Op::Get("add"), {bx, by})),
                  Config(), {{0, 0, "n", 1, 8, 1},
                             {1, 0, "n", 1, 8, 1}});
          }),
          "broadcast must fail before bounded request minting");

    const kxc::Var missing_rank("missing_rank", kxc::Type());
    CHECK(Throws([&] {
              (void)RestrictedSymbolicShapeAdapter::Prepare(
                  kxc::Function({missing_rank}, missing_rank), Config(),
                  {{0, 0, "n", 1, 8, 1}});
          }),
          "missing or dynamic rank must fail before bounded request minting");

    const shape::DimExpr n = shape::DimExpr::Symbol("n");
    const shape::DimExpr arithmetic =
        shape::DimExpr::Add({n, shape::DimExpr::Const(1)});
    const shape::GraphTemplate arithmetic_graph(
        kxc::api::Compiler::BuildGraphSemanticKey(UnaryGraph("nn_relu")),
        shape::ShapeProgram(
            {"n"}, {{"x", SymbolicContract({n})}},
            {{"y", SymbolicContract({arithmetic})}},
            {shape::Constraint::Range(n, 1, 8)}),
        {{shape::GraphLocalCallLocator("value.1"),
          kxc::api::UnitSemanticKey("bounded.test.relu"), {"x"}, {"y"}}});
    CHECK(Throws([&] {
              (void)compiler_internal::BuildDynamicUnitShapeContracts(
                  arithmetic_graph, {"nn_relu"});
          }),
          "complex output shape arithmetic must fail closed");

    CHECK(Throws([&] {
              const auto cuda_prepared =
                  RestrictedSymbolicShapeAdapter::Prepare(
                      UnaryGraph("nn_relu"), SyntheticCudaConfig(),
                      {{0, 0, "n", 1, 8, 1}});
              (void)RestrictedSymbolicShapeAdapter::MintBoundedCompileRequest(
                  cuda_prepared);
          }),
          "bounded admission must reject synthetic CUDA without CPU fallback");

    const shape::GraphTemplate broadcast_constraint_graph(
        kxc::api::Compiler::BuildGraphSemanticKey(UnaryGraph("nn_relu")),
        shape::ShapeProgram(
            {"n"}, {{"x", SymbolicContract({n})}},
            {{"y", SymbolicContract({n})}},
            {shape::Constraint::Range(n, 1, 8),
             shape::Constraint::BroadcastCompatible(
                 n, shape::DimExpr::Const(1))}),
        {{shape::GraphLocalCallLocator("value.1"),
          kxc::api::UnitSemanticKey("bounded.test.relu"), {"x"}, {"y"}}});
    CHECK(Throws([&] {
              (void)compiler_internal::BuildDynamicUnitShapeContracts(
                  broadcast_constraint_graph, {"nn_relu"});
          }),
          "broadcast constraints cannot masquerade as direct-axis proof");
    return true;
#endif
}

}  // namespace

int main() {
    std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"gate_and_restricted_exact_slice", TestGateAndRestrictedExactSlice},
        {"materialized_variant_closure", TestMaterializedVariantClosure},
        {"bounded_compile_admission_and_unit_contracts",
         TestBoundedCompileAdmissionAndUnitContracts},
        {"bounded_compile_fails_closed_structurally",
         TestBoundedCompileFailsClosedStructurally},
    };
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
