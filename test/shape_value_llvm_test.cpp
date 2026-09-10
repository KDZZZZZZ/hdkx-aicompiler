// M3 S1 形状值生产链验证：shape_of 把输入维度物化为真实 int64 结果，
// Shape→Reshape 的控制输入 lower 到既有 invocation/extent 合同。
// 所有负例都在 launch 前失败，且不触发编译或扩容（cache 统计不变）。

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "../src/compiler/internal/primitive_cache.h"
#include "../src/compiler/internal/dynamic_shape_contract.h"
#include "../src/compiler/internal/lowered_graph.h"
#include "../src/compiler/internal/te_to_tir.h"
#include "../src/runtime/internal/compiled_module_node.h"
#include "../src/runtime/internal/module_invocation_contract.h"
#include "kxc/compiler/compiler.h"
#include "kxc/compiler/restricted_symbolic_shape.h"
#include "kxc/relay/op.h"
#include "kxc/relay/type_infer.h"
#include "kxc/relay/transforms/infer_type.h"
#include "kxc/runtime/session.h"
#include "kxc/tir/printer/print_ir.h"
#include "support/primitive_lowering.h"

#include <sstream>

namespace {

namespace restricted = kxc::api::experimental::restricted_symbolic_shape::v1;
namespace compiler_internal = kxc::api::internal;

#define CHECK(condition, message)                                             \
    do {                                                                      \
        if (!(condition)) {                                                   \
            std::cerr << "[FAIL] " << __FUNCTION__ << ": " << message       \
                      << '\n';                                                \
            return false;                                                     \
        }                                                                     \
    } while (false)

bool Throws(const std::function<void()>& action) {
    try {
        action();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

kxc::api::CompileConfig CpuConfig() {
    return kxc::api::CompileConfig::Create(
        kxc::BuildTarget(kxc::Device::CPU()), 2);
}

kxc::Function ShapeOfGraph() {
    const kxc::TensorType data_type({4, 5, 3}, "float32");
    const kxc::Var x("x", data_type);
    return kxc::Function(
        {x}, kxc::Call(kxc::relay::Op::Get("shape_of"), {x}));
}

kxc::runtime::NDArray FloatTensor(std::vector<int64_t> shape, float seed) {
    size_t count = 1;
    for (int64_t extent : shape) count *= static_cast<size_t>(extent);
    std::vector<float> data(count);
    for (size_t index = 0; index < data.size(); ++index) {
        data[index] = static_cast<float>(index) * 0.25f + seed;
    }
    kxc::runtime::NDArray array = kxc::runtime::NDArray::Empty(
        shape, kxc::runtime::DataTypeFromString("float32"), kxc::Device::CPU());
    if (!data.empty()) {
        array.CopyFromBytes(data.data(), data.size() * sizeof(float));
    }
    return array;
}

kxc::runtime::NDArray Int64Tensor(std::vector<int64_t> shape,
                                  const std::vector<int64_t>& values) {
    kxc::runtime::NDArray array = kxc::runtime::NDArray::Empty(
        shape, kxc::runtime::DataTypeFromString("int64"), kxc::Device::CPU());
    if (!values.empty()) {
        array.CopyFromBytes(values.data(), values.size() * sizeof(int64_t));
    }
    return array;
}

bool SameStats(const compiler_internal::PrimitiveCacheStats& left,
               const compiler_internal::PrimitiveCacheStats& right) {
    return left.hits == right.hits && left.misses == right.misses &&
           left.entries == right.entries &&
           left.accounted_bytes == right.accounted_bytes &&
           left.evictions == right.evictions &&
           left.in_flight == right.in_flight &&
           left.merged_waiters == right.merged_waiters &&
           left.failures == right.failures &&
           left.rejections == right.rejections &&
           left.active_pins == right.active_pins;
}

// InferType 合同：shape_of 输出为 int64[rank(data)]，固定向量长度。
bool TestShapeOfInferType() {
    const kxc::Type inferred = kxc::relay::ShapeOfInferType(
        kxc::relay::Attrs(),
        kxc::Array<kxc::Type>{kxc::TensorType({4, 5, 3}, "float32")});
    const auto* tensor = inferred.As<kxc::TensorTypeNode>();
    CHECK(tensor && tensor->dtype == "int64" &&
              tensor->shape.size() == 1 && tensor->shape[0] == 3,
          "shape_of must infer int64[rank(data)] with a fixed vector length");

    CHECK(Throws([&] {
              (void)kxc::relay::ShapeOfInferType(
                  kxc::relay::Attrs(), kxc::Array<kxc::Type>{
                                    kxc::TensorType({}, "float32")});
          }),
          "shape_of must reject scalar (rank 0) inputs");
    CHECK(Throws([&] {
              (void)kxc::relay::ShapeOfInferType(
                  kxc::relay::Attrs(),
                  kxc::Array<kxc::Type>{kxc::TensorType(
                      {1, 2, 3, 4, 5, 6, 7, 8, 9}, "float32")});
          }),
          "shape_of must reject rank beyond its declared fixed-rank subset");
    return true;
}

// 生产 TIR：静态路径下 shape_of 的 kernel 把常量维度写入输出 buffer。
bool TestShapeOfPrimitiveTIR() {
    const kxc::Function function = kxc::relay::InferTypePass(ShapeOfGraph());
    const std::vector<kxc::relay::LoweredFunction> lowered =
        kxc::test_support::LowerPrimitiveUnits(function);
    CHECK(lowered.size() == 1, "shape_of lowers to exactly one primitive unit");
    const kxc::tir::PrimFunc& prim_func = lowered.front()->prim_func;
    CHECK(prim_func.defined() && prim_func->body.defined(),
          "shape_of primitive lowering must produce a defined PrimFunc");
    bool found_operator = false;
    for (const auto& item : prim_func->attrs) {
        if (item.first == "kxc.operator_name") {
            const auto* text = item.second.As<kxc::StringObj>();
            found_operator = text && text->data == "shape_of";
        }
    }
    CHECK(found_operator, "lowered shape_of unit must retain its operator identity");
    CHECK(kxc::relay::internal::GetTEScheduleContract(prim_func).find(
              "target-default-v1") != std::string::npos,
          "static shape_of keeps the default TE schedule policy");
    return true;
}

// 静态生产链：Shape→Gather(常量索引)→Concat 全部物化为真实输出。
bool TestStaticShapeChainProduction() {
    compiler_internal::ClearPrimitiveCacheForTesting();
    const kxc::TensorType data_type({4, 5, 3}, "float32");
    const kxc::Var x("x", data_type);
    const kxc::Expr shape_value =
        kxc::Call(kxc::relay::Op::Get("shape_of"), {x});
    const kxc::Expr head = kxc::Call(
        kxc::relay::Op::Get("gather"),
        {shape_value, kxc::Constant(Int64Tensor({2}, {0, 1}))},
        kxc::relay::GatherAttrs::Create(0));
    const kxc::Expr tail = kxc::Call(
        kxc::relay::Op::Get("gather"),
        {shape_value, kxc::Constant(Int64Tensor({1}, {2}))},
        kxc::relay::GatherAttrs::Create(0));
    const kxc::Expr joined = kxc::Call(
        kxc::relay::Op::Get("concatenate"), {head, tail},
        kxc::relay::ConcatenateAttrs::Create(0));
    const kxc::Function function = kxc::relay::InferTypePass(
        kxc::Function({x}, joined));

    const kxc::api::CompiledGraph compiled =
        kxc::api::Compiler::Compile(function, CpuConfig());
    CHECK(compiled.defined() && compiled.module().entry_count() == 4,
          "static shape chain compiles one unit per Relay call");
    const kxc::runtime::RuntimeSession session(compiled.module(),
                                               compiled.plan());
    const kxc::Array<kxc::runtime::NDArray> outputs =
        session.Run({FloatTensor({4, 5, 3}, -1.0f)});
    CHECK(outputs.size() == 1 && outputs[0].shape().size() == 1 &&
              outputs[0].shape()[0] == 3,
          "static shape chain returns an int64[3] shape value");
    std::vector<int64_t> actual(3, 0);
    outputs[0].CopyToBytes(actual.data(), actual.size() * sizeof(int64_t));
    CHECK(actual == std::vector<int64_t>({4, 5, 3}),
          "Shape→Gather→Concat must materialize the input dimensions");

    const compiler_internal::PrimitiveCacheStats after_compile =
        compiler_internal::GetPrimitiveCacheStats();
    const kxc::Array<kxc::runtime::NDArray> rerun =
        session.Run({FloatTensor({4, 5, 3}, 2.0f)});
    std::vector<int64_t> rerun_values(3, 0);
    rerun[0].CopyToBytes(rerun_values.data(), rerun_values.size() * sizeof(int64_t));
    CHECK(rerun_values == std::vector<int64_t>({4, 5, 3}) &&
              SameStats(after_compile,
                        compiler_internal::GetPrimitiveCacheStats()),
          "static rerun stays numeric-identical without compiling");
    return true;
}

// S1 有界物化：同一产物在两组合法输入形状下都返回正确 shape 值，
// runtime 期间 primitive cache 统计不变。
bool TestBoundedShapeValueMaterialization() {
#if !KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH || !KXC_USE_LLVM
    std::cerr << "bounded shape-value test requires all gates and LLVM\n";
    return false;
#else
    compiler_internal::ClearPrimitiveCacheForTesting();
    const auto prepared = restricted::RestrictedSymbolicShapeAdapter::Prepare(
        ShapeOfGraph(), CpuConfig(),
        {{0, 0, "B", 2, 8, 1}, {0, 1, "S", 1, 8, 1}});
    const restricted::BoundedCompileRequest request =
        restricted::RestrictedSymbolicShapeAdapter::MintBoundedCompileRequest(
            prepared);
    const kxc::api::CompiledGraph compiled =
        kxc::api::Compiler::CompileBounded(request);
    const compiler_internal::PrimitiveCacheStats after_compile =
        compiler_internal::GetPrimitiveCacheStats();

    CHECK(compiled.defined() &&
              compiled.plan().mode() ==
                  kxc::runtime::ExecutablePlanMode::kDynamicFreshOutputV1 &&
              compiled.plan().calls().size() == 1 &&
              compiled.module().entry_count() == 1 &&
              after_compile.entries == 1 && after_compile.misses == 1,
          "bounded shape-value graph publishes one dynamic unit");

    const kxc::Array<kxc::runtime::ValueSpec> plan_values =
        compiled.plan().values();
    CHECK(plan_values.size() == 2 &&
              plan_values[0]->is_input &&
              plan_values[0].shape().size() == 3 &&
              plan_values[0].shape()[0] ==
                  kxc::codegen::kDynamicDimension &&
              plan_values[0].shape()[1] ==
                  kxc::codegen::kDynamicDimension &&
              plan_values[0].shape()[2] == 3,
          "plan keeps the wildcard data input boundary");
    CHECK(plan_values[1].shape().size() == 1 && plan_values[1].shape()[0] == 3 &&
              plan_values[1]->dtype.bits == 64,
          "the shape value is a fixed-length int64 plan value");
    CHECK(compiled.plan().graph_input_guards().size() == 2,
          "graph preflight guards cover both symbolic axes");

    const std::string symbol = std::string(compiled.plan().calls()[0]->symbol);
    const kxc::codegen::KernelSignature signature =
        compiled.module().signature(symbol);
    size_t runtime_extents = 0;
    for (const kxc::codegen::KernelArgSpec& argument : signature.arguments()) {
        if (argument->role == kxc::codegen::KernelArgRole::kRuntimeExtent) {
            ++runtime_extents;
        }
    }
    const kxc::api::ModuleInvocationContract& invocation =
        kxc::api::internal::BorrowCompiledModuleInvocationContract(
            compiled.module(), symbol);
    CHECK(runtime_extents == 2 &&
              invocation.runtime_extent_scalars().size() == 2,
          "the ordered extent ABI carries the consumed input axes in order");
    // 形状值单元的 extent 全部来自输入轴，没有 M2 的 state 来源标量。
    using ExtentSource =
        kxc::api::ModuleRuntimeExtentScalar::Source;
    CHECK(invocation.runtime_extent_scalars()[0].source == ExtentSource::kInputAxis &&
              invocation.runtime_extent_scalars()[1].source == ExtentSource::kInputAxis &&
              invocation.runtime_extent_scalars()[0].expression.has_value() &&
              invocation.runtime_extent_scalars()[1].expression.has_value(),
          "shape-value extents are input-axis sourced and carry an expression");
    // extent 表达式按单元自身输入轴求值：{x} 的 shape[0]/shape[1]。
    CHECK(invocation.runtime_extent_scalars()[0].expression->Evaluate({{4, 5, 3}}) == 4 &&
              invocation.runtime_extent_scalars()[1].expression->Evaluate({{4, 5, 3}}) == 5,
          "extent scalars evaluate to the consumed input axes");
    CHECK(invocation.runtime_extent_scalars()[0].expression->Evaluate({{6, 7, 3}}) == 6 &&
              invocation.runtime_extent_scalars()[1].expression->Evaluate({{6, 7, 3}}) == 7,
          "extent scalars follow the second legal shape");
    CHECK(invocation.outputs().size() == 1 &&
              invocation.outputs()[0].max_bytes == 3 * sizeof(int64_t),
          "the shape-value output stays byte-capped by its fixed length");
    // M2 的 state-sourced extent 把调用合同编码升到 V3；形状值单元沿用
    // 同一个版本化编码，不另立一套。
    CHECK(invocation.CanonicalBytes().find(
              "KXC_MODULE_INVOKE_V4") != std::string::npos,
          "the invocation contract keeps its versioned canonical encoding");

    const kxc::runtime::RuntimeSession session(compiled.module(),
                                               compiled.plan());
    const auto run_shape = [&](std::vector<int64_t> shape,
                               std::vector<int64_t> expected) -> void {
        const kxc::Array<kxc::runtime::NDArray> outputs =
            session.Run({FloatTensor(shape, 0.5f)});
        if (outputs.size() != 1 || outputs[0].shape().size() != 1 ||
            outputs[0].shape()[0] != 3) {
            std::cerr << "[FAIL] shape-value output keeps its fixed length\n";
            std::exit(1);
        }
        std::vector<int64_t> actual(3, 0);
        outputs[0].CopyToBytes(actual.data(), actual.size() * sizeof(int64_t));
        if (actual != expected) {
            std::cerr << "[FAIL] shape value differs from the reference\n";
            std::exit(1);
        }
    };
    run_shape({4, 5, 3}, {4, 5, 3});
    CHECK(SameStats(after_compile,
                    compiler_internal::GetPrimitiveCacheStats()),
          "first legal shape run must not compile");
    run_shape({6, 7, 3}, {6, 7, 3});
    CHECK(SameStats(after_compile,
                    compiler_internal::GetPrimitiveCacheStats()),
          "second legal shape run must not compile or extend the cache");

    const auto rejected_without_compile = [&](const auto& input) {
        const compiler_internal::PrimitiveCacheStats before =
            compiler_internal::GetPrimitiveCacheStats();
        const bool rejected = Throws([&] { (void)session.Run(input); });
        return rejected && SameStats(before,
                                     compiler_internal::GetPrimitiveCacheStats());
    };
    CHECK(rejected_without_compile(
              kxc::Array<kxc::runtime::NDArray>{
                  FloatTensor({10, 5, 3}, 0.0f)}),
          "out-of-bounds extent is rejected before launch");
    CHECK(rejected_without_compile(
              kxc::Array<kxc::runtime::NDArray>{
                  FloatTensor({4, 5, 3, 1}, 0.0f)}),
          "rank mismatch is rejected before launch");
    return true;
#endif
}

// 失败边界：受限形状链的其余算子在本切片仍 fail closed，不许静默放行。
bool TestBoundedShapeChainFailures() {
#if !KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE
    return true;
#else
    // 越界 Gather 索引：链在受限准备时即被拒绝，cache 统计不变。
    const kxc::TensorType data_type({4, 5, 3}, "float32");
    const kxc::Var x("x", data_type);
    const kxc::Expr shape_value =
        kxc::Call(kxc::relay::Op::Get("shape_of"), {x});
    const kxc::Expr out_of_bounds = kxc::Call(
        kxc::relay::Op::Get("gather"),
        {shape_value, kxc::Constant(Int64Tensor({1}, {7}))},
        kxc::relay::GatherAttrs::Create(0));
    const compiler_internal::PrimitiveCacheStats before =
        compiler_internal::GetPrimitiveCacheStats();
    CHECK(Throws([&] {
              (void)restricted::RestrictedSymbolicShapeAdapter::Prepare(
                  kxc::relay::InferTypePass(kxc::Function({x}, out_of_bounds)),
                  CpuConfig(), {{0, 0, "B", 2, 8, 1}, {0, 1, "S", 1, 8, 1}});
          }),
          "bounded admission rejects out-of-bounds gather indices");
    CHECK(SameStats(before, compiler_internal::GetPrimitiveCacheStats()),
          "admission rejection must not touch the primitive cache");

    // 多源链：两份不同来源的 shape 值 concat 被拒绝（唯一表达式来源）。
    const kxc::Var y("y", kxc::TensorType({2, 2}, "float32"));
    const kxc::Expr shape_x =
        kxc::Call(kxc::relay::Op::Get("shape_of"), {x});
    const kxc::Expr shape_y =
        kxc::Call(kxc::relay::Op::Get("shape_of"), {y});
    const kxc::Expr mixed = kxc::Call(
        kxc::relay::Op::Get("concatenate"), {shape_x, shape_y},
        kxc::relay::ConcatenateAttrs::Create(0));
    CHECK(Throws([&] {
              (void)restricted::RestrictedSymbolicShapeAdapter::Prepare(
                  kxc::relay::InferTypePass(kxc::Function({x, y}, mixed)),
                  CpuConfig(), {{0, 0, "B", 2, 8, 1}, {0, 1, "S", 1, 8, 1}});
          }),
          "bounded admission rejects shape chains with two sources");
    return true;
#endif
}

// ---------------------------------------------------------------------------
// S2：受限形状表达式算子（shape_expr / reshape_dynamic / expand /
// constant_of_shape / squeeze / unsqueeze）的类型规则与生产链。
// ---------------------------------------------------------------------------

// 静态 InferType 合同：shape_expr 按 attrs 表达式输出 int64[len]，
// 非法 kind / 越界轴 / 缺 attrs 拒绝。
bool TestShapeExprInferType() {
    using kxc::relay::ShapeExprAttrs;
    const kxc::TensorType data_type({4, 5, 3}, "float32");
    const kxc::Var x("x", data_type);

    const kxc::Type inferred = kxc::relay::ShapeExprInferType(
        ShapeExprAttrs::Create(
            kxc::Array<int64_t>({kxc::relay::kShapeExprKindInputAxis,
                                 kxc::relay::kShapeExprKindConst,
                                 kxc::relay::kShapeExprKindInputAxis}),
            kxc::Array<int64_t>({0, 6, 0}),
            kxc::Array<int64_t>({0, 0, 2})),
        kxc::Array<kxc::Type>{data_type});
    const auto* tensor = inferred.As<kxc::TensorTypeNode>();
    CHECK(tensor && tensor->dtype == "int64" && tensor->shape.size() == 1 &&
              tensor->shape[0] == 3,
          "shape_expr infers int64[len(expr)]");

    CHECK(Throws([&] {
              (void)kxc::relay::ShapeExprInferType(
                  kxc::relay::ShapeExprAttrs::Create(
                      kxc::Array<int64_t>(
                          {kxc::relay::kShapeExprKindInputAxis}),
                      kxc::Array<int64_t>({0}),
                      kxc::Array<int64_t>({3})),
                  kxc::Array<kxc::Type>{data_type});
          }),
          "shape_expr rejects axis references beyond the source rank");
    CHECK(Throws([&] {
              (void)kxc::relay::ShapeExprInferType(
                  kxc::relay::Attrs(),
                  kxc::Array<kxc::Type>{data_type});
          }),
          "shape_expr without resolver attrs is rejected");
    return true;
}

// 静态 reshape_dynamic：控制输入来自受限形状表达式（0-copy 与元素数
// 证明在 attrs 内），InferType 推导目标形状并拒绝元素数不等。
bool TestReshapeDynamicInferType() {
    using kxc::relay::ReshapeDynamicAttrs;
    const kxc::TensorType data_type({4, 5, 3}, "float32");
    const kxc::Var x("x", data_type);
    const kxc::Type control = kxc::TensorType({3}, "int64");
    // target = [axis1, axis0, axis2] = [5,4,3]：维度重排，元素总数可证明。
    const kxc::Type reshaped = kxc::relay::ReshapeDynamicInferType(
        ReshapeDynamicAttrs::Create(
            kxc::Array<int64_t>({kxc::relay::kShapeExprKindInputAxis,
                                 kxc::relay::kShapeExprKindInputAxis,
                                 kxc::relay::kShapeExprKindInputAxis}),
            kxc::Array<int64_t>({0, 0, 0}),
            kxc::Array<int64_t>({1, 0, 2})),
        kxc::Array<kxc::Type>{data_type, control});
    const auto* tensor = reshaped.As<kxc::TensorTypeNode>();
    CHECK(tensor && tensor->dtype == "float32" &&
              tensor->shape.size() == 3 && tensor->shape[0] == 5 &&
              tensor->shape[1] == 4 && tensor->shape[2] == 3,
          "reshape_dynamic infers the expression target");
    CHECK(Throws([&] {
              (void)kxc::relay::ReshapeDynamicInferType(
                  ReshapeDynamicAttrs::Create(
                      kxc::Array<int64_t>(
                          {kxc::relay::kShapeExprKindInputAxis,
                           kxc::relay::kShapeExprKindInputAxis,
                           kxc::relay::kShapeExprKindInputAxis}),
                      kxc::Array<int64_t>({0, 0, 0}),
                      kxc::Array<int64_t>({1, 1, 2})),
                  kxc::Array<kxc::Type>{data_type, control});
          }),
          "reshape_dynamic rejects a provable element count mismatch");
    CHECK(Throws([&] {
              (void)kxc::relay::ReshapeDynamicInferType(
                  ReshapeDynamicAttrs::Create(
                      kxc::Array<int64_t>(
                          {kxc::relay::kShapeExprKindConst}),
                      kxc::Array<int64_t>({-1}),
                      kxc::Array<int64_t>({0})),
                  kxc::Array<kxc::Type>{data_type,
                                        kxc::TensorType({1}, "int64")});
          }),
          "reshape_dynamic rejects unresolved -1 inference targets");
    return true;
}

// 静态 expand/constant_of_shape/squeeze/unsqueeze InferType 合同。
bool TestS2StaticInferType() {
    const kxc::TensorType data_type({1, 5, 3}, "float32");
    const kxc::Type control = kxc::TensorType({3}, "int64");

    // expand：axis0 广播 1→4，其余等值。
    const kxc::Type expanded = kxc::relay::ExpandDynamicInferType(
        kxc::relay::ExpandDynamicAttrs::Create(
            kxc::Array<int64_t>({kxc::relay::kShapeExprKindConst,
                                 kxc::relay::kShapeExprKindInputAxis,
                                 kxc::relay::kShapeExprKindInputAxis}),
            kxc::Array<int64_t>({4, 0, 0}),
            kxc::Array<int64_t>({0, 1, 2})),
        kxc::Array<kxc::Type>{data_type, control});
    const auto* expand_tensor = expanded.As<kxc::TensorTypeNode>();
    CHECK(expand_tensor && expand_tensor->shape.size() == 3 &&
              expand_tensor->shape[0] == 4 && expand_tensor->shape[1] == 5 &&
              expand_tensor->shape[2] == 3,
          "expand infers the restricted broadcast target");
    CHECK(Throws([&] {
              (void)kxc::relay::ExpandDynamicInferType(
                  kxc::relay::ExpandDynamicAttrs::Create(
                      kxc::Array<int64_t>(
                          {kxc::relay::kShapeExprKindConst,
                           kxc::relay::kShapeExprKindInputAxis,
                           kxc::relay::kShapeExprKindInputAxis}),
                      kxc::Array<int64_t>({4, 0, 0}),
                      kxc::Array<int64_t>({0, 1, 2})),
                  kxc::Array<kxc::Type>{kxc::TensorType({2, 5, 3}, "float32"),
                                        control});
          }),
          "expand rejects a non-broadcastable data axis");

    // constant_of_shape：int32 标量填充、空目标（0 维元素）、字节上限。
    const kxc::Type filled = kxc::relay::ConstantOfShapeInferType(
        kxc::relay::ConstantOfShapeAttrs::Create(
            kxc::Array<int64_t>({2, 3}), /*dtype_code=*/1, 7.0),
        kxc::Array<kxc::Type>{kxc::TensorType({2}, "int64")});
    const auto* fill_tensor = filled.As<kxc::TensorTypeNode>();
    CHECK(fill_tensor && fill_tensor->dtype == "int32" &&
              fill_tensor->shape.size() == 2 && fill_tensor->shape[0] == 2 &&
              fill_tensor->shape[1] == 3,
          "constant_of_shape infers its constant target and fill dtype");
    const kxc::Type empty = kxc::relay::ConstantOfShapeInferType(
        kxc::relay::ConstantOfShapeAttrs::Create(
            kxc::Array<int64_t>({0}), 0, 1.0),
        kxc::Array<kxc::Type>{kxc::TensorType({1}, "int64")});
    const auto* empty_tensor = empty.As<kxc::TensorTypeNode>();
    CHECK(empty_tensor && empty_tensor->shape.size() == 1 &&
              empty_tensor->shape[0] == 0,
          "constant_of_shape admits an empty target");
    CHECK(Throws([&] {
              (void)kxc::relay::ConstantOfShapeInferType(
                  kxc::relay::ConstantOfShapeAttrs::Create(
                      kxc::Array<int64_t>({1 << 20, 1 << 20}), 1, 0.0),
                  kxc::Array<kxc::Type>{kxc::TensorType({2}, "int64")});
          }),
          "constant_of_shape rejects targets over its byte cap");
    CHECK(Throws([&] {
              (void)kxc::relay::ConstantOfShapeInferType(
                  kxc::relay::ConstantOfShapeAttrs::Create(
                      kxc::Array<int64_t>({2}), 1, 0.5),
                  kxc::Array<kxc::Type>{kxc::TensorType({1}, "int64")});
          }),
          "constant_of_shape rejects fractional integer fills");

    // squeeze：被移除维必须证明为 1；重复/越界 axis 拒绝。
    const kxc::TensorType squeezable({4, 1, 3}, "float32");
    const kxc::Type squeezed = kxc::relay::SqueezeInferType(
        kxc::relay::SqueezeAttrs::Create(kxc::Array<int64_t>({1})),
        kxc::Array<kxc::Type>{squeezable});
    const auto* squeeze_tensor = squeezed.As<kxc::TensorTypeNode>();
    CHECK(squeeze_tensor && squeeze_tensor->shape.size() == 2 &&
              squeeze_tensor->shape[0] == 4 && squeeze_tensor->shape[1] == 3,
          "squeeze removes the provably unit axis");
    CHECK(Throws([&] {
              (void)kxc::relay::SqueezeInferType(
                  kxc::relay::SqueezeAttrs::Create(
                      kxc::Array<int64_t>({0})),
                  kxc::Array<kxc::Type>{squeezable});
          }),
          "squeeze rejects removing a non-unit axis");
    CHECK(Throws([&] {
              (void)kxc::relay::SqueezeInferType(
                  kxc::relay::SqueezeAttrs::Create(
                      kxc::Array<int64_t>({1, 1})),
                  kxc::Array<kxc::Type>{squeezable});
          }),
          "squeeze rejects duplicate axes");

    // unsqueeze：插入 1 维，结果 rank 可证明；重复/越界拒绝。
    const kxc::Type unsqueezed = kxc::relay::UnsqueezeInferType(
        kxc::relay::UnsqueezeAttrs::Create(kxc::Array<int64_t>({0, 2})),
        kxc::Array<kxc::Type>{data_type});
    const auto* unsqueeze_tensor = unsqueezed.As<kxc::TensorTypeNode>();
    CHECK(unsqueeze_tensor && unsqueeze_tensor->shape.size() == 5 &&
              unsqueeze_tensor->shape[0] == 1 && unsqueeze_tensor->shape[2] == 1,
          "unsqueeze inserts unit axes at explicit positions");
    CHECK(Throws([&] {
              (void)kxc::relay::UnsqueezeInferType(
                  kxc::relay::UnsqueezeAttrs::Create(
                      kxc::Array<int64_t>({0, 0})),
                  kxc::Array<kxc::Type>{data_type});
          }),
          "unsqueeze rejects duplicate axes");
    CHECK(Throws([&] {
              (void)kxc::relay::UnsqueezeInferType(
                  kxc::relay::UnsqueezeAttrs::Create(
                      kxc::Array<int64_t>({9})),
                  kxc::Array<kxc::Type>{data_type});
          }),
          "unsqueeze rejects out-of-range axes");
    return true;
}

// S2 静态生产链：shape_expr / reshape_dynamic / expand / squeeze /
// unsqueeze / constant_of_shape 以受限 attrs 进入真实编译执行，
// 数值与参考一致；所有单元保留算子身份。
bool TestS2StaticProduction() {
    compiler_internal::ClearPrimitiveCacheForTesting();

    // reshape_dynamic(x[4,5,3] → [5,4,3])，控制输入是折叠后的 shape_expr。
    {
        const kxc::Var x("x", kxc::TensorType({4, 5, 3}, "float32"));
        const kxc::Expr control = kxc::Call(
            kxc::relay::Op::Get("shape_expr"), {x},
            kxc::relay::ShapeExprAttrs::Create(
                kxc::Array<int64_t>({kxc::relay::kShapeExprKindInputAxis,
                                     kxc::relay::kShapeExprKindInputAxis,
                                     kxc::relay::kShapeExprKindInputAxis}),
                kxc::Array<int64_t>({0, 0, 0}),
                kxc::Array<int64_t>({1, 0, 2})));
        const kxc::Expr reshaped = kxc::Call(
            kxc::relay::Op::Get("reshape_dynamic"), {x, control},
            kxc::relay::ReshapeDynamicAttrs::Create(
                kxc::Array<int64_t>({kxc::relay::kShapeExprKindInputAxis,
                                     kxc::relay::kShapeExprKindInputAxis,
                                     kxc::relay::kShapeExprKindInputAxis}),
                kxc::Array<int64_t>({0, 0, 0}),
                kxc::Array<int64_t>({1, 0, 2})));
        const kxc::Function function =
            kxc::relay::InferTypePass(kxc::Function({x}, reshaped));
        const kxc::api::CompiledGraph compiled =
            kxc::api::Compiler::Compile(function, CpuConfig());
        const kxc::runtime::RuntimeSession session(compiled.module(),
                                                   compiled.plan());
        const kxc::runtime::NDArray input = FloatTensor({4, 5, 3}, 0.0f);
        const kxc::Array<kxc::runtime::NDArray> outputs =
            session.Run({input});
        CHECK(outputs.size() == 1 && outputs[0].shape().size() == 3 &&
                  outputs[0].shape()[0] == 5 &&
                  outputs[0].shape()[1] == 4 &&
                  outputs[0].shape()[2] == 3,
              "reshape_dynamic returns the expression target shape");
        std::vector<float> actual(60, 0.0f);
        outputs[0].CopyToBytes(actual.data(), actual.size() * sizeof(float));
        std::vector<float> expected(60, 0.0f);
        for (size_t i = 0; i < expected.size(); ++i) {
            expected[i] = static_cast<float>(i) * 0.25f;
        }
        CHECK(actual == expected,
              "reshape_dynamic is a row-major element-preserving copy");
    }

    // expand(x[1,5,3] → [4,5,3])：axis0 广播。
    {
        const kxc::Var x("x", kxc::TensorType({1, 5, 3}, "float32"));
        const kxc::Expr control = kxc::Call(
            kxc::relay::Op::Get("shape_expr"), {x},
            kxc::relay::ShapeExprAttrs::Create(
                kxc::Array<int64_t>({kxc::relay::kShapeExprKindConst,
                                     kxc::relay::kShapeExprKindInputAxis,
                                     kxc::relay::kShapeExprKindInputAxis}),
                kxc::Array<int64_t>({4, 0, 0}),
                kxc::Array<int64_t>({0, 1, 2})));
        const kxc::Expr expanded = kxc::Call(
            kxc::relay::Op::Get("expand_dynamic"), {x, control},
            kxc::relay::ExpandDynamicAttrs::Create(
                kxc::Array<int64_t>({kxc::relay::kShapeExprKindConst,
                                     kxc::relay::kShapeExprKindInputAxis,
                                     kxc::relay::kShapeExprKindInputAxis}),
                kxc::Array<int64_t>({4, 0, 0}),
                kxc::Array<int64_t>({0, 1, 2})));
        const kxc::Function function =
            kxc::relay::InferTypePass(kxc::Function({x}, expanded));
        const kxc::api::CompiledGraph compiled =
            kxc::api::Compiler::Compile(function, CpuConfig());
        const kxc::runtime::RuntimeSession session(compiled.module(),
                                                   compiled.plan());
        const kxc::Array<kxc::runtime::NDArray> outputs =
            session.Run({FloatTensor({1, 5, 3}, 1.0f)});
        CHECK(outputs.size() == 1 && outputs[0].shape().size() == 3 &&
                  outputs[0].shape()[0] == 4 && outputs[0].shape()[1] == 5 &&
                  outputs[0].shape()[2] == 3,
              "expand returns the restricted broadcast target");
        std::vector<float> actual(60, 0.0f);
        outputs[0].CopyToBytes(actual.data(), actual.size() * sizeof(float));
        std::vector<float> expected;
        for (size_t row = 0; row < 4; ++row) {
            for (size_t i = 0; i < 15; ++i) {
                expected.push_back(static_cast<float>(i) * 0.25f + 1.0f);
            }
        }
        CHECK(actual == expected, "expand broadcasts axis 0 correctly");
    }

    // squeeze(x[4,1,3] → [4,3]) + unsqueeze(→ [1,4,1,3])。
    {
        const kxc::Var x("x", kxc::TensorType({4, 1, 3}, "float32"));
        const kxc::Expr squeezed = kxc::Call(
            kxc::relay::Op::Get("squeeze"), {x},
            kxc::relay::SqueezeAttrs::Create(kxc::Array<int64_t>({1})));
        const kxc::Expr unsqueezed = kxc::Call(
            kxc::relay::Op::Get("unsqueeze"), {squeezed},
            kxc::relay::UnsqueezeAttrs::Create(
                kxc::Array<int64_t>({0, 2})));
        const kxc::Function function =
            kxc::relay::InferTypePass(kxc::Function({x}, unsqueezed));
        const kxc::api::CompiledGraph compiled =
            kxc::api::Compiler::Compile(function, CpuConfig());
        const kxc::runtime::RuntimeSession session(compiled.module(),
                                                   compiled.plan());
        const kxc::Array<kxc::runtime::NDArray> outputs =
            session.Run({FloatTensor({4, 1, 3}, 2.0f)});
        CHECK(outputs.size() == 1 && outputs[0].shape().size() == 4 &&
                  outputs[0].shape()[0] == 1 && outputs[0].shape()[1] == 4 &&
                  outputs[0].shape()[2] == 1 && outputs[0].shape()[3] == 3,
              "squeeze/unsqueeze composition proves its output rank");
        std::vector<float> actual(12, 0.0f);
        outputs[0].CopyToBytes(actual.data(), actual.size() * sizeof(float));
        std::vector<float> expected;
        for (size_t i = 0; i < 12; ++i) {
            expected.push_back(static_cast<float>(i) * 0.25f + 2.0f);
        }
        CHECK(actual == expected, "squeeze/unsqueeze preserves elements");
    }

    // constant_of_shape：int32 标量填充。
    {
        const kxc::Var shape("shape", kxc::TensorType({2}, "int64"));
        const kxc::Expr filled = kxc::Call(
            kxc::relay::Op::Get("constant_of_shape"), {shape},
            kxc::relay::ConstantOfShapeAttrs::Create(
                kxc::Array<int64_t>({2, 3}), /*dtype_code=*/1, 7.0));
        const kxc::Function function =
            kxc::relay::InferTypePass(kxc::Function({shape}, filled));
        const kxc::api::CompiledGraph compiled =
            kxc::api::Compiler::Compile(function, CpuConfig());
        const kxc::runtime::RuntimeSession session(compiled.module(),
                                                   compiled.plan());
        const kxc::Array<kxc::runtime::NDArray> outputs =
            session.Run({Int64Tensor({2}, {2, 3})});
        CHECK(outputs.size() == 1 && outputs[0].shape().size() == 2 &&
                  outputs[0].shape()[0] == 2 && outputs[0].shape()[1] == 3,
              "constant_of_shape returns its bounded constant target");
        std::vector<int32_t> actual(6, 0);
        outputs[0].CopyToBytes(actual.data(), actual.size() * sizeof(int32_t));
        CHECK(actual == std::vector<int32_t>(6, 7),
              "constant_of_shape fills the explicit scalar value");
    }
    return true;
}

// S2 算子的生产 primitive lowering 合同：每个受限形状算子都必须下降为
// 一个保留自身算子身份的 PrimFunc 单元，而不是被折叠进别的单元或只留
// metadata。这条也是契约检查器要求的 tir lowering 证据。
bool TestS2PrimitiveTIR() {
    const auto lowers_to = [](const kxc::Function& function,
                              const std::string& op_name) {
        const std::vector<kxc::relay::LoweredFunction> lowered =
            kxc::test_support::LowerPrimitiveUnits(function);
        for (const kxc::relay::LoweredFunction& unit : lowered) {
            const kxc::tir::PrimFunc& prim_func = unit->prim_func;
            if (!prim_func.defined() || !prim_func->body.defined()) continue;
            for (const auto& item : prim_func->attrs) {
                if (item.first == "kxc.operator_name") {
                    const auto* text = item.second.As<kxc::StringObj>();
                    if (text && text->data == op_name) return true;
                }
            }
        }
        return false;
    };
    const kxc::Array<int64_t> permute_kinds(
        {kxc::relay::kShapeExprKindInputAxis,
         kxc::relay::kShapeExprKindInputAxis,
         kxc::relay::kShapeExprKindInputAxis});

    // shape_expr 与 reshape_dynamic：[4,5,3] → [5,4,3]。
    {
        const kxc::Var x("x", kxc::TensorType({4, 5, 3}, "float32"));
        const kxc::Expr control = kxc::Call(
            kxc::relay::Op::Get("shape_expr"), {x},
            kxc::relay::ShapeExprAttrs::Create(
                permute_kinds, kxc::Array<int64_t>({0, 0, 0}),
                kxc::Array<int64_t>({1, 0, 2})));
        const kxc::Expr reshaped = kxc::Call(
            kxc::relay::Op::Get("reshape_dynamic"), {x, control},
            kxc::relay::ReshapeDynamicAttrs::Create(
                permute_kinds, kxc::Array<int64_t>({0, 0, 0}),
                kxc::Array<int64_t>({1, 0, 2})));
        const kxc::Function function =
            kxc::relay::InferTypePass(kxc::Function({x}, reshaped));
        CHECK(lowers_to(function, "shape_expr"),
              "shape_expr lowers to its own primitive unit");
        CHECK(lowers_to(function, "reshape_dynamic"),
              "reshape_dynamic lowers to its own primitive unit");
    }

    // expand：[1,5,3] → [4,5,3]。
    {
        const kxc::Var x("x", kxc::TensorType({1, 5, 3}, "float32"));
        const kxc::Array<int64_t> kinds(
            {kxc::relay::kShapeExprKindConst,
             kxc::relay::kShapeExprKindInputAxis,
             kxc::relay::kShapeExprKindInputAxis});
        const kxc::Expr control = kxc::Call(
            kxc::relay::Op::Get("shape_expr"), {x},
            kxc::relay::ShapeExprAttrs::Create(
                kinds, kxc::Array<int64_t>({4, 0, 0}),
                kxc::Array<int64_t>({0, 1, 2})));
        const kxc::Expr expanded = kxc::Call(
            kxc::relay::Op::Get("expand_dynamic"), {x, control},
            kxc::relay::ExpandDynamicAttrs::Create(
                kinds, kxc::Array<int64_t>({4, 0, 0}),
                kxc::Array<int64_t>({0, 1, 2})));
        CHECK(lowers_to(kxc::relay::InferTypePass(kxc::Function({x}, expanded)),
                        "expand_dynamic"),
              "expand lowers to its own primitive unit");
    }

    // squeeze / unsqueeze：[4,1,3] → [4,3] → [1,4,1,3]。
    {
        const kxc::Var x("x", kxc::TensorType({4, 1, 3}, "float32"));
        const kxc::Expr squeezed = kxc::Call(
            kxc::relay::Op::Get("squeeze"), {x},
            kxc::relay::SqueezeAttrs::Create(kxc::Array<int64_t>({1})));
        const kxc::Expr unsqueezed = kxc::Call(
            kxc::relay::Op::Get("unsqueeze"), {squeezed},
            kxc::relay::UnsqueezeAttrs::Create(kxc::Array<int64_t>({0, 2})));
        const kxc::Function function =
            kxc::relay::InferTypePass(kxc::Function({x}, unsqueezed));
        CHECK(lowers_to(function, "squeeze"),
              "squeeze lowers to its own primitive unit");
        CHECK(lowers_to(function, "unsqueeze"),
              "unsqueeze lowers to its own primitive unit");
    }

    // constant_of_shape：int32 标量填充到 [2,3]。
    {
        const kxc::Var shape("shape", kxc::TensorType({2}, "int64"));
        const kxc::Expr filled = kxc::Call(
            kxc::relay::Op::Get("constant_of_shape"), {shape},
            kxc::relay::ConstantOfShapeAttrs::Create(
                kxc::Array<int64_t>({2, 3}), /*dtype_code=*/1, 7.0));
        CHECK(lowers_to(
                  kxc::relay::InferTypePass(kxc::Function({shape}, filled)),
                  "constant_of_shape"),
              "constant_of_shape lowers to its own primitive unit");
    }
    return true;
}

// S2 有界链：Shape→Gather→Concat→ReshapeDynamic 同一产物两组合法输入
// 形状执行（S1 纵向链的有界扩展）；runtime 期间编译/cache 统计不变。
bool TestBoundedShapeToReshapeProduction() {
#if !KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH || !KXC_USE_LLVM
    std::cerr << "bounded shape-to-reshape test requires all gates and LLVM\n";
    return false;
#else
    compiler_internal::ClearPrimitiveCacheForTesting();
    // x[B,S,3] → shape_of → gather([1,0]) ‖ gather([2]) → concat →
    // reshape_dynamic(x → [S,B,3])。目标 [S,B,3] 与 [B,S,3] 元素总数在
    // DimExpr canonical 乘积下可证明相等；gather/concat 折叠成 shape_expr。
    const kxc::TensorType data_type({4, 5, 3}, "float32");
    const kxc::Var x("x", data_type);
    const kxc::Expr shape_value =
        kxc::Call(kxc::relay::Op::Get("shape_of"), {x});
    const kxc::Expr head = kxc::Call(
        kxc::relay::Op::Get("gather"),
        {shape_value, kxc::Constant(Int64Tensor({2}, {1, 0}))},
        kxc::relay::GatherAttrs::Create(0));
    const kxc::Expr tail = kxc::Call(
        kxc::relay::Op::Get("gather"),
        {shape_value, kxc::Constant(Int64Tensor({1}, {2}))},
        kxc::relay::GatherAttrs::Create(0));
    const kxc::Expr control = kxc::Call(
        kxc::relay::Op::Get("concatenate"), {head, tail},
        kxc::relay::ConcatenateAttrs::Create(0));
    const kxc::Expr reshaped = kxc::Call(
        kxc::relay::Op::Get("reshape_dynamic"), {x, control});
    // 受限入口接收未推导类型的链：目标表达式由 Prepare 内的解析器证明并
    // 投影成 attrs，之后才重新推导类型。调用方先跑 InferTypePass 会在链
    // 折叠前就要求 attrs，等于绕过唯一表达式来源。
    const kxc::Function function = kxc::Function({x}, reshaped);

    // B 带整除约束（2 的倍数），以便下面的负例真正触发整除 guard 而不是
    // 只触发范围 guard；两个合法 shape 的 B=4、6 都满足它。
    const auto prepared = restricted::RestrictedSymbolicShapeAdapter::Prepare(
        function, CpuConfig(), {{0, 0, "B", 2, 8, 2}, {0, 1, "S", 1, 8, 1}});
    const restricted::BoundedCompileRequest request =
        restricted::RestrictedSymbolicShapeAdapter::MintBoundedCompileRequest(
            prepared);
    const kxc::api::CompiledGraph compiled =
        kxc::api::Compiler::CompileBounded(request);
    const compiler_internal::PrimitiveCacheStats after_compile =
        compiler_internal::GetPrimitiveCacheStats();
    CHECK(compiled.defined() &&
              compiled.plan().calls().size() == 2 &&
              compiled.module().entry_count() == 2,
          "the folded chain lowers to a shape_expr unit and a "
          "reshape_dynamic unit");
    CHECK(compiled.plan().mode() ==
              kxc::runtime::ExecutablePlanMode::kDynamicFreshOutputV1,
          "the bounded shape-to-reshape plan stays dynamic fresh output");

    const kxc::runtime::RuntimeSession session(compiled.module(),
                                               compiled.plan());
    const auto run_shape = [&](std::vector<int64_t> shape) {
        const kxc::runtime::NDArray input = FloatTensor(shape, 0.0f);
        const kxc::Array<kxc::runtime::NDArray> outputs =
            session.Run({input});
        // 目标 [S, B, 3]：行-major 拷贝保持线性元素顺序。
        if (outputs.size() != 1 || outputs[0].shape().size() != 3 ||
            outputs[0].shape()[0] != shape[1] ||
            outputs[0].shape()[1] != shape[0] ||
            outputs[0].shape()[2] != shape[2]) {
            std::cerr << "[FAIL] reshape target shape differs\n";
            std::exit(1);
        }
        std::vector<float> actual(
            static_cast<size_t>(shape[0] * shape[1] * shape[2]), 0.0f);
        outputs[0].CopyToBytes(actual.data(),
                               actual.size() * sizeof(float));
        std::vector<float> expected = actual;
        for (size_t i = 0; i < expected.size(); ++i) {
            expected[i] = static_cast<float>(i) * 0.25f;
        }
        if (actual != expected) {
            std::cerr << "[FAIL] reshape values differ from the reference\n";
            std::exit(1);
        }
    };
    run_shape({4, 5, 3});
    CHECK(SameStats(after_compile,
                    compiler_internal::GetPrimitiveCacheStats()),
          "first legal shape run must not compile");
    run_shape({6, 7, 3});
    CHECK(SameStats(after_compile,
                    compiler_internal::GetPrimitiveCacheStats()),
          "second legal shape run must not compile");

    const auto rejected_without_compile = [&](const auto& input) {
        const compiler_internal::PrimitiveCacheStats before =
            compiler_internal::GetPrimitiveCacheStats();
        const bool rejected = Throws([&] { (void)session.Run(input); });
        return rejected && SameStats(before,
                                     compiler_internal::GetPrimitiveCacheStats());
    };
    CHECK(rejected_without_compile(
              kxc::Array<kxc::runtime::NDArray>{
                  FloatTensor({10, 5, 3}, 0.0f)}),
          "out-of-bounds extent is rejected before launch");
    CHECK(rejected_without_compile(
              kxc::Array<kxc::runtime::NDArray>{
                  FloatTensor({5, 5, 3}, 0.0f)}),
          "an extent violating the divisibility guard is rejected");
    return true;
#endif
}

// S2 有界负例：不可证明的目标（元素数不等）、-1 推断、非法广播在
// 受限准备（launch 前）拒绝，cache 统计不变。
bool TestBoundedS2Failures() {
#if !KXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE
    return true;
#else
    const kxc::TensorType data_type({4, 5, 3}, "float32");
    const kxc::Var x("x", data_type);
    const kxc::Expr shape_value =
        kxc::Call(kxc::relay::Op::Get("shape_of"), {x});
    const kxc::Expr control = kxc::Call(
        kxc::relay::Op::Get("gather"),
        {shape_value, kxc::Constant(Int64Tensor({3}, {0, 1, 2}))},
        kxc::relay::GatherAttrs::Create(0));
    const std::vector<kxc::Expr> bad_targets = {
        // 元素总数不等：额外长度 2 的轴使目标元素数翻倍。
        // 长度 1 的附加轴合法，不能用来测试元素数拒绝。
        kxc::Call(kxc::relay::Op::Get("concatenate"),
                  {control,
                   kxc::Constant(Int64Tensor({1}, {2}))},
                  kxc::relay::ConcatenateAttrs::Create(0)),
        // B/S cancellation leaves 3/2, not a positive integral inferred axis.
        kxc::Call(kxc::relay::Op::Get("concatenate"),
                  {kxc::Call(kxc::relay::Op::Get("gather"),
                             {shape_value,
                              kxc::Constant(Int64Tensor({2}, {0, 1}))},
                             kxc::relay::GatherAttrs::Create(0)),
                   kxc::Constant(Int64Tensor({2}, {-1, 2}))},
                  kxc::relay::ConcatenateAttrs::Create(0)),
    };
    const compiler_internal::PrimitiveCacheStats before =
        compiler_internal::GetPrimitiveCacheStats();
    const kxc::Expr unit_axis_target = kxc::Call(
        kxc::relay::Op::Get("concatenate"),
        {control, kxc::Constant(Int64Tensor({1}, {1}))},
        kxc::relay::ConcatenateAttrs::Create(0));
    CHECK(!Throws([&] {
              (void)restricted::RestrictedSymbolicShapeAdapter::Prepare(
                  kxc::Function({x}, kxc::Call(kxc::relay::Op::Get("reshape_dynamic"),
                                               {x, unit_axis_target})),
                  CpuConfig(), {{0, 0, "B", 2, 8, 1}, {0, 1, "S", 1, 8, 1}});
          }),
          "an appended unit axis preserves the reshape element count");
    // 这些负例必须由受限准备（链式证明）拒绝，因此和正例一样不预先跑
    // InferTypePass —— 否则 Throws 来自 attrs 缺失的类型推导，测不到
    // bounded admission 本身。
    for (const kxc::Expr& target : bad_targets) {
        CHECK(Throws([&] {
                  (void)restricted::RestrictedSymbolicShapeAdapter::Prepare(
                      kxc::Function(
                          {x},
                          kxc::Call(kxc::relay::Op::Get("reshape_dynamic"),
                                    {x, target})),
                      CpuConfig(),
                      {{0, 0, "B", 2, 8, 1}, {0, 1, "S", 1, 8, 1}});
              }),
              "bounded admission rejects unprovable reshape targets");
    }
    // 非法广播：expand 目标把非 1 轴放大。轴 0 的数据维是符号 B，目标是
    // 常量 8；B 既不可证明等于 8 也不可证明等于 1，受限规则必须拒绝。
    // （用 x 自身的形状做目标是合法的恒等广播，测不到这条规则。）
    const kxc::Expr widened_target = kxc::Call(
        kxc::relay::Op::Get("concatenate"),
        {kxc::Constant(Int64Tensor({1}, {8})),
         kxc::Call(kxc::relay::Op::Get("gather"),
                   {shape_value, kxc::Constant(Int64Tensor({2}, {1, 2}))},
                   kxc::relay::GatherAttrs::Create(0))},
        kxc::relay::ConcatenateAttrs::Create(0));
    CHECK(Throws([&] {
              (void)restricted::RestrictedSymbolicShapeAdapter::Prepare(
                  kxc::Function(
                      {x},
                      kxc::Call(kxc::relay::Op::Get("expand_dynamic"),
                                {x, widened_target})),
                  CpuConfig(), {{0, 0, "B", 2, 8, 1}, {0, 1, "S", 1, 8, 1}});
          }),
          "bounded admission rejects illegal expand broadcasts");
    // squeeze 非 1 轴。
    CHECK(Throws([&] {
              (void)restricted::RestrictedSymbolicShapeAdapter::Prepare(
                  kxc::Function(
                      {x},
                      kxc::Call(kxc::relay::Op::Get("squeeze"), {x},
                                kxc::relay::SqueezeAttrs::Create(
                                    kxc::Array<int64_t>({0})))),
                  CpuConfig(), {{0, 0, "B", 2, 8, 1}, {0, 1, "S", 1, 8, 1}});
          }),
          "bounded admission rejects squeezing a non-unit axis");
    CHECK(SameStats(before, compiler_internal::GetPrimitiveCacheStats()),
          "S2 admission rejections must not touch the primitive cache");
    return true;
#endif
}

// 身份：受限形状表达式 attrs 进入版本化 canonical 身份；同结构异名
// 保身份，不同表达式不误复用。
bool TestS2Identity() {
    const auto key_of = [](const kxc::Function& function) {
        return kxc::api::Compiler::BuildGraphSemanticKey(
            kxc::relay::InferTypePass(function)).canonical_bytes();
    };
    // 两个目标表达式都必须是合法的（元素总数与 [4,5,3] 的 60 相等），
    // 但结构不同：一个把轴 0 写成常量 4，另一个把轴 2 写成常量 3。二者
    // 求值到同一形状 [4,5,3]，因此身份必须跟踪**表达式**而不是结果形状。
    const auto reshape_graph =
        [](const std::vector<int64_t>& kinds, const std::vector<int64_t>& values,
           const std::vector<int64_t>& axes) -> kxc::Function {
        const kxc::Var x("x", kxc::TensorType({4, 5, 3}, "float32"));
        const kxc::Expr control = kxc::Call(
            kxc::relay::Op::Get("shape_expr"), {x},
            kxc::relay::ShapeExprAttrs::Create(
                kxc::Array<int64_t>(std::vector<int64_t>(kinds)),
                kxc::Array<int64_t>(std::vector<int64_t>(values)),
                kxc::Array<int64_t>(std::vector<int64_t>(axes))));
        const kxc::Expr reshaped = kxc::Call(
            kxc::relay::Op::Get("reshape_dynamic"), {x, control},
            kxc::relay::ReshapeDynamicAttrs::Create(
                kxc::Array<int64_t>(std::vector<int64_t>(kinds)),
                kxc::Array<int64_t>(std::vector<int64_t>(values)),
                kxc::Array<int64_t>(std::vector<int64_t>(axes))));
        return kxc::Function({x}, reshaped);
    };
    // [const 4, axis1, axis2] → [4,5,3]
    const std::vector<int64_t> first_kinds = {
        kxc::relay::kShapeExprKindConst, kxc::relay::kShapeExprKindInputAxis,
        kxc::relay::kShapeExprKindInputAxis};
    const std::string first =
        key_of(reshape_graph(first_kinds, {4, 0, 0}, {0, 1, 2}));
    // 参数名不同（临时变量名）不影响身份。
    {
        const kxc::Var data("data", kxc::TensorType({4, 5, 3}, "float32"));
        const kxc::Expr control = kxc::Call(
            kxc::relay::Op::Get("shape_expr"), {data},
            kxc::relay::ShapeExprAttrs::Create(
                kxc::Array<int64_t>(std::vector<int64_t>(first_kinds)),
                kxc::Array<int64_t>({4, 0, 0}),
                kxc::Array<int64_t>({0, 1, 2})));
        const kxc::Expr reshaped = kxc::Call(
            kxc::relay::Op::Get("reshape_dynamic"), {data, control},
            kxc::relay::ReshapeDynamicAttrs::Create(
                kxc::Array<int64_t>(std::vector<int64_t>(first_kinds)),
                kxc::Array<int64_t>({4, 0, 0}),
                kxc::Array<int64_t>({0, 1, 2})));
        const std::string renamed =
            key_of(kxc::Function({data}, reshaped));
        CHECK(first == renamed,
              "same structure with different variable names keeps identity");
    }
    // 不同目标表达式 → 不同身份：[axis0, axis1, const 3] 同样求值到
    // [4,5,3]，但表达式结构不同，canonical bytes 必须不同。
    const std::string second = key_of(reshape_graph(
        {kxc::relay::kShapeExprKindInputAxis,
         kxc::relay::kShapeExprKindInputAxis,
         kxc::relay::kShapeExprKindConst},
        {0, 0, 3}, {0, 1, 0}));
    CHECK(first != second,
          "different restricted target expressions must not share identity");
    return true;
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"shape_of_infer_type", TestShapeOfInferType},
        {"shape_of_primitive_tir", TestShapeOfPrimitiveTIR},
        {"static_shape_chain_production", TestStaticShapeChainProduction},
        {"bounded_shape_value_materialization",
         TestBoundedShapeValueMaterialization},
        {"bounded_shape_chain_failures", TestBoundedShapeChainFailures},
        {"shape_expr_infer_type", TestShapeExprInferType},
        {"reshape_dynamic_infer_type", TestReshapeDynamicInferType},
        {"s2_static_infer_type", TestS2StaticInferType},
        {"s2_primitive_tir", TestS2PrimitiveTIR},
        {"s2_static_production", TestS2StaticProduction},
        {"bounded_shape_to_reshape_production",
         TestBoundedShapeToReshapeProduction},
        {"bounded_s2_failures", TestBoundedS2Failures},
        {"s2_identity", TestS2Identity},
    };
    for (const auto& [name, test] : tests) {
        try {
            if (!test()) return 1;
            std::cout << "[PASS] " << name << '\n';
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
            return 1;
        }
    }
    return 0;
}
