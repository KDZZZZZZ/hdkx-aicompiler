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
    // extent 表达式按单元自身输入轴求值：{x} 的 shape[0]/shape[1]。
    CHECK(invocation.runtime_extent_scalars()[0].expression.Evaluate({{4, 5, 3}}) == 4 &&
              invocation.runtime_extent_scalars()[1].expression.Evaluate({{4, 5, 3}}) == 5,
          "extent scalars evaluate to the consumed input axes");
    CHECK(invocation.runtime_extent_scalars()[0].expression.Evaluate({{6, 7, 3}}) == 6 &&
              invocation.runtime_extent_scalars()[1].expression.Evaluate({{6, 7, 3}}) == 7,
          "extent scalars follow the second legal shape");
    CHECK(invocation.outputs().size() == 1 &&
              invocation.outputs()[0].max_bytes == 3 * sizeof(int64_t),
          "the shape-value output stays byte-capped by its fixed length");
    CHECK(invocation.CanonicalBytes().find(
              "KXC_MODULE_INVOKE_V2") != std::string::npos,
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
    const kxc::TensorType data_type({4, 5, 3}, "float32");
    const kxc::Var x("x", data_type);
    const kxc::Expr shape_value =
        kxc::Call(kxc::relay::Op::Get("shape_of"), {x});
    const kxc::Expr gathered = kxc::Call(
        kxc::relay::Op::Get("gather"),
        {shape_value, kxc::Constant(Int64Tensor({2}, {0, 1}))},
        kxc::relay::GatherAttrs::Create(0));
    const compiler_internal::PrimitiveCacheStats before =
        compiler_internal::GetPrimitiveCacheStats();
    CHECK(Throws([&] {
              (void)restricted::RestrictedSymbolicShapeAdapter::Prepare(
                  kxc::relay::InferTypePass(kxc::Function({x}, gathered)),
                  CpuConfig(), {{0, 0, "B", 2, 8, 1}, {0, 1, "S", 1, 8, 1}});
          }),
          "bounded admission still rejects gather shape chains (fail closed; "
          "S3/B-line will admit static-shaped chain units)");
    CHECK(SameStats(before, compiler_internal::GetPrimitiveCacheStats()),
          "admission rejection must not touch the primitive cache");
    return true;
#endif
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
