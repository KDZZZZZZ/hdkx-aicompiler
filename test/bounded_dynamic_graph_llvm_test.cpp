#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "../src/compiler/internal/primitive_cache.h"
#include "../src/runtime/internal/compiled_module_node.h"
#include "../src/runtime/internal/module_invocation_contract.h"
#include "kxc/compiler/compiler.h"
#include "kxc/compiler/experimental_identity.h"
#include "kxc/compiler/restricted_symbolic_shape.h"
#include "kxc/relay/op.h"
#include "kxc/runtime/session.h"

namespace {

namespace restricted =
    kxc::api::experimental::restricted_symbolic_shape::v1;
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

kxc::Function BoundedChain() {
    const kxc::TensorType type({4, 4}, "float32");
    const kxc::Var x("x", type), y("y", type);
    const kxc::Expr add =
        kxc::Call(kxc::relay::Op::Get("add"), {x, y});
    const kxc::Expr relu =
        kxc::Call(kxc::relay::Op::Get("nn_relu"), {add});
    return kxc::Function(
        {x, y}, kxc::Call(kxc::relay::Op::Get("sqrt"), {relu}));
}

restricted::PreparedRestrictedSymbolicTemplate PrepareBoundedChain(
    const kxc::api::CompileConfig& config = CpuConfig()) {
    return restricted::RestrictedSymbolicShapeAdapter::Prepare(
        BoundedChain(), config,
        {{0, 0, "N", 2, 8, 2}, {1, 0, "N", 2, 8, 2}});
}

kxc::runtime::NDArray FloatTensor(
    std::int64_t rows, float offset, std::vector<float>* values = nullptr) {
    std::vector<float> data(static_cast<std::size_t>(rows * 4));
    for (std::size_t index = 0; index < data.size(); ++index) {
        data[index] = static_cast<float>(index) * 0.25f + offset;
    }
    kxc::runtime::NDArray array = kxc::runtime::NDArray::Empty(
        {rows, 4}, kxc::runtime::DataTypeFromString("float32"),
        kxc::Device::CPU());
    if (!data.empty()) {
        array.CopyFromBytes(data.data(), data.size() * sizeof(float));
    }
    if (values) *values = std::move(data);
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

bool CheckNumericRun(const kxc::runtime::RuntimeSession& session,
                     std::int64_t rows,
                     kxc::Storage* output_storage) {
    std::vector<float> x_values;
    std::vector<float> y_values;
    const kxc::runtime::NDArray x =
        FloatTensor(rows, -2.0f, &x_values);
    const kxc::runtime::NDArray y =
        FloatTensor(rows, 0.5f, &y_values);
    const kxc::Array<kxc::runtime::NDArray> outputs = session.Run({x, y});
    CHECK(outputs.size() == 1 && outputs[0].shape().size() == 2 &&
              outputs[0].shape()[0] == rows &&
              outputs[0].shape()[1] == 4,
          "dynamic graph returned the wrong logical output shape");
    std::vector<float> actual(x_values.size());
    outputs[0].CopyToBytes(actual.data(), actual.size() * sizeof(float));
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const float expected =
            std::sqrt(std::max(0.0f, x_values[index] + y_values[index]));
        CHECK(std::fabs(actual[index] - expected) < 1e-5f,
              "add -> relu -> sqrt numerical result differs");
    }
    *output_storage = outputs[0].storage();
    return true;
}

bool TestCompileOnceRunTwoShapes() {
#if !KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH || !KXC_USE_LLVM
    std::cerr << "bounded dynamic graph production test requires all gates and LLVM\n";
    return false;
#else
    compiler_internal::ClearPrimitiveCacheForTesting();
    const auto prepared = PrepareBoundedChain();
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
              compiled.plan().calls().size() == 3 &&
              compiled.module().entry_count() == 3 &&
              compiled.artifact_pins().size() == 3 &&
              after_compile.entries == 3 && after_compile.misses == 3,
          "CompileBounded did not publish one three-unit dynamic graph");

    std::set<std::int64_t> storage_ids;
    for (const kxc::runtime::ValueSpec& value : compiled.plan().values()) {
        storage_ids.insert(value->storage_id);
        const kxc::Array<std::int64_t> shape = value.shape();
        CHECK(shape.size() == 2 && shape[0] == -1 && shape[1] == 4,
              "dynamic plan lost wildcard/static axis separation");
    }
    CHECK(storage_ids.size() == compiled.plan().values().size() &&
              compiled.plan().graph_input_guards().size() == 2,
          "dynamic plan must use unique storage and prepared graph guards");

    const kxc::api::PlanAbiFingerprint plan_abi =
        kxc::api::BuildPlanAbiFingerprint(compiled);
    CHECK(plan_abi.defined() &&
              plan_abi.canonical_bytes().find(
                  "KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH.v1") !=
                  std::string::npos,
          "bounded gate/version is absent from plan ABI identity");

    for (const kxc::runtime::KernelCall& call : compiled.plan().calls()) {
        const kxc::codegen::KernelSignature signature =
            compiled.module().signature(call->symbol);
        std::size_t runtime_extents = 0;
        for (const kxc::codegen::KernelArgSpec& argument :
             signature.arguments()) {
            if (argument->role ==
                kxc::codegen::KernelArgRole::kRuntimeExtent) {
                ++runtime_extents;
            }
            if (argument->role == kxc::codegen::KernelArgRole::kInput ||
                argument->role == kxc::codegen::KernelArgRole::kOutput) {
                CHECK(argument.shape().size() == 2 &&
                          argument.shape()[0] ==
                              kxc::codegen::kDynamicDimension &&
                          argument.shape()[1] == 4,
                      "Const axis incorrectly entered the dynamic kernel ABI");
            }
        }
        const kxc::api::ModuleInvocationContract& invocation =
            kxc::api::internal::BorrowCompiledModuleInvocationContract(
                compiled.module(), call->symbol);
        CHECK(runtime_extents == 1 &&
                  invocation.runtime_extent_scalars().size() == 1 &&
                  invocation.outputs().size() == 1 &&
                  invocation.outputs()[0].max_bytes ==
                      8 * 4 * sizeof(float) &&
                  invocation.run_byte_budget() == 8 * 4 * sizeof(float),
              "extent ABI and invocation scalar order did not share W2 authority");
    }
    for (const kxc::api::ArtifactPin& pin : compiled.artifact_pins()) {
        const std::string identity = pin.record().artifact_key.canonical_bytes();
        CHECK(identity.find("bounded_dynamic_graph_version") !=
                  std::string::npos &&
                  identity.find("bounded-dynamic-serial-v3") !=
                  std::string::npos,
              "bounded gate/schedule version is absent from primitive identity");
    }

    const std::vector<std::string> pin_identities = [&] {
        std::vector<std::string> result;
        for (const kxc::api::ArtifactPin& pin : compiled.artifact_pins()) {
            result.push_back(pin.record().artifact_key.canonical_bytes());
        }
        return result;
    }();
    const kxc::runtime::RuntimeSession session(
        compiled.module(), compiled.plan());
    kxc::Storage first_output;
    kxc::Storage second_output;
    CHECK(CheckNumericRun(session, 2, &first_output) &&
              SameStats(after_compile,
                        compiler_internal::GetPrimitiveCacheStats()),
          "N=2 execution changed primitive cache state");
    CHECK(CheckNumericRun(session, 6, &second_output) &&
              first_output.get() != second_output.get() &&
              SameStats(after_compile,
                        compiler_internal::GetPrimitiveCacheStats()),
          "N=6 execution recompiled or reused dynamic output storage");
    for (std::size_t index = 0; index < pin_identities.size(); ++index) {
        CHECK(pin_identities[index] ==
                  compiled.artifact_pins()[index]
                      .record().artifact_key.canonical_bytes(),
              "artifact pin identity changed across runtime shapes");
    }

    const auto rejected_without_compile = [&](const auto& inputs) {
        const compiler_internal::PrimitiveCacheStats before =
            compiler_internal::GetPrimitiveCacheStats();
        const bool rejected = Throws([&] { (void)session.Run(inputs); });
        return rejected && SameStats(
                               before,
                               compiler_internal::GetPrimitiveCacheStats());
    };
    CHECK(rejected_without_compile(
              kxc::Array<kxc::runtime::NDArray>{FloatTensor(10, 0.0f),
                                                FloatTensor(10, 0.0f)}) &&
              rejected_without_compile(
              kxc::Array<kxc::runtime::NDArray>{FloatTensor(3, 0.0f),
                                                FloatTensor(3, 0.0f)}) &&
              rejected_without_compile(
              kxc::Array<kxc::runtime::NDArray>{FloatTensor(2, 0.0f),
                                                FloatTensor(4, 0.0f)}),
          "bounds, divisibility, or shared symbol escaped graph preflight");
    const kxc::runtime::NDArray wrong_rank = kxc::runtime::NDArray::Zeros(
        {2, 4, 1}, kxc::runtime::DataTypeFromString("float32"),
        kxc::Device::CPU());
    CHECK(rejected_without_compile(
              kxc::Array<kxc::runtime::NDArray>{wrong_rank,
                                                FloatTensor(2, 0.0f)}),
          "rank mismatch escaped graph preflight");
    return true;
#endif
}

bool TestUnsupportedGraphsFailBeforeBackend() {
#if !KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH || !KXC_USE_LLVM
    return false;
#else
    compiler_internal::ClearPrimitiveCacheForTesting();
    const compiler_internal::PrimitiveCacheStats before =
        compiler_internal::GetPrimitiveCacheStats();
    const kxc::TensorType type({4}, "float32");
    const kxc::Var x("x", type);
    CHECK(Throws([&] {
              (void)restricted::RestrictedSymbolicShapeAdapter::Prepare(
                  kxc::Function(
                      {x}, kxc::Call(kxc::relay::Op::Get("exp"), {x})),
                  CpuConfig(), {{0, 0, "N", 2, 8, 2}});
          }) &&
              Throws([&] {
                  (void)restricted::RestrictedSymbolicShapeAdapter::Prepare(
                      kxc::Function({x}, kxc::If(x, x, x)), CpuConfig(),
                      {{0, 0, "N", 2, 8, 2}});
              }) &&
              SameStats(before,
                        compiler_internal::GetPrimitiveCacheStats()),
          "unsupported op or control reached primitive backend/cache");
    const auto cuda = restricted::RestrictedSymbolicShapeAdapter::MintBoundedCompileRequest(
        PrepareBoundedChain(SyntheticCudaConfig()));
    CHECK(cuda.target()->device_type == kxc::kCUDA &&
          SameStats(before, compiler_internal::GetPrimitiveCacheStats()),
          "CUDA bounded admission must not compile or change targets");
    return true;
#endif
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"compile_once_run_two_shapes", TestCompileOnceRunTwoShapes},
        {"unsupported_before_backend", TestUnsupportedGraphsFailBeforeBackend},
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
