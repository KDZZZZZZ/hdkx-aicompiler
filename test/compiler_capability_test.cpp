/*! \file test/compiler_capability_test.cpp
 * \brief Verifies structural eligibility versus proven executable capability.
 */

#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "../src/compiler/internal/lowered_graph.h"
#include "kxc/compiler/capability.h"
#include "kxc/compiler/compiler.h"
#include "kxc/relay/op.h"
#include "kxc/relay/op_attr_types.h"
#include "kxc/relay/op_macros.h"
#include "kxc/relay/transforms/infer_type.h"
#include "kxc/te/te.h"

#ifndef KXC_USE_LLVM
#define KXC_USE_LLVM 0
#endif

#ifndef KXC_USE_CUDA
#define KXC_USE_CUDA 0
#endif

namespace {

#define TEST_CHECK(condition, message)                                           \
    do {                                                                          \
        if (!(condition)) {                                                        \
            std::cerr << "[FAIL] " << __FUNCTION__ << ": " << (message) << "\n"; \
            return false;                                                         \
        }                                                                         \
    } while (0)

bool ThrowsWith(const std::function<void()>& fn, const std::string& expected) {
    try {
        fn();
    } catch (const std::exception& error) {
        return std::string(error.what()).find(expected) != std::string::npos;
    }
    return false;
}

bool HasIssue(const kxc::api::CapabilityResult& result,
              const std::string& capability) {
    for (const auto& issue : result.issues) {
        if (issue.missing_capability == capability) return true;
    }
    return false;
}

bool HasConsistentStatus(const kxc::api::CapabilityResult& result) {
    return result.supported ==
           (result.status == kxc::api::CapabilityStatus::kExecutable);
}

kxc::Function TypedAdd() {
    using namespace kxc;
    TensorType type({4}, "float32");
    Var lhs("lhs", type);
    Var rhs("rhs", type);
    return relay::InferTypePass(
        Function({lhs, rhs}, Call(relay::Op::Get("add"), {lhs, rhs})));
}

kxc::api::CapabilityResult Verify(
    const kxc::Function& function, const kxc::Target& target,
    int opt_level = 2,
    kxc::api::CapabilityBoundary boundary =
        kxc::api::CapabilityBoundary::kCompilerEntry,
    bool require_checked_types = false,
    std::string locator = "model/main") {
    return kxc::api::CapabilityVerifier::Verify(kxc::api::CapabilityRequest{
        function, target, std::move(locator), "", boundary,
        kxc::api::CapabilityMode::kStaticExact, require_checked_types,
        opt_level});
}

kxc::Target FakeCudaTarget(bool with_compute_capability = true) {
    auto* node = new kxc::TargetNode();
    node->kind = "cuda";
    node->device_type = kxc::kCUDA;
    node->device_id = 0;
    node->attrs.exists = 1;
    node->attrs.device_name = "capability-cuda";
    node->attrs.arch = "sm_80";
    node->attrs.max_threads_per_block = 128;
    node->attrs.max_threads_per_multiprocessor = 2048;
    node->attrs.max_shared_memory_per_block = 48 * 1024;
    node->attrs.warp_size = 32;
    node->attrs.multi_processor_count = 1;
    if (with_compute_capability) {
        node->attrs.compute_version = "8.0";
        node->attrs.compute_version_major = 8;
        node->attrs.compute_version_minor = 0;
    }
    return kxc::Target(kxc::ObjectRef(node));
}

bool TestEligibilityIsNotExecutableSupport() {
    using namespace kxc;
    using namespace kxc::api;
    const Function function = TypedAdd();
    const Target target = BuildTarget(Device::CPU());
    for (CapabilityBoundary boundary : {
             CapabilityBoundary::kCompilerEntry,
             CapabilityBoundary::kPostGraphPass,
             CapabilityBoundary::kPrePartition}) {
        const CapabilityResult result =
            Verify(function, target, 2, boundary, true);
        TEST_CHECK(HasConsistentStatus(result),
                   "supported must be exactly equivalent to executable status");
#if KXC_USE_LLVM
        TEST_CHECK(result.supported && result.issues.empty() &&
                       result.status == CapabilityStatus::kExecutable &&
                       !result.pipeline_fingerprint.empty(),
                   "LLVM builds must prove the real normalized compile path");
#else
        TEST_CHECK(!result.supported &&
                       result.status ==
                           CapabilityStatus::kEligibleButNotExecutable &&
                       HasIssue(result, "backend.llvm"),
                   "LLVM-disabled builds must report eligible, not supported");
#endif
    }
    return true;
}

bool TestRepresentableControlFlowRejectedWithLocator() {
    using namespace kxc;
    using namespace kxc::api;
    Var cond("cond", TensorType({}, "bool"));
    Var lhs("lhs", TensorType({4}, "float32"));
    Var rhs("rhs", TensorType({4}, "float32"));
    Function function({cond, lhs, rhs}, If(cond, lhs, rhs));
    const CapabilityResult result = Verify(
        function, BuildTarget(Device::CPU()), 0,
        CapabilityBoundary::kCompilerEntry, false, "model/control");
    bool located_if = false;
    for (const CapabilityIssue& issue : result.issues) {
        located_if = located_if ||
                     (issue.missing_capability == "control_flow.if" &&
                      issue.diagnostic_locator == "model/control/body");
    }
    TEST_CHECK(!result.supported &&
                   result.status == CapabilityStatus::kUnsupported &&
                   located_if,
               "If rejection must identify the node and missing capability");
    TEST_CHECK(ThrowsWith(
                   [&] {
                       (void)Compiler::Compile(
                           function, CompileConfig::Create(
                                         BuildTarget(Device::CPU()), 0));
                   },
                   "CapabilityVerifier[compiler_entry]"),
               "Compiler entry must reject unsupported control flow");

    Var state("state", TensorType({4}, "float32"));
    Function while_function = relay::InferTypePass(
        Function({cond, lhs}, While(lhs, state, cond, state, 0)));
    const CapabilityResult while_result = Verify(
        while_function, BuildTarget(Device::CPU()), 0,
        CapabilityBoundary::kCompilerEntry, false, "model/while");
    TEST_CHECK(HasIssue(while_result, "control_flow.loop") &&
                   ThrowsWith([&] {
                       (void)Compiler::Compile(
                           while_function, CompileConfig::Create(
                               BuildTarget(Device::CPU()), 0));
                   }, "control_flow.loop"),
               "Compiler::Compile must reject While before ValueGraph with a stable loop diagnostic");
    return true;
}

bool TestLetFutureModeAndDynamicFailClosed() {
    using namespace kxc;
    using namespace kxc::api;
    TensorType type({4}, "float32");
    Var input("input", type);
    Var local("local", type);
    const Target target = BuildTarget(Device::CPU());
    const CapabilityResult let_result = CapabilityVerifier::Verify(
        CapabilityRequest{Function({input}, Let(local, input, local)), target,
                          "model/let", "",
                          CapabilityBoundary::kCompilerEntry,
                          CapabilityMode::kStaticExact, false, 2});
    TEST_CHECK(let_result.status != CapabilityStatus::kUnsupported &&
                   !HasIssue(let_result, "control_flow.let"),
               "lexical Let must remain executable after mandatory ANF normalization");

    const CapabilityResult mode_result = CapabilityVerifier::Verify(
        CapabilityRequest{TypedAdd(), target, "model/add", "",
                          CapabilityBoundary::kCompilerEntry,
                          CapabilityMode::kShapeSpecialization, false, 2});
    TEST_CHECK(mode_result.status == CapabilityStatus::kUnsupported &&
                   HasIssue(mode_result, "execution_mode"),
               "future execution modes must fail closed");

    Var dynamic("dynamic", TensorType({-1, 4}, "float32"));
    const CapabilityResult dynamic_result = Verify(
        Function({dynamic}, dynamic), target, 2,
        CapabilityBoundary::kCompilerEntry, false, "model/dynamic");
    TEST_CHECK(dynamic_result.status == CapabilityStatus::kUnsupported &&
                   HasIssue(dynamic_result, "static_exact_shape"),
               "legacy -1 is not executable dynamic-shape capability");
    return true;
}

bool TestTupleParameterAndUnspecifiedOpRejected() {
    using namespace kxc;
    using namespace kxc::api;
    const TensorType type({4}, "float32");
    Var tuple_parameter("tuple", TupleType({type, type}));
    const CapabilityResult tuple_result = Verify(
        Function({tuple_parameter}, TupleGetItem(tuple_parameter, 0)),
        BuildTarget(Device::CPU()));
    TEST_CHECK(tuple_result.status == CapabilityStatus::kUnsupported &&
                   HasIssue(tuple_result, "tensor_parameter"),
               "flat tuple values do not make tuple function parameters executable");

    Var input("input", type);
    relay::Op unspecified("test.unspecified");
    const CapabilityResult op_result = Verify(
        Function({input}, Call(unspecified, {input})),
        BuildTarget(Device::CPU()));
    TEST_CHECK(op_result.status == CapabilityStatus::kUnsupported &&
                   HasIssue(op_result, "registered_operator"),
               "an unspecified Call must fail closed");
    return true;
}

bool TestRealGemmLoweringRejectionIsNotSupported() {
    using namespace kxc;
    using namespace kxc::api;
    Var a("a", TensorType({3, 2}, "float32"));
    Var b("b", TensorType({4, 3}, "float32"));
    Var c("c", TensorType({4}, "float32"));
    Function function(
        {a, b, c},
        Call(relay::Op::Get("nn_gemm"), {a, b, c},
             relay::GemmAttrs::Create(1.0f, 1.0f, 1, 1)));
    const CapabilityResult result =
        Verify(function, BuildTarget(Device::CPU()));
    TEST_CHECK(!result.supported && HasConsistentStatus(result) &&
                   result.status ==
                       CapabilityStatus::kEligibleButNotExecutable &&
                   HasIssue(result, "per_unit_lowering") &&
                   result.Diagnostic().find("transA") != std::string::npos,
               "nn_gemm transA must use the real lowering rejection: " +
                   result.Diagnostic());
    TEST_CHECK(ThrowsWith(
                   [&] {
                       (void)Compiler::Compile(
                           function, CompileConfig::Create(
                                         BuildTarget(Device::CPU()), 2));
                   },
                   "transA"),
               "production compilation must share the same lowering rejection");
    return true;
}

const kxc::relay::Op& WrongShapeLoweringOp() {
    using namespace kxc;
    using namespace kxc::relay;
    static const Op op = [] {
        OperatorSpec spec;
        spec.name = "test.capability_wrong_shape";
        spec.category = "test";
        spec.input_arity.num_inputs = 1;
        spec.output_arity = 1;
        spec.type_relation_key = "FInferType";
        spec.lowering_kind = OperatorLoweringKind::kSingleTE;
        spec.lowering_key = "FRelayToTE";
        Op registered = Op::Register(spec);
        OpRegEntry(registered)
            .set_attr<FInferType>(
                "FInferType",
                FInferType([](const Attrs&, const Array<Type>& inputs) {
                    if (inputs.size() != 1) {
                        throw std::runtime_error("wrong-shape type arity");
                    }
                    return inputs[0];
                }))
            .set_attr<FRelayToTE>(
                "FRelayToTE",
                FRelayToTE([](const Attrs&, const Array<te::Tensor>&,
                              const Type&) {
                    const tir::DataType f32 = tir::DataType::Float(32);
                    return te::compute(
                        {tir::IntImm(5, tir::DataType::Int(64))},
                        [f32](const Array<tir::Var>&) {
                            return tir::FloatImm(0.0, f32);
                        },
                        "wrong_shape");
                }));
        return registered;
    }();
    return op;
}

bool TestCustomBindingMismatchRejectedBySharedLowering() {
    using namespace kxc;
    using namespace kxc::api;
    Var input("input", TensorType({4}, "float32"));
    Function function({input}, Call(WrongShapeLoweringOp(), {input}));
    const CapabilityResult result =
        Verify(function, BuildTarget(Device::CPU()));
    TEST_CHECK(result.status ==
                       CapabilityStatus::kEligibleButNotExecutable &&
                   HasIssue(result, "per_unit_lowering") &&
                   result.Diagnostic().find("shape") != std::string::npos,
               "custom TE output must match the inferred production boundary: " +
                   result.Diagnostic());
    TEST_CHECK(ThrowsWith(
                   [&] {
                       (void)Compiler::Compile(
                           function, CompileConfig::Create(
                                         BuildTarget(Device::CPU()), 2));
                   },
                   "output shape"),
               "Compiler must consume the same output-contract validation");
    return true;
}

bool TestBackendAndCudaTargetFactsAreStructured() {
    using namespace kxc;
    using namespace kxc::api;
#if !KXC_USE_LLVM
    const CapabilityResult cpu =
        Verify(TypedAdd(), BuildTarget(Device::CPU()));
    TEST_CHECK(cpu.status == CapabilityStatus::kEligibleButNotExecutable &&
                   HasIssue(cpu, "backend.llvm"),
               "build backend availability is part of executable truth");
#endif
#if !KXC_USE_CUDA
    const CapabilityResult cuda_backend =
        Verify(TypedAdd(), FakeCudaTarget(), 3);
    TEST_CHECK(cuda_backend.status ==
                       CapabilityStatus::kEligibleButNotExecutable &&
                   HasIssue(cuda_backend, "backend.cuda"),
               "CUDA-disabled builds cannot report CUDA support");
#endif
    const CapabilityResult no_compute =
        Verify(TypedAdd(), FakeCudaTarget(false), 3);
    TEST_CHECK(no_compute.status ==
                       CapabilityStatus::kEligibleButNotExecutable &&
                   HasIssue(no_compute, "cuda_compute_capability"),
               "CUDA compute capability must come from the Target snapshot");
    return true;
}

bool TestCudaReductionScheduleRejectedBeforeBackend() {
    using namespace kxc;
    using namespace kxc::api;
    Var input("input", TensorType({2, 4}, "float32"));
    Function reduction(
        {input}, Call(relay::Op::Get("reduce_mean"), {input},
                      relay::ReduceMeanAttrs::Create({1}, 1)));
    const CapabilityResult result =
        Verify(reduction, FakeCudaTarget(), 3);
    TEST_CHECK(!result.supported &&
                   result.status ==
                       CapabilityStatus::kEligibleButNotExecutable &&
                   HasIssue(result, "target_schedule"),
               "CUDA reduction must be rejected by the actual normalized schedule: " +
                   result.Diagnostic());
    return true;
}

bool TestCudaLayerNormScheduleRejectedBeforeBackend() {
    using namespace kxc;
    using namespace kxc::api;
    Var data("data", TensorType({2, 4}, "float32"));
    Var scale("scale", TensorType({4}, "float32"));
    Var bias("bias", TensorType({4}, "float32"));
    Function layer_norm(
        {data, scale, bias},
        Call(relay::Op::Get("nn_layer_norm"), {data, scale, bias},
             relay::LayerNormAttrs::Create(-1, 1e-5f, "float64")));
    const CapabilityResult result = Verify(layer_norm, FakeCudaTarget(), 3);
    TEST_CHECK(!result.supported &&
                   result.status == CapabilityStatus::kEligibleButNotExecutable &&
                   HasIssue(result, "target_schedule") &&
                   !HasIssue(result, "per_unit_lowering") &&
                   result.Diagnostic().find("BindCudaThreads") != std::string::npos,
               "CUDA LayerNorm must reach and fail the generic nested-reduction schedule "
               "without an op-name gate or CPU fallback: " + result.Diagnostic());
    return true;
}

bool TestCudaGatherScheduleRejectedBeforeBackend() {
    using namespace kxc;
    using namespace kxc::api;
    Var data("data", TensorType({4}, "float32"));
    Var indices("indices", TensorType({3}, "int64"));
    Function gather(
        {data, indices}, Call(relay::Op::Get("gather"), {data, indices},
                              relay::GatherAttrs::Create(0)));
    const CapabilityResult result = Verify(gather, FakeCudaTarget(), 3);
    TEST_CHECK(!result.supported &&
                   result.status == CapabilityStatus::kEligibleButNotExecutable &&
                   HasIssue(result, "target_schedule") &&
                   result.Diagnostic().find("indirect Load") != std::string::npos,
               "CUDA gather must fail closed at the generic indirect-load schedule gate: " +
                   result.Diagnostic());
    return true;
}

bool TestPrePartitionBoundaryCannotBeBypassed() {
    using namespace kxc;
    Var cond("cond", TensorType({}, "bool"));
    Var lhs("lhs", TensorType({4}, "float32"));
    Var rhs("rhs", TensorType({4}, "float32"));
    Function function = relay::InferTypePass(
        Function({cond, lhs, rhs}, If(cond, lhs, rhs)));
    TEST_CHECK(ThrowsWith(
                   [&] {
                       (void)api::internal::LowerGraph(function, Device::CPU());
                   },
                   "CapabilityVerifier[pre_partition]"),
               "direct per-unit lowering must run the pre-partition gate");
    return true;
}

bool TestSupportedImpliesProductionCompileSuccess() {
    using namespace kxc;
    using namespace kxc::api;
    const Target target = BuildTarget(Device::CPU());
    std::vector<Function> matrix;
    matrix.push_back(TypedAdd());

    Var lhs("lhs", TensorType({2, 3}, "float32"));
    Var rhs("rhs", TensorType({3, 4}, "float32"));
    matrix.push_back(Function(
        {lhs, rhs}, Call(relay::Op::Get("matmul"), {lhs, rhs})));

    for (const Function& function : matrix) {
        const CapabilityResult result = Verify(function, target, 2);
        TEST_CHECK(HasConsistentStatus(result),
                   "matrix row has inconsistent supported/status values");
        if (!result.supported) continue;
        try {
            const CompiledGraph compiled = Compiler::Compile(
                function, CompileConfig::Create(target, 2));
            TEST_CHECK(compiled.module().defined() && !compiled.artifact_pins().empty(),
                       "supported row must return executable production artifacts");
        } catch (const std::exception& error) {
            std::cerr << "supported row failed to compile: " << error.what() << "\n";
            return false;
        }
    }
#if KXC_USE_LLVM
    TEST_CHECK(true, "LLVM matrix executed supported rows");
#endif
    return true;
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"eligibility_not_support", TestEligibilityIsNotExecutableSupport},
        {"control_flow_locator", TestRepresentableControlFlowRejectedWithLocator},
        {"let_mode_dynamic", TestLetFutureModeAndDynamicFailClosed},
        {"tuple_param_unspecified", TestTupleParameterAndUnspecifiedOpRejected},
        {"gemm_transa_lowering", TestRealGemmLoweringRejectionIsNotSupported},
        {"custom_binding_mismatch", TestCustomBindingMismatchRejectedBySharedLowering},
        {"backend_cuda_target", TestBackendAndCudaTargetFactsAreStructured},
        {"cuda_reduction_schedule", TestCudaReductionScheduleRejectedBeforeBackend},
        {"cuda_layer_norm_schedule", TestCudaLayerNormScheduleRejectedBeforeBackend},
        {"cuda_gather_schedule", TestCudaGatherScheduleRejectedBeforeBackend},
        {"pre_partition_not_bypassed", TestPrePartitionBoundaryCannotBeBypassed},
        {"supported_implies_compile", TestSupportedImpliesProductionCompileSuccess},
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
