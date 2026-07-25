/*! \file test/pipeline_resolver_test.cpp
 * \brief Verifies normalized production execution plans and their executor.
 */

#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "../src/compiler/internal/execution_contract.h"
#include "kxc/compiler/pipeline.h"
#include "kxc/relay/op.h"
#include "kxc/relay/transforms/infer_type.h"
#include "kxc/tir/transforms/bind_cuda_threads.h"

namespace {

#define TEST_CHECK(condition, message)                                           \
    do {                                                                          \
        if (!(condition)) {                                                       \
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

bool Contains(const kxc::Array<kxc::String>& values, const char* expected) {
    for (const auto& value : values) {
        if (std::string(value) == expected) return true;
    }
    return false;
}

kxc::Target FakeCudaTarget() {
    auto* node = new kxc::TargetNode();
    node->kind = "cuda";
    node->device_type = kxc::kCUDA;
    node->device_id = 0;
    node->attrs.exists = 1;
    node->attrs.device_name = "contract-cuda";
    node->attrs.arch = "sm_80";
    node->attrs.max_threads_per_block = 128;
    node->attrs.max_shared_memory_per_block = 48 * 1024;
    node->attrs.warp_size = 32;
    node->attrs.multi_processor_count = 1;
    return kxc::Target(kxc::ObjectRef(node));
}

kxc::api::PipelineRequest Request(kxc::IRDialect dialect, int opt_level,
                                  kxc::Target target) {
    kxc::api::PipelineRequest request;
    request.dialect = dialect;
    request.requested_scope = dialect == kxc::IRDialect::kRelay
                                  ? kxc::PassScope::kGraph
                                  : kxc::PassScope::kPrimFunc;
    request.target = std::move(target);
    request.opt_level = opt_level;
    request.named_pipeline = kxc::String("compiler");
    return request;
}

kxc::Function MakeRelayFunction() {
    kxc::Var x("x", kxc::TensorType({4}, "float32"));
    kxc::Var y("y", kxc::TensorType({4}, "float32"));
    kxc::Call add(kxc::relay::Op::Get("add"), {x, y});
    return kxc::Function({x, y}, add);
}

kxc::tir::PrimFunc MakeElementwiseTIR() {
    using namespace kxc;
    using namespace kxc::tir;
    const DataType f32 = DataType::Float(32);
    const DataType i64 = DataType::Int(64);
    tir::Var a("a", f32);
    tir::Var b("b", f32);
    tir::Var out("out", f32);
    tir::Var i("i", i64);
    tir::Stmt body = tir::For(
        i, tir::IntImm(0, i64), tir::IntImm(65, i64), tir::ForType::Serial,
        tir::Store(out, tir::Add(tir::Load(a, i), tir::Load(b, i)), i));
    Array<tir::Var> params{a, b, out};
    Map<tir::Var, tir::Buffer> buffers;
    for (const tir::Var& param : params) {
        buffers.Set(param, tir::Buffer(param, f32, {tir::IntImm(65, i64)}, {},
                                       tir::IntImm(0, i64), param->name_hint, 4, 0));
    }
    return tir::PrimFunc(params, body, buffers, {});
}

bool TestProductionPlanHasExplicitInferBoundaries() {
    using namespace kxc;
    using namespace kxc::api;
    const Target cpu = BuildTarget(Device::CPU());
    const std::vector<size_t> relay_sizes = {4, 6, 9, 9};
    const std::vector<size_t> tir_sizes = {0, 2, 4, 8};
    for (int level = 0; level <= 3; ++level) {
        const NormalizedPipeline relay =
            PipelineResolver::Resolve(Request(IRDialect::kRelay, level, cpu));
        const NormalizedPipeline tir =
            PipelineResolver::Resolve(Request(IRDialect::kTIR, level, cpu));
        TEST_CHECK(relay.defined() && relay.execution_steps.size() == relay_sizes[level] &&
                       relay.invariant_transitions.size() == relay_sizes[level] &&
                       tir.execution_steps.size() == tir_sizes[level],
                   "compiler plans must contain exactly their executed pass steps");
        TEST_CHECK(relay.execution_steps.front().pass_name == String("infer_type") &&
                       relay.execution_steps[relay.execution_steps.size() - 2].pass_name ==
                           String("infer_type") &&
                       relay.execution_steps.back().pass_name ==
                           String("normalize_to_anf") &&
                       relay.execution_steps.front().occurrence == 0 &&
                       relay.execution_steps[relay.execution_steps.size() - 2].occurrence == 1 &&
                       relay.execution_steps.back().occurrence == 0,
                   "production Relay policy must retain pre/post InferType and finish with ANF");
    }
    const NormalizedPipeline plan =
        PipelineResolver::Resolve(Request(IRDialect::kRelay, 1, cpu));
    const Function result = PipelineExecutor::ExecuteRelay(plan, MakeRelayFunction(), cpu);
    TEST_CHECK(result->body.checked_type().defined(),
               "executor must verify the concrete InferType postcondition");
    return true;
}

bool TestTamperedAndUndeclaredStepsFailClosed() {
    using namespace kxc;
    using namespace kxc::api;
    const Target cpu = BuildTarget(Device::CPU());
    const NormalizedPipeline plan =
        PipelineResolver::Resolve(Request(IRDialect::kRelay, 1, cpu));

    NormalizedPipeline changed_transition = plan;
    changed_transition.invariant_transitions.back().produced = {};
    TEST_CHECK(Throws([&] { PipelineExecutor::Validate(changed_transition, cpu); }),
               "a missing invariant transition must be rejected");

    NormalizedPipeline extra_step = plan;
    extra_step.execution_steps.push_back(plan.execution_steps.back());
    TEST_CHECK(Throws([&] { PipelineExecutor::Validate(extra_step, cpu); }),
               "an extra execution step without a declared transition must be rejected");

    NormalizedPipeline undeclared = plan;
    undeclared.execution_steps[1].pass_name = String("not_a_declared_pass");
    TEST_CHECK(Throws([&] { PipelineExecutor::Validate(undeclared, cpu); }),
               "an undeclared pass must never enter execution");
    return true;
}

bool TestCudaScheduleIsCanonicalAndVerified() {
    using namespace kxc;
    using namespace kxc::api;
    const Target cuda = FakeCudaTarget();
    const NormalizedPipeline cuda_plan =
        PipelineResolver::Resolve(Request(IRDialect::kTIR, 3, cuda));
    const NormalizedPipeline cpu_plan = PipelineResolver::Resolve(
        Request(IRDialect::kTIR, 3, BuildTarget(Device::CPU())));
    TEST_CHECK(cuda_plan.execution_steps.size() == 5 &&
                   cuda_plan.execution_steps.back().pass_name == String("bind_cuda_threads") &&
                   Contains(cuda_plan.target_requirements, "cuda_thread_binding") &&
                   std::string(cuda_plan.execution_steps.back().phase) ==
                       "tir_schedule" &&
                   cuda_plan.fingerprint != cpu_plan.fingerprint,
               "CUDA scheduling and target requirements must be canonical execution identity");
    const tir::PrimFunc scheduled =
        PipelineExecutor::ExecuteTIR(cuda_plan, MakeElementwiseTIR(), cuda);
    const tir::CudaLaunchConfig launch = tir::GetCudaLaunchConfig(scheduled);
    TEST_CHECK(launch.grid_x == 1 && launch.block_x == 128,
               "executor must verify and retain CUDA schedule metadata");

    NormalizedPipeline target_tamper = cuda_plan;
    target_tamper.target_requirements.erase(
        target_tamper.target_requirements.begin() +
        target_tamper.target_requirements.size() - 1);
    TEST_CHECK(Throws([&] { PipelineExecutor::Validate(target_tamper, cuda); }),
               "missing CUDA target capability requirement must be rejected");
    return true;
}

bool TestExecutableInvariantValidationFailsClosed() {
    using namespace kxc;
    using namespace kxc::api;
    PassSpec fake_pass;
    fake_pass.name = String("fake_claims_checked_type");
    fake_pass.dialect = IRDialect::kRelay;
    fake_pass.scope = PassScope::kGraph;
    fake_pass.phase = String("relay_optimize");
    fake_pass.implementation_key = String("kxc.test.fake_claims_checked_type");
    fake_pass.produced_invariants = {String("checked_type")};
    PipelineInvariantValidator::ValidateProductionContract(fake_pass);
    const Function stale_claim = relay::InferTypePass(MakeRelayFunction());
    SetCheckedType(stale_claim->body, TensorType({5}, "float32"));
    TEST_CHECK(
        Throws([&] {
            PipelineInvariantValidator::ValidateRelay(
                fake_pass.produced_invariants, stale_claim);
        }),
        "a pass declaration with complete but stale types cannot prove an invariant");

    PassSpec declarative = fake_pass;
    declarative.name = String("fake_declarative_shape");
    declarative.produced_invariants = {String("shape_solved")};
    declarative.declarative_only_invariants = {String("shape_solved")};
    PipelineInvariantValidator::ValidateProductionContract(declarative);
    declarative.required_invariants = {String("shape_solved")};
    TEST_CHECK(
        Throws([&] {
            PipelineInvariantValidator::ValidateProductionContract(declarative);
        }),
        "declarative-only metadata cannot become a production precondition");

    PipelineRequest unsupported = Request(
        IRDialect::kRelay, 0, BuildTarget(Device::CPU()));
    unsupported.initial_invariants = {String("declarative_shape_solved")};
    TEST_CHECK(
        Throws([&] { (void)PipelineResolver::Resolve(unsupported); }),
        "an invariant without an executable validator cannot be a production precondition");
    return true;
}

bool TestCompilerArtifactIdentityUsesExecutedCanonicalPlan() {
    using namespace kxc;
    using namespace kxc::api;
    const CompileConfig config =
        CompileConfig::Create(BuildTarget(Device::CPU()), 2);
    const internal::CompilerExecutionContract contract =
        internal::ResolveCompilerExecutionContract(config);
    const std::string relay_bytes = contract.relay_pipeline.canonical_bytes;
    const std::string tir_bytes = contract.tir_pipeline.canonical_bytes;
    TEST_CHECK(!contract.canonical_bytes.empty() &&
                   !contract.fingerprint.empty() &&
                   contract.canonical_bytes.find(relay_bytes) !=
                       std::string::npos &&
                   contract.canonical_bytes.find(tir_bytes) !=
                       std::string::npos &&
                   contract.canonical_bytes.find(
                       "per-unit-boundary-lowering-v1") !=
                       std::string::npos &&
                   contract.relay_pipeline.execution_steps.front().pass_name ==
                       String("infer_type") &&
                   contract.relay_pipeline.execution_steps[
                       contract.relay_pipeline.execution_steps.size() - 2].pass_name ==
                       String("infer_type") &&
                   contract.relay_pipeline.execution_steps.back().pass_name ==
                       String("normalize_to_anf"),
               "artifact identity must use the exact executed Relay/lowering/TIR plan");
    return true;
}

bool TestCanonicalChangesAndNoHiddenCompatibilityPass() {
    using namespace kxc;
    using namespace kxc::api;
    const Target cpu = BuildTarget(Device::CPU());
    PipelineRequest request = Request(IRDialect::kRelay, 1, cpu);
    const NormalizedPipeline baseline = PipelineResolver::Resolve(request);
    request.disabled = {String("simplify_expr")};
    const NormalizedPipeline changed = PipelineResolver::Resolve(request);
    TEST_CHECK(baseline.fingerprint != changed.fingerprint &&
                   baseline.canonical_bytes != changed.canonical_bytes &&
                   changed.execution_steps.size() == 5 &&
                   changed.execution_steps.front().pass_name == String("infer_type") &&
                   changed.execution_steps[changed.execution_steps.size() - 2].pass_name ==
                       String("infer_type") &&
                   changed.execution_steps.back().pass_name ==
                       String("normalize_to_anf"),
               "the exact ordered execution steps must determine canonical identity");

    PipelineRequest forbidden = Request(IRDialect::kRelay, 0, cpu);
    forbidden.disabled = {String("infer_type")};
    TEST_CHECK(Throws([&] { (void)PipelineResolver::Resolve(forbidden); }),
               "mandatory compiler InferType cannot be hidden by a compatibility view");
    forbidden.disabled = {String("normalize_to_anf")};
    TEST_CHECK(Throws([&] { (void)PipelineResolver::Resolve(forbidden); }),
               "mandatory compiler ANF normalization cannot be disabled");
    forbidden.disabled = {String("fold_constant")};
    TEST_CHECK(Throws([&] { (void)PipelineResolver::Resolve(forbidden); }),
               "mandatory compiler control simplification cannot be disabled");

    PipelineRequest control = Request(IRDialect::kRelay, 2, cpu);
    control.required_control_capabilities = {
        String("conditional_branch")};
    const NormalizedPipeline control_safe =
        PipelineResolver::Resolve(control);
    TEST_CHECK(
        control_safe.required_control_capabilities.size() == 1 &&
            Contains(control_safe.required_control_capabilities,
                     "conditional_branch"),
        "the normalized pipeline must retain its structural control requirements");
    control.enabled = {String("eliminate_common_subexpr")};
    TEST_CHECK(
        Throws([&] { (void)PipelineResolver::Resolve(control); }),
        "a graph pass without an explicit control-safety declaration must fail closed");
    return true;
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"explicit_infer_boundaries", TestProductionPlanHasExplicitInferBoundaries},
        {"tamper_and_undeclared_rejection", TestTamperedAndUndeclaredStepsFailClosed},
        {"cuda_schedule_execution_identity", TestCudaScheduleIsCanonicalAndVerified},
        {"executable_invariant_fail_closed",
         TestExecutableInvariantValidationFailsClosed},
        {"compiler_execution_artifact_identity",
         TestCompilerArtifactIdentityUsesExecutedCanonicalPlan},
        {"canonical_change_no_hidden_pass", TestCanonicalChangesAndNoHiddenCompatibilityPass},
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
