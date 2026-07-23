/*! \file test/pipeline_resolver_test.cpp
 * \brief Verifies normalized production pass policy and fingerprints.
 */

#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "kxc/compiler/compiler.h"
#include "kxc/compiler/pipeline.h"

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
    node->attrs.max_threads_per_block = 1024;
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

bool TestCompilerPoliciesAreNormalizedOnce() {
    using namespace kxc;
    using namespace kxc::api;
    const Target cpu = BuildTarget(Device::CPU());
    const std::vector<size_t> relay_sizes = {0, 3, 6, 6};
    const std::vector<size_t> tir_sizes = {0, 2, 4, 8};
    for (int level = 0; level <= 3; ++level) {
        PipelineRequest relay_request = Request(IRDialect::kRelay, level, cpu);
        relay_request.initial_invariants = {String("checked_type")};
        const NormalizedPipeline first =
            PipelineResolver::Resolve(relay_request);
        const NormalizedPipeline second =
            PipelineResolver::Resolve(relay_request);
        const NormalizedPipeline tir = PipelineResolver::Resolve(
            Request(IRDialect::kTIR, level, cpu));
        TEST_CHECK(first.defined() && tir.defined() &&
                       first.ordered_passes.size() == relay_sizes[level] &&
                       tir.ordered_passes.size() == tir_sizes[level] &&
                       first.fingerprint == second.fingerprint &&
                       first.canonical_bytes == second.canonical_bytes,
                   "same request must produce deterministic current policy");
        TEST_CHECK(Compiler::RelayPassPolicy(level).size() ==
                           relay_sizes[level] &&
                       Compiler::TIRPassPolicy(level, cpu).size() ==
                           tir_sizes[level],
                   "compatibility policy helpers must delegate to the resolver");
    }
    return true;
}

bool TestTargetSpecificScheduleIsInProductionPipeline() {
    using namespace kxc;
    using namespace kxc::api;
    const Target cuda = FakeCudaTarget();
    const NormalizedPipeline pipeline = PipelineResolver::Resolve(
        Request(IRDialect::kTIR, 3, cuda));
    TEST_CHECK(pipeline.ordered_passes.size() == 5 &&
                   Contains(pipeline.ordered_passes, "bind_cuda_threads") &&
                   Contains(pipeline.target_requirements,
                            "cuda_thread_binding") &&
                   std::string(pipeline.invariant_transitions.back().phase) ==
                       "tir_schedule",
               "CUDA scheduling must be an audited normalized transition");
    TEST_CHECK(Compiler::TIRPassPolicy(3, cuda).size() == 4 &&
                   !Contains(Compiler::TIRPassPolicy(3, cuda),
                             "bind_cuda_threads"),
               "legacy policy view may omit scheduling while production cannot");
    return true;
}

bool TestEnableDisableAndPhaseConflictsFailClosed() {
    using namespace kxc;
    using namespace kxc::api;
    PipelineRequest request =
        Request(IRDialect::kRelay, 1, BuildTarget(Device::CPU()));
    request.initial_invariants = {String("checked_type")};
    request.disabled = {String("fold_constant")};
    request.enabled = {String("canonicalize_cast")};
    const NormalizedPipeline normalized = PipelineResolver::Resolve(request);
    TEST_CHECK(normalized.ordered_passes.size() == 3 &&
                   !Contains(normalized.ordered_passes, "fold_constant") &&
                   Contains(normalized.ordered_passes, "canonicalize_cast"),
               "explicit modifications must be normalized deterministically");

    request.enabled.push_back(String("fold_constant"));
    TEST_CHECK(Throws([&] { (void)PipelineResolver::Resolve(request); }),
               "one pass cannot be both enabled and disabled");

    PipelineRequest bad_scope =
        Request(IRDialect::kRelay, 1, BuildTarget(Device::CPU()));
    bad_scope.requested_scope = PassScope::kPrimFunc;
    TEST_CHECK(Throws([&] { (void)PipelineResolver::Resolve(bad_scope); }),
               "dialect/scope mismatch must fail before execution");

    PipelineRequest bad_phase = Request(IRDialect::kTIR, 3, FakeCudaTarget());
    bad_phase.enabled = {String("loop_partition")};
    TEST_CHECK(Throws([&] { (void)PipelineResolver::Resolve(bad_phase); }),
               "an optimize pass cannot be appended after schedule phase");
    return true;
}

bool TestInvariantTransitionsAndFingerprintInputs() {
    using namespace kxc;
    using namespace kxc::api;
    PipelineRequest request =
        Request(IRDialect::kRelay, 0, BuildTarget(Device::CPU()));
    request.named_pipeline = String("relay.optimize_default");
    const NormalizedPipeline pipeline = PipelineResolver::Resolve(request);
    TEST_CHECK(pipeline.ordered_passes.size() == 9 &&
                   pipeline.invariant_transitions.size() == 9 &&
                   std::string(
                       pipeline.invariant_transitions.back().pass_name) ==
                       "infer_type" &&
                   Contains(pipeline.invariant_transitions.back()
                                .invariants_after,
                            "checked_type") &&
                   pipeline.contract_versions.size() == 10,
               "required/produced invariant state and versions must be auditable");

    PipelineRequest changed = request;
    changed.disabled = {String("annotate_memory_scope")};
    const NormalizedPipeline changed_pipeline =
        PipelineResolver::Resolve(changed);
    TEST_CHECK(pipeline.fingerprint != changed_pipeline.fingerprint &&
                   pipeline.canonical_bytes !=
                       changed_pipeline.canonical_bytes,
               "normalized configuration changes must alter artifact fingerprint");

    request.initial_invariants = {String("checked_type"),
                                  String("checked_type")};
    TEST_CHECK(Throws([&] { (void)PipelineResolver::Resolve(request); }),
               "duplicate invariant state must fail closed");
    return true;
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"compiler_policy_normalization", TestCompilerPoliciesAreNormalizedOnce},
        {"target_schedule_transition",
         TestTargetSpecificScheduleIsInProductionPipeline},
        {"enable_disable_phase_conflicts",
         TestEnableDisableAndPhaseConflictsFailClosed},
        {"invariants_and_fingerprint",
         TestInvariantTransitionsAndFingerprintInputs},
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
