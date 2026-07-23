/*! \file test/compiler_foundation_contract_test.cpp
 * \brief Cross-track conformance tests using only frozen CoreContract v1 DTOs/fakes.
 */

#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "kxc/compiler/capability.h"
#include "kxc/compiler/foundation_contract.h"
#include "kxc/compiler/pipeline.h"
#include "support/compiler_foundation_fakes.h"

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

kxc::api::ArtifactKey MakeKey(const std::string& unit) {
    return kxc::api::ArtifactKey(
        kxc::api::UnitSemanticKey("unit=" + unit), "target=cpu-v1",
        "pipeline=frozen-v1", 1, "schedule=v1", "backend=fake-v1");
}

kxc::api::ArtifactHandle MakeArtifact(const kxc::api::ArtifactKey& key) {
    return kxc::api::ArtifactHandle(kxc::api::ArtifactRecord{
        key, "fake-executable", "signature-v1", "launch-v1", "fake-test",
        64, "validated"});
}

kxc::api::CompileRequest MakeRequest(const kxc::api::ArtifactKey& key,
                                     const std::string& cancellation_id) {
    kxc::api::CompileRequest request;
    request.artifact_key = key;
    request.priority = kxc::api::CompilePriority::kNormal;
    request.budget_class = kxc::api::CompileBudgetClass::kGlobal;
    request.request_origin = "core-contract-test";
    request.cancellation.id = cancellation_id;
    return request;
}

bool TestImmutableHandleAndEvictionPin() {
    using namespace kxc::api;
    using namespace kxc::api::testing;
    const ArtifactKey key = MakeKey("pin");
    ArtifactRecord mutable_record{
        key, "original", "signature-v1", "launch-v1", "fake", 32,
        "validated"};
    const ArtifactHandle handle(mutable_record);
    mutable_record.executable_token = "mutated";
    TEST_CHECK(handle.record().executable_token == "original",
               "ArtifactHandle must own an immutable value copy");

    FakeArtifactStore store;
    store.Store(handle);
    ArtifactLookup lookup = store.Lookup(key);
    TEST_CHECK(lookup.kind == ArtifactLookupKind::kHit &&
                   lookup.pin.defined(),
               "ready lookup must return a strong pin");
    store.Evict(key);
    TEST_CHECK(store.Lookup(key).kind == ArtifactLookupKind::kMiss &&
                   lookup.pin.handle().record().executable_token == "original" &&
                   store.events().back().kind == CacheEventKind::kMiss,
               "fake eviction removes discovery but cannot invalidate a pin");
    return true;
}

bool TestCompileRequestSingleflightAndCancellation() {
    using namespace kxc::api;
    using namespace kxc::api::testing;
    FakeArtifactStore store;
    FakeCompileCoordinator coordinator(2);
    const ArtifactKey key = MakeKey("singleflight");
    CompileRequest first_request = MakeRequest(key, "caller-a");
    CompileRequest second_request = MakeRequest(key, "caller-b");
    second_request.priority = CompilePriority::kUrgent;
    second_request.request_origin = "another-origin";
    TEST_CHECK(first_request.CanonicalSingleflightKey() ==
                   second_request.CanonicalSingleflightKey(),
               "priority/origin/caller cancellation id must not split full-key work");
    const auto first = coordinator.Request(first_request, &store);
    const auto second = coordinator.Request(second_request, &store);
    TEST_CHECK(first.get() == second.get() &&
                   first->merged_waiter_count == 1 &&
                   first->state == CompileRequestState::kQueued,
               "same artifact+dispatch request must share one ticket");
    coordinator.Begin(first);
    const CompileOutcome cancelled = coordinator.CancelWaiter(second);
    TEST_CHECK(cancelled.state == CompileRequestState::kCancelled &&
                   first->state == CompileRequestState::kCompiling,
               "cancelling one waiter must not cancel shared owner work");
    coordinator.Validate(first);
    const ArtifactHandle handle = MakeArtifact(key);
    coordinator.Ready(first, ArtifactPin(handle));
    TEST_CHECK(first->state == CompileRequestState::kReady &&
                   first->outcome->pin.defined(),
               "owner publishes one immutable ready result");
    return true;
}

bool TestFailureRetryAndBackpressureContracts() {
    using namespace kxc::api;
    using namespace kxc::api::testing;
    FakeArtifactStore store;
    FakeCompileCoordinator coordinator(1);
    const CompileRequest first_request =
        MakeRequest(MakeKey("failure"), "failure-a");
    const CompileRequest second_request =
        MakeRequest(MakeKey("saturated"), "saturated-a");
    const auto first = coordinator.Request(first_request, &store);
    const auto rejected = coordinator.Request(second_request, &store);
    TEST_CHECK(rejected->state == CompileRequestState::kRejected &&
                   rejected->outcome->failure->category ==
                       CompileFailureCategory::kBackpressure,
               "bounded saturation must return structured backpressure");
    coordinator.Begin(first);
    coordinator.Fail(first, CompileFailure{
                                CompileFailureCategory::kValidation, 100,
                                "signature mismatch"});
    const auto negative = coordinator.Request(first_request, &store);
    TEST_CHECK(negative->state == CompileRequestState::kFailed &&
                   negative->outcome->failure->retry_after_millis == 100,
               "negative result must retain a deterministic retry deadline");
    coordinator.Advance(100);
    const auto retry = coordinator.Request(first_request, &store);
    TEST_CHECK(retry->state == CompileRequestState::kQueued,
               "request may retry only after the failure deadline");
    return true;
}

bool TestExplicitCapabilityAndPipelineFakes() {
    using namespace kxc;
    using namespace kxc::api;
    using namespace kxc::api::testing;
    CapabilityRequest capability;
    capability.graph_locator = "graph/fake";
    capability.boundary = CapabilityBoundary::kPrePartition;
    capability.requested_mode = CapabilityMode::kStaticExact;
    FakeCapabilityVerifier verifier;
    TEST_CHECK(!verifier.Verify(capability).supported,
               "fake capability must fail closed without an explicit result");
    CapabilityResult accepted;
    accepted.supported = true;
    verifier.Set(capability, accepted);
    TEST_CHECK(verifier.Verify(capability).supported,
               "fake capability uses an explicit request key, not op-name guesses");

    auto* target_node = new TargetNode();
    target_node->kind = "llvm";
    target_node->device_type = kCPU;
    target_node->device_id = 0;
    PipelineRequest request;
    request.dialect = IRDialect::kRelay;
    request.requested_scope = PassScope::kGraph;
    request.target = Target(ObjectRef(target_node));
    request.opt_level = 2;
    request.named_pipeline = String("compiler");
    NormalizedPipeline normalized;
    normalized.dialect = IRDialect::kRelay;
    normalized.scope = PassScope::kGraph;
    normalized.ordered_passes = {String("fold_constant")};
    normalized.canonical_bytes = String("fake-normalized-pipeline");
    normalized.fingerprint = String("fake-fingerprint");
    FakePipelineResolver resolver;
    resolver.Set(request, normalized);
    TEST_CHECK(resolver.Resolve(request).fingerprint ==
                   String("fake-fingerprint"),
               "cross-track tests can inject a deterministic normalized pipeline");
    return true;
}

bool TestFrozenGenerationZeroPlanAndSentinelRejection() {
    using namespace kxc::api;
    using namespace kxc::api::testing;
    const ArtifactHandle handle = MakeArtifact(MakeKey("plan"));
    const SelectedArtifact selected{
        ArtifactPin(handle), 0, "static-exact", "signature-v1"};
    const FakePlanAssembler assembler;
    const FrozenPlanInput plan = assembler.Assemble(
        "graph-template-v1", {GraphValueLocator{"graph-v1", 0, 0}},
        {selected}, "logical=[4];physical=[4];valid=[4]", "memory-v1");
    TEST_CHECK(plan.selected_artifacts[0].generation == 0,
               "static exact M1 handoff uses immutable generation zero");
    TEST_CHECK(Throws([&] {
                   (void)assembler.Assemble(
                       "graph-template-v1",
                       {GraphValueLocator{"graph-v1", 0, 0}}, {selected},
                       "logical=[-1];physical=[4];valid=[4]", "memory-v1");
               }) &&
                   Throws([] {
                       (void)DispatchKey("family", "shape=[-1]", "exact-v1");
                   }),
               "legacy -1 sentinel cannot enter frozen plan or dispatch identity");
    return true;
}

}  // namespace

int main() {
    static_assert(kxc::api::kCompilerFoundationContractVersion == 1,
                  "test consumes CoreContract v1");
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"immutable_pin_eviction", TestImmutableHandleAndEvictionPin},
        {"singleflight_and_cancel", TestCompileRequestSingleflightAndCancellation},
        {"failure_retry_backpressure", TestFailureRetryAndBackpressureContracts},
        {"explicit_capability_pipeline_fakes",
         TestExplicitCapabilityAndPipelineFakes},
        {"generation_zero_and_no_sentinel",
         TestFrozenGenerationZeroPlanAndSentinelRejection},
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
