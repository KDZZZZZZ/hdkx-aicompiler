/*! \file test/adaptive_preparation_v2_test.cpp
 * \brief CPU tests for adaptive preparation and primitive replacement.
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "kxc/compiler/adaptive_production_experimental.h"
#if KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2
#include "kxc/compiler/adaptive_hot_swap_v2.h"
#endif
#include "kxc/relay/op.h"
#include "../src/compiler/internal/compiled_graph_access.h"
#include "../src/compiler/internal/primitive_cache.h"
#include "../src/runtime/internal/compiled_module_node.h"

namespace {

#define TEST_CHECK(condition, message)                                           \
    do {                                                                          \
        if (!(condition)) {                                                       \
            std::cerr << "[FAIL] " << __FUNCTION__ << ": " << (message) << '\n'; \
            return false;                                                         \
        }                                                                         \
    } while (0)

namespace production_path =
    kxc::api::adaptive::experimental::production_path;
using ProductionRequest = production_path::ProductionCompileRequest;
using ExecutionRequest = production_path::ProductionExecutionRequest;

bool Throws(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

DLDataType Float32() {
    return kxc::runtime::DataTypeFromString("float32");
}

kxc::runtime::NDArray Tensor(const std::vector<float>& values) {
    kxc::runtime::NDArray tensor = kxc::runtime::NDArray::Empty(
        {static_cast<int64_t>(values.size())}, Float32(), kxc::Device::CPU());
    tensor.CopyFromBytes(values.data(), values.size() * sizeof(float));
    return tensor;
}

bool TensorEquals(const kxc::runtime::NDArray& tensor,
                  const std::vector<float>& expected) {
    std::vector<float> actual(expected.size());
    tensor.CopyToBytes(actual.data(), actual.size() * sizeof(float));
    return actual == expected;
}

class FixtureLauncher final : public kxc::codegen::KernelLauncher {
public:
    bool IsReady() const noexcept override { return true; }

    kxc::AsyncOperation Launch(
        const kxc::Array<kxc::runtime::NDArray>& arguments,
        const kxc::DeviceStream& stream,
        const kxc::ObjectRef&) const override {
        kxc::Array<kxc::Storage> retained;
        for (const auto& argument : arguments) retained.push_back(argument.storage());
        return kxc::AsyncOperation::Completed(stream, std::move(retained));
    }
};

std::string Symbol(size_t index) {
    return "adaptive_fixture_" + std::to_string(index);
}

kxc::runtime::ExecutablePlan MakePlan(size_t call_count = 1) {
    using namespace kxc;
    Array<runtime::ValueSpec> values;
    values.push_back(runtime::ValueSpec(0, 0, {2}, Float32(), Device::CPU(), true));
    Array<runtime::KernelCall> calls;
    for (size_t index = 0; index < call_count; ++index) {
        const int64_t output = static_cast<int64_t>(index + 1);
        values.push_back(runtime::ValueSpec(
            output, output, {2}, Float32(), Device::CPU(), false, false,
            index + 1 == call_count));
        calls.push_back(runtime::KernelCall(
            Symbol(index), {static_cast<int64_t>(index)}, {output}));
    }
    return runtime::ExecutablePlan(
        std::move(values), std::move(calls), {0}, {},
        {static_cast<int64_t>(call_count)});
}

kxc::Function MakeFunction() {
    using namespace kxc;
    Var input("input", TensorType({2}, "float32"));
    return Function(
        {input}, Call(relay::Op::Get("nn_relu"), {input},
                      relay::ReluAttrs::Create()));
}

kxc::Function MakeTwoPrimitiveFunction() {
    using namespace kxc;
    Var lhs("lhs", TensorType({2}, "float32"));
    Var rhs("rhs", TensorType({2}, "float32"));
    Call add(relay::Op::Get("add"), {lhs, rhs});
    return Function({lhs, rhs}, Call(relay::Op::Get("mul"), {add, rhs}));
}

kxc::api::PrimitiveArtifactKey MakePrimitiveKey(
    const kxc::api::CompileConfig& config, size_t index) {
    return kxc::api::internal::BuildPrimitiveArtifactKey(
        kxc::api::UnitSemanticKey(
            "adaptive-fixture-primitive-v1:" + std::to_string(index)),
        config->target,
        "adaptive-fixture-pipeline-v1:opt=" +
            std::to_string(config->opt_level),
        "adaptive-fixture-schedule-v1", "adaptive-fixture-backend-v1");
}

kxc::api::internal::PrimitiveArtifactPin PinPrimitive(
    const kxc::api::PrimitiveArtifactKey& key,
    const kxc::codegen::KernelSignature& signature) {
    using namespace kxc;
    api::internal::PrimitiveCacheLease lease =
        api::internal::AcquirePrimitiveCache(key);
    if (lease.access() == api::internal::PrimitiveCacheAccess::kHit) {
        return lease.pin();
    }
    if (lease.access() != api::internal::PrimitiveCacheAccess::kOwner) {
        throw std::runtime_error("fixture failed to acquire a primitive cache pin");
    }
    const codegen::KernelLaunchMetadata metadata(
        Device::CPU(), codegen::CodeGenBackend::kLLVM);
    return api::internal::PublishPrimitiveCacheLease(
        lease, api::internal::CachedPrimitive{
                   signature, metadata,
                   codegen::CompiledKernel(
                       signature, metadata, std::make_shared<FixtureLauncher>()),
                   1, "adaptive-production-path-fixture", "validated"});
}

kxc::api::CompiledGraph MakeGraph(
    const kxc::api::GraphSemanticKey& graph_semantic_key,
    const kxc::api::CompileConfig& config, size_t call_count = 1,
    uint64_t output_alignment = 16) {
    using namespace kxc;
    using namespace kxc::codegen;
    std::vector<api::internal::CompiledModuleEntry> entries;
    std::vector<api::ArtifactPin> pins;
    entries.reserve(call_count);
    pins.reserve(call_count);
    for (size_t index = 0; index < call_count; ++index) {
        KernelSignature signature(
            Symbol(index),
            {KernelArgSpec("input", KernelArgRole::kInput, Float32(), {2},
                           Device::CPU()),
             KernelArgSpec("output", KernelArgRole::kOutput, Float32(), {2},
                           Device::CPU(), output_alignment, true)});
        const auto primitive = PinPrimitive(MakePrimitiveKey(config, index), signature);
        const KernelLaunchMetadata metadata(
            Device::CPU(), CodeGenBackend::kLLVM);
        entries.push_back(api::internal::CompiledModuleEntry{
            signature, metadata,
            CompiledKernel(signature, metadata,
                           primitive.artifact().kernel->launcher)});
        pins.push_back(api::internal::ArtifactPinAccess::Wrap(primitive));
    }
    return api::internal::CompiledGraphAccess::Create(
        api::internal::BuildCompiledModule(
            BuildTarget(Device::CPU()), std::move(entries), {}),
        MakePlan(call_count), std::move(pins), graph_semantic_key);
}

ProductionRequest MakeRequest(
    kxc::Function graph = MakeFunction(), int opt_level = 1,
    size_t call_count = 1, std::vector<std::int64_t> requested_ids = {}) {
    using namespace kxc;
    const api::CompileConfig config = api::CompileConfig::Create(
        BuildTarget(Device::CPU()), opt_level);
    if (requested_ids.empty()) {
        for (size_t index = 0; index < call_count; ++index) {
            requested_ids.push_back(static_cast<std::int64_t>(index));
        }
    }
    const api::GraphSemanticKey key = api::Compiler::BuildGraphSemanticKey(graph);
    return ProductionRequest(
        std::move(graph), config, MakeGraph(key, config, call_count),
        std::move(requested_ids));
}

ExecutionRequest Execute(const ProductionRequest& request) {
    return ExecutionRequest(request.dispatch_key(), request.plan_abi());
}

bool TestRequestSnapshotAndExactIdentity() {
    using namespace kxc;
    api::internal::ClearPrimitiveCacheForTesting();
    const Function graph = MakeFunction();
    api::CompileConfig original = api::CompileConfig::Create(
        BuildTarget(Device::CPU()), 1);
    original->profile_options.enabled = false;
    original->profile_options.bundle_dir = "before";
    const Target original_target = original->target;
    const std::string frozen_target =
        api::internal::BuildTargetCapabilityFingerprint(original_target);
    const api::GraphSemanticKey graph_key = api::Compiler::BuildGraphSemanticKey(graph);
    const ProductionRequest request(
        graph, original, MakeGraph(graph_key, original), {0});

    original->opt_level = 3;
    original->profile_options.enabled = true;
    original->profile_options.bundle_dir = "after";
    auto* mutable_target = const_cast<TargetNode*>(original_target.operator->());
    mutable_target->attrs.arch += "-mutated";
    TEST_CHECK(request.config()->opt_level == 1 &&
                   !request.config()->profile_options.enabled &&
                   request.config()->profile_options.bundle_dir == "before" &&
                   api::internal::BuildTargetCapabilityFingerprint(
                       request.config()->target) == frozen_target &&
                   request.baseline_graph().defined() &&
                   request.requested_unit_ids() == std::vector<std::int64_t>{0},
               "request must freeze config, baseline graph, and replacement ids");
    TEST_CHECK(request.graph_semantic_key() !=
                   api::Compiler::BuildGraphSemanticKey(MakeTwoPrimitiveFunction()) &&
                   request.dispatch_key().defined() && request.plan_abi().defined(),
               "request identity must be exact and typed");
    return true;
}

bool TestRequestIdNormalizationAndRejections() {
    using namespace kxc;
    api::internal::ClearPrimitiveCacheForTesting();
    const Function graph = MakeFunction();
    const api::CompileConfig config = api::CompileConfig::Create(
        BuildTarget(Device::CPU()), 1);
    const api::GraphSemanticKey key = api::Compiler::BuildGraphSemanticKey(graph);
    const api::CompiledGraph baseline = MakeGraph(key, config, 2);
    const ProductionRequest normalized(graph, config, baseline, {1, 0});
    TEST_CHECK(normalized.requested_unit_ids() ==
                   std::vector<std::int64_t>({0, 1}),
               "request replacement ids must be canonicalized in unit order");
    TEST_CHECK(Throws([&] { ProductionRequest(graph, config, baseline, {}); }) &&
                   Throws([&] { ProductionRequest(graph, config, baseline, {0, 0}); }) &&
                   Throws([&] { ProductionRequest(graph, config, baseline, {-1}); }) &&
                   Throws([&] { ProductionRequest(graph, config, baseline, {2}); }),
               "empty, duplicate, negative, and out-of-range ids must fail");
    return true;
}

bool TestPreparedCandidateValidationAndRetention() {
    using namespace kxc;
    using namespace production_path;
    api::internal::ClearPrimitiveCacheForTesting();
    const ProductionRequest request = MakeRequest();
    auto candidate = PrepareCandidate(
        request, MakeGraph(request.graph_semantic_key(), request.config()),
        "fixture-receipt");
    TEST_CHECK(candidate && candidate->compiled_graph().defined() &&
                   candidate->session() && candidate->session()->defined() &&
                   candidate->selection_plan_key().defined() &&
                   candidate->validation_receipt() == "fixture-receipt",
               "preparation must bind graph, pins, session, and receipt");
    TEST_CHECK(Throws([&] {
                   (void)PrepareCandidate(
                       request,
                       MakeGraph(request.graph_semantic_key(), request.config(), 1,
                                 32),
                       "fixture-receipt");
               }),
               "a candidate that changes its physical ABI must fail closed");
    std::weak_ptr<const runtime::RuntimeSession> session = candidate->session();
    api::internal::ClearPrimitiveCacheForTesting();
    auto run = candidate->session()->RunAsync(
        {runtime::NDArray::Zeros({2}, Float32(), Device::CPU())},
        DeviceStream::Default(Device::CPU()));
    run.completion.Wait();
    TEST_CHECK(session.lock() && run.outputs.size() == 1,
               "prepared candidate must retain a launchable session");
    candidate.reset();
    TEST_CHECK(session.expired(), "candidate release must release its session");
    return true;
}

bool TestMalformedGraphRejectedBeforePreparation() {
    using namespace kxc;
    api::internal::ClearPrimitiveCacheForTesting();
    ProductionRequest request = MakeRequest();
    auto* function = const_cast<FunctionNode*>(request.graph().operator->());
    function->body = Expr();
    TEST_CHECK(Throws([&] { request.Validate(); }),
               "an undefined Relay expression must fail request validation");
    return true;
}

#if KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2
namespace v2 = kxc::api::adaptive::hot_swap::v2;

bool TestAcquireWithoutRouteHasNoCompileSideEffect() {
    using namespace kxc;
    api::internal::ClearPrimitiveCacheForTesting();
    const ProductionRequest request = MakeRequest();
    const api::internal::PrimitiveCacheStats before =
        api::internal::GetPrimitiveCacheStats();
    std::atomic<uint64_t> events{0};
    v2::Options options;
    options.observer = [&](const v2::Event&) { events.fetch_add(1); };
    v2::AdaptiveHotSwapController controller(options);
    TEST_CHECK(Throws([&] { (void)controller.Acquire(Execute(request)); }),
               "Acquire must fail when no exact route has been published");
    const api::internal::PrimitiveCacheStats after =
        api::internal::GetPrimitiveCacheStats();
    TEST_CHECK(after.misses == before.misses && after.in_flight == before.in_flight &&
                   events.load() == 0,
               "Acquire without a route must not compile or emit events");
    return true;
}

class ValidationGate final : public v2::CandidateValidationAuthority {
public:
    v2::ValidationReceipt Validate(const ProductionRequest&,
                                   const kxc::api::CompiledGraph&) override {
        std::unique_lock<std::mutex> lock(mutex_);
        ++calls_;
        wake_.notify_all();
        if (calls_ == 1) {
            wake_.wait(lock, [this] { return released_; });
        }
        if (reject_subsequent_ && calls_ > 1) {
            throw std::runtime_error("validation gate rejection");
        }
        return IssueReceipt("validation-gate-" + std::to_string(calls_));
    }

    bool WaitForFirstCall() {
        std::unique_lock<std::mutex> lock(mutex_);
        return wake_.wait_for(lock, std::chrono::seconds(5),
                              [this] { return calls_ >= 1; });
    }

    void Release() {
        std::lock_guard<std::mutex> lock(mutex_);
        released_ = true;
        wake_.notify_all();
    }

    void RejectSubsequent() {
        std::lock_guard<std::mutex> lock(mutex_);
        reject_subsequent_ = true;
    }

    size_t calls() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return calls_;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable wake_;
    size_t calls_{0};
    bool released_{false};
    bool reject_subsequent_{false};
};

bool TestPreCancelledSubmitHasNoCompileSideEffect() {
    using namespace kxc;
    api::internal::ClearPrimitiveCacheForTesting();
    const ProductionRequest request = MakeRequest();
    const api::internal::PrimitiveCacheStats before =
        api::internal::GetPrimitiveCacheStats();
    v2::AdaptiveHotSwapController controller;
    v2::CancellationSource cancellation;
    cancellation.Cancel();
    const v2::CompileResult result = controller.Submit(
        {request, std::chrono::steady_clock::time_point::max(),
         cancellation.token()}).Wait();
    const api::internal::PrimitiveCacheStats after =
        api::internal::GetPrimitiveCacheStats();
    TEST_CHECK(!result.ready() &&
                   result.failure.category == v2::FailureCategory::kCancelled &&
                   after.misses == before.misses && after.in_flight == before.in_flight &&
                   Throws([&] { (void)controller.Acquire(Execute(request)); }),
               "a pre-cancelled Submit must not compile or publish a route");
    return true;
}
#endif

bool TestRealValidationSingleflightAndCancellation() {
    using namespace kxc;
    using namespace kxc::api;
    internal::ClearPrimitiveCacheForTesting();
    const Function graph = MakeTwoPrimitiveFunction();
    const CompileConfig config =
        CompileConfig::Create(BuildTarget(Device::CPU()), 2);
    const CompiledGraph baseline = Compiler::Compile(graph, config);
    const ProductionRequest request(graph, config, baseline, {0, 1});

    auto validation = std::make_shared<ValidationGate>();
    v2::Options options;
    options.worker_count = 1;
    options.validation_authority = validation;
    v2::AdaptiveHotSwapController controller(options);
    const v2::CompileTicket first = controller.Submit({request});
    if (!validation->WaitForFirstCall()) {
        validation->Release();
        TEST_CHECK(false, "singleflight validation gate was not reached");
    }
    const v2::CompileTicket second = controller.Submit({request});
    validation->Release();
    const v2::CompileResult first_result = first.Wait();
    const v2::CompileResult second_result = second.Wait();
    TEST_CHECK(first_result.ready() && second_result.ready() &&
                   first_result.lease == second_result.lease &&
                   validation->calls() == 1,
               "identical in-flight Submit requests must share validation and generation");

    auto cancelled_validation = std::make_shared<ValidationGate>();
    v2::Options cancelled_options;
    cancelled_options.worker_count = 1;
    cancelled_options.validation_authority = cancelled_validation;
    v2::AdaptiveHotSwapController cancelled(cancelled_options);
    v2::CancellationSource cancellation;
    const ProductionRequest cancelled_request(graph, config, baseline, {0, 1});
    const v2::CompileTicket cancelled_ticket = cancelled.Submit(
        {cancelled_request, std::chrono::steady_clock::time_point::max(),
         cancellation.token()});
    if (!cancelled_validation->WaitForFirstCall()) {
        cancelled_validation->Release();
        TEST_CHECK(false, "cancellation validation gate was not reached");
    }
    cancellation.Cancel();
    cancelled_validation->RejectSubsequent();
    cancelled_validation->Release();
    const ProductionRequest follow_up(graph, config, baseline, {1});
    TEST_CHECK(Throws([&] { (void)cancelled.CompileAndPublish({follow_up}); }) &&
                   !cancelled_ticket.Wait().ready() &&
                   cancelled_ticket.Wait().failure.category ==
                       v2::FailureCategory::kCancelled &&
                   Throws([&] { (void)cancelled.Acquire(Execute(cancelled_request)); }),
               "final cancellation must suppress publication after validation");
    return true;
}

bool TestRealPrimitiveReplacementProductionPath() {
    using namespace kxc;
    using namespace kxc::api;
    internal::ClearPrimitiveCacheForTesting();
    const Function graph = MakeTwoPrimitiveFunction();
    const CompileConfig baseline_config =
        CompileConfig::Create(BuildTarget(Device::CPU()), 1);
    const CompileConfig replacement_config =
        CompileConfig::Create(BuildTarget(Device::CPU()), 2);
    const CompileConfig later_config =
        CompileConfig::Create(BuildTarget(Device::CPU()), 3);
    const CompiledGraph baseline_graph = Compiler::Compile(graph, baseline_config);
    TEST_CHECK(baseline_graph.artifact_pins().size() == 2,
               "real replacement fixture requires exactly two primitive units");
    std::mutex observer_mutex;
    std::condition_variable observer_wake;
    uint64_t events{0};
    uint64_t publications{0};
    v2::Options options;
    options.observer = [&](const v2::Event& event) {
        std::lock_guard<std::mutex> lock(observer_mutex);
        ++events;
        if (event.kind == v2::EventKind::kPublished) ++publications;
        observer_wake.notify_all();
    };
    v2::AdaptiveHotSwapController controller(options);
    const ProductionRequest initial_request(
        graph, baseline_config, baseline_graph, {0, 1});
    const auto initial = controller.CompileAndPublish({initial_request});
    const ProductionRequest replacement_request(
        graph, replacement_config, initial->compiled_graph(), {1});
    const internal::PrimitiveCacheStats before_replacement =
        internal::GetPrimitiveCacheStats();
    const auto replacement = controller.CompileAndPublish({replacement_request});
    const internal::PrimitiveCacheStats after_replacement =
        internal::GetPrimitiveCacheStats();
    const internal::PrimitiveArtifactPin initial_unit0 =
        internal::ArtifactPinAccess::Unwrap(
            initial->compiled_graph().artifact_pins()[0]);
    const internal::PrimitiveArtifactPin replacement_unit0 =
        internal::ArtifactPinAccess::Unwrap(
            replacement->compiled_graph().artifact_pins()[0]);
    const internal::PrimitiveArtifactPin initial_unit1 =
        internal::ArtifactPinAccess::Unwrap(
            initial->compiled_graph().artifact_pins()[1]);
    const internal::PrimitiveArtifactPin replacement_unit1 =
        internal::ArtifactPinAccess::Unwrap(
            replacement->compiled_graph().artifact_pins()[1]);
    TEST_CHECK(after_replacement.misses == before_replacement.misses + 1 &&
                   &initial_unit0.artifact() == &replacement_unit0.artifact() &&
                   &initial_unit1.artifact() != &replacement_unit1.artifact() &&
                   initial_unit1.key() != replacement_unit1.key() &&
                   initial_unit1.artifact().kernel->launcher !=
                       replacement_unit1.artifact().kernel->launcher &&
                   initial->plan_abi() == replacement->plan_abi(),
               "one-unit replacement must preserve the other artifact owner and Plan ABI");
    const ExecutionRequest execution(replacement->dispatch_key(), replacement->plan_abi());
    const Array<runtime::NDArray> inputs = {
        Tensor({1.0f, 2.0f}), Tensor({3.0f, 4.0f})};
    const std::vector<float> expected{12.0f, 24.0f};
    auto old_run = initial->session()->RunAsync(inputs, DeviceStream::Default(Device::CPU()));
    old_run.completion.Wait();
    auto new_run = replacement->session()->RunAsync(inputs, DeviceStream::Default(Device::CPU()));
    new_run.completion.Wait();
    TEST_CHECK(old_run.outputs.size() == 1 && new_run.outputs.size() == 1 &&
                   TensorEquals(old_run.outputs[0], expected) &&
                   TensorEquals(new_run.outputs[0], expected),
               "old and replacement generations must compute the expected result");

    const ProductionRequest same_request(
        graph, replacement_config, initial->compiled_graph(), {1});
    const auto same = controller.CompileAndPublish({same_request});
    {
        std::unique_lock<std::mutex> lock(observer_mutex);
        if (!observer_wake.wait_for(lock, std::chrono::seconds(5),
                                    [&] { return publications == 2; })) {
            std::cerr << "[FAIL] " << __FUNCTION__
                      << ": expected publication events were not observed\n";
            return false;
        }
    }
    TEST_CHECK(same->generation() == replacement->generation(),
               "same keys from a stale baseline must not publish a new generation");

    const internal::PrimitiveCacheStats before_acquire =
        internal::GetPrimitiveCacheStats();
    uint64_t events_before_acquire = 0;
    {
        std::lock_guard<std::mutex> lock(observer_mutex);
        events_before_acquire = events;
    }
    const auto acquired = controller.Acquire(execution);
    auto current_run = controller.RunAsync(
        execution, inputs, DeviceStream::Default(Device::CPU()));
    current_run.completion.Wait();
    const internal::PrimitiveCacheStats after_acquire =
        internal::GetPrimitiveCacheStats();
    uint64_t events_after_acquire = 0;
    {
        std::lock_guard<std::mutex> lock(observer_mutex);
        events_after_acquire = events;
    }
    TEST_CHECK(acquired == replacement && current_run.outputs.size() == 1 &&
                   TensorEquals(current_run.outputs[0], expected) &&
                   after_acquire.misses == before_acquire.misses &&
                   after_acquire.in_flight == before_acquire.in_flight &&
                   events_after_acquire == events_before_acquire,
               "Acquire and RunAsync must have no compilation side effects");

    const auto old_lease = controller.Acquire(execution);
    const ProductionRequest later_request(
        graph, later_config, replacement->compiled_graph(), {1});
    const v2::CompileTicket ticket = controller.Submit({later_request});
    auto held_run = old_lease->session()->RunAsync(
        inputs, DeviceStream::Default(Device::CPU()));
    held_run.completion.Wait();
    const auto later = ticket.Wait();
    auto held_after = old_lease->session()->RunAsync(
        inputs, DeviceStream::Default(Device::CPU()));
    held_after.completion.Wait();
    TEST_CHECK(later.ready() && old_lease->generation() < later.lease->generation() &&
                   held_run.outputs.size() == 1 && held_after.outputs.size() == 1 &&
                   TensorEquals(held_run.outputs[0], expected) &&
                   TensorEquals(held_after.outputs[0], expected),
               "an acquired old lease must remain executable across publication");
    return true;
}

}  // namespace

int main() {
    std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"request_snapshot_exact_identity", TestRequestSnapshotAndExactIdentity},
        {"request_id_normalization_rejections", TestRequestIdNormalizationAndRejections},
        {"prepared_candidate_validation_retention", TestPreparedCandidateValidationAndRetention},
        {"malformed_graph_rejected_before_preparation", TestMalformedGraphRejectedBeforePreparation},
    };
#if KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2
    tests.push_back({"acquire_without_route_no_compile_side_effect",
                     TestAcquireWithoutRouteHasNoCompileSideEffect});
    tests.push_back({"pre_cancelled_submit_no_compile_side_effect",
                     TestPreCancelledSubmitHasNoCompileSideEffect});
#endif
#if KXC_USE_LLVM
    tests.push_back({"real_validation_singleflight_cancellation",
                     TestRealValidationSingleflightAndCancellation});
    tests.push_back({"real_primitive_replacement_production_path",
                     TestRealPrimitiveReplacementProductionPath});
#endif
    int failures = 0;
    for (const auto& test : tests) {
        try {
            if (test.second()) {
                std::cout << "[PASS] " << test.first << '\n';
            } else {
                ++failures;
            }
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << test.first << ": " << error.what()
                      << '\n';
            ++failures;
        }
    }
    return failures == 0 ? 0 : 1;
}
