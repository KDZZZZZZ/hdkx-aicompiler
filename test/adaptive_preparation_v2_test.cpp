/*! \file test/adaptive_preparation_v2_test.cpp
 * \brief CPU tests for preparation contracts and v2 adaptive authority.
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "kxc/compiler/adaptive_production_experimental.h"
#if KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2
#include "kxc/compiler/adaptive_hot_swap_v2.h"
#endif
#include "../src/support/hash.h"
#include "kxc/relay/op.h"
#include "kxc/runtime/compiled_module.h"
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

class UnregisteredDerivedCallNode final : public kxc::CallNode {
public:
    std::string hidden_semantics;
};

kxc::Expr MakeDerivedCall(const kxc::Expr& argument) {
    auto* call = new UnregisteredDerivedCallNode();
    call->op = kxc::relay::Op::Get("nn_relu");
    call->args = {argument};
    call->hidden_semantics = "must-not-be-omitted";
    return kxc::Expr(kxc::ObjectRef(call));
}

std::string ExceptionMessage(const std::exception_ptr& error) {
    try {
        if (error) std::rethrow_exception(error);
    } catch (const std::exception& failure) {
        return failure.what();
    } catch (...) {
        return "non-standard exception";
    }
    return {};
}

DLDataType Float32() {
    return kxc::runtime::DataTypeFromString("float32");
}

class Gate final {
public:
    void EnterAndWait() {
        std::unique_lock<std::mutex> lock(mutex_);
        ++entered_;
        condition_.notify_all();
        condition_.wait(lock, [this] { return released_; });
    }

    bool WaitUntilEntered(size_t count = 1) {
        std::unique_lock<std::mutex> lock(mutex_);
        return condition_.wait_for(lock, std::chrono::seconds(5),
                                   [this, count] { return entered_ >= count; });
    }

    void Release() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            released_ = true;
        }
        condition_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable condition_;
    size_t entered_{0};
    bool released_{false};
};

class FixtureLauncher final : public kxc::codegen::KernelLauncher {
public:
    explicit FixtureLauncher(std::shared_ptr<Gate> gate = nullptr)
        : gate_(std::move(gate)) {}

    bool IsReady() const noexcept override { return true; }

    kxc::AsyncOperation Launch(
        const kxc::Array<kxc::runtime::NDArray>& arguments,
        const kxc::DeviceStream& stream,
        const kxc::ObjectRef&) const override {
        launches.fetch_add(1, std::memory_order_relaxed);
        if (gate_) gate_->EnterAndWait();
        kxc::Array<kxc::Storage> retained;
        for (const auto& argument : arguments) {
            retained.push_back(argument.storage());
        }
        return kxc::AsyncOperation::Completed(stream, std::move(retained));
    }

    mutable std::atomic<int> launches{0};

private:
    std::shared_ptr<Gate> gate_;
};

std::string Symbol(size_t index) {
    return "adaptive_fixture_" + std::to_string(index);
}

kxc::runtime::ExecutablePlan MakePlan(size_t call_count = 1,
                                      int64_t input_extent = 2) {
    using namespace kxc;
    Array<runtime::ValueSpec> values;
    values.push_back(runtime::ValueSpec(0, 0, {input_extent}, Float32(),
                                        Device::CPU(), true));
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

kxc::Function MakeAddFunction() {
    using namespace kxc;
    Var lhs("lhs", TensorType({2}, "float32"));
    Var rhs("rhs", TensorType({2}, "float32"));
    return Function({lhs, rhs}, Call(relay::Op::Get("add"), {lhs, rhs}));
}

kxc::Function MakeConstantFunction(float value) {
    using namespace kxc;
    runtime::NDArray data =
        runtime::NDArray::Empty({1}, Float32(), Device::CPU());
    data.CopyFromBytes(&value, sizeof(value));
    return Function({}, Constant(std::move(data)));
}

kxc::api::PrimitiveArtifactKey MakePrimitiveKey(
    const kxc::api::CompileConfig& config, size_t index,
    const std::string& suffix = "baseline") {
    return kxc::api::internal::BuildPrimitiveArtifactKey(
        kxc::api::UnitSemanticKey(
            "adaptive-fixture-primitive-v1:" + std::to_string(index) + ":" +
            suffix),
        config->target,
        "adaptive-fixture-pipeline-v1:opt=" +
            std::to_string(config->opt_level),
        "adaptive-fixture-schedule-v1", "adaptive-fixture-backend-v1");
}

kxc::api::internal::PrimitiveArtifactPin PinPrimitive(
    const kxc::api::PrimitiveArtifactKey& key,
    const kxc::codegen::KernelSignature& signature,
    const kxc::codegen::KernelLaunchMetadata& metadata,
    std::shared_ptr<const kxc::codegen::KernelLauncher> launcher,
    uint64_t byte_size = 1,
    std::string provenance = "adaptive-production-path-fixture") {
    using namespace kxc;
    api::internal::PrimitiveCacheLease lease =
        api::internal::AcquirePrimitiveCache(key);
    switch (lease.access()) {
        case api::internal::PrimitiveCacheAccess::kOwner:
            return api::internal::PublishPrimitiveCacheLease(
                lease, api::internal::CachedPrimitive{
                           signature, metadata,
                           codegen::CompiledKernel(signature, metadata,
                                                   std::move(launcher)),
                           byte_size, std::move(provenance),
                           "typed-launcher-validated"});
        case api::internal::PrimitiveCacheAccess::kHit:
            return lease.pin();
        case api::internal::PrimitiveCacheAccess::kWait:
            return api::internal::WaitPrimitiveCacheLease(lease);
        case api::internal::PrimitiveCacheAccess::kFailed:
        case api::internal::PrimitiveCacheAccess::kRejected:
            throw std::runtime_error(
                "fixture failed to acquire a primitive cache pin");
    }
    throw std::logic_error("unknown primitive cache access");
}

struct GraphOptions final {
    int64_t input_extent{2};
    uint64_t output_alignment{16};
    bool foreign_launch_metadata{false};
    uint64_t primitive_byte_size{1};
    std::string primitive_provenance{"adaptive-production-path-fixture"};
    std::string cached_symbol_prefix;
    std::vector<std::shared_ptr<const kxc::codegen::KernelLauncher>> launchers;
    bool reverse_pins{false};
};

kxc::api::CompiledGraph MakeGraph(
    const kxc::api::GraphSemanticKey& graph_semantic_key,
    const std::vector<kxc::api::PrimitiveArtifactKey>& primitive_keys,
    GraphOptions options = {}) {
    using namespace kxc;
    using namespace kxc::codegen;
    if (primitive_keys.empty()) {
        throw std::invalid_argument("fixture graph requires primitive keys");
    }

    std::vector<api::internal::CompiledModuleEntry> entries;
    std::vector<api::ArtifactPin> pins;
    entries.reserve(primitive_keys.size());
    pins.reserve(primitive_keys.size());
    for (size_t index = 0; index < primitive_keys.size(); ++index) {
        const int64_t input_extent = index == 0 ? options.input_extent : 2;
        KernelSignature signature(
            Symbol(index),
            {KernelArgSpec("input", KernelArgRole::kInput, Float32(),
                           {input_extent}, Device::CPU()),
             KernelArgSpec("output", KernelArgRole::kOutput, Float32(), {2},
                           Device::CPU(), options.output_alignment, true)});
        KernelLaunchMetadata metadata = options.foreign_launch_metadata
            ? KernelLaunchMetadata(Device::CUDA(), CodeGenBackend::kCUDA,
                                   Dim3{1, 1, 1}, Dim3{1, 1, 1})
            : KernelLaunchMetadata(Device::CPU(), CodeGenBackend::kLLVM);
        std::shared_ptr<const KernelLauncher> launcher =
            index < options.launchers.size() && options.launchers[index]
                ? options.launchers[index]
                : std::make_shared<FixtureLauncher>();
        const KernelSignature cached_signature =
            options.cached_symbol_prefix.empty()
                ? signature
                : KernelSignature(
                      String(options.cached_symbol_prefix +
                             std::to_string(index)),
                      signature.arguments());
        const api::internal::PrimitiveArtifactPin primitive = PinPrimitive(
            primitive_keys[index], cached_signature, metadata,
            std::move(launcher), options.primitive_byte_size,
            options.primitive_provenance);
        const api::ArtifactPin pin = api::internal::ArtifactPinAccess::Wrap(primitive);
        const CompiledKernel module_kernel(
            signature, metadata, primitive.artifact().kernel->launcher);
        entries.push_back(api::internal::CompiledModuleEntry{
            tir::PrimFunc(), signature, metadata, module_kernel});
        pins.push_back(pin);
    }
    if (options.reverse_pins && pins.size() >= 2) {
        std::swap(pins[0], pins[1]);
    }
    api::CompiledModule module = api::internal::BuildCompiledModule(
        BuildTarget(Device::CPU()), std::move(entries), {});
    return api::internal::CompiledGraphAccess::Create(
        std::move(module), MakePlan(primitive_keys.size(), options.input_extent),
        std::move(pins), graph_semantic_key);
}

ProductionRequest MakeRequestFromConfig(
    kxc::Function graph, const kxc::api::CompileConfig& config,
    int64_t expected_input_extent = 2,
    uint64_t expected_output_alignment = 16,
    size_t call_count = 1,
    std::vector<kxc::api::PrimitiveArtifactKey> primitive_keys = {},
    std::shared_ptr<FixtureLauncher> baseline_launcher = nullptr) {
    using namespace kxc::api;
    const GraphSemanticKey graph_semantic_key =
        Compiler::BuildGraphSemanticKey(graph);
    if (primitive_keys.empty()) {
        primitive_keys.reserve(call_count);
        for (size_t index = 0; index < call_count; ++index) {
            primitive_keys.push_back(MakePrimitiveKey(config, index));
        }
    }
    if (!baseline_launcher) {
        baseline_launcher = std::make_shared<FixtureLauncher>();
    }
    GraphOptions options;
    options.input_extent = expected_input_extent;
    options.output_alignment = expected_output_alignment;
    options.launchers.assign(primitive_keys.size(),
                             std::move(baseline_launcher));
    const CompiledGraph baseline =
        MakeGraph(graph_semantic_key, primitive_keys, std::move(options));
    return ProductionRequest(std::move(graph), config, baseline);
}

ProductionRequest MakeRequest(
    int opt_level = 1, int64_t expected_input_extent = 2,
    uint64_t expected_output_alignment = 16, size_t call_count = 1,
    std::vector<kxc::api::PrimitiveArtifactKey> primitive_keys = {},
    std::shared_ptr<FixtureLauncher> baseline_launcher = nullptr) {
    const kxc::Function graph = MakeFunction();
    const kxc::api::CompileConfig config = kxc::api::CompileConfig::Create(
        kxc::BuildTarget(kxc::Device::CPU()), opt_level);
    return MakeRequestFromConfig(
        graph, config, expected_input_extent, expected_output_alignment,
        call_count, std::move(primitive_keys), std::move(baseline_launcher));
}

ExecutionRequest Execute(const ProductionRequest& request) {
    return ExecutionRequest(request.dispatch_key(), request.plan_abi());
}

std::vector<kxc::api::PrimitiveArtifactKey> SelectedKeys(
    const ProductionRequest& request) {
    std::vector<kxc::api::PrimitiveArtifactKey> keys;
    for (const auto& artifact : request.ordered_artifacts()) {
        keys.push_back(artifact.artifact_key);
    }
    return keys;
}

enum class Attack {
    kNone,
    kDistinctSelection,
    kWrongSignature,
    kWrongMetadata,
    kWrongOrder,
};

class FixtureCompiler final : public production_path::ProductionPathCompilerAdapter {
public:
    kxc::api::CompiledGraph Compile(
        const ProductionRequest& request) override {
        calls.fetch_add(1, std::memory_order_relaxed);
        observed_opt_level.store(request.config()->opt_level,
                                 std::memory_order_relaxed);
        const int active_now = active.fetch_add(1, std::memory_order_acq_rel) + 1;
        int peak = peak_active.load(std::memory_order_relaxed);
        while (peak < active_now &&
               !peak_active.compare_exchange_weak(
                   peak, active_now, std::memory_order_relaxed)) {
        }
        try {
            if (compile_gate) compile_gate->EnterAndWait();
            if (ConsumeFailure()) {
                throw std::runtime_error("injected adapter failure");
            }

            std::vector<kxc::api::PrimitiveArtifactKey> keys =
                SelectedKeys(request);
            std::vector<std::shared_ptr<const kxc::codegen::KernelLauncher>>
                launchers;
            launchers.reserve(request.verified_artifact_pins().size());
            for (const auto& pin : request.verified_artifact_pins()) {
                launchers.push_back(
                    kxc::api::internal::ArtifactPinAccess::Unwrap(pin)
                        .artifact().kernel->launcher);
            }
            if (attack == Attack::kDistinctSelection) {
                keys[0] = MakePrimitiveKey(request.config(), 0, "distinct-selection");
                launchers[0] = std::make_shared<FixtureLauncher>();
            }

            GraphOptions options;
            options.input_extent = candidate_input_extent;
            options.output_alignment =
                attack == Attack::kWrongSignature ? 32
                                                  : candidate_output_alignment;
            options.foreign_launch_metadata = attack == Attack::kWrongMetadata;
            options.primitive_byte_size = candidate_primitive_byte_size;
            options.primitive_provenance = candidate_primitive_provenance;
            options.launchers = launchers;
            options.reverse_pins = attack == Attack::kWrongOrder;
            kxc::api::CompiledGraph graph =
                MakeGraph(request.graph_semantic_key(), keys,
                          std::move(options));
            {
                std::lock_guard<std::mutex> lock(launcher_mutex);
                last_launcher = std::dynamic_pointer_cast<const FixtureLauncher>(
                    kxc::api::internal::ArtifactPinAccess::Unwrap(
                        graph.artifact_pins()[0]).artifact().kernel->launcher);
            }
            active.fetch_sub(1, std::memory_order_acq_rel);
            return graph;
        } catch (...) {
            active.fetch_sub(1, std::memory_order_acq_rel);
            throw;
        }
    }

    std::shared_ptr<const FixtureLauncher> LastLauncher() const {
        std::lock_guard<std::mutex> lock(launcher_mutex);
        return last_launcher;
    }

    bool ConsumeFailure() {
        int remaining = failures_remaining.load(std::memory_order_relaxed);
        while (remaining > 0) {
            if (failures_remaining.compare_exchange_weak(
                    remaining, remaining - 1, std::memory_order_relaxed)) {
                return true;
            }
        }
        return false;
    }

    int64_t candidate_input_extent{2};
    uint64_t candidate_output_alignment{16};
    uint64_t candidate_primitive_byte_size{1};
    Attack attack{Attack::kNone};
    std::shared_ptr<Gate> compile_gate;
    std::atomic<int> failures_remaining{0};
    std::atomic<int> calls{0};
    std::atomic<int> active{0};
    std::atomic<int> peak_active{0};
    std::atomic<int> observed_opt_level{-1};
    std::string candidate_primitive_provenance{"adaptive-production-path-fixture"};

private:
    mutable std::mutex launcher_mutex;
    std::shared_ptr<const FixtureLauncher> last_launcher;
};

bool WaitFor(const std::function<bool()>& predicate) {
    for (int attempt = 0; attempt < 5000; ++attempt) {
        if (predicate()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

bool TestRequestSnapshotAndExactIdentity() {
    using namespace kxc;
    using namespace production_path;
    api::internal::ClearPrimitiveCacheForTesting();
    const Function graph = MakeFunction();
    api::CompileConfig original = api::CompileConfig::Create(
        BuildTarget(Device::CPU()), 1);
    original->profile_options.enabled = false;
    original->profile_options.bundle_dir = "before";
    const Target original_target = original->target;
    const ProductionRequest request = MakeRequestFromConfig(graph, original);
    const std::string frozen_target =
        api::internal::BuildTargetCapabilityFingerprint(request.config()->target);

    original->opt_level = 3;
    original->profile_options.enabled = true;
    original->profile_options.bundle_dir = "after";
    auto* mutable_target = const_cast<TargetNode*>(original_target.operator->());
    mutable_target->attrs.arch += "-mutated";
    TEST_CHECK(request.config()->opt_level == 1 &&
                   !request.config()->profile_options.enabled &&
                   request.config()->profile_options.bundle_dir == "before" &&
                   api::internal::BuildTargetCapabilityFingerprint(
                       request.config()->target) == frozen_target,
               "request must deep-freeze config and target fields");
    api::CompileConfig exported_copy = request.config();
    exported_copy->opt_level = 2;
    TEST_CHECK(request.config()->opt_level == 1,
               "adapter config copies must not alias the request snapshot");
    TEST_CHECK(request.graph_semantic_key() !=
                   api::Compiler::BuildGraphSemanticKey(MakeAddFunction()) &&
                   request.dispatch_key().defined() && request.plan_abi().defined(),
               "request identity must be exact and typed");
    return true;
}

bool TestPreparedCandidateValidationAndExactIdentity() {
    using namespace production_path;
    kxc::api::internal::ClearPrimitiveCacheForTesting();
    const ProductionRequest request = MakeRequest();
    const auto candidate = PrepareCandidate(
        request, MakeGraph(request.graph_semantic_key(), SelectedKeys(request)),
        "fixture-receipt");
    TEST_CHECK(candidate && candidate->compiled_graph().defined() &&
                   candidate->session() && candidate->session()->defined() &&
                   candidate->selection_plan_key().defined() &&
                   candidate->selected_artifacts() == request.ordered_artifacts() &&
                   candidate->validation_receipt() == "fixture-receipt",
               "preparation must bind the validated graph, pins, session, and receipt");

    // Primitive cache entries may originate at a different graph-local link
    // symbol. The current module symbol remains exact while only the physical
    // ordered calling convention is reused.
    kxc::api::internal::ClearPrimitiveCacheForTesting();
    GraphOptions relocated;
    relocated.cached_symbol_prefix = "cached_graph_symbol_";
    const auto relocated_candidate = PrepareCandidate(
        request,
        MakeGraph(request.graph_semantic_key(), SelectedKeys(request),
                  relocated),
        "relocated-receipt");
    TEST_CHECK(relocated_candidate &&
                   relocated_candidate->compiled_graph().defined(),
               "same physical ABI must permit graph-local symbol relocation");

    GraphOptions malformed;
    malformed.output_alignment = 32;
    TEST_CHECK(Throws([&] {
                   (void)PrepareCandidate(
                       request,
                       MakeGraph(request.graph_semantic_key(),
                                 SelectedKeys(request), malformed),
                       "fixture-receipt");
               }),
               "a candidate that changes the exact callable ABI must fail closed");

    GraphOptions foreign_metadata;
    foreign_metadata.foreign_launch_metadata = true;
    TEST_CHECK(Throws([&] {
                   (void)PrepareCandidate(
                       request,
                       MakeGraph(request.graph_semantic_key(),
                                 SelectedKeys(request), foreign_metadata),
                       "fixture-receipt");
               }),
               "a candidate with foreign launch metadata must fail closed");

    const ProductionRequest two_calls = MakeRequest(1, 2, 16, 2);
    GraphOptions reversed_pins;
    reversed_pins.reverse_pins = true;
    TEST_CHECK(Throws([&] {
                   (void)PrepareCandidate(
                       two_calls,
                       MakeGraph(two_calls.graph_semantic_key(),
                                 SelectedKeys(two_calls), reversed_pins),
                       "fixture-receipt");
               }),
               "reversed ordered pins must fail preparation");

    const kxc::api::PrimitiveArtifactKey foreign_target_key(
        kxc::api::UnitSemanticKey("adaptive-fixture-foreign-target-v1"),
        "foreign-target-capability", "adaptive-fixture-pipeline-v1", 1,
        "adaptive-fixture-schedule-v1", "adaptive-fixture-backend-v1");
    TEST_CHECK(foreign_target_key.target_capability_fingerprint() ==
                   "foreign-target-capability" &&
                   Throws([&] {
                       (void)MakeGraph(request.graph_semantic_key(),
                                       {foreign_target_key});
                   }),
               "the candidate factory must reject foreign target capability");

    TEST_CHECK(Throws([&] {
                   (void)PrepareCandidate(
                       request,
                       MakeGraph(request.graph_semantic_key(),
                                 SelectedKeys(request)), "");
               }),
               "preparation requires a non-empty injected receipt");
    return true;
}

bool TestMalformedGraphRejectedBeforePreparation() {
    using namespace kxc;
    using namespace production_path;
    api::internal::ClearPrimitiveCacheForTesting();
    ProductionRequest undefined_request = MakeRequest();
    auto* undefined_function =
        const_cast<FunctionNode*>(undefined_request.graph().operator->());
    undefined_function->body = Expr();
    TEST_CHECK(Throws([&] { undefined_request.Validate(); }),
               "an undefined Relay expression must fail request validation");

    ProductionRequest derived_request = MakeRequest();
    auto* derived_function =
        const_cast<FunctionNode*>(derived_request.graph().operator->());
    derived_function->body = MakeDerivedCall(derived_function->params[0]);
    TEST_CHECK(Throws([&] { derived_request.Validate(); }),
               "an unknown derived Relay expression must fail request validation");
    return true;
}

bool TestPreparedCandidateRetainsPinsAndSession() {
    using namespace kxc;
    using namespace production_path;
    api::internal::ClearPrimitiveCacheForTesting();
    const ProductionRequest request = MakeRequest();
    auto candidate = PrepareCandidate(
        request, MakeGraph(request.graph_semantic_key(), SelectedKeys(request)),
        "fixture-receipt");
    std::weak_ptr<const runtime::RuntimeSession> session = candidate->session();
    api::internal::ClearPrimitiveCacheForTesting();
    auto run = candidate->session()->RunAsync(
        {runtime::NDArray::Zeros({2}, Float32(), Device::CPU())},
        DeviceStream::Default(Device::CPU()));
    run.completion.Wait();
    TEST_CHECK(session.lock() && run.outputs.size() == 1,
               "prepared candidate pins must keep its session launchable after cache clear");
    candidate.reset();
    TEST_CHECK(session.expired(),
               "dropping the prepared candidate releases its session owner");
    return true;
}

#if KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2
namespace v2 = kxc::api::adaptive::hot_swap::v2;

class OneShotHealth final : public v2::HealthAuthority {
public:
    explicit OneShotHealth(v2::Generation generation) : generation_(generation) {}
    v2::HealthDecision Evaluate(const v2::GenerationLease&) override {
        return {generation_, v2::HealthDisposition::kQuarantine, "fixture-health", "one"};
    }
    bool VerifyAndConsume(const v2::HealthDecision& decision,
                          const v2::GenerationLease&) noexcept override {
        if (consumed_ || decision.generation != generation_) return false;
        consumed_ = true;
        return true;
    }
private:
    v2::Generation generation_;
    bool consumed_{false};
};

class EveryGenerationHealth final : public v2::HealthAuthority {
public:
    v2::HealthDecision Evaluate(const v2::GenerationLease& lease) override {
        return {lease.generation(), v2::HealthDisposition::kQuarantine,
                "fixture-health", std::to_string(lease.generation())};
    }
    bool VerifyAndConsume(const v2::HealthDecision& decision,
                          const v2::GenerationLease& lease) noexcept override {
        return decision.generation == lease.generation();
    }
};

class ReentrantHealth final : public v2::HealthAuthority {
public:
    void SetController(v2::AdaptiveHotSwapController* controller,
                       ExecutionRequest request) {
        controller_ = controller;
        request_ = std::move(request);
    }
    v2::HealthDecision Evaluate(const v2::GenerationLease& lease) override {
        return {lease.generation(), v2::HealthDisposition::kHealthy,
                "fixture-health", "reentry"};
    }
    bool VerifyAndConsume(const v2::HealthDecision&,
                          const v2::GenerationLease&) noexcept override {
        reentry_rejected_ = Throws(
            [&] { (void)controller_->Acquire(*request_); });
        return reentry_rejected_;
    }
    bool reentry_rejected() const noexcept { return reentry_rejected_; }
private:
    v2::AdaptiveHotSwapController* controller_{nullptr};
    std::optional<ExecutionRequest> request_;
    bool reentry_rejected_{false};
};

class UniqueReceiptValidation final : public v2::CandidateValidationAuthority {
public:
    v2::ValidationReceipt Validate(const ProductionRequest&, const kxc::api::CompiledGraph&) override {
        return IssueReceipt("fixture-receipt-" + std::to_string(next_++));
    }
private:
    uint64_t next_{1};
};

class ReplayGenerationAuthority final : public v2::GenerationAuthority {
public:
    std::shared_ptr<const v2::GenerationLease> Issue(
        const v2::GenerationAuthorityRequest& request) override {
        if (first_) return first_;
        first_ = MakeLease(9, request);
        return first_;
    }
private:
    std::shared_ptr<const v2::GenerationLease> first_;
};

class ScriptedGenerationAuthority final : public v2::GenerationAuthority {
public:
    explicit ScriptedGenerationAuthority(std::vector<v2::Generation> generations)
        : generations_(std::move(generations)) {}
    std::shared_ptr<const v2::GenerationLease> Issue(
        const v2::GenerationAuthorityRequest& request) override {
        return MakeLease(generations_.at(index_++), request);
    }
private:
    std::vector<v2::Generation> generations_;
    size_t index_{0};
};

class BlockingGenerationAuthority final : public v2::GenerationAuthority {
public:
    explicit BlockingGenerationAuthority(std::shared_ptr<Gate> gate) : gate_(std::move(gate)) {}
    void SetController(v2::AdaptiveHotSwapController* controller,
                       ExecutionRequest request) {
        controller_ = controller;
        request_ = std::move(request);
    }
    bool reentry_rejected() const noexcept { return reentry_rejected_; }
    std::shared_ptr<const v2::GenerationLease> Issue(
        const v2::GenerationAuthorityRequest& request) override {
        if (controller_) {
            reentry_rejected_ = Throws(
                [&] { (void)controller_->Acquire(*request_); });
        }
        gate_->EnterAndWait();
        return MakeLease(1, request);
    }
private:
    std::shared_ptr<Gate> gate_;
    v2::AdaptiveHotSwapController* controller_{nullptr};
    std::optional<ExecutionRequest> request_;
    bool reentry_rejected_{false};
};

class RecordingGenerationAuthority final : public v2::GenerationAuthority {
public:
    std::shared_ptr<const v2::GenerationLease> Issue(
        const v2::GenerationAuthorityRequest& request) override {
        ++issues_;
        last_selection_ = request.selection_plan;
        last_candidate_ = request.candidate;
        return MakeLease(next_++, request);
    }


    size_t issues() const noexcept { return issues_; }
    const kxc::api::PlanVariantKey& last_selection() const noexcept {
        return last_selection_;
    }
    const std::shared_ptr<const production_path::PreparedCandidate>&
    last_candidate() const noexcept {
        return last_candidate_;
    }

private:
    v2::Generation next_{41};
    size_t issues_{0};
    kxc::api::PlanVariantKey last_selection_;
    std::shared_ptr<const production_path::PreparedCandidate> last_candidate_;
};

bool TestV2AuthorityBindingMonotonicityAndCancellationRace() {
    using namespace production_path;
    kxc::api::internal::ClearPrimitiveCacheForTesting();
    const ProductionRequest request = MakeRequest();

    auto replay_authority = std::make_shared<ReplayGenerationAuthority>();
    v2::Options replay_options;
    replay_options.validation_authority = std::make_shared<UniqueReceiptValidation>();
    replay_options.generation_authority = replay_authority;
    v2::AdaptiveHotSwapController replay(std::make_shared<FixtureCompiler>(), replay_options);
    const auto first = replay.CompileAndPublish({request});
    const auto replayed = replay.Submit({request}).Wait();
    TEST_CHECK(!replayed.ready() && replayed.failure.category == v2::FailureCategory::kPermanent &&
                   replay.Acquire(Execute(request)) == first,
               "a returned lease must bind this candidate validation receipt");

    auto scripted_authority = std::make_shared<ScriptedGenerationAuthority>(
        std::vector<v2::Generation>{7, 6, 7, 8});
    v2::Options monotonic_options;
    monotonic_options.generation_authority = scripted_authority;
    v2::AdaptiveHotSwapController monotonic(std::make_shared<FixtureCompiler>(), monotonic_options);
    const auto seven = monotonic.CompileAndPublish({request});
    const auto lower = monotonic.Submit({MakeRequest(2)}).Wait();
    const auto duplicate = monotonic.Submit({MakeRequest(3)}).Wait();
    const auto eight = monotonic.CompileAndPublish(
        {MakeRequest(1, 2, 16, 1,
                     {MakePrimitiveKey(request.config(), 0,
                                       "monotonic-fourth")})});
    TEST_CHECK(seven->generation() == 7 && !lower.ready() && !duplicate.ready() &&
                   eight->generation() == 8 && monotonic.Acquire(Execute(request)) == eight,
               "injected authorities cannot publish decreasing or duplicate generations");

    auto gate = std::make_shared<Gate>();
    auto race_authority = std::make_shared<BlockingGenerationAuthority>(gate);
    v2::Options race_options;
    race_options.generation_authority = race_authority;
    v2::AdaptiveHotSwapController race(std::make_shared<FixtureCompiler>(), race_options);
    race_authority->SetController(&race, Execute(request));
    v2::CancellationSource source;
    const auto ticket = race.Submit({request, std::chrono::steady_clock::time_point::max(), source.token()});
    TEST_CHECK(gate->WaitUntilEntered(), "authority must reach the publication linearization race");
    std::thread cancelling([&] { source.Cancel(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    gate->Release();
    cancelling.join();
    TEST_CHECK(ticket.Wait().failure.category == v2::FailureCategory::kCancelled &&
                   race.Acquire(Execute(request))->generation() == 1 &&
                   race_authority->reentry_rejected(),
               "commit may win only when its publication-mutex linearization precedes Cancel, and Issue reentry must fail fast");
    return true;
}

bool TestV2WaitersCacheEvictionAndOverflow() {
    using namespace production_path;
    kxc::api::internal::ClearPrimitiveCacheForTesting();
    const ProductionRequest request = MakeRequest();
    auto compiler = std::make_shared<FixtureCompiler>();
    compiler->compile_gate = std::make_shared<Gate>();
    v2::Options options;
    options.worker_count = 1;
    options.max_queued_flights = 1;
    options.max_in_flight = 1;
    options.max_waiters_per_flight = 32;
    options.max_discoverable_generations = 1;
    options.max_producer_reported_bytes = 1;
    options.transient_backoff = std::chrono::seconds(1);
    v2::AdaptiveHotSwapController controller(compiler, options);
    v2::CancellationSource cancelled;
    const auto first = controller.Submit({request});
    TEST_CHECK(compiler->compile_gate->WaitUntilEntered(), "v2 worker must enter fixture compiler");
    const auto merged = controller.Submit({request, std::chrono::steady_clock::time_point::max(),
                                           cancelled.token()});
    cancelled.Cancel();
    TEST_CHECK(merged.Wait().failure.category == v2::FailureCategory::kCancelled,
               "one cancelled waiter must not cancel the shared flight");
    std::vector<v2::CompileTicket> stress;
    std::mutex stress_mutex;
    std::vector<std::thread> submitters;
    for (int index = 0; index < 12; ++index) {
        submitters.emplace_back([&] {
            auto ticket = controller.Submit({request});
            std::lock_guard<std::mutex> lock(stress_mutex);
            stress.push_back(std::move(ticket));
        });
    }
    for (auto& thread : submitters) thread.join();
    compiler->compile_gate->Release();
    const auto first_result = first.Wait();
    TEST_CHECK(first_result.ready() && first_result.lease->generation() == 1,
               "v2 must publish its own first monotonic generation");
    for (const auto& ticket : stress) {
        TEST_CHECK(ticket.Wait().ready(), "bounded same-key stress must fan out one flight");
    }
    TEST_CHECK(compiler->calls.load() == 1,
               "v2 must singleflight bounded same-key waiters");
    const auto expired = controller.Submit({request, std::chrono::steady_clock::now()});
    TEST_CHECK(expired.Wait().failure.category == v2::FailureCategory::kTimeout,
               "expired waiter must fail without changing a shared flight");
    const auto second = controller.CompileAndPublish({request});
    TEST_CHECK(second->generation() == 2,
               "producer byte/discoverability eviction must retain the new route");
    TEST_CHECK(first_result.lease->candidate() != nullptr,
               "eviction must not revoke an external generation lease");
    TEST_CHECK(controller.Acquire(Execute(request))->generation() == 2,
               "routing must atomically select the published generation");

    kxc::api::internal::ClearPrimitiveCacheForTesting();
    auto failing = std::make_shared<FixtureCompiler>();
    failing->failures_remaining.store(1);
    v2::AdaptiveHotSwapController retries(failing, options);
    TEST_CHECK(!retries.Submit({request}).Wait().ready(), "adapter failure must be categorized");
    const int after_failure = failing->calls.load();
    TEST_CHECK(retries.Submit({request}).Wait().failure.category == v2::FailureCategory::kTransient &&
               failing->calls.load() == after_failure,
               "negative cache must suppress retry during TTL");
    v2::AdaptiveHotSwapController fresh_retries(
        std::make_shared<FixtureCompiler>(), options);
    TEST_CHECK(fresh_retries.Submit({request}).Wait().ready(),
               "a fresh controller must not inherit negative-cache state");

    v2::Options overflow = options;
    overflow.initial_generation = std::numeric_limits<v2::Generation>::max();
    auto overflow_compiler = std::make_shared<FixtureCompiler>();
    v2::AdaptiveHotSwapController exhausted(overflow_compiler, overflow);
    TEST_CHECK(exhausted.Submit({request}).Wait().failure.category == v2::FailureCategory::kPermanent,
               "generation exhaustion must fail closed rather than wrap");
    return true;
}

bool TestV2CriticalAuditFixes() {
    using namespace production_path;
    kxc::api::internal::ClearPrimitiveCacheForTesting();
    const ProductionRequest request = MakeRequest();

    // Cancellation must be observed after Wait has entered, not just before it.
    auto blocked_compiler = std::make_shared<FixtureCompiler>();
    blocked_compiler->compile_gate = std::make_shared<Gate>();
    v2::Options blocked_options;
    blocked_options.worker_count = 1;
    v2::AdaptiveHotSwapController blocked(blocked_compiler, blocked_options);
    const auto owner = blocked.Submit({request});
    TEST_CHECK(blocked_compiler->compile_gate->WaitUntilEntered(),
               "v2 cancellation fixture worker must block");
    v2::CancellationSource cancellation;
    const auto waiting = blocked.Submit(
        {request, std::chrono::steady_clock::time_point::max(), cancellation.token()});
    std::atomic<bool> wait_started{false};
    std::promise<v2::CompileResult> cancelled_result;
    auto cancelled_future = cancelled_result.get_future();
    std::thread waiter([&] {
        wait_started.store(true, std::memory_order_release);
        cancelled_result.set_value(waiting.Wait());
    });
    TEST_CHECK(WaitFor([&] { return wait_started.load(std::memory_order_acquire); }),
               "v2 cancellation waiter must start");
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    cancellation.Cancel();
    TEST_CHECK(cancelled_future.wait_for(std::chrono::milliseconds(100)) ==
                   std::future_status::ready &&
                   cancelled_future.get().failure.category == v2::FailureCategory::kCancelled,
               "v2 waiter cancellation must be promptly polled after Wait begins");
    waiter.join();
    blocked_compiler->compile_gate->Release();
    TEST_CHECK(owner.Wait().ready(), "cancelled waiter must not cancel its shared flight");

    // A rejected oversized successor must leave the predecessor routed.
    kxc::api::internal::ClearPrimitiveCacheForTesting();
    const ProductionRequest predecessor_request = MakeRequest(1);
    const ProductionRequest oversized_request = MakeRequest(2);
    auto budget_compiler = std::make_shared<FixtureCompiler>();
    v2::Options budget_options;
    budget_options.max_producer_reported_bytes = 1;
    v2::AdaptiveHotSwapController budget(budget_compiler, budget_options);
    const auto predecessor = budget.CompileAndPublish({predecessor_request});
    kxc::api::internal::ClearPrimitiveCacheForTesting();
    budget_compiler->candidate_primitive_byte_size = 2;
    const auto oversized = budget.Submit({oversized_request}).Wait();
    TEST_CHECK(!oversized.ready() &&
                   oversized.failure.category == v2::FailureCategory::kPermanent &&
                   budget.Acquire(Execute(predecessor_request)) == predecessor,
               "over-budget candidate must be rejected before replacing its predecessor");

    // A first generation has no predecessor, but a verified quarantine still unroutes it.
    kxc::api::internal::ClearPrimitiveCacheForTesting();
    v2::Options quarantine_options;
    quarantine_options.health_authority = std::make_shared<OneShotHealth>(1);
    auto quarantine_compiler = std::make_shared<FixtureCompiler>();
    v2::AdaptiveHotSwapController quarantine(quarantine_compiler, quarantine_options);
    const auto only = quarantine.CompileAndPublish({request});
    TEST_CHECK(quarantine.EvaluateHealth(only) &&
                   Throws([&] { (void)quarantine.Acquire(Execute(request)); }) &&
                   !quarantine.Submit({request}).Wait().ready(),
               "quarantining the first generation must durably unroute and reject it");

    // Destroying the final controller from its worker observer must not self-join or UAF.
    kxc::api::internal::ClearPrimitiveCacheForTesting();
    std::shared_ptr<v2::AdaptiveHotSwapController> released;
    std::weak_ptr<v2::AdaptiveHotSwapController> released_weak;
    std::atomic<bool> observer_released{false};
    v2::Options release_options;
    release_options.worker_count = 1;
    release_options.observer = [&](const v2::Event& event) {
        if (event.kind == v2::EventKind::kPublished &&
            !observer_released.exchange(true, std::memory_order_acq_rel)) {
            released.reset();
        }
    };
    released = std::make_shared<v2::AdaptiveHotSwapController>(
        std::make_shared<FixtureCompiler>(), release_options);
    released_weak = released;
    const auto release_ticket = released->Submit({request});
    TEST_CHECK(release_ticket.Wait().ready() &&
                   WaitFor([&] { return observer_released.load(std::memory_order_acquire); }) &&
                   released_weak.expired(),
               "worker observer may release the final controller safely");
    return true;
}

bool TestV2BoundedFailureAndQuarantineMetadata() {
    using namespace production_path;
    kxc::api::internal::ClearPrimitiveCacheForTesting();
    const ProductionRequest base = MakeRequest();
    const std::vector<kxc::api::PrimitiveArtifactKey> keys =
        SelectedKeys(base);
    const auto distinct_request = [&](int token) {
        kxc::api::CompileConfig config = kxc::api::CompileConfig::Create(
            kxc::BuildTarget(kxc::Device::CPU()), token % 4);
        std::vector<kxc::api::PrimitiveArtifactKey> distinct_keys = keys;
        distinct_keys[0] = MakePrimitiveKey(
            config, 0, "adaptive-v2-metadata-" + std::to_string(token));
        return MakeRequestFromConfig(
            MakeFunction(), config, 2, 16, 1, std::move(distinct_keys));
    };
    std::atomic<uint64_t> negative_evicted_events{0};
    v2::Options transient_options;
    transient_options.worker_count = 1;
    transient_options.max_negative_cache_entries = 3;
    transient_options.max_negative_diagnostic_bytes = 32;
    transient_options.transient_backoff = std::chrono::seconds(10);
    transient_options.observer = [&](const v2::Event& event) {
        if (event.kind == v2::EventKind::kNegativeEvicted) {
            negative_evicted_events.fetch_add(1, std::memory_order_relaxed);
        }
    };
    auto transient_compiler = std::make_shared<FixtureCompiler>();
    transient_compiler->failures_remaining.store(16);
    v2::AdaptiveHotSwapController transient(transient_compiler, transient_options);
    std::vector<ProductionRequest> transient_requests;
    for (int index = 0; index < 8; ++index) {
        transient_requests.push_back(distinct_request(index));
        TEST_CHECK(!transient.Submit({transient_requests.back()}).Wait().ready(),
                   "distinct transient failures must be negative-cached or evicted");
    }
    TEST_CHECK(WaitFor([&] { return negative_evicted_events.load() == 5; }),
               "negative cache must evict transient records at configured capacity");
    const int before_old_retry = transient_compiler->calls.load();
    TEST_CHECK(!transient.Submit({transient_requests.front()}).Wait().ready() &&
                   transient_compiler->calls.load() == before_old_retry + 1,
               "the deterministically evicted transient record must be retried");
    const int before_new_retry = transient_compiler->calls.load();
    TEST_CHECK(!transient.Submit({transient_requests.back()}).Wait().ready() &&
                   transient_compiler->calls.load() == before_new_retry,
               "the newest transient record must remain cached");

    kxc::api::internal::ClearPrimitiveCacheForTesting();
    std::atomic<uint64_t> negative_saturated_events{0};
    v2::Options permanent_options = transient_options;
    permanent_options.max_negative_cache_entries = 3;
    permanent_options.observer = [&](const v2::Event& event) {
        if (event.kind == v2::EventKind::kNegativeCacheSaturated) {
            negative_saturated_events.fetch_add(1, std::memory_order_relaxed);
        }
    };
    auto permanent_compiler = std::make_shared<FixtureCompiler>();
    permanent_compiler->attack = Attack::kWrongSignature;
    v2::AdaptiveHotSwapController permanent(permanent_compiler, permanent_options);
    std::vector<ProductionRequest> permanent_requests;
    for (int index = 0; index < 4; ++index) {
        permanent_requests.push_back(distinct_request(20 + index));
        TEST_CHECK(permanent.Submit({permanent_requests.back()}).Wait().failure.category ==
                       v2::FailureCategory::kPermanent,
                   "distinct permanent failures must fail closed");
    }
    TEST_CHECK(WaitFor([&] { return negative_saturated_events.load() == 1; }),
               "permanent negative-cache saturation must block publication rather than forget a failure");
    const int before_cached_permanent = permanent_compiler->calls.load();
    TEST_CHECK(permanent.Submit({permanent_requests.front()}).Wait().failure.category ==
                       v2::FailureCategory::kPermanent &&
                   permanent_compiler->calls.load() == before_cached_permanent,
               "permanent failures within the configured bound must remain fail-closed");
    const int before_blocked_permanent = permanent_compiler->calls.load();
    TEST_CHECK(permanent.Submit({distinct_request(24)}).Wait().failure.category ==
                       v2::FailureCategory::kPermanent &&
                   permanent_compiler->calls.load() == before_blocked_permanent,
               "saturated permanent failure state must fail closed before compilation");

    kxc::api::internal::ClearPrimitiveCacheForTesting();
    std::atomic<uint64_t> quarantine_saturated_events{0};
    v2::Options quarantine_options;
    quarantine_options.worker_count = 1;
    quarantine_options.max_discoverable_generations = 8;
    quarantine_options.max_producer_reported_bytes = 1024;
    quarantine_options.max_quarantine_tombstones_per_route = 2;
    quarantine_options.health_authority = std::make_shared<EveryGenerationHealth>();
    quarantine_options.observer = [&](const v2::Event& event) {
        if (event.kind == v2::EventKind::kQuarantineSaturated) {
            quarantine_saturated_events.fetch_add(1, std::memory_order_relaxed);
        }
    };
    auto quarantine_compiler = std::make_shared<FixtureCompiler>();
    v2::AdaptiveHotSwapController quarantine(quarantine_compiler, quarantine_options);
    const auto healthy = quarantine.CompileAndPublish({base});
    const ProductionRequest bad_one = distinct_request(100);
    const ProductionRequest bad_two = distinct_request(101);
    const ProductionRequest bad_three = distinct_request(102);
    TEST_CHECK(quarantine.EvaluateHealth(quarantine.CompileAndPublish({bad_one})) &&
                   quarantine.Acquire(Execute(base)) == healthy &&
                   quarantine.Submit({bad_one}).Wait().failure.category ==
                       v2::FailureCategory::kPermanent,
               "a quarantined artifact must never be republished");
    TEST_CHECK(quarantine.EvaluateHealth(quarantine.CompileAndPublish({bad_two})) &&
                   quarantine.EvaluateHealth(quarantine.CompileAndPublish({bad_three})),
               "health authority must drive per-route tombstone saturation");
    TEST_CHECK(WaitFor([&] { return quarantine_saturated_events.load() == 1; }) &&
                   quarantine.Acquire(Execute(base)) == healthy,
               "tombstone saturation must preserve a healthy predecessor and fail closed");
    const int before_blocked_publish = quarantine_compiler->calls.load();
    TEST_CHECK(quarantine.Submit({distinct_request(103)}).Wait().failure.category ==
                       v2::FailureCategory::kPermanent &&
                   quarantine_compiler->calls.load() == before_blocked_publish + 1,
               "a saturated route must reject further publication rather than forget a quarantine");
    TEST_CHECK(Throws([&] {
                   v2::Options invalid;
                   invalid.max_negative_cache_entries = 0;
                   v2::AdaptiveHotSwapController rejected(
                       std::make_shared<FixtureCompiler>(), invalid);
               }) &&
                   Throws([&] {
                       v2::Options invalid;
                       invalid.initial_generation = 0;
                       v2::AdaptiveHotSwapController rejected(
                           std::make_shared<FixtureCompiler>(), invalid);
                   }),
               "metadata bounds and the initial generation must be validated");
    return true;
}

bool TestV2HealthRollbackObserverAndAbi() {
    using namespace production_path;
    kxc::api::internal::ClearPrimitiveCacheForTesting();
    const ProductionRequest request = MakeRequest();
    const ProductionRequest successor = MakeRequest(2);
    TEST_CHECK(successor.dispatch_key() == request.dispatch_key() &&
                   successor.plan_abi() == request.plan_abi(),
               "fixture successor must be exact-route compatible");
    auto compiler = std::make_shared<FixtureCompiler>();
    v2::Options options;
    options.max_discoverable_generations = 4;
    options.max_producer_reported_bytes = 1024;
    v2::AdaptiveHotSwapController controller(compiler, options);
    const auto old_lease = controller.CompileAndPublish({request});
    const auto current = controller.CompileAndPublish({successor});
    TEST_CHECK(current->generation() > old_lease->generation(), "generations must be monotonic");
    v2::Options health_options;
    health_options.max_discoverable_generations = 4;
    health_options.max_producer_reported_bytes = 1024;
    health_options.health_authority = std::make_shared<OneShotHealth>(current->generation());
    auto health_compiler = std::make_shared<FixtureCompiler>();
    v2::AdaptiveHotSwapController health(health_compiler, health_options);
    const auto predecessor = health.CompileAndPublish({request});
    const auto regressed = health.CompileAndPublish({successor});
    TEST_CHECK(health.EvaluateHealth(regressed) &&
               health.Acquire(Execute(request))->generation() == predecessor->generation(),
               "one-shot accepted quarantine must atomically roll future routing back");
    TEST_CHECK(!health.EvaluateHealth(regressed), "consumed health evidence must not replay");
    TEST_CHECK(Throws([&] { (void)controller.Acquire(Execute(MakeRequest(1, 2, 32))); }),
               "different exact PlanAbi must never route a physical/layout contract replacement");

    bool reentry_rejected = false;
    v2::AdaptiveHotSwapController* observed = nullptr;
    v2::Options observed_options;
    observed_options.max_producer_reported_bytes = 1024;
    observed_options.observer = [&](const v2::Event&) {
        reentry_rejected = Throws(
            [&] { (void)observed->Acquire(Execute(request)); });
    };
    auto observed_compiler = std::make_shared<FixtureCompiler>();
    v2::AdaptiveHotSwapController observed_controller(observed_compiler, observed_options);
    observed = &observed_controller;
    TEST_CHECK(observed_controller.Submit({request}).Wait().ready() && reentry_rejected,
               "observer reentry must fail fast and remain isolated");
    return true;
}

bool TestV2DistinctSelectionAndOpaqueLeaseBinding() {
    using namespace production_path;
    using namespace kxc;
    kxc::api::internal::ClearPrimitiveCacheForTesting();
    auto baseline_launcher = std::make_shared<FixtureLauncher>();
    const ProductionRequest request = MakeRequest(1, 2, 16, 1, {}, baseline_launcher);
    auto compiler = std::make_shared<FixtureCompiler>();
    auto authority = std::make_shared<RecordingGenerationAuthority>();
    v2::Options options;
    options.generation_authority = authority;
    v2::AdaptiveHotSwapController controller(compiler, options);

    const auto baseline = controller.CompileAndPublish({request});
    controller.RunAsync(Execute(request), {runtime::NDArray::Zeros({2}, Float32(), Device::CPU())},
                        DeviceStream::Default(Device::CPU())).completion.Wait();
    compiler->attack = Attack::kDistinctSelection;
    compiler->candidate_primitive_byte_size = 2;
    compiler->candidate_primitive_provenance = "distinct-producer";
    const auto selected = controller.CompileAndPublish({request});
    const auto selected_launcher = compiler->LastLauncher();
    controller.RunAsync(Execute(request), {runtime::NDArray::Zeros({2}, Float32(), Device::CPU())},
                        DeviceStream::Default(Device::CPU())).completion.Wait();
    baseline->session()->RunAsync(
        {runtime::NDArray::Zeros({2}, Float32(), Device::CPU())},
        DeviceStream::Default(Device::CPU())).completion.Wait();

    TEST_CHECK(request.dispatch_key() == baseline->dispatch_key() &&
                   request.dispatch_key() == selected->dispatch_key() &&
                   request.plan_abi() == baseline->plan_abi() &&
                   request.plan_abi() == selected->plan_abi() &&
                   baseline->generation() < selected->generation(),
               "same callable ABI may publish distinct selected generations");
    TEST_CHECK(baseline->selection_plan_key() != selected->selection_plan_key() &&
                   selected_launcher && selected_launcher != baseline_launcher &&
                   baseline_launcher->launches.load() == 2 &&
                   selected_launcher->launches.load() == 1,
               "N and N+1 must retain and independently execute distinct launchers");
    TEST_CHECK(authority->issues() == 2 && authority->last_candidate() &&
                   authority->last_selection() == selected->selection_plan_key() &&
                   selected->candidate() == authority->last_candidate(),
               "only the authority may bind the opaque generation lease to its prepared selection");
    return true;
}

bool TestV2TransactionalCancellationAndGlobalBounds() {
    using namespace production_path;
    kxc::api::internal::ClearPrimitiveCacheForTesting();
    const ProductionRequest base = MakeRequest();
    const ProductionRequest successor = MakeRequest(2);
    v2::Options transactional_options;
    transactional_options.worker_count = 1;
    transactional_options.max_producer_reported_bytes = 1;
    auto transactional_compiler = std::make_shared<FixtureCompiler>();
    v2::AdaptiveHotSwapController transactional(transactional_compiler,
                                                 transactional_options);
    const auto predecessor = transactional.CompileAndPublish({base});
    transactional_compiler->attack = Attack::kDistinctSelection;
    transactional_compiler->candidate_primitive_byte_size = 2;
    const auto rejected = transactional.Submit({successor}).Wait();
    TEST_CHECK(!rejected.ready() &&
                   transactional.Acquire(Execute(base)) == predecessor,
               "a failed pre-publication candidate must preserve the routed generation");

    kxc::api::internal::ClearPrimitiveCacheForTesting();
    auto gated = std::make_shared<FixtureCompiler>();
    gated->compile_gate = std::make_shared<Gate>();
    v2::Options cancellation_options;
    cancellation_options.worker_count = 1;
    v2::AdaptiveHotSwapController cancellation(gated, cancellation_options);
    const auto owner = cancellation.Submit({base});
    TEST_CHECK(gated->compile_gate->WaitUntilEntered(), "owner must occupy the only worker");
    v2::CancellationSource queued_cancel;
    const auto queued = cancellation.Submit(
        {successor, std::chrono::steady_clock::time_point::max(), queued_cancel.token()});
    queued_cancel.Cancel();  // No Wait: controller owns/rechecks this demand.
    gated->compile_gate->Release();
    TEST_CHECK(owner.Wait().ready() &&
                   queued.Wait().failure.category == v2::FailureCategory::kCancelled &&
                   gated->calls.load() == 1,
               "all-cancelled queued flight must not enter the backend");

    kxc::api::internal::ClearPrimitiveCacheForTesting();
    auto inflight_compiler = std::make_shared<FixtureCompiler>();
    inflight_compiler->compile_gate = std::make_shared<Gate>();
    v2::AdaptiveHotSwapController inflight(inflight_compiler, cancellation_options);
    v2::CancellationSource inflight_cancel;
    const auto inflight_ticket = inflight.Submit(
        {base, std::chrono::steady_clock::time_point::max(), inflight_cancel.token()});
    TEST_CHECK(inflight_compiler->compile_gate->WaitUntilEntered(), "in-flight backend must enter");
    inflight_cancel.Cancel();  // Backend hard cancel is intentionally unsupported.
    inflight_compiler->compile_gate->Release();
    TEST_CHECK(inflight_ticket.Wait().failure.category == v2::FailureCategory::kCancelled &&
                   Throws([&] { (void)inflight.Acquire(Execute(base)); }),
               "all-cancelled in-flight work may finish but must not publish");

    kxc::api::internal::ClearPrimitiveCacheForTesting();
    std::atomic<int> route_events{0};
    v2::Options global_options;
    global_options.worker_count = 1;
    global_options.max_routes = 1;
    const auto framed_size = [](const std::string& value) {
        return value.size() + std::to_string(value.size()).size() + 2U;
    };
    global_options.max_route_metadata_bytes =
        framed_size(base.dispatch_key().canonical_bytes()) +
        framed_size(base.plan_abi().canonical_bytes()) + 64U;
    global_options.observer = [&](const v2::Event& event) {
        if (event.kind == v2::EventKind::kRouteSaturated) ++route_events;
    };
    auto global_compiler = std::make_shared<FixtureCompiler>();
    v2::AdaptiveHotSwapController global(global_compiler, global_options);
    (void)global.CompileAndPublish({base});
    global_compiler->candidate_output_alignment = 32;
    const kxc::api::CompileConfig other_abi_config =
        kxc::api::CompileConfig::Create(kxc::BuildTarget(kxc::Device::CPU()), 1);
    const ProductionRequest other_abi = MakeRequestFromConfig(
        MakeFunction(), other_abi_config, 2, 32, 1,
        {MakePrimitiveKey(other_abi_config, 0, "global-other-abi")});
    TEST_CHECK(other_abi.plan_abi() != base.plan_abi(),
               "global saturation fixture must identify a distinct exact route");
    const auto route_rejection = global.Submit({other_abi}).Wait();
    TEST_CHECK(!route_rejection.ready() &&
                   WaitFor([&] { return route_events.load() == 1; }) &&
                   global.Acquire(Execute(base)) != nullptr,
               "global route count saturation must fail closed and preserve routing");
    return true;
}

bool TestV2LeaseRetainsCandidateAfterEvictionAndControllerDestruction() {
    using namespace kxc;
    using namespace production_path;
    api::internal::ClearPrimitiveCacheForTesting();
    const ProductionRequest request = MakeRequest();
    std::shared_ptr<const v2::GenerationLease> retained;
    std::weak_ptr<const PreparedCandidate> candidate;
    std::weak_ptr<const runtime::RuntimeSession> session;
    {
        v2::Options options;
        options.max_discoverable_generations = 1;
        options.max_producer_reported_bytes = 1;
        v2::AdaptiveHotSwapController controller(
            std::make_shared<FixtureCompiler>(), options);
        retained = controller.CompileAndPublish({request});
        candidate = retained->candidate();
        session = retained->session();
        (void)controller.CompileAndPublish({request});
    }
    api::internal::ClearPrimitiveCacheForTesting();
    {
        auto result = retained->session()->RunAsync(
            {runtime::NDArray::Zeros({2}, Float32(), Device::CPU())},
            DeviceStream::Default(Device::CPU()));
        result.completion.Wait();
        TEST_CHECK(candidate.lock() && session.lock() && result.outputs.size() == 1,
                   "an external lease must retain candidate pins and session after eviction and controller destruction");
    }
    retained.reset();
    TEST_CHECK(candidate.expired() && session.expired(),
               "candidate and session may release after the final lease drops");
    return true;
}

bool TestV2ConcurrentHealthConsumption() {
    using namespace production_path;
    kxc::api::internal::ClearPrimitiveCacheForTesting();
    const ProductionRequest base = MakeRequest();
    const ProductionRequest successor = MakeRequest(2);
    v2::Options options;
    options.health_authority = std::make_shared<OneShotHealth>(2);
    options.max_producer_reported_bytes = 1024;
    v2::AdaptiveHotSwapController controller(std::make_shared<FixtureCompiler>(), options);
    const auto old = controller.CompileAndPublish({base});
    const auto current = controller.CompileAndPublish({successor});
    std::atomic<int> consumed{0};
    std::thread first([&] { if (controller.EvaluateHealth(current)) ++consumed; });
    std::thread second([&] { if (controller.EvaluateHealth(current)) ++consumed; });
    first.join(); second.join();
    TEST_CHECK(consumed.load() == 1 && controller.Acquire(Execute(base)) == old,
               "serialized health consumption must consume a current lease once without stale burn");

    auto reentrant_health = std::make_shared<ReentrantHealth>();
    v2::Options reentrant_options;
    reentrant_options.health_authority = reentrant_health;
    v2::AdaptiveHotSwapController reentrant(std::make_shared<FixtureCompiler>(),
                                             reentrant_options);
    reentrant_health->SetController(&reentrant, Execute(base));
    TEST_CHECK(reentrant.EvaluateHealth(reentrant.CompileAndPublish({base})) &&
                   reentrant_health->reentry_rejected(),
               "health verification reentry must fail fast rather than deadlock");
    return true;
}
#endif

#if KXC_USE_LLVM
bool TestRealCompilerLLVMIntegration() {
    using namespace kxc;
    using namespace kxc::api;
    using namespace production_path;
    internal::ClearPrimitiveCacheForTesting();
    const Function graph = MakeAddFunction();
    const CompileConfig config =
        CompileConfig::Create(BuildTarget(Device::CPU()), 2);
    const CompiledGraph baseline = Compiler::Compile(graph, config);
    const ProductionRequest request(graph, config, baseline);
    const auto candidate = PrepareCandidate(
        request, Compiler::Compile(request.graph(), request.config()),
        "llvm-preparation-receipt");
    auto result = candidate->session()->RunAsync(
        {runtime::NDArray::Zeros({2}, Float32(), Device::CPU()),
         runtime::NDArray::Zeros({2}, Float32(), Device::CPU())},
        DeviceStream::Default(Device::CPU()));
    result.completion.Wait();
    TEST_CHECK(candidate->compiled_graph().artifact_pins().size() ==
                   candidate->compiled_graph().plan().calls().size() &&
                   result.outputs.size() == 1,
               "real Compiler output must pass the preparation structural gate");
    return true;
}
#endif

}  // namespace

int main() {
    std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"request_snapshot_exact_identity", TestRequestSnapshotAndExactIdentity},
        {"prepared_candidate_validation_exact_identity",
         TestPreparedCandidateValidationAndExactIdentity},
        {"malformed_graph_rejected_before_preparation",
         TestMalformedGraphRejectedBeforePreparation},
        {"prepared_candidate_retains_pins_session",
         TestPreparedCandidateRetainsPinsAndSession},
    };
#if KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2
    tests.push_back({"v2_authority_binding_monotonicity_cancellation_race",
                     TestV2AuthorityBindingMonotonicityAndCancellationRace});
    tests.push_back({"v2_waiters_cache_eviction_overflow",
                     TestV2WaitersCacheEvictionAndOverflow});
    tests.push_back({"v2_critical_audit_fixes", TestV2CriticalAuditFixes});
    tests.push_back({"v2_bounded_failure_quarantine_metadata",
                     TestV2BoundedFailureAndQuarantineMetadata});
    tests.push_back({"v2_health_rollback_observer_abi",
                     TestV2HealthRollbackObserverAndAbi});
    tests.push_back({"v2_distinct_selection_opaque_lease",
                     TestV2DistinctSelectionAndOpaqueLeaseBinding});
    tests.push_back({"v2_transactional_cancellation_global_bounds",
                     TestV2TransactionalCancellationAndGlobalBounds});
    tests.push_back({"v2_lease_retains_candidate_after_eviction",
                     TestV2LeaseRetainsCandidateAfterEvictionAndControllerDestruction});
    tests.push_back({"v2_concurrent_health_consumption",
                     TestV2ConcurrentHealthConsumption});
#endif
#if KXC_USE_LLVM
    tests.push_back(
        {"real_compiler_llvm_integration", TestRealCompilerLLVMIntegration});
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
