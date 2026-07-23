/*! \file test/adaptive_production_experimental_test.cpp
 * \brief CPU tests for the default-off production-path experimental adapter.
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
#include "kxc/profiling/profiling.h"
#include "kxc/relay/op.h"
#include "kxc/runtime/compiled_module.h"
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

kxc::api::ArtifactKey MakePrimitiveKey(
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
    const kxc::api::ArtifactKey& key,
    const kxc::codegen::KernelSignature& signature,
    const kxc::codegen::KernelLaunchMetadata& metadata,
    std::shared_ptr<const kxc::codegen::KernelLauncher> launcher,
    uint64_t byte_size = 1) {
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
                           byte_size, "adaptive-production-path-fixture",
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
    std::vector<std::shared_ptr<const kxc::codegen::KernelLauncher>> launchers;
};

kxc::api::CompiledGraph MakeGraph(
    const kxc::api::ArtifactKey& graph_artifact,
    const std::vector<kxc::api::ArtifactKey>& primitive_keys,
    GraphOptions options = {}) {
    using namespace kxc;
    using namespace kxc::codegen;
    if (primitive_keys.empty()) {
        throw std::invalid_argument("fixture graph requires primitive keys");
    }

    std::vector<api::internal::CompiledModuleEntry> entries;
    std::vector<api::ArtifactPin> pins;
    std::vector<api::ArtifactPlanBinding> bindings;
    entries.reserve(primitive_keys.size());
    pins.reserve(primitive_keys.size());
    bindings.reserve(primitive_keys.size());
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
        const api::internal::PrimitiveArtifactPin primitive = PinPrimitive(
            primitive_keys[index], signature, metadata, std::move(launcher),
            options.primitive_byte_size);
        const api::ArtifactPin pin = api::internal::ToArtifactPin(primitive);
        const CompiledKernel module_kernel(
            signature, metadata, primitive.artifact().kernel->launcher);
        entries.push_back(api::internal::CompiledModuleEntry{
            tir::PrimFunc(), signature, metadata, module_kernel});
        pins.push_back(pin);
        bindings.push_back(api::ArtifactPlanBinding{
            index, api::LinkSymbol{Symbol(index)}, pin,
            profiling::HashText(signature.ToString()),
            profiling::HashText(metadata.ToString())});
    }

    api::CompiledModule module = api::internal::BuildCompiledModule(
        BuildTarget(Device::CPU()), std::move(entries), {});
    return api::CompiledGraph{
        std::move(module),
        MakePlan(primitive_keys.size(), options.input_extent),
        std::move(pins), runtime::PlanVariant(), std::move(bindings),
        graph_artifact};
}

ProductionRequest MakeRequestFromConfig(
    kxc::Function graph, const kxc::api::CompileConfig& config,
    int64_t expected_input_extent = 2,
    uint64_t expected_output_alignment = 16,
    size_t call_count = 1,
    std::vector<kxc::api::ArtifactKey> primitive_keys = {},
    std::shared_ptr<FixtureLauncher> baseline_launcher = nullptr) {
    using namespace kxc::api;
    const ArtifactKey graph_artifact =
        Compiler::BuildGraphArtifactKey(graph, config);
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
        MakeGraph(graph_artifact, primitive_keys, std::move(options));
    return ProductionRequest(std::move(graph), config, baseline);
}

ProductionRequest MakeRequest(
    int opt_level = 1, int64_t expected_input_extent = 2,
    uint64_t expected_output_alignment = 16, size_t call_count = 1,
    std::vector<kxc::api::ArtifactKey> primitive_keys = {},
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

std::vector<kxc::api::ArtifactKey> SelectedKeys(
    const ProductionRequest& request) {
    std::vector<kxc::api::ArtifactKey> keys;
    for (const auto& artifact : request.ordered_artifacts()) {
        keys.push_back(artifact.artifact_key);
    }
    return keys;
}

enum class Attack {
    kNone,
    kWrongKeySameLauncher,
    kWrongSignature,
    kWrongMetadata,
    kWrongOrder,
    kCorruptBinding,
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

            std::vector<kxc::api::ArtifactKey> keys = SelectedKeys(request);
            std::vector<std::shared_ptr<const kxc::codegen::KernelLauncher>>
                launchers;
            launchers.reserve(request.verified_artifact_pins().size());
            for (const auto& pin : request.verified_artifact_pins()) {
                launchers.push_back(
                    kxc::api::internal::ProductionArtifactAccess::Pin(pin)
                        .artifact().kernel->launcher);
            }
            if (attack == Attack::kWrongKeySameLauncher) {
                keys[0] = MakePrimitiveKey(request.config(), 0, "wrong-key");
                same_launcher_attack.store(true, std::memory_order_relaxed);
            }

            GraphOptions options;
            options.input_extent = candidate_input_extent;
            options.output_alignment =
                attack == Attack::kWrongSignature ? 32
                                                  : candidate_output_alignment;
            options.foreign_launch_metadata = attack == Attack::kWrongMetadata;
            options.primitive_byte_size = candidate_primitive_byte_size;
            options.launchers = launchers;
            kxc::api::CompiledGraph graph =
                MakeGraph(request.artifact_key(), keys, std::move(options));
            if (attack == Attack::kWrongOrder &&
                graph.artifact_pins.size() >= 2) {
                std::swap(graph.artifact_pins[0], graph.artifact_pins[1]);
                std::swap(graph.artifact_plan_bindings[0],
                          graph.artifact_plan_bindings[1]);
            }
            if (attack == Attack::kCorruptBinding) {
                graph.artifact_plan_bindings[0].signature_digest = "foreign";
            }
            {
                std::lock_guard<std::mutex> lock(launcher_mutex);
                last_launcher = std::dynamic_pointer_cast<const FixtureLauncher>(
                    kxc::api::internal::ProductionArtifactAccess::Pin(
                        graph.artifact_pins[0]).artifact().kernel->launcher);
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
    std::atomic<bool> same_launcher_attack{false};

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

bool TestSameKeySingleflightAndGraphIdentity() {
    using namespace production_path;
    kxc::api::internal::ClearPrimitiveCacheForTesting();
    auto compiler = std::make_shared<FixtureCompiler>();
    compiler->compile_gate = std::make_shared<Gate>();
    AdaptiveController controller(compiler);
    const ProductionRequest request = MakeRequest();
    std::shared_ptr<const FrozenPlanVariant> first;
    std::shared_ptr<const FrozenPlanVariant> second;
    std::exception_ptr first_error;
    std::exception_ptr second_error;
    std::thread owner([&] {
        try {
            first = controller.CompileAndPublish(request);
        } catch (...) {
            first_error = std::current_exception();
        }
    });
    const bool started = compiler->compile_gate->WaitUntilEntered();
    std::thread waiter([&] {
        try {
            second = controller.CompileAndPublish(request);
        } catch (...) {
            second_error = std::current_exception();
        }
    });
    const bool merged = WaitFor([&] {
        return controller.Snapshot().merged_compiles == 1;
    });
    compiler->compile_gate->Release();
    owner.join();
    waiter.join();
    if (first_error) std::rethrow_exception(first_error);
    if (second_error) std::rethrow_exception(second_error);

    const auto snapshot = controller.Snapshot();
    TEST_CHECK(started && merged && first && first == second &&
                   first->generation() == 1,
               "same exact in-flight request must share one generation");
    TEST_CHECK(compiler->calls.load() == 1 && snapshot.published == 1 &&
                   snapshot.merged_compiles == 1 &&
                   snapshot.in_flight_compiles == 0,
               "same-key compilation must singleflight and publish atomically");
    TEST_CHECK(request.ordered_artifacts()[0].artifact_key !=
                   request.artifact_key(),
               "fixture must use a primitive key, never the whole-graph key");
    TEST_CHECK(controller.Acquire(Execute(request)) == first,
               "exact acquire must return the published frozen variant");
    TEST_CHECK(kxc::api::Compiler::BuildGraphArtifactKey(
                   MakeAddFunction(), request.config()) !=
                   request.artifact_key(),
               "different graphs must not share an artifact identity");
    TEST_CHECK(kxc::api::Compiler::BuildGraphArtifactKey(
                   MakeConstantFunction(1.0F), request.config()) !=
                   kxc::api::Compiler::BuildGraphArtifactKey(
                       MakeConstantFunction(2.0F), request.config()),
               "graph identity must include constant payload bytes");
    return true;
}

bool TestArtifactAuthorityRejectsAttacks() {
    using namespace production_path;
    const std::vector<std::pair<Attack, const char*>> attacks = {
        {Attack::kWrongKeySameLauncher, "same launcher with wrong primitive key"},
        {Attack::kWrongSignature, "same launcher with wrong signature"},
        {Attack::kWrongMetadata, "same launcher with wrong launch metadata"},
        {Attack::kCorruptBinding, "forged binding digest"},
    };
    for (const auto& [attack, description] : attacks) {
        kxc::api::internal::ClearPrimitiveCacheForTesting();
        const ProductionRequest request = MakeRequest();
        auto compiler = std::make_shared<FixtureCompiler>();
        compiler->attack = attack;
        AdaptiveController controller(compiler);
        TEST_CHECK(Throws([&] {
                       (void)controller.CompileAndPublish(request);
                   }),
                   description);
        TEST_CHECK(controller.Snapshot().published == 0,
                   "an artifact authority failure must never publish");
        if (attack == Attack::kWrongKeySameLauncher) {
            TEST_CHECK(compiler->same_launcher_attack.load(),
                       "wrong-key attack must intentionally reuse the launcher");
        }
    }

    kxc::api::internal::ClearPrimitiveCacheForTesting();
    const ProductionRequest ordered = MakeRequest(1, 2, 16, 2);
    auto compiler = std::make_shared<FixtureCompiler>();
    compiler->attack = Attack::kWrongOrder;
    AdaptiveController controller(compiler);
    TEST_CHECK(Throws([&] {
                   (void)controller.CompileAndPublish(ordered);
               }),
               "wrong ordered call-to-artifact mapping must fail closed");
    return true;
}

bool TestConfigSnapshotAndUnknownRelayFailClosed() {
    using namespace kxc;
    using namespace production_path;
    api::internal::ClearPrimitiveCacheForTesting();
    const Function graph = MakeFunction();
    api::CompileConfig original = api::CompileConfig::Create(
        BuildTarget(Device::CPU()), 1);
    original->profile_options.enabled = false;
    original->profile_options.bundle_dir = "before";
    const Target original_target = original->target;
    const ProductionRequest request =
        MakeRequestFromConfig(graph, original);
    const std::string frozen_target =
        api::internal::BuildTargetCapabilityFingerprint(
            request.config()->target);

    original->opt_level = 3;
    original->profile_options.enabled = true;
    original->profile_options.bundle_dir = "after";
    auto* mutable_target =
        const_cast<TargetNode*>(original_target.operator->());
    mutable_target->attrs.arch += "-mutated";
    TEST_CHECK(request.config()->opt_level == 1 &&
                   !request.config()->profile_options.enabled &&
                   request.config()->profile_options.bundle_dir == "before" &&
                   api::internal::BuildTargetCapabilityFingerprint(
                       request.config()->target) == frozen_target,
               "request must deep-freeze opt/profile/Target codegen fields");
    api::CompileConfig exported_copy = request.config();
    exported_copy->opt_level = 2;
    exported_copy->target = BuildTarget(Device::CPU());
    TEST_CHECK(request.config()->opt_level == 1,
               "mutable config copies returned to adapters must not alias the request");

    std::atomic<bool> readers_ok{true};
    std::thread mutator([&] {
        for (int index = 0; index < 2000; ++index) {
            original->opt_level = index % 2 == 0 ? 2 : 3;
            original->profile_options.enabled = index % 2 == 0;
            mutable_target->attrs.max_clock_rate_khz = index;
        }
    });
    std::vector<std::thread> readers;
    for (int thread_index = 0; thread_index < 4; ++thread_index) {
        readers.emplace_back([&] {
            for (int index = 0; index < 2000; ++index) {
                if (request.config()->opt_level != 1 ||
                    request.config()->profile_options.enabled ||
                    api::internal::BuildTargetCapabilityFingerprint(
                        request.config()->target) != frozen_target) {
                    readers_ok.store(false, std::memory_order_relaxed);
                }
            }
        });
    }
    mutator.join();
    for (auto& reader : readers) reader.join();
    TEST_CHECK(readers_ok.load(),
               "frozen request reads must not alias concurrent source mutation");

    auto compiler = std::make_shared<FixtureCompiler>();
    AdaptiveController controller(compiler);
    (void)controller.CompileAndPublish(request);
    TEST_CHECK(compiler->observed_opt_level.load() == 1,
               "adapter must consume the same frozen config snapshot");

    const Expr unknown(new RelayNode());
    const Function unknown_graph({}, unknown);
    TEST_CHECK(Throws([&] {
                   (void)api::Compiler::BuildGraphArtifactKey(
                       unknown_graph, request.config());
               }),
               "unknown Relay semantic identity must fail closed");
    return true;
}

enum class ObserverChildResult {
    kSucceeded,
    kLogicError,
    kUnexpectedError,
};

template <typename Callback>
ObserverChildResult InvokeObserverChild(Callback&& callback) noexcept {
    try {
        callback();
        return ObserverChildResult::kSucceeded;
    } catch (const std::logic_error&) {
        return ObserverChildResult::kLogicError;
    } catch (...) {
        return ObserverChildResult::kUnexpectedError;
    }
}

bool TestCrossThreadObserverWindowFailsFast() {
    using namespace production_path;
    constexpr auto callback_timeout = std::chrono::seconds(2);
    constexpr auto cleanup_timeout = std::chrono::seconds(5);

    kxc::api::internal::ClearPrimitiveCacheForTesting();
    const ProductionRequest request = MakeRequest();
    auto compiler = std::make_shared<FixtureCompiler>();
    AdaptiveController* controller_ptr = nullptr;
    std::atomic<bool> child_started{false};
    std::promise<ObserverChildResult> child_promise;
    std::future<ObserverChildResult> child_result = child_promise.get_future();
    std::thread child;
    bool joined_in_callback = false;
    bool callback_timed_out = false;

    AdaptiveControllerOptions options;
    options.observer = [&](const AdaptiveControllerEvent& event) {
        if (event.kind != AdaptiveControllerEventKind::kCompileStarted ||
            child_started.exchange(true)) {
            return;
        }
        child = std::thread([&] {
            child_promise.set_value(InvokeObserverChild([&] {
                (void)controller_ptr->CompileAndPublish(request);
            }));
        });
        if (child_result.wait_for(callback_timeout) ==
            std::future_status::ready) {
            child.join();
            joined_in_callback = true;
        } else {
            // Let the owner settle its flight so the pre-fix implementation
            // also terminates and reports a bounded regression instead of hanging.
            callback_timed_out = true;
        }
    };
    AdaptiveController controller(compiler, std::move(options));
    controller_ptr = &controller;
    std::exception_ptr owner_error;
    std::shared_ptr<const FrozenPlanVariant> variant;
    try {
        variant = controller.CompileAndPublish(request);
    } catch (...) {
        owner_error = std::current_exception();
    }
    const bool child_completed =
        child_result.wait_for(cleanup_timeout) == std::future_status::ready;
    if (!child_completed) {
        std::cerr << "[FAIL] " << __FUNCTION__
                  << ": observer child did not terminate within cleanup bound\n";
        std::abort();
    }
    if (child.joinable()) child.join();
    const ObserverChildResult same_key_result = child_result.get();
    if (owner_error) std::rethrow_exception(owner_error);

    TEST_CHECK(child_started && joined_in_callback && !callback_timed_out &&
                   same_key_result == ObserverChildResult::kLogicError,
               "callback-spawned same-key compile must fail fast before join");
    TEST_CHECK(variant && compiler->calls.load() == 1 &&
                   controller.Snapshot().compile_requests == 1 &&
                   controller.Snapshot().published == 1,
               "rejected callback child must not merge, compile, or publish");

    kxc::api::internal::ClearPrimitiveCacheForTesting();
    const ProductionRequest snapshot_request = MakeRequest();
    auto snapshot_compiler = std::make_shared<FixtureCompiler>();
    AdaptiveController other_controller;
    AdaptiveController* observed_ptr = nullptr;
    std::atomic<bool> snapshot_child_started{false};
    struct ApiResults final {
        ObserverChildResult observed{ObserverChildResult::kUnexpectedError};
        ObserverChildResult other{ObserverChildResult::kUnexpectedError};
    };
    std::promise<ApiResults> api_promise;
    std::future<ApiResults> api_result = api_promise.get_future();
    std::thread api_child;
    bool api_joined_in_callback = false;
    bool api_callback_timed_out = false;

    AdaptiveControllerOptions snapshot_options;
    snapshot_options.observer = [&](const AdaptiveControllerEvent& event) {
        if (event.kind != AdaptiveControllerEventKind::kCompileStarted ||
            snapshot_child_started.exchange(true)) {
            return;
        }
        api_child = std::thread([&] {
            ApiResults results;
            results.observed = InvokeObserverChild(
                [&] { (void)observed_ptr->Snapshot(); });
            results.other = InvokeObserverChild(
                [&] { (void)other_controller.Snapshot(); });
            api_promise.set_value(results);
        });
        if (api_result.wait_for(callback_timeout) ==
            std::future_status::ready) {
            api_child.join();
            api_joined_in_callback = true;
        } else {
            api_callback_timed_out = true;
        }
    };
    AdaptiveController observed(snapshot_compiler,
                                std::move(snapshot_options));
    observed_ptr = &observed;
    std::exception_ptr snapshot_owner_error;
    try {
        (void)observed.CompileAndPublish(snapshot_request);
    } catch (...) {
        snapshot_owner_error = std::current_exception();
    }
    const bool api_child_completed =
        api_result.wait_for(cleanup_timeout) == std::future_status::ready;
    if (!api_child_completed) {
        std::cerr << "[FAIL] " << __FUNCTION__
                  << ": observer API child did not terminate within cleanup bound\n";
        std::abort();
    }
    if (api_child.joinable()) api_child.join();
    const ApiResults api_results = api_result.get();
    if (snapshot_owner_error) std::rethrow_exception(snapshot_owner_error);

    TEST_CHECK(snapshot_child_started && api_joined_in_callback &&
                   !api_callback_timed_out &&
                   api_results.observed == ObserverChildResult::kLogicError,
               "callback-spawned different API must fail fast before join");
    TEST_CHECK(api_results.other == ObserverChildResult::kSucceeded,
               "one controller callback window must not block another controller");
    return true;
}

bool TestObserverReentryFailsFastAndThrowsAreIsolated() {
    using namespace kxc;
    using namespace production_path;
    api::internal::ClearPrimitiveCacheForTesting();
    const ProductionRequest old_request = MakeRequest(1);
    const ProductionRequest new_request =
        MakeRequest(2, 2, 16, 1, SelectedKeys(old_request));
    const ExecutionRequest execution = Execute(old_request);
    auto compiler = std::make_shared<FixtureCompiler>();
    AdaptiveController* controller_ptr = nullptr;
    std::unique_ptr<AdministrativeQuarantineRequest> quarantine;
    std::atomic<bool> compile_checked{false};
    std::atomic<bool> snapshot_checked{false};
    std::atomic<bool> acquire_checked{false};
    std::atomic<bool> run_checked{false};
    std::atomic<bool> rollback_checked{false};
    std::atomic<int> fail_fast_count{0};

    AdaptiveControllerOptions options;
    options.observer = [&](const AdaptiveControllerEvent& event) {
        auto expect_fail_fast = [&](const std::function<void()>& callback) {
            if (Throws(callback)) {
                fail_fast_count.fetch_add(1, std::memory_order_relaxed);
            }
        };
        bool expected = false;
        if (event.kind == AdaptiveControllerEventKind::kCompileStarted &&
            compile_checked.compare_exchange_strong(expected, true)) {
            expect_fail_fast([&] {
                (void)controller_ptr->CompileAndPublish(old_request);
            });
        }
        expected = false;
        if (event.kind == AdaptiveControllerEventKind::kValidated &&
            snapshot_checked.compare_exchange_strong(expected, true)) {
            expect_fail_fast([&] {
                (void)controller_ptr->Snapshot();
            });
        }
        expected = false;
        if (event.kind == AdaptiveControllerEventKind::kPublished &&
            acquire_checked.compare_exchange_strong(expected, true)) {
            expect_fail_fast([&] {
                (void)controller_ptr->Acquire(execution);
            });
        }
        expected = false;
        if (event.kind == AdaptiveControllerEventKind::kAcquired &&
            run_checked.compare_exchange_strong(expected, true)) {
            expect_fail_fast([&] {
                (void)controller_ptr->RunAsync(
                    execution,
                    {runtime::NDArray::Zeros({2}, Float32(), Device::CPU())},
                    DeviceStream::Default(Device::CPU()));
            });
        }
        expected = false;
        if (event.kind == AdaptiveControllerEventKind::kQuarantined &&
            rollback_checked.compare_exchange_strong(expected, true)) {
            expect_fail_fast([&] {
                (void)controller_ptr->RollbackAdministrative(*quarantine);
            });
        }
        throw std::runtime_error("observer failure must be ignored");
    };
    AdaptiveController controller(compiler, std::move(options));
    controller_ptr = &controller;
    const auto old = controller.CompileAndPublish(old_request);
    (void)controller.Acquire(execution);
    auto run = controller.RunAsync(
        execution,
        {runtime::NDArray::Zeros({2}, Float32(), Device::CPU())},
        DeviceStream::Default(Device::CPU()));
    run.completion.Wait();
    const auto regressed = controller.CompileAndPublish(new_request);
    quarantine = std::make_unique<AdministrativeQuarantineRequest>(
        AdministrativeQuarantineRequest::ForTrustedControlPlane(
            regressed, "trusted-control-plane-regression"));
    const AdaptiveHandoffResult rollback =
        controller.RollbackAdministrative(*quarantine, old->generation());

    TEST_CHECK(rollback.changed && controller.Acquire(execution) == old,
               "observer throws must not alter publication or administrative routing");
    TEST_CHECK(compile_checked && snapshot_checked && acquire_checked &&
                   run_checked && rollback_checked &&
                   fail_fast_count.load() == 5,
               "all same-controller observer API reentry must fail fast");
    return true;
}

bool TestMalformedGraphNeverPublishes() {
    using namespace kxc;
    using namespace production_path;

    api::internal::ClearPrimitiveCacheForTesting();
    ProductionRequest undefined_request = MakeRequest();
    auto* undefined_function =
        const_cast<FunctionNode*>(undefined_request.graph().operator->());
    undefined_function->body = Expr();
    auto undefined_compiler = std::make_shared<FixtureCompiler>();
    AdaptiveController undefined_controller(undefined_compiler);
    TEST_CHECK(Throws([&] {
                   (void)api::Compiler::BuildGraphArtifactKey(
                       undefined_request.graph(), undefined_request.config());
               }),
               "an undefined Expr must not produce a graph artifact key");
    TEST_CHECK(Throws([&] {
                   (void)undefined_controller.CompileAndPublish(
                       undefined_request);
               }) &&
                   undefined_compiler->calls.load() == 0 &&
                   undefined_controller.Snapshot().published == 0,
               "an undefined Expr must be rejected before adapter invocation or publish");

    api::internal::ClearPrimitiveCacheForTesting();
    ProductionRequest derived_request = MakeRequest();
    auto* derived_function =
        const_cast<FunctionNode*>(derived_request.graph().operator->());
    derived_function->body = MakeDerivedCall(derived_function->params[0]);
    auto derived_compiler = std::make_shared<FixtureCompiler>();
    AdaptiveController derived_controller(derived_compiler);
    TEST_CHECK(Throws([&] {
                   (void)api::Compiler::BuildGraphArtifactKey(
                       derived_request.graph(), derived_request.config());
               }),
               "a derived Call must not produce a graph artifact key");
    TEST_CHECK(Throws([&] {
                   (void)derived_controller.CompileAndPublish(derived_request);
               }) &&
                   derived_compiler->calls.load() == 0 &&
                   derived_controller.Snapshot().published == 0,
               "a derived Call must be rejected before adapter invocation or publish");
    return true;
}

bool TestSameFlightFailureFansOutAndRetry() {
    using namespace production_path;
    kxc::api::internal::ClearPrimitiveCacheForTesting();
    const ProductionRequest request = MakeRequest();
    auto compiler = std::make_shared<FixtureCompiler>();
    compiler->compile_gate = std::make_shared<Gate>();
    compiler->failures_remaining.store(1);
    AdaptiveController controller(compiler);
    std::exception_ptr owner_error;
    std::exception_ptr waiter_error;
    std::thread owner([&] {
        try {
            (void)controller.CompileAndPublish(request);
        } catch (...) {
            owner_error = std::current_exception();
        }
    });
    const bool started = compiler->compile_gate->WaitUntilEntered();
    std::thread waiter([&] {
        try {
            (void)controller.CompileAndPublish(request);
        } catch (...) {
            waiter_error = std::current_exception();
        }
    });
    const bool merged = WaitFor([&] {
        return controller.Snapshot().merged_compiles == 1;
    });
    compiler->compile_gate->Release();
    owner.join();
    waiter.join();
    TEST_CHECK(started && merged &&
                   ExceptionMessage(owner_error) == "injected adapter failure" &&
                   ExceptionMessage(waiter_error) == "injected adapter failure",
               "one adapter failure must reach every same-flight waiter");
    TEST_CHECK(controller.Snapshot().in_flight_compiles == 0 &&
                   controller.Snapshot().published == 0,
               "failed flight must be removed before retry");
    compiler->compile_gate.reset();
    const auto retry = controller.CompileAndPublish(request);
    TEST_CHECK(retry && retry->generation() == 1 && compiler->calls.load() == 2 &&
                   controller.Snapshot().published == 1,
               "controlled same-key retry must acquire a fresh flight");
    return true;
}

bool TestDifferentKeyParallelismAndBackpressure() {
    using namespace production_path;
    kxc::api::internal::ClearPrimitiveCacheForTesting();
    const ProductionRequest first_request = MakeRequest(1);
    const ProductionRequest second_request = MakeRequest(2);
    auto parallel_compiler = std::make_shared<FixtureCompiler>();
    parallel_compiler->compile_gate = std::make_shared<Gate>();
    AdaptiveControllerOptions parallel_options;
    parallel_options.max_in_flight_compiles = 2;
    AdaptiveController parallel(parallel_compiler, parallel_options);
    std::exception_ptr first_error;
    std::exception_ptr second_error;
    std::thread first([&] {
        try {
            (void)parallel.CompileAndPublish(first_request);
        } catch (...) {
            first_error = std::current_exception();
        }
    });
    std::thread second([&] {
        try {
            (void)parallel.CompileAndPublish(second_request);
        } catch (...) {
            second_error = std::current_exception();
        }
    });
    const bool both_entered =
        parallel_compiler->compile_gate->WaitUntilEntered(2);
    parallel_compiler->compile_gate->Release();
    first.join();
    second.join();
    if (first_error) std::rethrow_exception(first_error);
    if (second_error) std::rethrow_exception(second_error);
    TEST_CHECK(both_entered && parallel_compiler->peak_active.load() == 2 &&
                   parallel.Snapshot().published == 2,
               "different full keys must compile concurrently within the bound");

    kxc::api::internal::ClearPrimitiveCacheForTesting();
    const ProductionRequest bounded_first = MakeRequest(1);
    const ProductionRequest bounded_second = MakeRequest(2);
    auto bounded_compiler = std::make_shared<FixtureCompiler>();
    bounded_compiler->compile_gate = std::make_shared<Gate>();
    AdaptiveControllerOptions bounded_options;
    bounded_options.max_in_flight_compiles = 1;
    AdaptiveController bounded(bounded_compiler, bounded_options);
    std::exception_ptr bounded_error;
    std::thread blocked([&] {
        try {
            (void)bounded.CompileAndPublish(bounded_first);
        } catch (...) {
            bounded_error = std::current_exception();
        }
    });
    const bool one_entered = bounded_compiler->compile_gate->WaitUntilEntered();
    TEST_CHECK(one_entered && Throws([&] {
                   (void)bounded.CompileAndPublish(bounded_second);
               }),
               "a different key must observe global in-flight backpressure");
    TEST_CHECK(bounded_compiler->calls.load() == 1 &&
                   bounded.Snapshot().in_flight_compiles == 1,
               "backpressure must reject before invoking the adapter");
    bounded_compiler->compile_gate->Release();
    blocked.join();
    if (bounded_error) std::rethrow_exception(bounded_error);
    bounded_compiler->compile_gate.reset();
    const auto retry = bounded.CompileAndPublish(bounded_second);
    TEST_CHECK(retry && bounded_compiler->calls.load() == 2 &&
                   bounded.Snapshot().in_flight_compiles == 0,
               "backpressured key must be admissible after capacity is released");
    return true;
}

bool TestAdministrativeRollbackAndOldInFlightGeneration() {
    using namespace kxc;
    using namespace production_path;
    api::internal::ClearPrimitiveCacheForTesting();
    auto launch_gate = std::make_shared<Gate>();
    auto gated_launcher = std::make_shared<FixtureLauncher>(launch_gate);
    const ProductionRequest old_request =
        MakeRequest(1, 2, 16, 1, {}, gated_launcher);
    const ProductionRequest new_request =
        MakeRequest(2, 2, 16, 1, SelectedKeys(old_request));
    auto compiler = std::make_shared<FixtureCompiler>();
    AdaptiveController controller(compiler);
    const auto old = controller.CompileAndPublish(old_request);
    AdaptiveRunAsyncResult old_run;
    std::exception_ptr run_error;
    std::thread runner([&] {
        try {
            old_run = controller.RunAsync(
                Execute(old_request),
                {runtime::NDArray::Zeros({2}, Float32(), Device::CPU())},
                DeviceStream::Default(Device::CPU()));
        } catch (...) {
            run_error = std::current_exception();
        }
    });
    const bool launch_started = launch_gate->WaitUntilEntered();

    std::shared_ptr<const FrozenPlanVariant> regressed;
    AdaptiveHandoffResult rollback;
    try {
        regressed = controller.CompileAndPublish(new_request);
        const auto quarantine =
            AdministrativeQuarantineRequest::ForTrustedControlPlane(
                regressed, "trusted-control-plane numeric regression");
        rollback =
            controller.RollbackAdministrative(quarantine, old->generation());
    } catch (...) {
        launch_gate->Release();
        runner.join();
        throw;
    }
    launch_gate->Release();
    runner.join();
    if (run_error) std::rethrow_exception(run_error);

    TEST_CHECK(launch_started && old_run.variant == old &&
                   rollback.changed && rollback.generation == old->generation() &&
                   rollback.predecessor_generation == regressed->generation() &&
                   controller.Acquire(Execute(old_request)) == old,
               "administrative rollback must not change an in-flight generation");
    TEST_CHECK(old_run.variant == old && old_run.artifact_lease.valid() &&
                   old->artifact_lease().valid(),
               "existing strong snapshots must survive quarantine");
    TEST_CHECK(Throws([&] {
                   (void)controller.CompileAndPublish(new_request);
               }),
               "administratively quarantined graph identity must not republish");
    TEST_CHECK(controller.Snapshot().quarantined == 1,
               "administrative quarantine must be durable and observable");
    return true;
}

bool TestCacheClearStillLaunchesAndCompletionRetains() {
    using namespace kxc;
    using namespace production_path;
    api::internal::ClearPrimitiveCacheForTesting();
    AdaptiveRunAsyncResult result;
    std::weak_ptr<const FrozenPlanVariant> weak_variant;
    std::shared_ptr<const FixtureLauncher> launcher;
    {
        const ProductionRequest request = MakeRequest();
        auto compiler = std::make_shared<FixtureCompiler>();
        AdaptiveController controller(compiler);
        weak_variant = controller.CompileAndPublish(request);
        launcher = compiler->LastLauncher();
        TEST_CHECK(launcher != nullptr,
                   "published fixture must expose its typed launcher to the test");
        const int before = launcher->launches.load();
        api::internal::SetPrimitiveCacheLimitsForTesting(
            api::internal::PrimitiveCacheLimits{
                1, 64ULL * 1024ULL * 1024ULL, 1024, 256});
        const ProductionRequest evictor = MakeRequest(2);
        TEST_CHECK(!api::internal::LookupPrimitiveCache(
                        request.ordered_artifacts()[0].artifact_key).defined(),
                   "a second artifact must LRU-evict the published variant key");
        auto evicted_run = controller.RunAsync(
            Execute(request),
            {runtime::NDArray::Zeros({2}, Float32(), Device::CPU())},
            DeviceStream::Default(Device::CPU()));
        evicted_run.completion.Wait();
        TEST_CHECK(launcher->launches.load() == before + 1,
                   "a pinned variant must launch after normal LRU eviction");

        api::internal::ClearPrimitiveCacheForTesting();
        const auto stats = api::internal::GetPrimitiveCacheStats();
        TEST_CHECK(stats.entries == 0 && stats.active_pins == 0,
                   "active_pins intentionally excludes cleared/evicted pinned artifacts");
        result = controller.RunAsync(
            Execute(request),
            {runtime::NDArray::Zeros({2}, Float32(), Device::CPU())},
            DeviceStream::Default(Device::CPU()));
        TEST_CHECK(launcher->launches.load() == before + 2,
                   "a pinned variant must also launch after explicit cache clear");
        TEST_CHECK(result.completion->retained_contexts.size() >= 2,
                   "runtime and adaptive owners must both attach to completion");
        result.variant.reset();
        result.artifact_lease = ArtifactLease();
    }
    TEST_CHECK(!weak_variant.expired(),
               "completion must retain variant/session/pins after controller destruction");
    result.completion.Wait();
    result.completion = AsyncOperation();
    TEST_CHECK(weak_variant.expired(),
               "variant may release after the completion handle is destroyed");
    return true;
}

bool TestStaticExactRuntimeAndSlotBoundaries() {
    using namespace kxc;
    using namespace production_path;

    api::internal::ClearPrimitiveCacheForTesting();
    auto dynamic_compiler = std::make_shared<FixtureCompiler>();
    dynamic_compiler->candidate_input_extent = -1;
    AdaptiveController dynamic_controller(dynamic_compiler);
    TEST_CHECK(Throws([&] {
                   (void)dynamic_controller.CompileAndPublish(MakeRequest());
               }),
               "dynamic candidate must not publish as static exact");

    api::internal::ClearPrimitiveCacheForTesting();
    auto dispatch_compiler = std::make_shared<FixtureCompiler>();
    dispatch_compiler->candidate_input_extent = 3;
    AdaptiveController dispatch_controller(dispatch_compiler);
    TEST_CHECK(Throws([&] {
                   (void)dispatch_controller.CompileAndPublish(MakeRequest());
               }),
               "candidate dispatch must equal the exact request");

    api::internal::ClearPrimitiveCacheForTesting();
    auto compiler = std::make_shared<FixtureCompiler>();
    AdaptiveControllerOptions options;
    options.max_slots = 1;
    AdaptiveController controller(compiler, options);
    const ProductionRequest exact = MakeRequest();
    controller.CompileAndPublish(exact);
    TEST_CHECK(Throws([&] {
                   (void)controller.CompileAndPublish(
                       MakeRequest(1, 2, 32));
               }),
               "new exact ABI slot must respect the global slot bound");
    const int compiled = compiler->calls.load();
    TEST_CHECK(Throws([&] {
                   (void)controller.RunAsync(
                       Execute(exact),
                       {runtime::NDArray::Zeros({3}, Float32(), Device::CPU())},
                       DeviceStream::Default(Device::CPU()));
               }),
               "RunAsync must reject a different input shape");
    TEST_CHECK(compiler->calls.load() == compiled,
               "RunAsync/RuntimeSession must never compile or query a cache");
    TEST_CHECK(controller.Snapshot().discoverable_history_variants == 1,
               "snapshot count must describe controller history only");
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
    TEST_CHECK(compiler->calls.load() == 1 && controller.SnapshotForTesting().merged_waiters >= 12,
               "v2 must singleflight bounded same-key waiters");
    const auto expired = controller.Submit({request, std::chrono::steady_clock::now()});
    TEST_CHECK(expired.Wait().failure.category == v2::FailureCategory::kTimeout,
               "expired waiter must fail without changing a shared flight");
    const auto second = controller.CompileAndPublish({request});
    TEST_CHECK(second->generation() == 2 && controller.SnapshotForTesting().discoverable_generations == 1,
               "producer byte/discoverability eviction must retain only the new route");
    TEST_CHECK(first_result.lease->variant() != nullptr,
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
    retries.ClearNegativeCacheForTesting();
    TEST_CHECK(retries.Submit({request}).Wait().ready(), "cache clear must allow deterministic retry");

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
    const ProductionRequest oversized_request =
        MakeRequest(2, 2, 16, 1, SelectedKeys(predecessor_request));
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

bool TestV2HealthRollbackObserverAndAbi() {
    using namespace production_path;
    kxc::api::internal::ClearPrimitiveCacheForTesting();
    const ProductionRequest request = MakeRequest();
    const ProductionRequest successor =
        MakeRequest(2, 2, 16, 1, SelectedKeys(request));
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
        reentry_rejected = Throws([&] { (void)observed->SnapshotForTesting(); });
    };
    auto observed_compiler = std::make_shared<FixtureCompiler>();
    v2::AdaptiveHotSwapController observed_controller(observed_compiler, observed_options);
    observed = &observed_controller;
    TEST_CHECK(observed_controller.Submit({request}).Wait().ready() && reentry_rejected,
               "observer reentry must fail fast and remain isolated");
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
    AdaptiveController controller;
    const auto variant = controller.CompileAndPublish(request);
    AdaptiveRunAsyncResult result = controller.RunAsync(
        Execute(request),
        {runtime::NDArray::Zeros({2}, Float32(), Device::CPU()),
         runtime::NDArray::Zeros({2}, Float32(), Device::CPU())},
        DeviceStream::Default(Device::CPU()));
    result.completion.Wait();
    TEST_CHECK(variant->compiled_graph().artifact_plan_bindings.size() ==
                   variant->compiled_graph().plan.calls().size() &&
                   result.outputs.size() == 1,
               "real Compiler output must pass the production-path structural gate");
    return true;
}
#endif

}  // namespace

int main() {
    std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"same_key_singleflight_graph_identity",
         TestSameKeySingleflightAndGraphIdentity},
        {"artifact_authority_attacks", TestArtifactAuthorityRejectsAttacks},
        {"config_snapshot_unknown_relay",
         TestConfigSnapshotAndUnknownRelayFailClosed},
        {"cross_thread_observer_window_fail_fast",
         TestCrossThreadObserverWindowFailsFast},
        {"observer_reentry_exception_isolation",
         TestObserverReentryFailsFastAndThrowsAreIsolated},
        {"malformed_graph_never_publishes",
         TestMalformedGraphNeverPublishes},
        {"same_flight_failure_retry", TestSameFlightFailureFansOutAndRetry},
        {"different_key_parallel_backpressure",
         TestDifferentKeyParallelismAndBackpressure},
        {"administrative_rollback_old_snapshot",
         TestAdministrativeRollbackAndOldInFlightGeneration},
        {"cache_clear_launch_completion_retention",
         TestCacheClearStillLaunchesAndCompletionRetains},
        {"static_exact_runtime_slot_boundaries",
         TestStaticExactRuntimeAndSlotBoundaries},
    };
#if KXC_ENABLE_ADAPTIVE_HOT_SWAP_V2
    tests.push_back({"v2_waiters_cache_eviction_overflow",
                     TestV2WaitersCacheEvictionAndOverflow});
    tests.push_back({"v2_critical_audit_fixes", TestV2CriticalAuditFixes});
    tests.push_back({"v2_health_rollback_observer_abi",
                     TestV2HealthRollbackObserverAndAbi});
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
