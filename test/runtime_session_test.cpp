/*! \file test/runtime_session_test.cpp
 * \brief 验证 RuntimeSession 的输入校验、常量绑定、输出分配和异步完成契约。
 */

#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "kxc/runtime/compiled_module.h"
#include "kxc/runtime/control_execution_plan.h"
#include "kxc/runtime/session.h"
#include "../src/runtime/internal/compiled_module_node.h"
#include "../src/runtime/internal/memory_plan.h"

namespace {

static_assert(!std::is_constructible_v<
              kxc::runtime::RuntimeSession, kxc::api::CompiledModule,
              kxc::runtime::ControlExecutionPlan>,
              "RuntimeSession must not accept a control execution plan");

#define TEST_CHECK(condition, message)                                           \
    do {                                                                          \
        if (!(condition)) {                                                        \
            std::cerr << "[FAIL] " << __FUNCTION__ << ": " << (message) << "\n"; \
            return false;                                                         \
        }                                                                         \
    } while (0)

/*! \brief 捕获标准异常，并可选返回错误文本供上下文断言。 */
bool Throws(const std::function<void()>& function,
            std::string* message = nullptr) {
    try {
        function();
    } catch (const std::exception& error) {
        if (message) *message = error.what();
        return true;
    }
    return false;
}

/*! \brief 返回测试统一使用的 float32 DLPack dtype。 */
DLDataType Float32() { return kxc::runtime::DataTypeFromString("float32"); }

/*! \brief 逐维比较 shape 内容，不把 Array 容器节点身份当作值相等。 */
bool SameShape(const kxc::Array<int64_t>& actual,
               const std::vector<int64_t>& expected) {
    if (actual.size() != expected.size()) return false;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) return false;
    }
    return true;
}

/*!
 * \brief 记录 RuntimeSession 最终提交的有序参数，并同步完成 fake kernel。
 *
 * 该 launcher 不复制或重排参数，只用于证明 session 组装出的数组已经通过
 * CompiledModule 公共校验，并且 completion 保活全部参数 Storage。
 */
class RecordingLauncher final : public kxc::codegen::KernelLauncher {
public:
    /*! \brief 允许测试在 module 组装后模拟 executable 失效。 */
    explicit RecordingLauncher(bool is_ready = true) : ready(is_ready) {}

    /*! \brief fake executable 始终可启动。 */
    bool IsReady() const noexcept override { return ready; }

    /*! \brief 保存参数快照并返回保活所有 Storage 的同步完成句柄。 */
    kxc::AsyncOperation Launch(
        const kxc::Array<kxc::runtime::NDArray>& arguments,
        const kxc::DeviceStream& stream,
        const kxc::ObjectRef& executable_owner) const override {
        ++calls;
        if (fail_on_call > 0 && calls == fail_on_call) {
            throw std::runtime_error("recording launcher injected failure");
        }
        last_arguments = arguments;
        saw_compiled_kernel_owner =
            executable_owner.As<kxc::codegen::CompiledKernelNode>() != nullptr;
        kxc::Array<kxc::Storage> retained;
        for (const auto& argument : arguments) {
            retained.push_back(argument.storage());
        }
        return kxc::AsyncOperation::Completed(stream, std::move(retained));
    }

    /*! \brief 当前 fake executable 是否允许启动。 */
    bool ready{true};
    /*! \brief 后端实际收到的 launch 次数。 */
    mutable int calls{0};
    int fail_on_call{0};
    /*! \brief 记录 launcher 是否收到当前 CompiledKernel owner。 */
    mutable bool saw_compiled_kernel_owner{false};
    /*! \brief 最近一次调用按 KernelSignature 顺序保存的参数。 */
    mutable kxc::Array<kxc::runtime::NDArray> last_arguments;
};

/*! \brief 不保存参数快照的线程安全 launcher，用于验证并发参数装配。 */
class ConcurrentLauncher final : public kxc::codegen::KernelLauncher {
public:
    /*! \brief 并发测试 executable 始终 ready。 */
    bool IsReady() const noexcept override { return true; }

    /*! \brief 原子记录调用，将同次 launch 的 input 复制到 output。 */
    kxc::AsyncOperation Launch(
        const kxc::Array<kxc::runtime::NDArray>& arguments,
        const kxc::DeviceStream& stream,
        const kxc::ObjectRef&) const override {
        if (arguments.size() != 2) {
            throw std::invalid_argument(
                "concurrent launcher expects one input and one output");
        }
        calls.fetch_add(1, std::memory_order_relaxed);
        return arguments[1].CopyFromAsync(arguments[0], stream);
    }

    /*! \brief 所有线程累计到达 backend 的次数。 */
    mutable std::atomic<int> calls{0};
};

class StateAccumulatorLauncher final : public kxc::codegen::KernelLauncher {
public:
    explicit StateAccumulatorLauncher(bool delay = false) : delay_(delay) {}

    bool IsReady() const noexcept override { return true; }

    kxc::AsyncOperation Launch(
        const kxc::Array<kxc::runtime::NDArray>& arguments,
        const kxc::DeviceStream& stream,
        const kxc::ObjectRef&) const override {
        if (arguments.size() != 3 ||
            arguments[0].storage().get() != arguments[2].storage().get()) {
            throw std::invalid_argument(
                "state accumulator requires aliased source/output storage");
        }
        const int current =
            active.fetch_add(1, std::memory_order_relaxed) + 1;
        int observed = max_active.load(std::memory_order_relaxed);
        while (observed < current &&
               !max_active.compare_exchange_weak(
                   observed, current, std::memory_order_relaxed)) {
        }
        if (delay_) std::this_thread::sleep_for(std::chrono::milliseconds(2));
        {
            std::lock_guard<std::mutex> lock(update_mutex_);
            std::vector<float> state(4);
            std::vector<float> increment(4);
            arguments[0].CopyToBytes(state.data(), state.size() * sizeof(float));
            arguments[1].CopyToBytes(increment.data(),
                                     increment.size() * sizeof(float));
            for (size_t i = 0; i < state.size(); ++i) state[i] += increment[i];
            arguments[2].CopyFromBytes(state.data(), state.size() * sizeof(float));
        }
        calls.fetch_add(1, std::memory_order_relaxed);
        saw_storage_alias.store(true, std::memory_order_relaxed);
        active.fetch_sub(1, std::memory_order_relaxed);
        kxc::Array<kxc::Storage> retained;
        for (const auto& argument : arguments) {
            retained.push_back(argument.storage());
        }
        return kxc::AsyncOperation::Completed(stream, std::move(retained));
    }

    mutable std::atomic<int> calls{0};
    mutable std::atomic<int> active{0};
    mutable std::atomic<int> max_active{0};
    mutable std::atomic<bool> saw_storage_alias{false};

private:
    bool delay_{false};
    mutable std::mutex update_mutex_;
};

class BinaryElementwiseLauncher final : public kxc::codegen::KernelLauncher {
public:
    explicit BinaryElementwiseLauncher(bool multiply)
        : multiply_(multiply) {}

    bool IsReady() const noexcept override { return true; }

    kxc::AsyncOperation Launch(
        const kxc::Array<kxc::runtime::NDArray>& arguments,
        const kxc::DeviceStream& stream,
        const kxc::ObjectRef&) const override {
        if (arguments.size() != 3) {
            throw std::invalid_argument(
                "binary elementwise launcher expects two inputs and one output");
        }
        std::vector<float> lhs(4);
        std::vector<float> rhs(4);
        std::vector<float> output(4);
        arguments[0].CopyToBytes(lhs.data(), lhs.size() * sizeof(float));
        arguments[1].CopyToBytes(rhs.data(), rhs.size() * sizeof(float));
        for (size_t i = 0; i < output.size(); ++i) {
            output[i] = multiply_ ? lhs[i] * rhs[i] : lhs[i] + rhs[i];
        }
        arguments[2].CopyFromBytes(output.data(),
                                   output.size() * sizeof(float));
        ++calls;
        last_arguments = arguments;
        kxc::Array<kxc::Storage> retained;
        for (const auto& argument : arguments) {
            retained.push_back(argument.storage());
        }
        return kxc::AsyncOperation::Completed(stream, std::move(retained));
    }

    mutable int calls{0};
    mutable kxc::Array<kxc::runtime::NDArray> last_arguments;

private:
    bool multiply_{false};
};

struct DynamicLaunchLog final {
    std::mutex mutex;
    std::vector<std::string> symbols;
};

class DynamicRecordingLauncher final : public kxc::codegen::KernelLauncher {
public:
    DynamicRecordingLauncher(std::string symbol,
                             std::shared_ptr<DynamicLaunchLog> log)
        : symbol_(std::move(symbol)), log_(std::move(log)) {}

    bool IsReady() const noexcept override { return true; }

    kxc::AsyncOperation Launch(
        const kxc::Array<kxc::runtime::NDArray>& arguments,
        const kxc::DeviceStream& stream,
        const kxc::ObjectRef&) const override {
        ++calls;
        last_arguments = arguments;
        {
            std::lock_guard<std::mutex> lock(log_->mutex);
            log_->symbols.push_back(symbol_);
        }
        auto token = std::make_shared<int>(calls);
        last_operation_token = token;
        kxc::Array<kxc::Storage> retained;
        for (const auto& argument : arguments) {
            retained.push_back(argument.storage());
        }
        kxc::AsyncOperation operation =
            kxc::AsyncOperation::Completed(stream, std::move(retained));
        operation.RetainDependencies({}, std::move(token));
        return operation;
    }

    mutable int calls{0};
    mutable kxc::Array<kxc::runtime::NDArray> last_arguments;
    mutable std::weak_ptr<int> last_operation_token;

private:
    std::string symbol_;
    std::shared_ptr<DynamicLaunchLog> log_;
};

std::shared_ptr<const kxc::api::ModuleInvocationContract> Dynamic2DContract(
    size_t input_count, bool shared_inputs = false,
    bool logical_valid_match = true) {
    using namespace kxc::api;
    std::vector<ModuleInputContract> inputs;
    inputs.reserve(input_count);
    for (size_t input_index = 0; input_index < input_count; ++input_index) {
        ModuleInputContract input;
        input.axis_guards.push_back(ModuleAxisGuard{
            0, 2, 8, 2, std::nullopt,
            shared_inputs && input_index == 1
                ? std::optional<ModuleAxisReference>(
                      ModuleAxisReference{0, 0})
                : std::nullopt});
        input.axis_guards.push_back(
            ModuleAxisGuard{1, 4, 4, 1, ModuleExtent{4}, std::nullopt});
        inputs.push_back(std::move(input));
    }
    const ModuleShapeExpr rows = ModuleShapeExpr::InputAxis(0, 0);
    const ModuleShapeExpr columns = ModuleShapeExpr::Const(4);
    ModuleTensorContract output;
    output.logical = {rows, columns};
    output.physical = output.logical;
    output.valid = logical_valid_match
                       ? output.logical
                       : std::vector<ModuleShapeExpr>{
                             ModuleShapeExpr::Const(1), columns};
    output.max_bytes = 8 * 4 * sizeof(float);
    return std::make_shared<ModuleInvocationContract>(
        std::move(inputs), std::vector<ModuleTensorContract>{output},
        std::vector<ModuleRuntimeExtentScalar>{});
}

struct DynamicSessionFixture final {
    kxc::api::CompiledModule module;
    kxc::runtime::ExecutablePlan plan;
    std::shared_ptr<DynamicRecordingLauncher> first;
    std::shared_ptr<DynamicRecordingLauncher> second;
    std::shared_ptr<DynamicLaunchLog> log;
};

DynamicSessionFixture MakeDynamicSessionFixture(
    bool logical_valid_match = true) {
    using namespace kxc;
    using namespace kxc::api;
    using namespace kxc::codegen;
    const Device cpu = Device::CPU();
    const DLDataType dtype = Float32();
    const KernelSignature first_signature(
        "dynamic_add",
        {KernelArgSpec("lhs", KernelArgRole::kInput, dtype, {-1, 4}, cpu, 8),
         KernelArgSpec("rhs", KernelArgRole::kInput, dtype, {-1, 4}, cpu, 8),
         KernelArgSpec("bias", KernelArgRole::kConstant, dtype, {4}, cpu, 8,
                       false, "dynamic.bias"),
         KernelArgSpec("sum", KernelArgRole::kOutput, dtype, {-1, 4}, cpu,
                       64, true)});
    const KernelSignature second_signature(
        "dynamic_relu",
        {KernelArgSpec("input", KernelArgRole::kInput, dtype, {-1, 4}, cpu, 8),
         KernelArgSpec("output", KernelArgRole::kOutput, dtype, {-1, 4}, cpu,
                       32, true)});
    const KernelLaunchMetadata metadata(cpu, CodeGenBackend::kLLVM);
    auto log = std::make_shared<DynamicLaunchLog>();
    auto first =
        std::make_shared<DynamicRecordingLauncher>("dynamic_add", log);
    auto second =
        std::make_shared<DynamicRecordingLauncher>("dynamic_relu", log);
    Map<String, runtime::NDArray> constants;
    constants.Set("dynamic.bias",
                  runtime::NDArray::Zeros({4}, dtype, cpu, 8));
    api::CompiledModule module = api::internal::BuildCompiledModule(
        BuildTarget(cpu),
        {{first_signature, metadata,
          CompiledKernel(first_signature, metadata, first),
          Dynamic2DContract(2, true, logical_valid_match)},
         {second_signature, metadata,
          CompiledKernel(second_signature, metadata, second),
          Dynamic2DContract(1)}},
        std::move(constants));
    runtime::ExecutablePlan plan(
        {runtime::ValueSpec(0, 0, {-1, 4}, dtype, cpu, true),
         runtime::ValueSpec(1, 1, {-1, 4}, dtype, cpu, true),
         runtime::ValueSpec(2, 2, {4}, dtype, cpu, false, true),
         runtime::ValueSpec(3, 3, {-1, 4}, dtype, cpu),
         runtime::ValueSpec(4, 4, {-1, 4}, dtype, cpu, false, false, true)},
        {runtime::KernelCall("dynamic_add", {0, 1, 2}, {3}),
         runtime::KernelCall("dynamic_relu", {3}, {4})},
        {0, 1}, {2}, {4}, {},
        runtime::ExecutablePlanMode::kDynamicFreshOutputV1,
        {{0, 0, 2, 8, 2, std::nullopt},
         {1, 0, 2, 8, 2, runtime::GraphInputAxisReference{0, 0}}});
    return {std::move(module), std::move(plan), std::move(first),
            std::move(second), std::move(log)};
}

kxc::runtime::NDArray DynamicTensor(int64_t rows) {
    return kxc::runtime::NDArray::Zeros(
        {rows, 4}, Float32(), kxc::Device::CPU(), 8);
}

kxc::runtime::ExecutablePlan MakePlan(
    const kxc::codegen::KernelSignature& signature) {
    using namespace kxc;
    Array<runtime::ValueSpec> values;
    Array<int64_t> call_inputs;
    Array<int64_t> call_outputs;
    Array<int64_t> graph_inputs;
    Array<int64_t> constants;
    Array<int64_t> graph_outputs;
    int64_t value_id = 0;
    for (const auto& argument : signature.arguments()) {
        const bool is_input =
            argument->role == codegen::KernelArgRole::kInput;
        const bool is_constant =
            argument->role == codegen::KernelArgRole::kConstant;
        const bool is_output =
            argument->role == codegen::KernelArgRole::kOutput;
        values.push_back(runtime::ValueSpec(
            value_id, value_id, argument.shape(), argument->dtype,
            argument->device, is_input, is_constant, is_output));
        if (is_output) {
            call_outputs.push_back(value_id);
            graph_outputs.push_back(value_id);
        } else {
            call_inputs.push_back(value_id);
            if (is_input) graph_inputs.push_back(value_id);
            if (is_constant) constants.push_back(value_id);
        }
        ++value_id;
    }
    return runtime::ExecutablePlan(
        std::move(values),
        Array<runtime::KernelCall>{runtime::KernelCall(
            signature->symbol, std::move(call_inputs),
            std::move(call_outputs))},
        std::move(graph_inputs), std::move(constants),
        std::move(graph_outputs));
}

/*! \brief 保存 session 测试所需的模块、常量和 fake launcher。 */
struct SessionFixture {
    kxc::api::CompiledModule module;
    kxc::runtime::ExecutablePlan plan;
    kxc::runtime::NDArray constant;
    std::shared_ptr<RecordingLauncher> launcher;
};

/*! \brief 从任意合法签名和常量表构造 CPU fake CompiledModule。 */
kxc::api::CompiledModule MakeModule(
    const kxc::codegen::KernelSignature& signature,
    const kxc::Map<kxc::String, kxc::runtime::NDArray>& constants,
    std::shared_ptr<kxc::codegen::KernelLauncher> launcher) {
    using namespace kxc;
    using namespace kxc::codegen;
    KernelLaunchMetadata metadata(Device::CPU(), CodeGenBackend::kLLVM);
    CompiledKernel executable(signature, metadata, launcher);
    return api::internal::BuildCompiledModule(
        BuildTarget(Device::CPU()),
        {api::internal::CompiledModuleEntry{signature, metadata, executable}},
        constants);
}

/*! \brief 构造 input -> constant -> output 的静态 session fixture。 */
SessionFixture MakeStaticFixture() {
    using namespace kxc;
    using namespace kxc::codegen;
    const Device cpu = Device::CPU();
    KernelSignature signature(
        "session_fixture",
        {KernelArgSpec("input", KernelArgRole::kInput, Float32(), {2, 3}, cpu),
         KernelArgSpec("weight", KernelArgRole::kConstant, Float32(), {3}, cpu,
                       1, false, "relay.constant.0"),
         KernelArgSpec("output", KernelArgRole::kOutput, Float32(), {2, 3}, cpu,
                       64, true)});
    runtime::NDArray constant = runtime::NDArray::Zeros({3}, Float32(), cpu);
    Map<String, runtime::NDArray> constants;
    constants.Set(String("relay.constant.0"), constant);
    auto launcher = std::make_shared<RecordingLauncher>();
    return {MakeModule(signature, constants, launcher), MakePlan(signature),
            constant,
            std::move(launcher)};
}

struct StateFixture {
    kxc::api::CompiledModule module;
    kxc::runtime::ExecutablePlan plan;
    std::shared_ptr<StateAccumulatorLauncher> launcher;
};

kxc::codegen::KernelSignature AccumulatorSignature(
    kxc::String symbol, uint64_t output_alignment) {
    using namespace kxc;
    using namespace kxc::codegen;
    return KernelSignature(
        std::move(symbol),
        {KernelArgSpec("state", KernelArgRole::kInput, Float32(), {4},
                       Device::CPU(), 4),
         KernelArgSpec("increment", KernelArgRole::kInput, Float32(), {4},
                       Device::CPU(), 4),
         KernelArgSpec("next_state", KernelArgRole::kOutput, Float32(), {4},
                       Device::CPU(), output_alignment, true)});
}

kxc::codegen::KernelSignature StateSignature() {
    return AccumulatorSignature("state_accumulate", 64);
}

kxc::runtime::ExecutablePlan MakeStatePlan(
    kxc::Array<int64_t> state_shape = {4}, DLDataType state_dtype = Float32(),
    kxc::Device state_device = kxc::Device::CPU()) {
    using namespace kxc;
    using namespace kxc::runtime;
    return ExecutablePlan(
        {ValueSpec(0, 0, state_shape, state_dtype, state_device, false, false,
                   false, false, false, true),
         ValueSpec(1, 1, {4}, Float32(), Device::CPU(), true),
         ValueSpec(2, 0, std::move(state_shape), state_dtype,
                   std::move(state_device), false, false, true, true, false,
                   false, 0, ValueWriteMode::kInPlace)},
        {KernelCall("state_accumulate", {0, 1}, {2})}, {1}, {}, {2}, {0});
}

kxc::runtime::ExecutablePlan MakeDonationPlan() {
    using namespace kxc;
    using namespace kxc::runtime;
    return ExecutablePlan(
        {ValueSpec(0, 0, {4}, Float32(), Device::CPU(), true),
         ValueSpec(1, 1, {4}, Float32(), Device::CPU(), true),
         ValueSpec(2, 0, {4}, Float32(), Device::CPU(), false, false, true,
                   true, false, false, 0, ValueWriteMode::kInPlace)},
        {KernelCall("state_accumulate", {0, 1}, {2})}, {0, 1}, {}, {2});
}

kxc::runtime::ExecutablePlan MakeDonationChainPlan() {
    using namespace kxc;
    using namespace kxc::runtime;
    return ExecutablePlan(
        {ValueSpec(0, 0, {4}, Float32(), Device::CPU(), true),
         ValueSpec(1, 1, {4}, Float32(), Device::CPU(), true),
         ValueSpec(2, 0, {4}, Float32(), Device::CPU(), false, false, false,
                   true, false, false, 0, ValueWriteMode::kInPlace),
         ValueSpec(3, 0, {4}, Float32(), Device::CPU(), false, false, true,
                   true, false, false, 2, ValueWriteMode::kInPlace)},
        {KernelCall("donate_0", {0, 1}, {2}),
         KernelCall("donate_1", {2, 1}, {3})},
        {0, 1}, {}, {3});
}

StateFixture MakeStateFixture() {
    const kxc::codegen::KernelSignature signature = StateSignature();
    auto launcher = std::make_shared<StateAccumulatorLauncher>();
    return {MakeModule(signature, {}, launcher), MakeStatePlan(),
            std::move(launcher)};
}

/*! \brief 构造器必须拒绝 undefined module 和错误 ObjectRef 节点类型。 */
bool TestConstructionAndTypeChecks() {
    using namespace kxc;
    TEST_CHECK(Throws([] {
                   runtime::RuntimeSession invalid{
                       api::CompiledModule(ObjectRef()),
                       runtime::ExecutablePlan()};
               }),
               "undefined CompiledModule should fail");
    TEST_CHECK(Throws([] {
                   runtime::RuntimeSession invalid{ObjectRef(Device::CPU())};
               }),
               "Device ObjectRef should not become RuntimeSession");
    SessionFixture fixture = MakeStaticFixture();
    runtime::RuntimeSession session(fixture.module, fixture.plan);
    TEST_CHECK(session.defined(), "valid module should create a session");
    fixture.launcher->ready = false;
    TEST_CHECK(Throws([&] {
                   runtime::RuntimeSession invalid(fixture.module,
                                                   fixture.plan);
               }),
               "defined but no longer ready module should fail");
    return true;
}

bool TestConstantRealignmentAtModuleConstruction() {
    using namespace kxc;
    using namespace kxc::codegen;
    const Device cpu = Device::CPU();
    KernelSignature signature(
        "misaligned_constant",
        {KernelArgSpec("weight", KernelArgRole::kConstant, Float32(), {3},
                       cpu, 64, false, "weight"),
         KernelArgSpec("output", KernelArgRole::kOutput, Float32(), {3}, cpu,
                       1, true)});
    runtime::NDArray backing =
        runtime::NDArray::Zeros({4}, Float32(), cpu, 64);
    runtime::NDArray misaligned = backing.CreateView({3}, {1}, sizeof(float));
    Map<String, runtime::NDArray> constants;
    constants.Set("weight", misaligned);
    auto launcher = std::make_shared<RecordingLauncher>();
    const api::CompiledModule module =
        MakeModule(signature, constants, launcher);
    runtime::RuntimeSession session(module, MakePlan(signature));
    const Array<runtime::NDArray> outputs = session.Run({});
    TEST_CHECK(outputs.size() == 1 && launcher->calls == 1 &&
                   launcher->last_arguments.size() == 2 &&
                   launcher->last_arguments[0].get() != misaligned.get() &&
                   launcher->last_arguments[0]->byte_offset == 0 &&
                   launcher->last_arguments[0].storage()->alignment >= 64 &&
                   reinterpret_cast<uintptr_t>(
                       launcher->last_arguments[0].storage().data()) % 64 == 0,
               "BuildCompiledModule must freeze a signature-aligned constant "
               "before RuntimeSession construction");
    return true;
}

/*! \brief Run 必须自动绑定常量、分配输出并保持最终 ABI 参数顺序。 */
bool TestSynchronousAssembly() {
    using namespace kxc;
    SessionFixture fixture = MakeStaticFixture();
    runtime::RuntimeSession session(fixture.module, fixture.plan);
    runtime::NDArray input =
        runtime::NDArray::Zeros({2, 3}, Float32(), Device::CPU());
    Array<runtime::NDArray> outputs = session.Run({input});

    TEST_CHECK(outputs.size() == 1, "session should allocate one output");
    TEST_CHECK(SameShape(outputs[0].shape(), {2, 3}) &&
                   outputs[0].device() == Device::CPU(),
               "allocated output metadata should follow signature");
    TEST_CHECK(outputs[0].storage()->alignment == 64 &&
                   reinterpret_cast<uintptr_t>(outputs[0].storage().data()) % 64 ==
                       0,
               "allocated output should satisfy signature alignment");
    TEST_CHECK(fixture.launcher->calls == 1 &&
                   fixture.launcher->last_arguments.size() == 3,
               "session should launch exactly one three-argument kernel");
    TEST_CHECK(fixture.launcher->last_arguments[0].get() == input.get() &&
                   fixture.launcher->last_arguments[1].get() !=
                       fixture.constant.get() &&
                   fixture.launcher->last_arguments[2].get() == outputs[0].get(),
               "session must preserve ABI order and inject module-owned constants");
    TEST_CHECK(fixture.launcher->saw_compiled_kernel_owner,
               "launch should pass the compiled kernel owner to the launcher");
    return true;
}

/*! \brief 构建后修改源常量不得改变 RuntimeSession 的数值结果。 */
bool TestModuleOwnedConstantExecution() {
    using namespace kxc;
    using namespace kxc::codegen;
    const Device cpu = Device::CPU();
    const Array<int64_t> shape{4};
    const String key("relay.constant.runtime-owned");
    KernelSignature signature(
        "runtime_owned_constant",
        {KernelArgSpec("input", KernelArgRole::kInput, Float32(), shape, cpu),
         KernelArgSpec("weight", KernelArgRole::kConstant, Float32(), shape,
                       cpu, 64, false, key),
         KernelArgSpec("output", KernelArgRole::kOutput, Float32(), shape,
                       cpu, 64, true)});
    KernelLaunchMetadata metadata(cpu, CodeGenBackend::kLLVM);
    auto launcher = std::make_shared<BinaryElementwiseLauncher>(false);
    CompiledKernel executable(signature, metadata, launcher);

    runtime::NDArray source =
        runtime::NDArray::Empty(shape, Float32(), cpu);
    const std::vector<float> original{1, 1, 1, 1};
    source.CopyFromBytes(original.data(), original.size() * sizeof(float));
    Map<String, runtime::NDArray> constants;
    constants.Set(key, source);
    api::CompiledModule module = api::internal::BuildCompiledModule(
        BuildTarget(cpu),
        {api::internal::CompiledModuleEntry{signature, metadata, executable}},
        constants);

    const std::vector<float> mutation{9, 9, 9, 9};
    source.CopyFromBytes(mutation.data(), mutation.size() * sizeof(float));
    runtime::RuntimeSession session(module, MakePlan(signature));
    runtime::NDArray input =
        runtime::NDArray::Empty(shape, Float32(), cpu);
    const std::vector<float> input_values{1, 2, 3, 4};
    input.CopyFromBytes(input_values.data(),
                        input_values.size() * sizeof(float));
    const Array<runtime::NDArray> outputs = session.Run({input});
    std::vector<float> actual(4);
    outputs[0].CopyToBytes(actual.data(), actual.size() * sizeof(float));
    std::vector<float> launched_constant(4);
    launcher->last_arguments[1].CopyToBytes(
        launched_constant.data(), launched_constant.size() * sizeof(float));
    TEST_CHECK(actual == std::vector<float>({2, 3, 4, 5}) &&
                   launched_constant == original &&
                   launcher->last_arguments[1].get() != source.get(),
               "RuntimeSession must execute with the build-time constant snapshot");
    return true;
}

/*! \brief RunAsync 必须返回 outputs 和可独立保活全部参数的 completion。 */
bool TestAsyncResultLifetime() {
    using namespace kxc;
    SessionFixture fixture = MakeStaticFixture();
    runtime::RunAsyncResult result;
    {
        runtime::RuntimeSession session(fixture.module, fixture.plan);
        runtime::NDArray input =
            runtime::NDArray::Zeros({2, 3}, Float32(), Device::CPU());
        result = session.RunAsync(
            {input}, DeviceStream::Default(Device::CPU()));
    }
    TEST_CHECK(result.outputs.size() == 1 && result.completion.IsReady(),
               "async result should expose output and ready completion");
    TEST_CHECK(result.completion->retained_storage.size() == 3,
               "completion should retain input, constant, and output Storage");
    result.completion.Wait();
    return true;
}

bool TestStateAliasPersistenceAndLifetime() {
    using namespace kxc;
    StateFixture fixture = MakeStateFixture();
    TEST_CHECK(fixture.plan.mode() ==
                   runtime::ExecutablePlanMode::kStatic,
               "state/alias plans must retain the compatible static mode");
    AsyncOperation escaped_completion;
    const Object* state_storage = nullptr;
    {
        runtime::RuntimeSession session(fixture.module, fixture.plan);
        runtime::NDArray first_increment = runtime::NDArray::Empty(
            {4}, Float32(), Device::CPU());
        const std::vector<float> first_values{1, 2, 3, 4};
        first_increment.CopyFromBytes(first_values.data(),
                                      first_values.size() * sizeof(float));
        Array<runtime::NDArray> first = session.Run({first_increment});
        std::vector<float> first_actual(4);
        first[0].CopyToBytes(first_actual.data(),
                             first_actual.size() * sizeof(float));
        TEST_CHECK(first_actual == first_values,
                   "session state must be zero-initialized on its first run");

        runtime::NDArray second_increment = runtime::NDArray::Empty(
            {4}, Float32(), Device::CPU());
        const std::vector<float> second_values{10, 20, 30, 40};
        second_increment.CopyFromBytes(second_values.data(),
                                       second_values.size() * sizeof(float));
        runtime::RunAsyncResult second = session.RunAsync(
            {second_increment}, DeviceStream::Default(Device::CPU()));
        second.completion.Wait();
        std::vector<float> second_actual(4);
        second.outputs[0].CopyToBytes(second_actual.data(),
                                      second_actual.size() * sizeof(float));
        TEST_CHECK(second_actual == std::vector<float>({11, 22, 33, 44}) &&
                       first[0].storage().get() ==
                           second.outputs[0].storage().get() &&
                       reinterpret_cast<uintptr_t>(
                           second.outputs[0].storage().data()) % 64 == 0 &&
                       fixture.launcher->calls.load(std::memory_order_relaxed) == 2 &&
                       fixture.launcher->saw_storage_alias.load(
                           std::memory_order_relaxed),
                   "state must persist and use the alias output storage across runs");
        state_storage = second.outputs[0].storage().get();
        escaped_completion = second.completion;
    }

    bool completion_retains_state = false;
    for (const auto& storage : escaped_completion->retained_storage) {
        completion_retains_state =
            completion_retains_state || storage.get() == state_storage;
    }
    TEST_CHECK(escaped_completion.IsReady() && completion_retains_state,
               "completion must keep session-owned state alive after session destruction");

    runtime::RuntimeSession independent(fixture.module, fixture.plan);
    runtime::NDArray increment =
        runtime::NDArray::Empty({4}, Float32(), Device::CPU());
    const std::vector<float> values{1, 1, 1, 1};
    increment.CopyFromBytes(values.data(), values.size() * sizeof(float));
    const Array<runtime::NDArray> fresh = independent.Run({increment});
    std::vector<float> fresh_actual(4);
    fresh[0].CopyToBytes(fresh_actual.data(),
                         fresh_actual.size() * sizeof(float));
    TEST_CHECK(fresh_actual == values &&
                   fresh[0].storage().get() != state_storage,
               "each RuntimeSession must own an independent zeroed state");
    return true;
}

bool TestStateRunAsyncSerialization() {
    using namespace kxc;
    const codegen::KernelSignature signature = StateSignature();
    auto launcher = std::make_shared<StateAccumulatorLauncher>(true);
    runtime::RuntimeSession session(MakeModule(signature, {}, launcher),
                                    MakeStatePlan());
    constexpr int kThreads = 4;
    std::atomic<int> failures{0};
    std::vector<runtime::NDArray> outputs(kThreads);
    std::vector<std::thread> threads;
    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([&, i] {
            try {
                runtime::NDArray increment =
                    runtime::NDArray::Empty({4}, Float32(), Device::CPU());
                const std::vector<float> ones{1, 1, 1, 1};
                increment.CopyFromBytes(ones.data(),
                                        ones.size() * sizeof(float));
                runtime::RunAsyncResult result = session.RunAsync(
                    {increment}, DeviceStream::Default(Device::CPU()));
                result.completion.Wait();
                outputs[i] = result.outputs[0];
            } catch (...) {
                failures.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (auto& thread : threads) thread.join();
    std::vector<float> actual(4);
    if (outputs[0].defined()) {
        outputs[0].CopyToBytes(actual.data(), actual.size() * sizeof(float));
    }
    TEST_CHECK(failures.load(std::memory_order_relaxed) == 0 &&
                   launcher->calls.load(std::memory_order_relaxed) == kThreads &&
                   launcher->max_active.load(std::memory_order_relaxed) == 1 &&
                   actual == std::vector<float>({4, 4, 4, 4}),
               "stateful RunAsync submissions must serialize within one session");
    return true;
}

bool TestInputDonationAliasAndPreflight() {
    using namespace kxc;
    const codegen::KernelSignature signature = StateSignature();
    auto launcher = std::make_shared<StateAccumulatorLauncher>();
    runtime::RuntimeSession session(MakeModule(signature, {}, launcher),
                                    MakeDonationPlan());

    runtime::NDArray source =
        runtime::NDArray::Empty({4}, Float32(), Device::CPU(), 64);
    runtime::NDArray increment =
        runtime::NDArray::Empty({4}, Float32(), Device::CPU());
    const std::vector<float> source_values{1, 2, 3, 4};
    const std::vector<float> increment_values{10, 20, 30, 40};
    source.CopyFromBytes(source_values.data(), source_values.size() * sizeof(float));
    increment.CopyFromBytes(increment_values.data(),
                            increment_values.size() * sizeof(float));
    const Array<runtime::NDArray> outputs = session.Run({source, increment});
    std::vector<float> actual(4);
    outputs[0].CopyToBytes(actual.data(), actual.size() * sizeof(float));
    TEST_CHECK(actual == std::vector<float>({11, 22, 33, 44}) &&
                   outputs[0].storage().get() == source.storage().get() &&
                   launcher->calls.load(std::memory_order_relaxed) == 1,
               "declared input donation must bind the output to source storage");

    runtime::NDArray backing =
        runtime::NDArray::Zeros({5}, Float32(), Device::CPU(), 64);
    runtime::NDArray misaligned =
        backing.CreateView({4}, {1}, sizeof(float));
    TEST_CHECK(Throws([&] {
                   session.RunAsync(
                       {misaligned, increment},
                       DeviceStream::Default(Device::CPU()));
               }) &&
                   launcher->calls.load(std::memory_order_relaxed) == 1,
               "donated input must satisfy alias-output alignment before launch");

    const codegen::KernelSignature first =
        AccumulatorSignature("donate_0", 4);
    const codegen::KernelSignature second =
        AccumulatorSignature("donate_1", 64);
    const codegen::KernelLaunchMetadata metadata(
        Device::CPU(), codegen::CodeGenBackend::kLLVM);
    auto chain_launcher = std::make_shared<StateAccumulatorLauncher>();
    const api::CompiledModule chain_module = api::internal::BuildCompiledModule(
        BuildTarget(Device::CPU()),
        {{first, metadata,
          codegen::CompiledKernel(first, metadata, chain_launcher)},
         {second, metadata,
          codegen::CompiledKernel(second, metadata, chain_launcher)}},
        {});
    runtime::RuntimeSession chain_session(chain_module,
                                          MakeDonationChainPlan());
    TEST_CHECK(Throws([&] {
                   chain_session.RunAsync(
                       {misaligned, increment},
                       DeviceStream::Default(Device::CPU()));
               }) &&
                   chain_launcher->calls.load(std::memory_order_relaxed) == 0,
               "donation-chain alignment must fail before its first launch");
    return true;
}

bool TestStateContractMismatchRejected() {
    using namespace kxc;
    StateFixture fixture = MakeStateFixture();
    TEST_CHECK(Throws([&] {
                   runtime::RuntimeSession invalid(
                       fixture.module, MakeStatePlan({2}, Float32(), Device::CPU()));
               }),
               "state shape mismatch must fail at session construction");
    TEST_CHECK(Throws([&] {
                   runtime::RuntimeSession invalid(
                       fixture.module,
                       MakeStatePlan({4}, runtime::DataTypeFromString("int32"),
                                     Device::CPU()));
               }),
               "state dtype mismatch must fail at session construction");
    TEST_CHECK(Throws([&] {
                   runtime::RuntimeSession invalid(
                       fixture.module,
                       MakeStatePlan({4}, Float32(), Device::CUDA(0)));
               }),
               "state device mismatch must fail before state allocation");
    TEST_CHECK(fixture.launcher->calls.load(std::memory_order_relaxed) == 0,
               "invalid state contracts must never reach the backend");
    return true;
}

/*! \brief 输入数量、定义状态、dtype、rank、shape 和布局必须在 launch 前失败。 */
bool TestInputValidation() {
    using namespace kxc;
    SessionFixture fixture = MakeStaticFixture();
    runtime::RuntimeSession session(fixture.module, fixture.plan);
    const DeviceStream stream = DeviceStream::Default(Device::CPU());
    TEST_CHECK(Throws([&] { session.RunAsync({}, stream); }),
               "missing input should fail");
    TEST_CHECK(Throws([&] {
                   session.RunAsync(
                       {runtime::NDArray::Zeros({2, 3}, Float32(), Device::CPU()),
                        runtime::NDArray::Zeros({2, 3}, Float32(), Device::CPU())},
                       stream);
               }),
               "extra input should fail");
    TEST_CHECK(Throws([&] { session.RunAsync({runtime::NDArray()}, stream); }),
               "undefined input should fail");
    TEST_CHECK(Throws([&] {
                   session.RunAsync(
                       {runtime::NDArray::Zeros(
                           {2, 3}, runtime::DataTypeFromString("int32"),
                           Device::CPU())},
                       stream);
               }),
               "input dtype mismatch should fail");
    TEST_CHECK(Throws([&] {
                   session.RunAsync(
                       {runtime::NDArray::Zeros({6}, Float32(), Device::CPU())},
                       stream);
               }),
               "input rank mismatch should fail");
    TEST_CHECK(Throws([&] {
                   session.RunAsync(
                       {runtime::NDArray::Zeros({2, 4}, Float32(), Device::CPU())},
                       stream);
               }),
               "input shape mismatch should fail");

    runtime::NDArray backing =
        runtime::NDArray::Zeros({2, 3}, Float32(), Device::CPU());
    auto* node = const_cast<runtime::NDArrayNode*>(
        backing.As<runtime::NDArrayNode>());
    node->strides_storage = {4, 1};
    TEST_CHECK(Throws([&] { session.RunAsync({backing}, stream); }),
               "non-contiguous input should fail");
    TEST_CHECK(fixture.launcher->calls == 0,
               "invalid inputs must not reach the backend");
    return true;
}

/*! \brief 显式异步 stream 必须定义且与模块 Device 完全一致。 */
bool TestStreamValidation() {
    using namespace kxc;
    SessionFixture fixture = MakeStaticFixture();
    runtime::RuntimeSession session(fixture.module, fixture.plan);
    runtime::NDArray input =
        runtime::NDArray::Zeros({2, 3}, Float32(), Device::CPU());
    TEST_CHECK(Throws([&] { session.RunAsync({input}, DeviceStream()); }),
               "undefined stream should fail");
    TEST_CHECK(Throws([&] {
                   session.RunAsync(
                       {input}, DeviceStream::Default(Device::CUDA(0)));
               }),
               "stream on another device should fail");
    TEST_CHECK(fixture.launcher->calls == 0,
               "invalid stream must not reach the backend");
    return true;
}

/*!
 * \brief 输入 device 错误必须先于 CUDA 输出分配失败，证明校验顺序无副作用。
 */
bool TestInputDeviceValidationBeforeAllocation() {
    using namespace kxc;
    using namespace kxc::codegen;
    const Device cuda = Device::CUDA(0);
    KernelSignature signature(
        "cuda_device_contract",
        {KernelArgSpec("input", KernelArgRole::kInput, Float32(), {1}, cuda),
         KernelArgSpec("output", KernelArgRole::kOutput, Float32(), {1}, cuda,
                       1, true)});
    KernelLaunchMetadata metadata(cuda, CodeGenBackend::kCUDA);
    auto launcher = std::make_shared<RecordingLauncher>();
    CompiledKernel executable(signature, metadata, launcher);
    api::CompiledModule module = api::internal::BuildCompiledModule(
        BuildTarget(cuda),
        {api::internal::CompiledModuleEntry{signature, metadata, executable}},
        {});
    runtime::RuntimeSession session(module, MakePlan(signature));
    std::string message;
    TEST_CHECK(
        Throws(
            [&] {
                session.RunAsync(
                    {runtime::NDArray::Zeros({1}, Float32(), Device::CPU())},
                    DeviceStream::Default(cuda));
            },
            &message),
        "CPU input for CUDA signature should fail");
    TEST_CHECK(message.find("device expected cuda:0") != std::string::npos,
               "input device error should precede disabled CUDA allocation");
    TEST_CHECK(launcher->calls == 0,
               "device mismatch must not allocate output or launch backend");
    return true;
}

/*! \brief RuntimeSession has no dynamic graph memory plan and rejects it at construction. */
bool TestNonstaticAndScalarContractsRejectedAtConstruction() {
    using namespace kxc;
    using namespace kxc::api;
    using namespace kxc::codegen;
    const Device cpu = Device::CPU();
    KernelSignature dynamic_signature(
        "dynamic_session",
        {KernelArgSpec("input", KernelArgRole::kInput, Float32(), {-1, 4}, cpu),
         KernelArgSpec("output", KernelArgRole::kOutput, Float32(), {3, 4},
                       cpu, 1, true)});
    auto dynamic_launcher = std::make_shared<RecordingLauncher>();
    TEST_CHECK(Throws([&] {
                   (void)MakeModule(dynamic_signature, {}, dynamic_launcher);
               }) && dynamic_launcher->calls == 0,
               "dynamic signatures without complete finite guards must be rejected at module admission");

    const DLDataType u64{kDLUInt, 64, 1};
    KernelSignature scalar_signature(
        "scalar_session",
        {KernelArgSpec("input", KernelArgRole::kInput, Float32(), {-1}, cpu, 8),
         KernelArgSpec("extent", KernelArgRole::kRuntimeExtent, u64, {1}, cpu, 8),
         KernelArgSpec("output", KernelArgRole::kOutput, Float32(), {-1}, cpu,
                       8, true)});
    ModuleInputContract input{{{0, 0, 4, 1, std::nullopt, std::nullopt}}};
    const auto twice = ModuleShapeExpr::Mul(ModuleShapeExpr::InputAxis(0, 0),
                                             ModuleShapeExpr::Const(2));
    ModuleTensorContract output;
    output.logical = {twice}; output.physical = {twice}; output.valid = {twice};
    output.max_bytes = 32;
    auto contract = std::make_shared<ModuleInvocationContract>(
        std::vector<ModuleInputContract>{input},
        std::vector<ModuleTensorContract>{output},
        std::vector<ModuleRuntimeExtentScalar>{{twice}});
    auto scalar_launcher = std::make_shared<RecordingLauncher>();
    const KernelLaunchMetadata metadata(cpu, CodeGenBackend::kLLVM);
    const api::CompiledModule scalar_module = api::internal::BuildCompiledModule(
        BuildTarget(cpu), {{scalar_signature, metadata,
                            CompiledKernel(scalar_signature, metadata, scalar_launcher),
                            std::move(contract)}}, {});
    TEST_CHECK(Throws([&] {
                   (void)runtime::RuntimeSession(
                       scalar_module, MakePlan(scalar_signature));
               }) && scalar_launcher->calls == 0,
               "RuntimeSession must reject scalar ABI generation before execution");
    return true;
}

bool TestDynamicFreshOutputExecutionAndLifetime() {
    using namespace kxc;
    DynamicSessionFixture fixture = MakeDynamicSessionFixture();
#if !KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI
    TEST_CHECK(Throws([&] {
                   (void)runtime::RuntimeSession(fixture.module, fixture.plan);
               }) && fixture.first->calls == 0 && fixture.second->calls == 0,
               "the existing dynamic module ABI gate must fail closed at session construction");
    return true;
#else
    runtime::RuntimeSession session(fixture.module, fixture.plan);
    const DeviceStream stream = DeviceStream::Create(Device::CPU());
    const auto rejected_before_launch = [&](const Array<runtime::NDArray>& inputs) {
        return Throws([&] { (void)session.RunAsync(inputs, stream); }) &&
               fixture.first->calls == 0 && fixture.second->calls == 0;
    };
    TEST_CHECK(rejected_before_launch(
                   {DynamicTensor(10), DynamicTensor(10)}),
               "out-of-bound graph inputs must fail before the first launch");
    TEST_CHECK(rejected_before_launch(
                   {DynamicTensor(3), DynamicTensor(3)}),
               "non-divisible graph inputs must fail before the first launch");
    TEST_CHECK(rejected_before_launch(
                   {DynamicTensor(4), DynamicTensor(6)}),
               "shared-symbol disagreement must fail before the first launch");
    TEST_CHECK(rejected_before_launch(
                   {runtime::NDArray::Zeros({4}, Float32(), Device::CPU()),
                    DynamicTensor(4)}),
               "dynamic rank mismatch must fail before the first launch");

    runtime::RunAsyncResult first =
        session.RunAsync({DynamicTensor(2), DynamicTensor(2)}, stream);
    first.completion.Wait();
    TEST_CHECK(first.outputs.size() == 1 &&
                   SameShape(first.outputs[0].shape(), {2, 4}) &&
                   fixture.first->calls == 1 && fixture.second->calls == 1 &&
                   fixture.first->last_arguments.size() == 4 &&
                   fixture.second->last_arguments.size() == 2,
               "dynamic calls must inject constants and bind fresh resolved outputs");
    runtime::RunAsyncResult second =
        session.RunAsync({DynamicTensor(6), DynamicTensor(6)}, stream);
    second.completion.Wait();
    TEST_CHECK(second.outputs.size() == 1 &&
                   SameShape(second.outputs[0].shape(), {6, 4}) &&
                   first.outputs[0].storage().get() !=
                       second.outputs[0].storage().get() &&
                   fixture.first->last_arguments[3].storage().get() !=
                       fixture.second->last_arguments[1].storage().get() &&
                   fixture.first->calls == 2 && fixture.second->calls == 2 &&
                   fixture.log->symbols ==
                       std::vector<std::string>({"dynamic_add", "dynamic_relu",
                                                 "dynamic_add", "dynamic_relu"}),
               "one dynamic module/plan must run multiple shapes with fresh outputs in call order");

    runtime::RunAsyncResult escaped;
    std::weak_ptr<DynamicRecordingLauncher> first_launcher;
    std::weak_ptr<DynamicRecordingLauncher> final_launcher;
    std::weak_ptr<int> prior_token;
    std::weak_ptr<int> final_token;
    const Object* intermediate_storage = nullptr;
    {
        DynamicSessionFixture lifetime = MakeDynamicSessionFixture();
        first_launcher = lifetime.first;
        final_launcher = lifetime.second;
        runtime::RuntimeSession retained_session(lifetime.module,
                                                  lifetime.plan);
        escaped = retained_session.RunAsync(
            {DynamicTensor(4), DynamicTensor(4)}, stream);
        prior_token = lifetime.first->last_operation_token;
        final_token = lifetime.second->last_operation_token;
        intermediate_storage =
            lifetime.first->last_arguments[3].storage().get();
    }
    bool retains_intermediate = false;
    for (const auto& storage : escaped.completion->retained_storage) {
        retains_intermediate =
            retains_intermediate || storage.get() == intermediate_storage;
    }
    TEST_CHECK(!first_launcher.expired() && !final_launcher.expired() &&
                   !prior_token.expired() && !final_token.expired() &&
                   retains_intermediate,
               "final completion must retain module, intermediates, and prior operations");
    escaped.completion.Wait();
    escaped.completion = AsyncOperation();
    TEST_CHECK(first_launcher.expired() && final_launcher.expired() &&
                   prior_token.expired() && final_token.expired(),
               "dynamic execution owners must release with the final completion");
    return true;
#endif
}

bool TestDynamicFreshOutputConstructionRejections() {
    using namespace kxc;
    using namespace kxc::api;
    using namespace kxc::codegen;
    using namespace kxc::runtime;

    const ExecutablePlan state = MakeStatePlan();
    TEST_CHECK(Throws([&] {
                   (void)ExecutablePlan(
                       state.values(), state.calls(), state.input_value_ids(),
                       state.constant_value_ids(), state.output_value_ids(),
                       state.state_value_ids(),
                       ExecutablePlanMode::kDynamicFreshOutputV1);
               }),
               "dynamic plan construction must reject state and in-place aliasing");
    const ExecutablePlan donation = MakeDonationPlan();
    TEST_CHECK(Throws([&] {
                   (void)ExecutablePlan(
                       donation.values(), donation.calls(),
                       donation.input_value_ids(),
                       donation.constant_value_ids(),
                       donation.output_value_ids(), {},
                       ExecutablePlanMode::kDynamicFreshOutputV1);
               }),
               "dynamic plan construction must reject input donation");

    const Device cpu = Device::CPU();
    const KernelSignature static_signature(
        "static_in_dynamic_mode",
        {KernelArgSpec("input", KernelArgRole::kInput, Float32(), {4}, cpu),
         KernelArgSpec("output", KernelArgRole::kOutput, Float32(), {4}, cpu,
                       1, true)});
    auto static_launcher = std::make_shared<RecordingLauncher>();
    const CompiledModule static_module =
        MakeModule(static_signature, {}, static_launcher);
    const ExecutablePlan static_values_dynamic_mode(
        {ValueSpec(0, 0, {4}, Float32(), cpu, true),
         ValueSpec(1, 1, {4}, Float32(), cpu, false, false, true)},
        {KernelCall("static_in_dynamic_mode", {0}, {1})}, {0}, {}, {1}, {},
        ExecutablePlanMode::kDynamicFreshOutputV1);
    TEST_CHECK(Throws([&] {
                   (void)RuntimeSession(static_module,
                                        static_values_dynamic_mode);
               }) && static_launcher->calls == 0,
               "dynamic mode must reject an entry without a dynamic invocation contract");

#if KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI
    DynamicSessionFixture rank = MakeDynamicSessionFixture();
    const ExecutablePlan wrong_rank(
        {ValueSpec(0, 0, {-1}, Float32(), cpu, true),
         ValueSpec(1, 1, {-1}, Float32(), cpu, true),
         ValueSpec(2, 2, {4}, Float32(), cpu, false, true),
         ValueSpec(3, 3, {-1}, Float32(), cpu),
         ValueSpec(4, 4, {-1}, Float32(), cpu, false, false, true)},
        {KernelCall("dynamic_add", {0, 1, 2}, {3}),
         KernelCall("dynamic_relu", {3}, {4})},
        {0, 1}, {2}, {4}, {},
        ExecutablePlanMode::kDynamicFreshOutputV1,
        {{0, 0, 2, 8, 2, std::nullopt},
         {1, 0, 2, 8, 2, GraphInputAxisReference{0, 0}}});
    TEST_CHECK(Throws([&] { (void)RuntimeSession(rank.module, wrong_rank); }) &&
                   rank.first->calls == 0 && rank.second->calls == 0,
               "dynamic mode must reject rank drift at construction");

    DynamicSessionFixture valid_mismatch = MakeDynamicSessionFixture(false);
    TEST_CHECK(Throws([&] {
                   (void)RuntimeSession(valid_mismatch.module,
                                        valid_mismatch.plan);
               }) && valid_mismatch.first->calls == 0,
               "dynamic mode must reject valid extents that differ from logical extents");
    const Device cuda = Device::CUDA(0);
    const KernelSignature cuda_signature(
        "dynamic_cuda",
        {KernelArgSpec("input", KernelArgRole::kInput, Float32(), {-1, 4},
                       cuda, 8),
         KernelArgSpec("output", KernelArgRole::kOutput, Float32(), {-1, 4},
                       cuda, 8, true)});
    const KernelLaunchMetadata cuda_metadata(
        cuda, CodeGenBackend::kCUDA, {1, 1, 1}, {1, 1, 1});
    auto cuda_log = std::make_shared<DynamicLaunchLog>();
    auto cuda_launcher = std::make_shared<DynamicRecordingLauncher>(
        "dynamic_cuda", cuda_log);
    auto* target_node = new TargetNode();
    target_node->kind = "cuda";
    target_node->device_type = kCUDA;
    target_node->device_id = 0;
    target_node->attrs.exists = 1;
    const CompiledModule cuda_module =
        kxc::api::internal::BuildCompiledModule(
            Target(ObjectRef(target_node)),
            {{cuda_signature, cuda_metadata,
              CompiledKernel(cuda_signature, cuda_metadata, cuda_launcher),
              Dynamic2DContract(1)}},
            {});
    const ExecutablePlan cuda_plan(
        {ValueSpec(0, 0, {-1, 4}, Float32(), cuda, true),
         ValueSpec(1, 1, {-1, 4}, Float32(), cuda, false, false, true)},
        {KernelCall("dynamic_cuda", {0}, {1})}, {0}, {}, {1}, {},
        ExecutablePlanMode::kDynamicFreshOutputV1,
        {{0, 0, 2, 8, 2, std::nullopt}});
    TEST_CHECK(Throws([&] { (void)RuntimeSession(cuda_module, cuda_plan); }) &&
                   cuda_launcher->calls == 0,
               "dynamic mode must reject CUDA at construction without fallback");
#endif
    return true;
}

/*!
 * \brief 零输入调用仍应绑定常量，并按签名返回标量及零尺寸多个输出。
 */
bool TestZeroInputAndMultipleOutputs() {
    using namespace kxc;
    using namespace kxc::codegen;
    const Device cpu = Device::CPU();
    KernelSignature signature(
        "multi_output_session",
        {KernelArgSpec("scalar_constant", KernelArgRole::kConstant, Float32(),
                       {}, cpu, 1, false, "relay.constant.0"),
         KernelArgSpec("scalar_output", KernelArgRole::kOutput, Float32(), {},
                       cpu, 32, true),
         KernelArgSpec("empty_output", KernelArgRole::kOutput, Float32(), {0},
                       cpu, 64, true)});
    runtime::NDArray constant =
        runtime::NDArray::Zeros({}, Float32(), cpu);
    Map<String, runtime::NDArray> constants;
    constants.Set(String("relay.constant.0"), constant);
    auto launcher = std::make_shared<RecordingLauncher>();
    runtime::RuntimeSession session(
        MakeModule(signature, constants, launcher), MakePlan(signature));

    Array<runtime::NDArray> outputs = session.Run({});
    TEST_CHECK(outputs.size() == 2 && outputs[0].shape().empty() &&
                   SameShape(outputs[1].shape(), {0}),
               "session should preserve scalar and zero-size output order");
    TEST_CHECK(outputs[0].NBytes() == sizeof(float) &&
                   outputs[1].NBytes() == 0,
               "scalar and zero-size output allocation is incorrect");
    TEST_CHECK(launcher->last_arguments.size() == 3 &&
                   launcher->last_arguments[0].get() != constant.get() &&
                   launcher->last_arguments[1].get() == outputs[0].get() &&
                   launcher->last_arguments[2].get() == outputs[1].get(),
               "zero-input constant/output ABI order is incorrect");
    return true;
}

/*! \brief 在线程安全 launcher 上验证并发参数装配没有共享调用状态。 */
bool TestConcurrentArgumentAssembly() {
    using namespace kxc;
    using namespace kxc::codegen;
    KernelSignature signature(
        "concurrent_session",
        {KernelArgSpec("input", KernelArgRole::kInput, Float32(), {4},
                       Device::CPU()),
         KernelArgSpec("output", KernelArgRole::kOutput, Float32(), {4},
                       Device::CPU(), 16, true)});
    auto launcher = std::make_shared<ConcurrentLauncher>();
    KernelLaunchMetadata metadata(Device::CPU(), CodeGenBackend::kLLVM);
    CompiledKernel executable(signature, metadata, launcher);
    api::CompiledModule module = api::internal::BuildCompiledModule(
        BuildTarget(Device::CPU()),
        {api::internal::CompiledModuleEntry{signature, metadata, executable}},
        {});
    runtime::RuntimeSession session(module, MakePlan(signature));
    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    constexpr int kThreads = 8;
    std::vector<runtime::NDArray> concurrent_outputs(kThreads);
    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([&, i] {
            try {
                const float expected = static_cast<float>(i + 1);
                std::vector<float> input_values(4, expected);
                runtime::NDArray input = runtime::NDArray::Empty(
                    {4}, Float32(), Device::CPU());
                input.CopyFromBytes(input_values.data(),
                                    input_values.size() * sizeof(float));
                Array<runtime::NDArray> outputs = session.Run({input});
                if (outputs.size() != 1 ||
                    outputs[0].NBytes() != 4 * sizeof(float)) {
                    failures.fetch_add(1, std::memory_order_relaxed);
                } else {
                    std::vector<float> output_values(4);
                    outputs[0].CopyToBytes(
                        output_values.data(),
                        output_values.size() * sizeof(float));
                    for (float value : output_values) {
                        if (value != expected) {
                            failures.fetch_add(1, std::memory_order_relaxed);
                            break;
                        }
                    }
                    concurrent_outputs[i] = outputs[0];
                }
            } catch (...) {
                failures.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (auto& thread : threads) thread.join();
    bool outputs_are_distinct = true;
    for (int i = 0; i < kThreads; ++i) {
        if (!concurrent_outputs[i].defined()) outputs_are_distinct = false;
        for (int j = 0; j < i; ++j) {
            if (concurrent_outputs[i].defined() &&
                concurrent_outputs[j].defined() &&
                concurrent_outputs[i].storage().get() ==
                    concurrent_outputs[j].storage().get()) {
                outputs_are_distinct = false;
            }
        }
    }
    TEST_CHECK(failures.load(std::memory_order_relaxed) == 0 &&
                   launcher->calls.load(std::memory_order_relaxed) == kThreads &&
                   outputs_are_distinct,
               "concurrent RuntimeSession assembly should keep outputs independent");
    return true;
}

bool TestMultiEntryPlanExecution() {
    using namespace kxc;
    using namespace kxc::codegen;
    const Device cpu = Device::CPU();
    const Array<int64_t> shape{4};
    KernelSignature add_signature(
        "plan_add",
        {KernelArgSpec("lhs", KernelArgRole::kInput, Float32(), shape, cpu),
         KernelArgSpec("rhs", KernelArgRole::kInput, Float32(), shape, cpu),
         KernelArgSpec("out", KernelArgRole::kOutput, Float32(), shape, cpu,
                       32, true)});
    KernelSignature mul_signature(
        "plan_mul",
        {KernelArgSpec("lhs", KernelArgRole::kInput, Float32(), shape, cpu),
         KernelArgSpec("rhs", KernelArgRole::kInput, Float32(), shape, cpu),
         KernelArgSpec("out", KernelArgRole::kOutput, Float32(), shape, cpu,
                       64, true)});
    KernelLaunchMetadata add_metadata(cpu, CodeGenBackend::kLLVM);
    KernelLaunchMetadata mul_metadata(cpu, CodeGenBackend::kLLVM);
    auto add_launcher = std::make_shared<BinaryElementwiseLauncher>(false);
    auto mul_launcher = std::make_shared<BinaryElementwiseLauncher>(true);
    CompiledKernel add_kernel(add_signature, add_metadata, add_launcher);
    CompiledKernel mul_kernel(mul_signature, mul_metadata, mul_launcher);
    std::vector<api::internal::CompiledModuleEntry> entries{
        {add_signature, add_metadata, add_kernel},
        {mul_signature, mul_metadata, mul_kernel},
    };
    api::CompiledModule module = api::internal::BuildCompiledModule(
        BuildTarget(cpu), std::move(entries), {});

    Array<runtime::ValueSpec> values{
        runtime::ValueSpec(0, 0, shape, Float32(), cpu, true),
        runtime::ValueSpec(1, 1, shape, Float32(), cpu, true),
        runtime::ValueSpec(2, 2, shape, Float32(), cpu, true),
        runtime::ValueSpec(3, 3, shape, Float32(), cpu),
        runtime::ValueSpec(4, 4, shape, Float32(), cpu, false, false, true),
    };
    runtime::ExecutablePlan plan(
        values,
        {runtime::KernelCall("plan_add", {0, 1}, {3}),
         runtime::KernelCall("plan_mul", {3, 2}, {4})},
        {0, 1, 2}, {}, {4});
    runtime::RuntimeSession session(module, plan);

    auto filled = [&](float value) {
        runtime::NDArray array =
            runtime::NDArray::Empty(shape, Float32(), cpu);
        std::vector<float> payload(4, value);
        array.CopyFromBytes(payload.data(), payload.size() * sizeof(float));
        return array;
    };
    Array<runtime::NDArray> outputs =
        session.Run({filled(1.0f), filled(2.0f), filled(3.0f)});
    std::vector<float> actual(4);
    outputs[0].CopyToBytes(actual.data(), actual.size() * sizeof(float));
    bool numeric_ok = true;
    for (float value : actual) numeric_ok = numeric_ok && value == 9.0f;
    TEST_CHECK(outputs.size() == 1 && numeric_ok && add_launcher->calls == 1 &&
                   mul_launcher->calls == 1,
               "RuntimeSession must execute every plan call in order");
    TEST_CHECK(add_launcher->last_arguments[2].get() ==
                   mul_launcher->last_arguments[0].get() &&
                   outputs[0].get() == mul_launcher->last_arguments[2].get() &&
                   add_launcher->last_arguments[2].storage()->alignment >= 32 &&
                   outputs[0].storage()->alignment >= 64,
               "stable intermediate/output values or alignments were not preserved");

    runtime::ExecutablePlan bad_plan(
        values,
        {runtime::KernelCall("plan_add", {0, 1}, {3}),
         runtime::KernelCall("missing_mul", {3, 2}, {4})},
        {0, 1, 2}, {}, {4});
    TEST_CHECK(Throws([&] { runtime::RuntimeSession invalid(module, bad_plan); }),
               "session construction must reject plan/module symbol drift");
    return true;
}

bool TestPlannedIntermediateStorageReuse() {
    using namespace kxc;
    using namespace kxc::codegen;
    const Device cpu = Device::CPU();
    const Array<int64_t> shape{4};
    std::vector<std::shared_ptr<RecordingLauncher>> launchers;
    std::vector<api::internal::CompiledModuleEntry> entries;
    Array<runtime::KernelCall> calls;
    for (int i = 0; i < 4; ++i) {
        const String symbol("reuse_" + std::to_string(i));
        KernelSignature signature(
            symbol,
            {KernelArgSpec("input", KernelArgRole::kInput, Float32(), shape,
                           cpu),
             KernelArgSpec("output", KernelArgRole::kOutput, Float32(), shape,
                           cpu, 16, true)});
        KernelLaunchMetadata metadata(cpu, CodeGenBackend::kLLVM);
        auto launcher = std::make_shared<RecordingLauncher>();
        entries.push_back({signature, metadata,
                           CompiledKernel(signature, metadata, launcher)});
        launchers.push_back(std::move(launcher));
        calls.push_back(runtime::KernelCall(symbol, {i}, {i + 1}));
    }
    api::CompiledModule module = api::internal::BuildCompiledModule(
        BuildTarget(cpu), std::move(entries), {});
    Array<runtime::ValueSpec> values{
        runtime::ValueSpec(0, 0, shape, Float32(), cpu, true),
        runtime::ValueSpec(1, 1, shape, Float32(), cpu),
        runtime::ValueSpec(2, 2, shape, Float32(), cpu),
        runtime::ValueSpec(3, 3, shape, Float32(), cpu),
        runtime::ValueSpec(4, 4, shape, Float32(), cpu, false, false, true),
    };
    runtime::ExecutablePlan plan = runtime::internal::PlanMemory(
        runtime::ExecutablePlan(values, calls, {0}, {}, {4}));
    runtime::RuntimeSession session(module, plan);
    const Array<runtime::NDArray> outputs = session.Run(
        {runtime::NDArray::Zeros(shape, Float32(), cpu)});

    TEST_CHECK(plan.values()[1]->storage_id == plan.values()[3]->storage_id &&
                   launchers[0]->last_arguments[1].storage().get() ==
                       launchers[2]->last_arguments[1].storage().get() &&
                   launchers[1]->last_arguments[1].storage().get() !=
                       launchers[2]->last_arguments[1].storage().get() &&
                   outputs[0].storage().get() !=
                       launchers[2]->last_arguments[1].storage().get(),
               "runtime must realize planned intermediate reuse without reusing outputs");
    return true;
}

}  // namespace

/*! \brief 顺序执行 RuntimeSession 契约用例，并将任一失败转换为非零退出码。 */
int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"construction_and_type_checks", TestConstructionAndTypeChecks},
        {"constant_realignment_at_module_construction",
         TestConstantRealignmentAtModuleConstruction},
        {"synchronous_assembly", TestSynchronousAssembly},
        {"module_owned_constant_execution", TestModuleOwnedConstantExecution},
        {"async_result_lifetime", TestAsyncResultLifetime},
        {"state_alias_persistence_and_lifetime",
         TestStateAliasPersistenceAndLifetime},
        {"state_run_async_serialization", TestStateRunAsyncSerialization},
        {"input_donation_alias_and_preflight",
         TestInputDonationAliasAndPreflight},
        {"state_contract_mismatch_rejected",
         TestStateContractMismatchRejected},
        {"input_validation", TestInputValidation},
        {"stream_validation", TestStreamValidation},
        {"input_device_validation_before_allocation",
         TestInputDeviceValidationBeforeAllocation},
        {"nonstatic_and_scalar_contracts_rejected_at_construction",
         TestNonstaticAndScalarContractsRejectedAtConstruction},
        {"dynamic_fresh_output_execution_and_lifetime",
         TestDynamicFreshOutputExecutionAndLifetime},
        {"dynamic_fresh_output_construction_rejections",
         TestDynamicFreshOutputConstructionRejections},
        {"zero_input_and_multiple_outputs", TestZeroInputAndMultipleOutputs},
        {"concurrent_argument_assembly", TestConcurrentArgumentAssembly},
        {"multi_entry_plan_execution", TestMultiEntryPlanExecution},
        {"planned_intermediate_storage_reuse",
         TestPlannedIntermediateStorageReuse},
    };
    int failures = 0;
    for (const auto& test : tests) {
        try {
            if (!test.second()) {
                ++failures;
                continue;
            }
            std::cout << "[PASS] " << test.first << '\n';
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << test.first << ": " << error.what() << '\n';
            ++failures;
        }
    }
    return failures == 0 ? 0 : 1;
}
