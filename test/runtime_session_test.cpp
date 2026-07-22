/*! \file test/runtime_session_test.cpp
 * \brief 验证 RuntimeSession 的输入校验、常量绑定、输出分配和异步完成契约。
 */

#include <atomic>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "kxc/runtime/compiled_module.h"
#include "kxc/runtime/session.h"
#include "../src/runtime/internal/compiled_module_node.h"

namespace {

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

/*! \brief 保存 session 测试所需的模块、常量和 fake launcher。 */
struct SessionFixture {
    kxc::api::CompiledModule module;
    kxc::runtime::NDArray constant;
    std::shared_ptr<RecordingLauncher> launcher;
};

/*! \brief 从任意合法签名和常量表构造 CPU fake CompiledModule。 */
kxc::api::CompiledModule MakeModule(
    const kxc::codegen::KernelSignature& signature,
    const kxc::Map<kxc::String, kxc::runtime::NDArray>& constants,
    const std::shared_ptr<RecordingLauncher>& launcher) {
    using namespace kxc;
    using namespace kxc::codegen;
    KernelLaunchMetadata metadata(Device::CPU(), CodeGenBackend::kLLVM);
    CompiledKernel executable(signature, metadata, launcher);
    return api::internal::BuildCompiledModule(
        BuildTarget(Device::CPU()), tir::PrimFunc(), signature, metadata,
        constants, executable);
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
    return {MakeModule(signature, constants, launcher), constant,
            std::move(launcher)};
}

/*! \brief 构造器必须拒绝 undefined module 和错误 ObjectRef 节点类型。 */
bool TestConstructionAndTypeChecks() {
    using namespace kxc;
    TEST_CHECK(Throws([] {
                   runtime::RuntimeSession invalid{
                       api::CompiledModule(ObjectRef())};
               }),
               "undefined CompiledModule should fail");
    TEST_CHECK(Throws([] {
                   runtime::RuntimeSession invalid(ObjectRef(Device::CPU()));
               }),
               "Device ObjectRef should not become RuntimeSession");
    SessionFixture fixture = MakeStaticFixture();
    runtime::RuntimeSession session(fixture.module);
    TEST_CHECK(session.defined(), "valid module should create a session");
    fixture.launcher->ready = false;
    TEST_CHECK(Throws([&] { runtime::RuntimeSession invalid(fixture.module); }),
               "defined but no longer ready module should fail");
    return true;
}

/*! \brief Run 必须自动绑定常量、分配输出并保持最终 ABI 参数顺序。 */
bool TestSynchronousAssembly() {
    using namespace kxc;
    SessionFixture fixture = MakeStaticFixture();
    runtime::RuntimeSession session(fixture.module);
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
                   fixture.launcher->last_arguments[1].get() ==
                       fixture.constant.get() &&
                   fixture.launcher->last_arguments[2].get() == outputs[0].get(),
               "session changed input/constant/output ABI order or identity");
    TEST_CHECK(fixture.launcher->saw_compiled_kernel_owner,
               "launch should pass the compiled kernel owner to the launcher");
    return true;
}

/*! \brief RunAsync 必须返回 outputs 和可独立保活全部参数的 completion。 */
bool TestAsyncResultLifetime() {
    using namespace kxc;
    SessionFixture fixture = MakeStaticFixture();
    runtime::RunAsyncResult result;
    {
        runtime::RuntimeSession session(fixture.module);
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

/*! \brief 输入数量、定义状态、dtype、rank、shape 和布局必须在 launch 前失败。 */
bool TestInputValidation() {
    using namespace kxc;
    SessionFixture fixture = MakeStaticFixture();
    runtime::RuntimeSession session(fixture.module);
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
    runtime::RuntimeSession session(fixture.module);
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
        BuildTarget(cuda), tir::PrimFunc(), signature, metadata, {}, executable);
    runtime::RuntimeSession session(module);
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

/*! \brief 动态输入维接受实际非负 shape，输出仍按静态签名分配。 */
bool TestDynamicInput() {
    using namespace kxc;
    using namespace kxc::codegen;
    KernelSignature signature(
        "dynamic_session",
        {KernelArgSpec("input", KernelArgRole::kInput, Float32(), {-1, 4},
                       Device::CPU()),
         KernelArgSpec("output", KernelArgRole::kOutput, Float32(), {3, 4},
                       Device::CPU(), 1, true)});
    auto launcher = std::make_shared<RecordingLauncher>();
    runtime::RuntimeSession session(MakeModule(signature, {}, launcher));
    Array<runtime::NDArray> outputs = session.Run(
        {runtime::NDArray::Zeros({3, 4}, Float32(), Device::CPU())});
    TEST_CHECK(outputs.size() == 1 &&
                   SameShape(outputs[0].shape(), {3, 4}) &&
                   launcher->calls == 1,
               "dynamic input should launch with static output allocation");
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
        MakeModule(signature, constants, launcher));

    Array<runtime::NDArray> outputs = session.Run({});
    TEST_CHECK(outputs.size() == 2 && outputs[0].shape().empty() &&
                   SameShape(outputs[1].shape(), {0}),
               "session should preserve scalar and zero-size output order");
    TEST_CHECK(outputs[0].NBytes() == sizeof(float) &&
                   outputs[1].NBytes() == 0,
               "scalar and zero-size output allocation is incorrect");
    TEST_CHECK(launcher->last_arguments.size() == 3 &&
                   launcher->last_arguments[0].get() == constant.get() &&
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
        BuildTarget(Device::CPU()), tir::PrimFunc(), signature, metadata, {}, executable);
    runtime::RuntimeSession session(module);
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

}  // namespace

/*! \brief 顺序执行 RuntimeSession 契约用例，并将任一失败转换为非零退出码。 */
int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"construction_and_type_checks", TestConstructionAndTypeChecks},
        {"synchronous_assembly", TestSynchronousAssembly},
        {"async_result_lifetime", TestAsyncResultLifetime},
        {"input_validation", TestInputValidation},
        {"stream_validation", TestStreamValidation},
        {"input_device_validation_before_allocation",
         TestInputDeviceValidationBeforeAllocation},
        {"dynamic_input", TestDynamicInput},
        {"zero_input_and_multiple_outputs", TestZeroInputAndMultipleOutputs},
        {"concurrent_argument_assembly", TestConcurrentArgumentAssembly},
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
