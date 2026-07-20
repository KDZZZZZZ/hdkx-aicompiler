/*! \file test/compiled_module_test.cpp
 * \brief 验证 CompiledModule 的 ObjectRef 生命周期和 NDArray 启动门禁。
 */

#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "api/compiled_module.h"

namespace {

#define TEST_CHECK(condition, message)                                           \
    do {                                                                          \
        if (!(condition)) {                                                        \
            std::cerr << "[FAIL] " << __FUNCTION__ << ": " << (message) << "\n"; \
            return false;                                                         \
        }                                                                         \
    } while (0)

// 捕获标准异常，并可选返回错误文本供诊断上下文断言使用。
bool Throws(const std::function<void()>& function, std::string* message = nullptr) {
    try {
        function();
    } catch (const std::exception& error) {
        if (message) *message = error.what();
        return true;
    }
    return false;
}

// 测试统一使用项目运行时认可的 float32 DLPack 描述。
DLDataType Float32() { return kxc::runtime::DataTypeFromString("float32"); }

/*!
 * \brief 记录型 launcher，用于证明公共校验失败时后端不会收到调用。
 *
 * launcher 返回同步完成操作，并显式保活所有参数 Storage，从而同时验证
 * CompiledKernel 会把自身 ObjectRef 传递给后端。
 */
class RecordingLauncher final : public kxc::codegen::KernelLauncher {
public:
    explicit RecordingLauncher(bool ready = true) : ready_(ready) {}

    // 状态由测试构造参数控制，便于覆盖未就绪 executable。
    bool IsReady() const noexcept override { return ready_; }

    // 记录调用参数和 executable owner，再返回保活 Storage 的完成句柄。
    kxc::AsyncOperation Launch(
        const kxc::Array<kxc::runtime::NDArray>& arguments,
        const kxc::DeviceStream& stream,
        const kxc::ObjectRef& executable_owner) const override {
        ++calls;
        last_arguments = arguments;
        saw_compiled_kernel_owner =
            executable_owner.As<kxc::codegen::CompiledKernelNode>() != nullptr;
        kxc::Array<kxc::Storage> retained;
        for (const auto& argument : arguments) retained.push_back(argument.storage());
        return kxc::AsyncOperation::Completed(stream, std::move(retained));
    }

    bool ready_{true};
    mutable int calls{0};
    mutable bool saw_compiled_kernel_owner{false};
    mutable kxc::Array<kxc::runtime::NDArray> last_arguments;
};

// 保存一套模块、参数和 fake launcher，减少负例重复装配代码。
struct ModuleFixture {
    kxc::api::CompiledModule module;
    kxc::Array<kxc::runtime::NDArray> arguments;
    kxc::runtime::NDArray constant;
    std::shared_ptr<RecordingLauncher> launcher;
};

// 构造 input -> constant -> output 的合法 CPU 模块和调用参数。
ModuleFixture MakeStaticFixture(uint64_t alignment = 1) {
    using namespace kxc;
    using namespace kxc::codegen;

    const Device cpu = Device::CPU();
    KernelArgSpec input("input", KernelArgRole::kInput, Float32(), {2, 3}, cpu,
                        alignment, false);
    KernelArgSpec constant_spec("weight", KernelArgRole::kConstant, Float32(), {3},
                                cpu, alignment, false, "relay.constant.0");
    KernelArgSpec output("output", KernelArgRole::kOutput, Float32(), {2, 3}, cpu,
                         alignment, true);
    KernelSignature signature("fixture_kernel", {input, constant_spec, output});
    KernelLaunchMetadata metadata(cpu, CodeGenBackend::kLLVM);
    auto launcher = std::make_shared<RecordingLauncher>();
    CompiledKernel executable(signature, metadata, launcher);

    runtime::NDArray input_array = runtime::NDArray::Zeros({2, 3}, Float32(), cpu);
    runtime::NDArray constant = runtime::NDArray::Zeros({3}, Float32(), cpu);
    runtime::NDArray output_array = runtime::NDArray::Zeros({2, 3}, Float32(), cpu);
    Map<String, runtime::NDArray> constants;
    constants.Set(String("relay.constant.0"), constant);

    api::CompiledModule module(BuildTarget(cpu), tir::PrimFunc(), signature, metadata,
                               constants, executable);
    return ModuleFixture{module, {input_array, constant, output_array}, constant,
                         std::move(launcher)};
}

// 使用指定签名构造不含常量的模块，供动态 shape 和单参数负例复用。
kxc::api::CompiledModule MakeModule(
    const kxc::codegen::KernelSignature& signature,
    const std::shared_ptr<RecordingLauncher>& launcher) {
    using namespace kxc;
    using namespace kxc::codegen;
    KernelLaunchMetadata metadata(Device::CPU(), CodeGenBackend::kLLVM);
    CompiledKernel executable(signature, metadata, launcher);
    return api::CompiledModule(BuildTarget(Device::CPU()), tir::PrimFunc(), signature,
                               metadata, {}, executable);
}

// 验证合法调用到达后端一次，并返回保活 Storage 的同步完成操作。
bool TestValidLaunchAndAccessors() {
    ModuleFixture fixture = MakeStaticFixture();
    const kxc::DeviceStream stream = kxc::DeviceStream::Default(kxc::Device::CPU());
    kxc::AsyncOperation operation = fixture.module.Launch(fixture.arguments, stream);

    TEST_CHECK(operation.IsReady(), "CPU fake launch should complete inline");
    TEST_CHECK(fixture.launcher->calls == 1,
               "valid arguments should reach launcher exactly once");
    TEST_CHECK(fixture.launcher->saw_compiled_kernel_owner,
               "launcher should receive CompiledKernel owner for async retention");
    TEST_CHECK(operation->retained_storage.size() == fixture.arguments.size(),
               "operation should retain every argument Storage");
    TEST_CHECK(fixture.module.IsReady() &&
                   std::string(fixture.module.GetStatus()) == "ready",
               "assembled module should report ready");
    TEST_CHECK(fixture.module.signature()->symbol == "fixture_kernel" &&
                   fixture.module.launch_metadata()->device == kxc::Device::CPU(),
               "module accessors should preserve signature and metadata");

    // 修改返回 Map 不得删除或替换模块内部常量绑定。
    kxc::Map<kxc::String, kxc::runtime::NDArray> copied = fixture.module.constants();
    copied.Set(kxc::String("unexpected"), fixture.arguments[0]);
    TEST_CHECK(fixture.module.constants().size() == 1,
               "constants accessor must return an independent Map");
    return true;
}

// 错误 ObjectRef 类型和未就绪 launcher 必须在模块边界被拒绝。
bool TestObjectAndReadinessChecks() {
    using namespace kxc;
    using namespace kxc::codegen;

    ObjectRef device_ref = Device::CPU();
    TEST_CHECK(Throws([&] { api::CompiledModule invalid(device_ref); }),
               "CompiledModule should reject DeviceNode");
    TEST_CHECK(Throws([&] { CompiledKernel invalid(device_ref); }),
               "CompiledKernel should reject DeviceNode");

    KernelArgSpec output("output", KernelArgRole::kOutput, Float32(), {1},
                         Device::CPU(), 1, true);
    KernelSignature signature("not_ready", {output});
    KernelLaunchMetadata metadata(Device::CPU(), CodeGenBackend::kLLVM);
    auto launcher = std::make_shared<RecordingLauncher>(false);
    CompiledKernel executable(signature, metadata, launcher);
    TEST_CHECK(!executable.IsReady(), "disabled launcher should not be ready");
    TEST_CHECK(Throws([&] {
                   api::CompiledModule invalid(BuildTarget(Device::CPU()), tir::PrimFunc(),
                                               signature, metadata, {}, executable);
               }),
               "module assembly should reject a non-ready executable");
    return true;
}

// undefined stream、设备错误和参数数量错误都不能触发 launcher。
bool TestStreamAndCountChecks() {
    ModuleFixture fixture = MakeStaticFixture();
    TEST_CHECK(Throws([&] {
                   fixture.module.Launch(fixture.arguments, kxc::DeviceStream());
               }),
               "undefined stream should fail");
    TEST_CHECK(Throws([&] {
                   fixture.module.Launch(
                       fixture.arguments,
                       kxc::DeviceStream::Default(kxc::Device::CUDA()));
               }),
               "stream on another device should fail");

    kxc::Array<kxc::runtime::NDArray> too_few{fixture.arguments[0], fixture.constant};
    TEST_CHECK(Throws([&] {
                   fixture.module.Launch(
                       too_few, kxc::DeviceStream::Default(kxc::Device::CPU()));
               }),
               "argument count mismatch should fail");
    TEST_CHECK(fixture.launcher->calls == 0,
               "stream/count failures must not reach launcher");
    return true;
}

// dtype、rank、shape 和 undefined NDArray 错误必须携带参数上下文。
bool TestTensorMetadataChecks() {
    ModuleFixture fixture = MakeStaticFixture();
    const kxc::DeviceStream stream = kxc::DeviceStream::Default(kxc::Device::CPU());

    auto bad_arguments = fixture.arguments;
    bad_arguments[0] = kxc::runtime::NDArray();
    std::string message;
    TEST_CHECK(Throws([&] { fixture.module.Launch(bad_arguments, stream); }, &message),
               "undefined NDArray should fail");
    TEST_CHECK(message.find("fixture_kernel") != std::string::npos &&
                   message.find("argument[0]") != std::string::npos &&
                   message.find("input") != std::string::npos,
               "argument error should include symbol, index, and name");

    bad_arguments = fixture.arguments;
    bad_arguments[0] = kxc::runtime::NDArray::Empty(
        {2, 3}, kxc::runtime::DataTypeFromString("int32"), kxc::Device::CPU());
    TEST_CHECK(Throws([&] { fixture.module.Launch(bad_arguments, stream); }),
               "dtype mismatch should fail");

    bad_arguments = fixture.arguments;
    bad_arguments[0] = kxc::runtime::NDArray::Empty(
        {2, 3}, DLDataType{kDLFloat, 32, 2}, kxc::Device::CPU());
    TEST_CHECK(Throws([&] { fixture.module.Launch(bad_arguments, stream); }),
               "dtype lanes mismatch should fail");

    bad_arguments = fixture.arguments;
    bad_arguments[0] = kxc::runtime::NDArray::Empty({6}, Float32(), kxc::Device::CPU());
    TEST_CHECK(Throws([&] { fixture.module.Launch(bad_arguments, stream); }),
               "rank mismatch should fail");

    bad_arguments = fixture.arguments;
    bad_arguments[0] = kxc::runtime::NDArray::Empty({2, 4}, Float32(), kxc::Device::CPU());
    TEST_CHECK(Throws([&] { fixture.module.Launch(bad_arguments, stream); }),
               "static shape mismatch should fail");
    TEST_CHECK(fixture.launcher->calls == 0,
               "metadata failures must not reach launcher");
    return true;
}

// 动态输入维接受任意非负实际值，输出仍保持静态契约。
bool TestDynamicInputShape() {
    using namespace kxc;
    using namespace kxc::codegen;
    KernelArgSpec input("input", KernelArgRole::kInput, Float32(), {-1, 4},
                        Device::CPU());
    KernelArgSpec output("output", KernelArgRole::kOutput, Float32(), {3, 4},
                         Device::CPU(), 1, true);
    KernelSignature signature("dynamic_kernel", {input, output});
    auto launcher = std::make_shared<RecordingLauncher>();
    api::CompiledModule module = MakeModule(signature, launcher);
    Array<runtime::NDArray> arguments{
        runtime::NDArray::Empty({3, 4}, Float32(), Device::CPU()),
        runtime::NDArray::Empty({3, 4}, Float32(), Device::CPU()),
    };
    module.Launch(arguments, DeviceStream::Default(Device::CPU()));
    TEST_CHECK(launcher->calls == 1, "dynamic input shape should launch");
    return true;
}

// 非连续布局、越界 range 和未满足 alignment 的有效地址必须被拒绝。
bool TestLayoutRangeAndAlignmentChecks() {
    const kxc::DeviceStream stream = kxc::DeviceStream::Default(kxc::Device::CPU());

    ModuleFixture layout_fixture = MakeStaticFixture();
    auto* layout_node = const_cast<kxc::runtime::NDArrayNode*>(
        layout_fixture.arguments[0].As<kxc::runtime::NDArrayNode>());
    layout_node->strides_storage = {4, 1};
    TEST_CHECK(Throws([&] {
                   layout_fixture.module.Launch(layout_fixture.arguments, stream);
               }),
               "non-contiguous layout should fail");
    TEST_CHECK(layout_fixture.launcher->calls == 0,
               "layout failure must not reach launcher");

    ModuleFixture range_fixture = MakeStaticFixture();
    auto* range_node = const_cast<kxc::runtime::NDArrayNode*>(
        range_fixture.arguments[0].As<kxc::runtime::NDArrayNode>());
    range_node->byte_offset = range_node->storage.capacity_bytes();
    TEST_CHECK(Throws([&] {
                   range_fixture.module.Launch(range_fixture.arguments, stream);
               }),
               "out-of-range byte offset should fail");

    ModuleFixture alignment_fixture = MakeStaticFixture(/*alignment=*/4);
    // 多分配一个元素，确保 offset=1 的 view 落在容量内，失败原因只来自对齐。
    kxc::runtime::NDArray backing = kxc::runtime::NDArray::Empty(
        {7}, Float32(), kxc::Device::CPU());
    alignment_fixture.arguments[0] = backing.CreateView({2, 3}, {3, 1}, 1);
    TEST_CHECK(Throws([&] {
                   alignment_fixture.module.Launch(alignment_fixture.arguments, stream);
               }),
               "misaligned effective address should fail");
    TEST_CHECK(alignment_fixture.launcher->calls == 0,
               "range/alignment failures must not reach launcher");
    return true;
}

// 合法非零 offset 只要 range、布局和有效地址对齐就可以启动。
bool TestAlignedOffsetAndZeroSize() {
    using namespace kxc;
    using namespace kxc::codegen;

    KernelArgSpec input("input", KernelArgRole::kInput, Float32(), {1},
                        Device::CPU(), 4, false);
    KernelArgSpec output("output", KernelArgRole::kOutput, Float32(), {1},
                         Device::CPU(), 4, true);
    KernelSignature offset_signature("offset_kernel", {input, output});
    auto offset_launcher = std::make_shared<RecordingLauncher>();
    api::CompiledModule offset_module = MakeModule(offset_signature, offset_launcher);
    runtime::NDArray backing = runtime::NDArray::Empty({2}, Float32(), Device::CPU());
    runtime::NDArray offset_input = backing.CreateView({1}, {1}, sizeof(float));
    Array<runtime::NDArray> offset_arguments{
        offset_input, runtime::NDArray::Empty({1}, Float32(), Device::CPU())};
    offset_module.Launch(offset_arguments, DeviceStream::Default(Device::CPU()));
    TEST_CHECK(offset_launcher->calls == 1,
               "aligned non-zero offset should reach launcher");

    KernelArgSpec empty_input("empty", KernelArgRole::kInput, Float32(), {0},
                              Device::CPU(), 64, false);
    KernelArgSpec empty_output("empty_out", KernelArgRole::kOutput, Float32(), {0},
                               Device::CPU(), 64, true);
    KernelSignature empty_signature("empty_kernel", {empty_input, empty_output});
    auto empty_launcher = std::make_shared<RecordingLauncher>();
    api::CompiledModule empty_module = MakeModule(empty_signature, empty_launcher);
    Array<runtime::NDArray> empty_arguments{
        runtime::NDArray::Empty({0}, Float32(), Device::CPU()),
        runtime::NDArray::Empty({0}, Float32(), Device::CPU())};
    empty_module.Launch(empty_arguments, DeviceStream::Default(Device::CPU()));
    TEST_CHECK(empty_launcher->calls == 1,
               "zero-size null data should skip address alignment and launch");
    return true;
}

// constant 角色只能接收模块绑定的同一 NDArray，不能被等形等类型值替换。
bool TestConstantIdentityCheck() {
    ModuleFixture fixture = MakeStaticFixture();
    auto arguments = fixture.arguments;
    arguments[1] = kxc::runtime::NDArray::Zeros({3}, Float32(), kxc::Device::CPU());
    TEST_CHECK(Throws([&] {
                   fixture.module.Launch(
                       arguments, kxc::DeviceStream::Default(kxc::Device::CPU()));
               }),
               "replacement constant should fail");
    TEST_CHECK(fixture.launcher->calls == 0,
               "constant identity failure must not reach launcher");
    return true;
}

}  // namespace

// 顺序执行全部模块契约用例并汇总失败，保证 CI 收到可靠非零退出码。
int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"valid_launch_and_accessors", TestValidLaunchAndAccessors},
        {"object_and_readiness_checks", TestObjectAndReadinessChecks},
        {"stream_and_count_checks", TestStreamAndCountChecks},
        {"tensor_metadata_checks", TestTensorMetadataChecks},
        {"dynamic_input_shape", TestDynamicInputShape},
        {"layout_range_and_alignment_checks", TestLayoutRangeAndAlignmentChecks},
        {"aligned_offset_and_zero_size", TestAlignedOffsetAndZeroSize},
        {"constant_identity_check", TestConstantIdentityCheck},
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
