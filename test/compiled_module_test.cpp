/*! \file test/compiled_module_test.cpp
 * \brief 验证 CompiledModule 的 ObjectRef 生命周期和 NDArray 启动门禁。
 */

#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "kxc/runtime/compiled_module.h"
#include "../src/runtime/internal/compiled_module_node.h"

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

    api::CompiledModule module = api::internal::BuildCompiledModule(
        BuildTarget(cpu),
        {api::internal::CompiledModuleEntry{tir::PrimFunc(), signature,
                                            metadata, executable}},
        constants);
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
    return api::internal::BuildCompiledModule(
        BuildTarget(Device::CPU()),
        {api::internal::CompiledModuleEntry{tir::PrimFunc(), signature,
                                            metadata, executable}},
        {});
}

kxc::api::internal::CompiledModuleEntry MakeEntry(
    const kxc::codegen::KernelSignature& signature,
    const std::shared_ptr<RecordingLauncher>& launcher) {
    using namespace kxc;
    using namespace kxc::codegen;
    KernelLaunchMetadata metadata(Device::CPU(), CodeGenBackend::kLLVM);
    CompiledKernel executable(signature, metadata, launcher);
    return api::internal::CompiledModuleEntry{tir::PrimFunc(), signature,
                                              metadata, executable};
}

// 验证合法调用到达后端一次，并返回保活 Storage 的同步完成操作。
bool TestValidLaunchAndAccessors() {
    ModuleFixture fixture = MakeStaticFixture();
    const kxc::DeviceStream stream = kxc::DeviceStream::Default(kxc::Device::CPU());
    kxc::AsyncOperation operation =
        fixture.module.Launch("fixture_kernel", fixture.arguments, stream);

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
    TEST_CHECK(fixture.module.signature("fixture_kernel")->symbol ==
                       "fixture_kernel" &&
                   fixture.module.launch_metadata("fixture_kernel")->device ==
                       kxc::Device::CPU(),
               "module accessors should preserve signature and metadata");

    // 修改返回 Map 不得删除或替换模块内部常量绑定。
    kxc::Map<kxc::String, kxc::runtime::NDArray> copied = fixture.module.constants();
    const kxc::String constant_key("relay.constant.0");
    const std::vector<float> overwritten{1, 2, 3};
    copied.at(constant_key).CopyFromBytes(
        overwritten.data(), overwritten.size() * sizeof(float));
    copied.Set(kxc::String("unexpected"), fixture.arguments[0]);
    const auto fresh_constants = fixture.module.constants();
    std::vector<float> retained(3, -1.0F);
    fresh_constants.at(constant_key).CopyToBytes(
        retained.data(), retained.size() * sizeof(float));
    TEST_CHECK(fresh_constants.size() == 1 &&
                   retained == std::vector<float>({0, 0, 0}),
               "constants accessor must deep-copy its Map and NDArray payloads");
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
                   api::CompiledModule invalid = api::internal::BuildCompiledModule(
                       BuildTarget(Device::CPU()),
                       {api::internal::CompiledModuleEntry{
                           tir::PrimFunc(), signature, metadata, executable}},
                       {});
               }),
               "module assembly should reject a non-ready executable");
    return true;
}

// undefined stream、设备错误和参数数量错误都不能触发 launcher。
bool TestStreamAndCountChecks() {
    ModuleFixture fixture = MakeStaticFixture();
    TEST_CHECK(Throws([&] {
                   fixture.module.Launch("fixture_kernel", fixture.arguments,
                                         kxc::DeviceStream());
               }),
               "undefined stream should fail");
    TEST_CHECK(Throws([&] {
                   fixture.module.Launch(
                       "fixture_kernel", fixture.arguments,
                       kxc::DeviceStream::Default(kxc::Device::CUDA()));
               }),
               "stream on another device should fail");

    kxc::Array<kxc::runtime::NDArray> too_few{fixture.arguments[0], fixture.constant};
    TEST_CHECK(Throws([&] {
                   fixture.module.Launch(
                       "fixture_kernel", too_few,
                       kxc::DeviceStream::Default(kxc::Device::CPU()));
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
    TEST_CHECK(Throws([&] {
                   fixture.module.Launch("fixture_kernel", bad_arguments, stream);
               }, &message),
               "undefined NDArray should fail");
    TEST_CHECK(message.find("fixture_kernel") != std::string::npos &&
                   message.find("argument[0]") != std::string::npos &&
                   message.find("input") != std::string::npos,
               "argument error should include symbol, index, and name");

    bad_arguments = fixture.arguments;
    bad_arguments[0] = kxc::runtime::NDArray::Empty(
        {2, 3}, kxc::runtime::DataTypeFromString("int32"), kxc::Device::CPU());
    TEST_CHECK(Throws([&] {
                   fixture.module.Launch("fixture_kernel", bad_arguments, stream);
               }),
               "dtype mismatch should fail");

    bad_arguments = fixture.arguments;
    bad_arguments[0] = kxc::runtime::NDArray::Empty(
        {2, 3}, DLDataType{kDLFloat, 32, 2}, kxc::Device::CPU());
    TEST_CHECK(Throws([&] {
                   fixture.module.Launch("fixture_kernel", bad_arguments, stream);
               }),
               "dtype lanes mismatch should fail");

    bad_arguments = fixture.arguments;
    bad_arguments[0] = kxc::runtime::NDArray::Empty({6}, Float32(), kxc::Device::CPU());
    TEST_CHECK(Throws([&] {
                   fixture.module.Launch("fixture_kernel", bad_arguments, stream);
               }),
               "rank mismatch should fail");

    bad_arguments = fixture.arguments;
    bad_arguments[0] = kxc::runtime::NDArray::Empty({2, 4}, Float32(), kxc::Device::CPU());
    TEST_CHECK(Throws([&] {
                   fixture.module.Launch("fixture_kernel", bad_arguments, stream);
               }),
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
    module.Launch(signature->symbol, arguments,
                  DeviceStream::Default(Device::CPU()));
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
                   layout_fixture.module.Launch(
                       "fixture_kernel", layout_fixture.arguments, stream);
               }),
               "non-contiguous layout should fail");
    TEST_CHECK(layout_fixture.launcher->calls == 0,
               "layout failure must not reach launcher");

    ModuleFixture range_fixture = MakeStaticFixture();
    auto* range_node = const_cast<kxc::runtime::NDArrayNode*>(
        range_fixture.arguments[0].As<kxc::runtime::NDArrayNode>());
    range_node->byte_offset = range_node->storage.capacity_bytes();
    TEST_CHECK(Throws([&] {
                   range_fixture.module.Launch(
                       "fixture_kernel", range_fixture.arguments, stream);
               }),
               "out-of-range byte offset should fail");

    ModuleFixture alignment_fixture = MakeStaticFixture(/*alignment=*/4);
    // 多分配一个元素，确保 offset=1 的 view 落在容量内，失败原因只来自对齐。
    kxc::runtime::NDArray backing = kxc::runtime::NDArray::Empty(
        {7}, Float32(), kxc::Device::CPU());
    alignment_fixture.arguments[0] = backing.CreateView({2, 3}, {3, 1}, 1);
    TEST_CHECK(Throws([&] {
                   alignment_fixture.module.Launch(
                       "fixture_kernel", alignment_fixture.arguments, stream);
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
    offset_module.Launch(offset_signature->symbol, offset_arguments,
                         DeviceStream::Default(Device::CPU()));
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
    empty_module.Launch(empty_signature->symbol, empty_arguments,
                        DeviceStream::Default(Device::CPU()));
    TEST_CHECK(empty_launcher->calls == 1,
               "zero-size null data should skip address alignment and launch");
    return true;
}

// constant 角色比较完整 payload；launch 始终改用模块自有的 immutable binding。
bool TestConstantIdentityCheck() {
    ModuleFixture fixture = MakeStaticFixture();
    auto arguments = fixture.arguments;
    arguments[1] = kxc::runtime::NDArray::Zeros({3}, Float32(), kxc::Device::CPU());
    const std::vector<float> wrong_payload{1, 1, 1};
    arguments[1].CopyFromBytes(
        wrong_payload.data(), wrong_payload.size() * sizeof(float));
    TEST_CHECK(Throws([&] {
                   fixture.module.Launch(
                       "fixture_kernel", arguments,
                       kxc::DeviceStream::Default(kxc::Device::CPU()));
               }),
               "different constant payload should fail");
    TEST_CHECK(fixture.launcher->calls == 0,
               "constant payload failure must not reach launcher");

    arguments[1] = fixture.module.constants().at(
        kxc::String("relay.constant.0"));
    (void)fixture.module.Launch(
        "fixture_kernel", arguments,
        kxc::DeviceStream::Default(kxc::Device::CPU()));
    TEST_CHECK(fixture.launcher->calls == 1 &&
                   fixture.launcher->last_arguments[1].get() !=
                       fixture.constant.get() &&
                   fixture.launcher->last_arguments[1].get() !=
                       arguments[1].get(),
               "equal public snapshot must validate but launch the module-owned binding");
    return true;
}

// BuildCompiledModule 必须同步复制逻辑 payload，并为模块自有副本重新建立对齐。
bool TestBuildOwnsConstantPayloads() {
    using namespace kxc;
    using namespace kxc::codegen;
    const Device cpu = Device::CPU();
    const String key("relay.constant.offset");
    KernelArgSpec constant_spec("weight", KernelArgRole::kConstant, Float32(),
                                {3}, cpu, 64, false, key);
    KernelSignature signature(
        "owned_constant",
        {constant_spec,
         KernelArgSpec("output", KernelArgRole::kOutput, Float32(), {3},
                       cpu, 64, true)});
    auto launcher = std::make_shared<RecordingLauncher>();

    runtime::NDArray backing = runtime::NDArray::Empty(
        {4}, Float32(), cpu, 64);
    const std::vector<float> initial{99, 1, 2, 3};
    backing.CopyFromBytes(initial.data(), initial.size() * sizeof(float));
    runtime::NDArray source =
        backing.CreateView({3}, {1}, sizeof(float));
    Map<String, runtime::NDArray> constants;
    constants.Set(key, source);
    api::CompiledModule module = api::internal::BuildCompiledModule(
        BuildTarget(cpu), {MakeEntry(signature, launcher)}, constants);

    const std::vector<float> overwritten{9, 9, 9, 9};
    backing.CopyFromBytes(overwritten.data(),
                          overwritten.size() * sizeof(float));
    const runtime::NDArray snapshot = module.constants().at(key);
    std::vector<float> retained(3);
    snapshot.CopyToBytes(retained.data(), retained.size() * sizeof(float));
    TEST_CHECK(retained == std::vector<float>({1, 2, 3}) &&
                   snapshot.get() != source.get() &&
                   snapshot.storage().get() != source.storage().get() &&
                   snapshot->byte_offset == 0 &&
                   snapshot.storage()->alignment >= 64 &&
                   reinterpret_cast<uintptr_t>(snapshot.storage().data()) % 64 == 0,
               "module must own an aligned offset-normalized payload snapshot");

    runtime::NDArray output =
        runtime::NDArray::Empty({3}, Float32(), cpu, 64);
    module.Launch(signature->symbol, {snapshot, output},
                  DeviceStream::Default(cpu));
    std::vector<float> launched(3);
    launcher->last_arguments[0].CopyToBytes(
        launched.data(), launched.size() * sizeof(float));
    TEST_CHECK(launcher->calls == 1 && launched == retained &&
                   launcher->last_arguments[0].get() != source.get() &&
                   launcher->last_arguments[0].get() != snapshot.get(),
               "Launch must inject the immutable module-owned payload");

    const String empty_key("relay.constant.empty");
    KernelSignature empty_signature(
        "owned_empty_constant",
        {KernelArgSpec("empty", KernelArgRole::kConstant, Float32(), {0},
                       cpu, 64, false, empty_key),
         KernelArgSpec("empty_out", KernelArgRole::kOutput, Float32(), {0},
                       cpu, 64, true)});
    runtime::NDArray empty_source =
        runtime::NDArray::Empty({0}, Float32(), cpu);
    Map<String, runtime::NDArray> empty_constants;
    empty_constants.Set(empty_key, empty_source);
    auto empty_launcher = std::make_shared<RecordingLauncher>();
    api::CompiledModule empty_module = api::internal::BuildCompiledModule(
        BuildTarget(cpu), {MakeEntry(empty_signature, empty_launcher)},
        empty_constants);
    const runtime::NDArray empty_snapshot =
        empty_module.constants().at(empty_key);
    TEST_CHECK(empty_snapshot.NBytes() == 0 &&
                   empty_snapshot.storage().data() == nullptr &&
                   empty_snapshot.storage().get() !=
                       empty_source.storage().get() &&
                   empty_snapshot.storage()->alignment >= 64,
               "zero-byte constants must still have module-owned metadata/storage");
    empty_module.Launch(
        empty_signature->symbol,
        {empty_snapshot, runtime::NDArray::Empty({0}, Float32(), cpu, 64)},
        DeviceStream::Default(cpu));
    TEST_CHECK(empty_launcher->calls == 1,
               "zero-byte module-owned constants must remain launchable");

    runtime::NDArray non_contiguous =
        runtime::NDArray::Zeros({2, 2}, Float32(), cpu);
    auto* non_contiguous_node = const_cast<runtime::NDArrayNode*>(
        non_contiguous.As<runtime::NDArrayNode>());
    non_contiguous_node->strides_storage = {3, 1};
    const String bad_key("relay.constant.non_contiguous");
    KernelSignature bad_signature(
        "bad_constant_layout",
        {KernelArgSpec("bad", KernelArgRole::kConstant, Float32(), {2, 2},
                       cpu, 4, false, bad_key),
         KernelArgSpec("out", KernelArgRole::kOutput, Float32(), {2, 2},
                       cpu, 4, true)});
    Map<String, runtime::NDArray> bad_constants;
    bad_constants.Set(bad_key, non_contiguous);
    auto bad_launcher = std::make_shared<RecordingLauncher>();
    TEST_CHECK(Throws([&] {
                   (void)api::internal::BuildCompiledModule(
                       BuildTarget(cpu),
                       {MakeEntry(bad_signature, bad_launcher)}, bad_constants);
               }),
               "module build must fail closed on non-contiguous constants");
    return true;
}

bool TestMultiEntrySymbolDispatchAndAccessors() {
    using namespace kxc;
    using namespace kxc::codegen;
    KernelSignature first(
        "entry_a",
        {KernelArgSpec("output", KernelArgRole::kOutput, Float32(), {1},
                       Device::CPU(), 1, true)});
    KernelSignature second(
        "entry_b",
        {KernelArgSpec("output", KernelArgRole::kOutput, Float32(), {2},
                       Device::CPU(), 1, true)});
    auto first_launcher = std::make_shared<RecordingLauncher>();
    auto second_launcher = std::make_shared<RecordingLauncher>();
    api::CompiledModule module = api::internal::BuildCompiledModule(
        BuildTarget(Device::CPU()),
        {MakeEntry(first, first_launcher), MakeEntry(second, second_launcher)},
        {});

    TEST_CHECK(module.entry_count() == 2, "module should expose two entries");
    TEST_CHECK(module.HasFunction("entry_a") && module.HasFunction("entry_b") &&
                   !module.HasFunction("missing"),
               "HasFunction should answer by symbol");
    TEST_CHECK(module.symbols().size() == 2,
               "symbols accessor should return every entry symbol");
    TEST_CHECK(module.signature("entry_b")->symbol == "entry_b" &&
                   module.launch_metadata("entry_b")->device == Device::CPU(),
               "symbol accessors should return the requested entry metadata");
    module.Launch("entry_b",
                  {runtime::NDArray::Zeros({2}, Float32(), Device::CPU())},
                  DeviceStream::Default(Device::CPU()));
    TEST_CHECK(first_launcher->calls == 0 && second_launcher->calls == 1,
               "symbol launch should dispatch only to the requested entry");
    return true;
}

bool TestUnknownDuplicateAndMismatchedSymbolDispatch() {
    using namespace kxc;
    using namespace kxc::codegen;
    KernelSignature first(
        "entry_a",
        {KernelArgSpec("output", KernelArgRole::kOutput, Float32(), {1},
                       Device::CPU(), 1, true)});
    KernelSignature second(
        "entry_b",
        {KernelArgSpec("output", KernelArgRole::kOutput, Float32(), {2},
                       Device::CPU(), 1, true)});
    auto first_launcher = std::make_shared<RecordingLauncher>();
    auto second_launcher = std::make_shared<RecordingLauncher>();
    api::CompiledModule module = api::internal::BuildCompiledModule(
        BuildTarget(Device::CPU()),
        {MakeEntry(first, first_launcher), MakeEntry(second, second_launcher)},
        {});

    TEST_CHECK(Throws([&] { module.signature("unknown"); }),
               "unknown symbol metadata lookup should fail");
    TEST_CHECK(Throws([&] {
                   module.Launch(
                       "unknown",
                       {runtime::NDArray::Zeros({1}, Float32(), Device::CPU())},
                       DeviceStream::Default(Device::CPU()));
               }),
               "unknown symbol launch should fail");
    TEST_CHECK(Throws([&] {
                   module.Launch(
                       "entry_b",
                       {runtime::NDArray::Zeros({1}, Float32(), Device::CPU())},
                       DeviceStream::Default(Device::CPU()));
               }),
               "dispatch should validate against the requested symbol signature");
    TEST_CHECK(first_launcher->calls == 0 && second_launcher->calls == 0,
               "failed symbol dispatch must not reach any launcher");

    auto duplicate_launcher = std::make_shared<RecordingLauncher>();
    TEST_CHECK(Throws([&] {
                   api::internal::BuildCompiledModule(
                       BuildTarget(Device::CPU()),
                       {MakeEntry(first, first_launcher),
                        MakeEntry(first, duplicate_launcher)},
                       {});
               }),
               "duplicate symbols should fail during module assembly");
    return true;
}

bool TestSharedConstantsAcrossEntries() {
    using namespace kxc;
    using namespace kxc::codegen;
    const Device cpu = Device::CPU();
    KernelArgSpec shared("weight", KernelArgRole::kConstant, Float32(), {3},
                         cpu, 1, false, "relay.constant.shared");
    KernelSignature first(
        "constant_entry_a",
        {shared, KernelArgSpec("output", KernelArgRole::kOutput, Float32(), {3},
                               cpu, 1, true)});
    KernelArgSpec shared_aligned(
        "weight", KernelArgRole::kConstant, Float32(), {3}, cpu, 64,
        false, "relay.constant.shared");
    KernelSignature second(
        "constant_entry_b",
        {shared_aligned,
         KernelArgSpec("output", KernelArgRole::kOutput, Float32(), {3},
                       cpu, 1, true)});
    runtime::NDArray constant = runtime::NDArray::Zeros({3}, Float32(), cpu);
    Map<String, runtime::NDArray> constants;
    constants.Set(String("relay.constant.shared"), constant);
    auto first_launcher = std::make_shared<RecordingLauncher>();
    auto second_launcher = std::make_shared<RecordingLauncher>();
    api::CompiledModule module = api::internal::BuildCompiledModule(
        BuildTarget(cpu),
        {MakeEntry(first, first_launcher), MakeEntry(second, second_launcher)},
        constants);

    const runtime::NDArray shared_snapshot = module.constants().at(
        String("relay.constant.shared"));
    TEST_CHECK(module.constants().size() == 1 &&
                   shared_snapshot.storage()->alignment >= 64,
               "shared constants must be deduplicated at their maximum alignment");
    module.Launch("constant_entry_a",
                  {shared_snapshot,
                   runtime::NDArray::Zeros({3}, Float32(), cpu)},
                  DeviceStream::Default(cpu));
    module.Launch("constant_entry_b",
                  {shared_snapshot,
                   runtime::NDArray::Zeros({3}, Float32(), cpu)},
                  DeviceStream::Default(cpu));
    TEST_CHECK(first_launcher->calls == 1 && second_launcher->calls == 1,
               "both entries should launch with the shared module constant");

    Map<String, runtime::NDArray> unexpected = constants;
    unexpected.Set(String("unexpected"), constant);
    TEST_CHECK(Throws([&] {
                   api::internal::BuildCompiledModule(
                       BuildTarget(cpu),
                       {MakeEntry(first, first_launcher),
                        MakeEntry(second, second_launcher)},
                       unexpected);
               }),
               "constant union validation should reject unexpected keys");

    KernelArgSpec conflicting("weight", KernelArgRole::kConstant, Float32(), {4},
                              cpu, 1, false, "relay.constant.shared");
    KernelSignature conflict_signature(
        "constant_entry_conflict",
        {conflicting, KernelArgSpec("output", KernelArgRole::kOutput, Float32(),
                                    {4}, cpu, 1, true)});
    auto conflict_launcher = std::make_shared<RecordingLauncher>();
    TEST_CHECK(Throws([&] {
                   api::internal::BuildCompiledModule(
                       BuildTarget(cpu),
                       {MakeEntry(first, first_launcher),
                        MakeEntry(conflict_signature, conflict_launcher)},
                       constants);
               }),
               "shared constant keys with different contracts should fail");
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
        {"build_owns_constant_payloads", TestBuildOwnsConstantPayloads},
        {"multi_entry_symbol_dispatch_and_accessors",
         TestMultiEntrySymbolDispatchAndAccessors},
        {"unknown_duplicate_and_mismatched_symbol_dispatch",
         TestUnknownDuplicateAndMismatchedSymbolDispatch},
        {"shared_constants_across_entries", TestSharedConstantsAcrossEntries},
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
