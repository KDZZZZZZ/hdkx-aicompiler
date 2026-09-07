/*! \file test/runtime_profiling_test.cpp
 * \brief 验证执行观测与 ProfileContext 的关联：run/kernel/alloc 事件、错误
 * 状态、关闭态和完成回调结算契约。
 *
 * 改前证据（基线 e426f7e）：对未修改基线以 KXC_PROFILE_ENABLE=ON +
 * KXC_PROFILE_BUNDLE_DIR 运行 op_numeric_llvm_test，bundle 只含编译期事件
 * （compiler/relay_pass/relay_pipeline 共 13 条，phase 均为 complete），
 * 完全没有 runtime_session_run、kernel_submit、kernel_exec、alloc、copy；
 * RuntimeSession 的 Run/RunAsync 全程不产生任何事件。
 */

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "kxc/compiler/compiler.h"
#include "kxc/profiling/profiling.h"
#include "kxc/relay/op.h"
#include "kxc/runtime/compiled_module.h"
#include "kxc/runtime/session.h"
#include "../src/runtime/internal/compiled_module_node.h"
#include "../src/runtime/internal/memory_plan.h"

#ifndef KXC_USE_LLVM
#define KXC_USE_LLVM 0
#endif

namespace {

#define TEST_CHECK(condition, message)                                           \
    do {                                                                          \
        if (!(condition)) {                                                       \
            std::cerr << "[FAIL] " << __FUNCTION__ << ": " << (message) << "\n"; \
            return false;                                                         \
        }                                                                         \
    } while (0)

/*! \brief 捕获标准异常，并可选返回错误文本供上下文断言。 */
bool Throws(const std::function<void()>& function, std::string* message = nullptr) {
    try {
        function();
    } catch (const std::exception& error) {
        if (message) *message = error.what();
        return true;
    }
    return false;
}

DLDataType Float32() { return kxc::runtime::DataTypeFromString("float32"); }

/*! \brief 记录提交次数并把输入加到输出上的同步 fake kernel。 */
class AddLauncher final : public kxc::codegen::KernelLauncher {
public:
    bool IsReady() const noexcept override { return true; }

    kxc::AsyncOperation Launch(
        const kxc::Array<kxc::runtime::NDArray>& arguments,
        const kxc::DeviceStream& stream,
        const kxc::ObjectRef&) const override {
        if (arguments.size() != 3) {
            throw std::invalid_argument("add launcher expects two inputs and one output");
        }
        std::vector<float> lhs(4);
        std::vector<float> rhs(4);
        std::vector<float> output(4);
        arguments[0].CopyToBytes(lhs.data(), lhs.size() * sizeof(float));
        arguments[1].CopyToBytes(rhs.data(), rhs.size() * sizeof(float));
        for (size_t i = 0; i < output.size(); ++i) output[i] = lhs[i] + rhs[i];
        arguments[2].CopyFromBytes(output.data(), output.size() * sizeof(float));
        calls.fetch_add(1);
        kxc::Array<kxc::Storage> retained;
        for (const auto& argument : arguments) retained.push_back(argument.storage());
        return kxc::AsyncOperation::Completed(stream, std::move(retained));
    }

    mutable std::atomic<int> calls{0};
};

/*! \brief 记录提交次数并原样保留参数的 launcher，用于校验失败路径。 */
class RecordingLauncher final : public kxc::codegen::KernelLauncher {
public:
    explicit RecordingLauncher(bool is_ready = true) : ready(is_ready) {}

    bool IsReady() const noexcept override { return ready; }

    kxc::AsyncOperation Launch(
        const kxc::Array<kxc::runtime::NDArray>& arguments,
        const kxc::DeviceStream& stream,
        const kxc::ObjectRef&) const override {
        ++calls;
        if (fail_on_call > 0 && calls == fail_on_call) {
            throw std::runtime_error("recording launcher injected failure");
        }
        kxc::Array<kxc::Storage> retained;
        for (const auto& argument : arguments) retained.push_back(argument.storage());
        return kxc::AsyncOperation::Completed(stream, std::move(retained));
    }

    bool ready{true};
    mutable int calls{0};
    int fail_on_call{0};
};

/*! \brief 输出经 StorageCopyAsync 的 launcher，用于拷贝记账。 */
class CopyLauncher final : public kxc::codegen::KernelLauncher {
public:
    bool IsReady() const noexcept override { return true; }

    kxc::AsyncOperation Launch(
        const kxc::Array<kxc::runtime::NDArray>& arguments,
        const kxc::DeviceStream& stream,
        const kxc::ObjectRef&) const override {
        if (arguments.size() != 2) {
            throw std::invalid_argument("copy launcher expects input and output");
        }
        calls.fetch_add(1);
        return arguments[1].CopyFromAsync(arguments[0], stream);
    }

    mutable std::atomic<int> calls{0};
};

/*! \brief 状态累加 launcher：要求输出别名输入存储并就地累加。 */
class StateAccumulatorLauncher final : public kxc::codegen::KernelLauncher {
public:
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
        std::vector<float> state(4);
        std::vector<float> increment(4);
        arguments[0].CopyToBytes(state.data(), state.size() * sizeof(float));
        arguments[1].CopyToBytes(increment.data(), increment.size() * sizeof(float));
        for (size_t i = 0; i < state.size(); ++i) state[i] += increment[i];
        arguments[2].CopyFromBytes(state.data(), state.size() * sizeof(float));
        calls.fetch_add(1);
        kxc::Array<kxc::Storage> retained;
        for (const auto& argument : arguments) retained.push_back(argument.storage());
        return kxc::AsyncOperation::Completed(stream, std::move(retained));
    }

    mutable std::atomic<int> calls{0};
};

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
        const bool is_input = argument->role == codegen::KernelArgRole::kInput;
        const bool is_constant = argument->role == codegen::KernelArgRole::kConstant;
        const bool is_output = argument->role == codegen::KernelArgRole::kOutput;
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
            signature->symbol, std::move(call_inputs), std::move(call_outputs))},
        std::move(graph_inputs), std::move(constants), std::move(graph_outputs));
}

/*! \brief 构造带可选 profiling 上下文/显式观测器的 CPU fake CompiledModule。 */
kxc::api::CompiledModule MakeModule(
    const kxc::codegen::KernelSignature& signature,
    const kxc::Map<kxc::String, kxc::runtime::NDArray>& constants,
    std::shared_ptr<kxc::codegen::KernelLauncher> launcher,
    std::shared_ptr<kxc::profiling::ProfileContext> profile_context = nullptr,
    std::shared_ptr<kxc::runtime::ExecutionObserver> observer = nullptr) {
    using namespace kxc;
    using namespace kxc::codegen;
    KernelLaunchMetadata metadata(Device::CPU(), CodeGenBackend::kLLVM);
    CompiledKernel executable(signature, metadata, launcher);
    return api::internal::BuildCompiledModule(
        BuildTarget(Device::CPU()),
        {api::internal::CompiledModuleEntry{signature, metadata, executable}},
        constants, std::move(profile_context), std::move(observer));
}

struct BundleFixture {
    kxc::codegen::KernelSignature signature;
    kxc::api::CompiledModule module;
    kxc::runtime::ExecutablePlan plan;
    std::shared_ptr<AddLauncher> launcher;
    std::shared_ptr<kxc::profiling::ProfileContext> context;
};

/*! \brief input + constant -> output 的静态 fixture，绑定一个启用的 bundle。 */
BundleFixture MakeBundleFixture(const std::string& bundle_name) {
    using namespace kxc;
    using namespace kxc::codegen;
    const Device cpu = Device::CPU();
    KernelSignature signature(
        "profiled_session",
        {KernelArgSpec("input", KernelArgRole::kInput, Float32(), {4}, cpu),
         KernelArgSpec("weight", KernelArgRole::kConstant, Float32(), {4}, cpu,
                       1, false, "relay.constant.0"),
         KernelArgSpec("output", KernelArgRole::kOutput, Float32(), {4}, cpu,
                       64, true)});
    runtime::NDArray constant = runtime::NDArray::Zeros({4}, Float32(), cpu);
    Map<String, runtime::NDArray> constants;
    constants.Set(String("relay.constant.0"), constant);

    profiling::ProfileOptions options;
    options.enabled = true;
    options.bundle_dir = (std::filesystem::current_path() /
                          "runtime_profiling_output" / bundle_name)
                             .string();
    auto context = profiling::ProfileContext::Create(options);

    auto launcher = std::make_shared<AddLauncher>();
    return BundleFixture{signature,
                         MakeModule(signature, constants, launcher, context),
                         MakePlan(signature), std::move(launcher),
                         std::move(context)};
}

kxc::runtime::NDArray FilledInput(const kxc::Array<int64_t>& shape,
                                  const std::vector<float>& values) {
    kxc::runtime::NDArray array =
        kxc::runtime::NDArray::Empty(shape, Float32(), kxc::Device::CPU());
    array.CopyFromBytes(values.data(), values.size() * sizeof(float));
    return array;
}

std::vector<float> ReadOutput(const kxc::runtime::NDArray& array) {
    std::vector<float> values(array.NBytes() / sizeof(float), 0.0f);
    array.CopyToBytes(values.data(), values.size() * sizeof(float));
    return values;
}

// ---------------------------------------------------------------------------
// 极小的 events.jsonl 行解析：只取顶层引号字段与数值指标。
// ---------------------------------------------------------------------------

struct EventLine {
    std::string raw;

    std::string StringValue(const char* key) const {
        const std::string needle = std::string("\"") + key + "\":\"";
        const size_t begin = raw.find(needle);
        if (begin == std::string::npos) return "";
        const size_t value = begin + needle.size();
        const size_t end = raw.find('"', value);
        if (end == std::string::npos) return "";
        return raw.substr(value, end - value);
    }

    long long NumberValue(const char* key) const {
        const std::string needle = std::string("\"") + key + "\":";
        const size_t begin = raw.find(needle);
        if (begin == std::string::npos) return -1;
        const size_t value = begin + needle.size();
        size_t end = value;
        while (end < raw.size() &&
               (raw[end] == '-' || (raw[end] >= '0' && raw[end] <= '9'))) {
            ++end;
        }
        if (end == value) return -1;
        return std::stoll(raw.substr(value, end - value));
    }

    bool HasType(const char* event_type) const {
        return StringValue("event_type") == event_type;
    }
};

std::vector<EventLine> LoadEvents(const std::string& bundle_dir) {
    std::vector<EventLine> events;
    std::ifstream ifs(std::filesystem::path(bundle_dir) / "events.jsonl");
    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty()) events.push_back(EventLine{line});
    }
    return events;
}

std::vector<const EventLine*> Filter(const std::vector<EventLine>& events,
                                     const char* event_type) {
    std::vector<const EventLine*> matched;
    for (const auto& event : events) {
        if (event.HasType(event_type)) matched.push_back(&event);
    }
    return matched;
}

/*! \brief 同一 run_id 下 run/submit/exec/alloc 关联，parent 指向 run span。 */
bool TestBundleCorrelationAcrossRunsAndSessions() {
    using namespace kxc;
    BundleFixture fixture = MakeBundleFixture("correlation");
    const Array<int64_t> shape{4};
    runtime::RuntimeSession session(fixture.module, fixture.plan);

    Array<runtime::NDArray> first_outputs = session.Run(
        {FilledInput(shape, {1, 2, 3, 4})});
    Array<runtime::NDArray> second_outputs = session.Run(
        {FilledInput(shape, {10, 20, 30, 40})});
    // 独立 session 共享模块（同一观测器）时也不串数据。
    runtime::RuntimeSession independent(fixture.module, fixture.plan);
    Array<runtime::NDArray> third_outputs = independent.Run(
        {FilledInput(shape, {100, 100, 100, 100})});

    TEST_CHECK(ReadOutput(first_outputs[0]) ==
                   std::vector<float>({1, 2, 3, 4}),
               "first run should add zero constant");
    TEST_CHECK(ReadOutput(second_outputs[0]) ==
                   std::vector<float>({10, 20, 30, 40}),
               "second run should add zero constant");
    TEST_CHECK(ReadOutput(third_outputs[0]) ==
                   std::vector<float>({100, 100, 100, 100}),
               "independent session output should stay correct");
    TEST_CHECK(fixture.launcher->calls.load() == 3, "three kernels should run");

    fixture.context->Flush();
    const std::vector<EventLine> events = LoadEvents(
        fixture.context->bundle_dir());
    const std::vector<const EventLine*> runs = Filter(events, "runtime_session_run");
    TEST_CHECK(runs.size() == 3, "each run should produce one run span");
    const std::vector<const EventLine*> submits = Filter(events, "kernel_submit");
    const std::vector<const EventLine*> execs = Filter(events, "kernel_exec");
    const std::vector<const EventLine*> allocs = Filter(events, "alloc");
    TEST_CHECK(submits.size() == 3 && execs.size() == 3 && allocs.size() == 3,
               "each run should submit one kernel and allocate one output");

    std::string previous_run_id;
    for (const EventLine* run : runs) {
        const std::string run_id = run->StringValue("run_id");
        const std::string span_id = run->StringValue("span_id");
        TEST_CHECK(!run_id.empty() && !span_id.empty(),
                   "run span should carry run and span ids");
        TEST_CHECK(run->StringValue("parent_span_id").empty(),
                   "run span is the root of its run");
        TEST_CHECK(run->StringValue("status") == "ok",
                   "successful runs close with status ok");
        TEST_CHECK(run->StringValue("component") == "runtime_session",
                   "run span belongs to the runtime_session component");
        TEST_CHECK(run->raw.find("\"timing\":\"host_execute\"") != std::string::npos,
                   "run span timing must be host_execute");
        TEST_CHECK(run->NumberValue("input_count") == 1 &&
                       run->NumberValue("kernel_count") == 1 &&
                       run->NumberValue("submit_count") == 1,
                   "run metrics should count one input and one kernel");
        TEST_CHECK(previous_run_id != run_id,
                   "repeated runs must not share a run id");
        previous_run_id = run_id;

        int submits_for_run = 0;
        int execs_for_run = 0;
        int allocs_for_run = 0;
        long long submit_ts = -1;
        for (const EventLine* submit : submits) {
            if (submit->StringValue("run_id") != run_id) continue;
            ++submits_for_run;
            TEST_CHECK(submit->StringValue("phase") == "submit",
                       "kernel_submit is an instant submit-phase event");
            TEST_CHECK(submit->StringValue("parent_span_id") == span_id,
                       "kernel_submit parent must be the run span");
            TEST_CHECK(submit->raw.find("\"timing\":\"host_submit\"") != std::string::npos,
                       "kernel_submit timing must be host_submit");
            TEST_CHECK(submit->StringValue("kernel_symbol") == "profiled_session",
                       "kernel_submit carries the kernel symbol");
            submit_ts = submit->NumberValue("ts_ns");
        }
        for (const EventLine* exec : execs) {
            if (exec->StringValue("run_id") != run_id) continue;
            ++execs_for_run;
            TEST_CHECK(exec->StringValue("parent_span_id") == span_id,
                       "kernel_exec parent must be the run span");
            TEST_CHECK(exec->raw.find("\"timing\":\"host_execute\"") != std::string::npos,
                       "synchronous CPU completion is host_execute");
            TEST_CHECK(exec->NumberValue("ts_ns") <= submit_ts,
                       "kernel_exec starts before the host submit completes");
            TEST_CHECK(exec->StringValue("kernel_symbol") == "profiled_session",
                       "kernel_exec carries the kernel symbol");
        }
        for (const EventLine* alloc : allocs) {
            if (alloc->StringValue("run_id") != run_id) continue;
            ++allocs_for_run;
            TEST_CHECK(alloc->StringValue("parent_span_id") == span_id,
                       "alloc parent must be the run span");
            TEST_CHECK(alloc->NumberValue("bytes") == 16,
                       "output allocation covers 4 float32 elements");
            TEST_CHECK(alloc->raw.find("\"reused\":\"false\"") != std::string::npos,
                       "a fresh output allocation is not reused");
        }
        TEST_CHECK(submits_for_run == 1 && execs_for_run == 1 && allocs_for_run == 1,
                   "kernel and allocation events must correlate to one run");
    }
    return true;
}

/*! \brief profiling 关闭时没有 runtime 事件，输出与开启时逐位一致。 */
bool TestDisabledProfilingKeepsOutputIdentical() {
    using namespace kxc;
    BundleFixture observed = MakeBundleFixture("disabled_reference");
    runtime::RuntimeSession observed_session(observed.module, observed.plan);
    const Array<int64_t> shape{4};
    const Array<runtime::NDArray> observed_outputs =
        observed_session.Run({FilledInput(shape, {7, 8, 9, 10})});

    // 关闭态一：模块没有 profiling 上下文，也没有观测器。
    using namespace kxc::codegen;
    const Device cpu = Device::CPU();
    KernelSignature signature = observed.signature;
    Map<String, runtime::NDArray> constants;
    constants.Set(String("relay.constant.0"),
                  runtime::NDArray::Zeros({4}, Float32(), cpu));
    auto plain_launcher = std::make_shared<AddLauncher>();
    api::CompiledModule plain_module =
        MakeModule(signature, constants, plain_launcher);
    TEST_CHECK(std::string(plain_module.GetProfileBundlePath()).empty(),
               "module without profiling context has no bundle path");
    runtime::RuntimeSession plain_session(plain_module, observed.plan);
    const Array<runtime::NDArray> plain_outputs =
        plain_session.Run({FilledInput(shape, {7, 8, 9, 10})});

    // 关闭态二：上下文存在但 enabled=false，观测器不会被装配。
    profiling::ProfileOptions off_options;
    off_options.enabled = false;
    off_options.bundle_dir = (std::filesystem::current_path() /
                              "runtime_profiling_output" / "disabled_context")
                                 .string();
    auto off_context = profiling::ProfileContext::Create(off_options);
    auto off_launcher = std::make_shared<AddLauncher>();
    api::CompiledModule off_module =
        MakeModule(signature, constants, off_launcher, off_context);
    runtime::RuntimeSession off_session(off_module, observed.plan);
    const Array<runtime::NDArray> off_outputs =
        off_session.Run({FilledInput(shape, {7, 8, 9, 10})});

    const std::vector<float> expected = ReadOutput(observed_outputs[0]);
    TEST_CHECK(expected == std::vector<float>({7, 8, 9, 10}),
               "observed run should add the zero constant");
    TEST_CHECK(ReadOutput(plain_outputs[0]) == expected &&
                   ReadOutput(off_outputs[0]) == expected,
               "outputs must be bitwise identical with observation disabled");
    TEST_CHECK(plain_launcher->calls.load() == 1 && off_launcher->calls.load() == 1,
               "disabled observation must not change kernel submission");

    off_context->Flush();
    TEST_CHECK(LoadEvents(off_context->bundle_dir()).empty(),
               "profiling disabled must not produce runtime events");
    return true;
}

/*! \brief 每个钩子都抛异常的观测器不能改变执行结果。 */
class ThrowingObserver final : public kxc::runtime::ExecutionObserver {
public:
    kxc::runtime::ExecutionRunCorrelation OnRunStart(
        const kxc::runtime::ExecutionRunStart&) override {
        throw std::runtime_error("observer failure");
    }
    void OnRunEnd(const kxc::runtime::ExecutionRunEnd&,
                  const kxc::runtime::ExecutionRunCorrelation&) override {
        throw std::runtime_error("observer failure");
    }
    void OnKernelBegin(const kxc::runtime::KernelSubmitInfo&,
                       const kxc::runtime::ExecutionRunCorrelation&) override {
        throw std::runtime_error("observer failure");
    }
    kxc::runtime::ExecutionCompletionCallback OnKernelSubmitted(
        const kxc::runtime::KernelSubmitInfo&,
        const kxc::runtime::ExecutionRunCorrelation&) override {
        throw std::runtime_error("observer failure");
    }
    void OnAllocation(const kxc::runtime::AllocationInfo&,
                      const kxc::runtime::ExecutionRunCorrelation&) override {
        throw std::runtime_error("observer failure");
    }
    void OnCopy(const kxc::runtime::CopyInfo&,
                const kxc::runtime::ExecutionRunCorrelation&) override {
        throw std::runtime_error("observer failure");
    }
    kxc::runtime::ExecutionCompletionCallback OnCopySubmitted(
        const kxc::runtime::CopyInfo&,
        const kxc::runtime::ExecutionRunCorrelation&) override {
        throw std::runtime_error("observer failure");
    }
};

bool TestThrowingObserverDoesNotChangeExecution() {
    using namespace kxc;
    using namespace kxc::codegen;
    const Device cpu = Device::CPU();
    KernelSignature signature(
        "throwing_observer_session",
        {KernelArgSpec("input", KernelArgRole::kInput, Float32(), {4}, cpu),
         KernelArgSpec("weight", KernelArgRole::kConstant, Float32(), {4}, cpu,
                       1, false, "relay.constant.0"),
         KernelArgSpec("output", KernelArgRole::kOutput, Float32(), {4}, cpu,
                       64, true)});
    Map<String, runtime::NDArray> constants;
    constants.Set(String("relay.constant.0"),
                  runtime::NDArray::Zeros({4}, Float32(), cpu));
    auto launcher = std::make_shared<AddLauncher>();
    api::CompiledModule module = MakeModule(
        signature, constants, launcher, nullptr,
        std::make_shared<ThrowingObserver>());
    runtime::RuntimeSession session(module, MakePlan(signature));
    const Array<runtime::NDArray> outputs =
        session.Run({FilledInput({4}, {3, 3, 3, 3})});
    TEST_CHECK(ReadOutput(outputs[0]) == std::vector<float>({3, 3, 3, 3}) &&
                   launcher->calls.load() == 1,
               "a throwing observer must not alter execution results");
    return true;
}

/*! \brief 辅助：取 run span 的 run_id（单一 run 场景）。 */
std::string runs_run_id(const std::vector<EventLine>& events) {
    for (const auto& event : events) {
        if (event.HasType("runtime_session_run")) return event.StringValue("run_id");
    }
    return "";
}

/*! \brief 校验失败必须抛原异常，且 run 只有 error 状态、零 kernel_submit。 */
bool TestValidationErrorRecordsErrorRun() {
    using namespace kxc;
    BundleFixture fixture = MakeBundleFixture("validation_error");
    runtime::RuntimeSession session(fixture.module, fixture.plan);
    std::string message;
    TEST_CHECK(Throws([&] {
                   session.Run({FilledInput({4}, {1, 1, 1, 1}),
                                FilledInput({4}, {1, 1, 1, 1})});
               }, &message) &&
                   message.find("input count expected 1, actual 2") !=
                       std::string::npos,
               "input validation must keep its original message");
    TEST_CHECK(fixture.launcher->calls.load() == 0,
               "validation failure must not reach the backend");

    fixture.context->Flush();
    const std::vector<EventLine> events = LoadEvents(fixture.context->bundle_dir());
    const std::vector<const EventLine*> runs = Filter(events, "runtime_session_run");
    TEST_CHECK(runs.size() == 1, "the rejected run records exactly one span");
    TEST_CHECK(runs[0]->StringValue("status") == "error",
               "the rejected run closes with status error");
    TEST_CHECK(runs[0]->raw.find("input count expected 1, actual 2") !=
                   std::string::npos,
               "the run span carries the original exception text");
    TEST_CHECK(runs[0]->NumberValue("submit_count") == 0,
               "no kernel was submitted before validation failed");
    TEST_CHECK(Filter(events, "kernel_submit").empty() &&
                   Filter(events, "kernel_exec").empty(),
               "validation failure must not produce submit or exec events");
    return true;
}

/*! \brief launcher 注入失败必须抛原异常，并产生 error 事件。 */
bool TestLauncherFailureRecordsErrorRun() {
    using namespace kxc;
    using namespace kxc::codegen;
    const Device cpu = Device::CPU();
    KernelSignature signature(
        "failing_kernel_session",
        {KernelArgSpec("input", KernelArgRole::kInput, Float32(), {4}, cpu),
         KernelArgSpec("output", KernelArgRole::kOutput, Float32(), {4}, cpu,
                       64, true)});
    Map<String, runtime::NDArray> constants;
    auto launcher = std::make_shared<RecordingLauncher>();
    launcher->fail_on_call = 1;
    profiling::ProfileOptions options;
    options.enabled = true;
    options.bundle_dir = (std::filesystem::current_path() /
                          "runtime_profiling_output" / "launcher_error")
                             .string();
    auto context = profiling::ProfileContext::Create(options);
    api::CompiledModule module =
        MakeModule(signature, constants, launcher, context);
    runtime::RuntimeSession session(module, MakePlan(signature));

    std::string message;
    TEST_CHECK(Throws([&] { session.Run({FilledInput({4}, {1, 1, 1, 1})}); },
                      &message) &&
                   message == "recording launcher injected failure",
               "launcher failure must keep its type and text");

    context->Flush();
    const std::vector<EventLine> events = LoadEvents(context->bundle_dir());
    const std::vector<const EventLine*> runs = Filter(events, "runtime_session_run");
    TEST_CHECK(runs.size() == 1 && runs[0]->StringValue("status") == "error",
               "the failed run closes with status error");
    TEST_CHECK(runs[0]->raw.find("recording launcher injected failure") !=
                   std::string::npos,
               "the run span carries the launcher failure text");
    TEST_CHECK(runs[0]->NumberValue("submit_count") == 0,
               "a failed launch never completed its submit action");
    TEST_CHECK(Filter(events, "kernel_submit").empty() &&
                   Filter(events, "kernel_exec").empty(),
               "no submit or exec events exist for a failed launch");
    TEST_CHECK(launcher->calls == 1, "the backend did observe the launch attempt");
    return true;
}

/*! \brief 拷贝记账：异步拷贝记提交与观测完成两点，同一 run 关联。 */
bool TestCopyAccountingOnRunPath() {
    using namespace kxc;
    using namespace kxc::codegen;
    const Device cpu = Device::CPU();
    KernelSignature signature(
        "copy_kernel_session",
        {KernelArgSpec("input", KernelArgRole::kInput, Float32(), {4}, cpu),
         KernelArgSpec("output", KernelArgRole::kOutput, Float32(), {4}, cpu,
                       64, true)});
    Map<String, runtime::NDArray> constants;
    auto launcher = std::make_shared<CopyLauncher>();
    profiling::ProfileOptions options;
    options.enabled = true;
    options.bundle_dir = (std::filesystem::current_path() /
                          "runtime_profiling_output" / "copy_accounting")
                             .string();
    auto context = profiling::ProfileContext::Create(options);
    api::CompiledModule module =
        MakeModule(signature, constants, launcher, context);
    runtime::RuntimeSession session(module, MakePlan(signature));
    const Array<runtime::NDArray> outputs =
        session.Run({FilledInput({4}, {5, 5, 5, 5})});
    TEST_CHECK(ReadOutput(outputs[0]) == std::vector<float>({5, 5, 5, 5}),
               "copy-backed kernel must forward its input");
    TEST_CHECK(launcher->calls.load() == 1, "one copy-backed kernel ran");

    context->Flush();
    const std::vector<EventLine> events = LoadEvents(context->bundle_dir());
    const std::vector<const EventLine*> runs = Filter(events, "runtime_session_run");
    TEST_CHECK(runs.size() == 1 && runs[0]->StringValue("status") == "ok",
               "the copy run closes with status ok");
    const std::vector<const EventLine*> copies = Filter(events, "copy");
    // 模块构建期的常量快照复制也是 copy 事件（无 run 关联）；
    // 运行内应恰好有提交与观测完成两点。
    TEST_CHECK(copies.size() == 3,
               "module snapshot copy plus the run's async copy points");
    int submits = 0;
    int completes = 0;
    for (const EventLine* copy : copies) {
        if (copy->StringValue("run_id") != runs[0]->StringValue("run_id")) {
            TEST_CHECK(copy->StringValue("run_id").empty() &&
                           copy->raw.find("\"copy_kind\":\"constant_snapshot\"") !=
                               std::string::npos,
                       "the non-run copy is the module constant snapshot");
            continue;
        }
        TEST_CHECK(copy->NumberValue("bytes") == 16, "copy bytes cover 4 floats");
        TEST_CHECK(copy->StringValue("device") == "cpu:0" &&
                       copy->raw.find("\"from_device\":\"cpu:0\"") !=
                           std::string::npos &&
                       copy->raw.find("\"to_device\":\"cpu:0\"") !=
                           std::string::npos,
                   "copy endpoints carry both devices");
        if (copy->StringValue("phase") == "submit") {
            ++submits;
            TEST_CHECK(copy->raw.find("\"timing\":\"host_submit\"") !=
                           std::string::npos,
                       "copy submit timing is host_submit");
            TEST_CHECK(copy->StringValue("parent_span_id") ==
                           runs[0]->StringValue("span_id"),
                       "copy submit parents to the run span");
        } else {
            ++completes;
            TEST_CHECK(copy->raw.find("\"timing\":\"host_execute\"") !=
                           std::string::npos,
                       "a CPU copy observed complete at submit is host_execute");
            TEST_CHECK(copy->StringValue("parent_span_id") ==
                           runs[0]->StringValue("span_id"),
                       "copy completion parents to the run span");
        }
    }
    TEST_CHECK(submits == 1 && completes == 1,
               "exactly one submit point and one completion point");
    return true;
}

/*! \brief state 会话：构造期初始化分配、别名输出和状态复用的记账。 */
bool TestStateAliasAndConstructionAccounting() {
    using namespace kxc;
    using namespace kxc::codegen;
    const Device cpu = Device::CPU();
    KernelSignature signature(
        "state_accumulate",
        {KernelArgSpec("state", KernelArgRole::kInput, Float32(), {4}, cpu, 4),
         KernelArgSpec("increment", KernelArgRole::kInput, Float32(), {4}, cpu, 4),
         KernelArgSpec("next_state", KernelArgRole::kOutput, Float32(), {4}, cpu,
                       64, true)});
    auto launcher = std::make_shared<StateAccumulatorLauncher>();
    profiling::ProfileOptions options;
    options.enabled = true;
    options.bundle_dir = (std::filesystem::current_path() /
                          "runtime_profiling_output" / "state_accounting")
                             .string();
    auto context = profiling::ProfileContext::Create(options);
    api::CompiledModule module =
        MakeModule(signature, {}, launcher, context);
    runtime::ExecutablePlan plan(
        {runtime::ValueSpec(0, 0, {4}, Float32(), cpu, false, false, false,
                             false, false, true),
         runtime::ValueSpec(1, 1, {4}, Float32(), cpu, true),
         runtime::ValueSpec(2, 0, {4}, Float32(), cpu, false, false, true,
                             true, false, false, 0,
                             runtime::ValueWriteMode::kInPlace)},
        {runtime::KernelCall("state_accumulate", {0, 1}, {2})},
        {1}, {}, {2}, {0});
    runtime::RuntimeSession session(module, plan);
    Array<runtime::NDArray> first = session.Run({FilledInput({4}, {1, 2, 3, 4})});
    // 别名语义：输出与 state 共享存储，必须在第二次运行前取值。
    const std::vector<float> first_actual = ReadOutput(first[0]);
    TEST_CHECK(first_actual == std::vector<float>({1, 2, 3, 4}),
               "first state run should zero-accumulate");
    Array<runtime::NDArray> second =
        session.Run({FilledInput({4}, {10, 20, 30, 40})});
    const std::vector<float> second_actual = ReadOutput(second[0]);
    TEST_CHECK(second_actual == std::vector<float>({11, 22, 33, 44}),
               "second state run should accumulate across runs");

    context->Flush();
    const std::vector<EventLine> events = LoadEvents(context->bundle_dir());
    const std::vector<const EventLine*> allocs = Filter(events, "alloc");
    // 构造期 state 初始化分配 + 每次运行的别名输出 = 3 条 alloc。
    TEST_CHECK(allocs.size() == 3, "construction and both runs allocate once each");
    int empty_run_allocs = 0;
    int alias_allocs = 0;
    for (const EventLine* alloc : allocs) {
        if (alloc->StringValue("run_id").empty()) {
            ++empty_run_allocs;
            TEST_CHECK(alloc->raw.find("\"alloc_kind\":\"fresh\"") !=
                           std::string::npos,
                       "the state initialization is a fresh allocation");
        } else {
            TEST_CHECK(alloc->raw.find("\"alloc_kind\":\"alias\"") !=
                           std::string::npos,
                       "in-place state output is recorded as alias");
            TEST_CHECK(alloc->raw.find("\"reused\":\"true\"") !=
                           std::string::npos,
                       "alias bindings are not counted as new allocations");
            ++alias_allocs;
        }
    }
    TEST_CHECK(empty_run_allocs == 1 && alias_allocs == 2,
               "construction allocation has no run id; each run aliases once");
    TEST_CHECK(Filter(events, "kernel_exec").size() == 2,
               "each state run executes one kernel");
    return true;
}

/*! \brief 计划内的中间存储复用必须以 reused=true 记账。 */
bool TestPlannedReuseAccounting() {
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
            {KernelArgSpec("input", KernelArgRole::kInput, Float32(), shape, cpu),
             KernelArgSpec("output", KernelArgRole::kOutput, Float32(), shape,
                           cpu, 16, true)});
        KernelLaunchMetadata metadata(cpu, CodeGenBackend::kLLVM);
        auto launcher = std::make_shared<RecordingLauncher>();
        entries.push_back({signature, metadata,
                           codegen::CompiledKernel(signature, metadata, launcher)});
        launchers.push_back(std::move(launcher));
        calls.push_back(runtime::KernelCall(symbol, {i}, {i + 1}));
    }
    profiling::ProfileOptions options;
    options.enabled = true;
    options.bundle_dir = (std::filesystem::current_path() /
                          "runtime_profiling_output" / "reuse_accounting")
                             .string();
    auto context = profiling::ProfileContext::Create(options);
    api::CompiledModule module = api::internal::BuildCompiledModule(
        BuildTarget(cpu), std::move(entries), {}, context);
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
    session.Run({FilledInput(shape, {1, 1, 1, 1})});

    context->Flush();
    const std::vector<EventLine> events = LoadEvents(context->bundle_dir());
    const std::vector<const EventLine*> allocs = Filter(events, "alloc");
    TEST_CHECK(allocs.size() == 4, "four outputs allocate across the chain");
    int fresh = 0;
    int reused = 0;
    for (const EventLine* alloc : allocs) {
        TEST_CHECK(alloc->StringValue("run_id") == runs_run_id(events),
                   "chain allocations correlate to the run");
        if (alloc->raw.find("\"alloc_kind\":\"reuse\"") != std::string::npos) {
            TEST_CHECK(alloc->raw.find("\"reused\":\"true\"") != std::string::npos,
                       "reuse events are marked reused");
            ++reused;
        } else {
            TEST_CHECK(alloc->raw.find("\"alloc_kind\":\"fresh\"") !=
                           std::string::npos,
                       "non-reuse allocations are fresh");
            ++fresh;
        }
        TEST_CHECK(alloc->NumberValue("bytes") == 16, "chain allocations hold 4 floats");
    }
    TEST_CHECK(fresh == 3 && reused == 1,
               "the planned shared intermediate is recorded as reuse");
    return true;
}

/*! \brief 完成回调在注册即触发、Wait/IsReady/析构之间恰好结算一次。 */
int g_completion_fires = 0;  // NOLINT — 简单计数器，测试单线程使用

bool TestCompletionSettlesExactlyOnce() {
    using namespace kxc;
    const DeviceStream stream = DeviceStream::Default(Device::CPU());
    std::vector<bool> registrations;

    // 已完成句柄：注册即触发一次（at_registration=true）。
    {
        g_completion_fires = 0;
        AsyncOperation completed = AsyncOperation::Completed(stream);
        completed.ObserveCompletion([&registrations](bool at_registration) {
            registrations.push_back(at_registration);
            ++g_completion_fires;
        });
        TEST_CHECK(g_completion_fires == 1 && registrations.size() == 1 &&
                       registrations.back(),
                   "registering on a completed handle fires immediately");
        completed.Wait();
        TEST_CHECK(completed.IsReady() && g_completion_fires == 1,
                   "Wait/IsReady after settlement must not fire again");
    }
    TEST_CHECK(g_completion_fires == 1,
               "destruction after settlement must not fire again");

    // 未完成句柄：IsReady 是第一个观测点；等待中不提前报完成。
    {
        g_completion_fires = 0;
        registrations.clear();
        AsyncOperation pending =
            AsyncOperation::Pending(stream, nullptr, {});
        pending.ObserveCompletion([&registrations](bool at_registration) {
            registrations.push_back(at_registration);
            ++g_completion_fires;
        });
        TEST_CHECK(g_completion_fires == 0,
                   "a pending handle must not report completion early");
        TEST_CHECK(pending.IsReady(),
                   "a null-event pending handle completes on observation");
        TEST_CHECK(g_completion_fires == 1 && registrations.size() == 1 &&
                       !registrations.back(),
                   "IsReady settles once with at_registration=false");
        pending.Wait();
        TEST_CHECK(pending.IsReady() && g_completion_fires == 1,
                   "Wait/IsReady after settlement must not fire again");
    }
    TEST_CHECK(g_completion_fires == 1,
               "destruction after settlement must not fire again");

    // Wait 观测点结算一次。
    {
        g_completion_fires = 0;
        AsyncOperation pending =
            AsyncOperation::Pending(stream, nullptr, {});
        pending.ObserveCompletion([](bool) { ++g_completion_fires; });
        TEST_CHECK(g_completion_fires == 0, "not fired before observation");
        pending.Wait();
        TEST_CHECK(g_completion_fires == 1, "Wait settles exactly once");
    }
    TEST_CHECK(g_completion_fires == 1, "destruction after Wait must not fire again");

    // 析构观测点结算一次（回调不得持有句柄，否则析构永不发生）。
    {
        g_completion_fires = 0;
        AsyncOperation pending;
        {
            AsyncOperation scoped =
                AsyncOperation::Pending(stream, nullptr, {});
            scoped.ObserveCompletion([](bool) { ++g_completion_fires; });
            TEST_CHECK(g_completion_fires == 0, "not fired before destruction");
            pending = scoped;
        }
        // pending 仍持有引用：句柄未析构，尚未结算。
        TEST_CHECK(g_completion_fires == 0,
                   "holding a second reference delays settlement");
        pending = AsyncOperation();
        TEST_CHECK(g_completion_fires == 1,
                   "destruction settles the observation exactly once");
    }

    // 回调内不得再等待同一句柄：completed 已为 true，嵌套 Wait 直接返回。
    {
        AsyncOperation pending =
            AsyncOperation::Pending(stream, nullptr, {});
        pending.ObserveCompletion([&pending](bool) { pending.Wait(); });
        pending.Wait();
        TEST_CHECK(true, "a callback re-entering Wait must not deadlock");
    }

    // 回调抛出不得传播；已完成句柄上的迟到注册独立触发一次（注册即触发），
    // 已结算的存储回调不会被重复触发。
    {
        g_completion_fires = 0;
        AsyncOperation pending =
            AsyncOperation::Pending(stream, nullptr, {});
        pending.ObserveCompletion([](bool) {
            ++g_completion_fires;
            throw std::runtime_error("observer failure");
        });
        pending.Wait();
        TEST_CHECK(g_completion_fires == 1,
                   "a throwing callback still settles and never propagates");
        bool threw = false;
        try {
            pending.ObserveCompletion([](bool) { ++g_completion_fires; });
        } catch (...) {
            threw = true;
        }
        TEST_CHECK(!threw && g_completion_fires == 2,
                   "a late registration on a completed handle fires once more "
                   "without disturbing the settled callback");
    }
    return true;
}

/*! \brief 记账路径自身的分配不得递归触发钩子（窄的重入保护）。 */
class ReentrantAllocObserver final : public kxc::runtime::ExecutionObserver {
public:
    kxc::runtime::ExecutionRunCorrelation OnRunStart(
        const kxc::runtime::ExecutionRunStart&) override {
        return {};
    }
    void OnRunEnd(const kxc::runtime::ExecutionRunEnd&,
                  const kxc::runtime::ExecutionRunCorrelation&) override {}
    void OnKernelBegin(const kxc::runtime::KernelSubmitInfo&,
                       const kxc::runtime::ExecutionRunCorrelation&) override {}
    kxc::runtime::ExecutionCompletionCallback OnKernelSubmitted(
        const kxc::runtime::KernelSubmitInfo&,
        const kxc::runtime::ExecutionRunCorrelation&) override {
        return nullptr;
    }
    void OnAllocation(const kxc::runtime::AllocationInfo&,
                      const kxc::runtime::ExecutionRunCorrelation&) override {
        ++allocations;
        // 观测记录路径自身分配内存：内层 Storage::Alloc 不得再次上报。
        kxc::runtime::NDArray nested =
            kxc::runtime::NDArray::Empty({4}, Float32(), kxc::Device::CPU());
        (void)nested;
    }
    void OnCopy(const kxc::runtime::CopyInfo&,
                const kxc::runtime::ExecutionRunCorrelation&) override {}
    kxc::runtime::ExecutionCompletionCallback OnCopySubmitted(
        const kxc::runtime::CopyInfo&,
        const kxc::runtime::ExecutionRunCorrelation&) override {
        return nullptr;
    }

    int allocations{0};
};

bool TestRecordPathDoesNotRecurse() {
    ReentrantAllocObserver observer;
    {
        kxc::runtime::ExecutionObservationScope scope(&observer,
                                                      kxc::runtime::ExecutionRunCorrelation{});
        kxc::runtime::NDArray array = kxc::runtime::NDArray::Empty(
            {4}, Float32(), kxc::Device::CPU());
        (void)array;
    }
    TEST_CHECK(observer.allocations == 1,
               "the nested allocation inside the record path must not re-dispatch");
    return true;
}

#if KXC_USE_LLVM
/*! \brief 值返回 helper 内不能使用 TEST_CHECK（它 return false）；
 *  违约直接抛出，由测试入口统一转为失败。 */
void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

/*! \brief 把主机字节负载按输入顺序填进计划输入数组。 */
kxc::Array<kxc::runtime::NDArray> FillPlanInputs(
    const kxc::runtime::ExecutablePlan& plan,
    const std::vector<std::vector<uint8_t>>& payloads) {
    using namespace kxc;
    const auto values = plan.values();
    const auto find_value = [&](int64_t value_id) {
        for (const auto& value : values) {
            if (value->value_id == value_id) return value;
        }
        throw std::runtime_error("plan references an unknown value");
    };
    Array<runtime::NDArray> inputs;
    const auto input_ids = plan.input_value_ids();
    Require(payloads.size() == input_ids.size(),
            "host payload count must match the graph ABI");
    for (size_t i = 0; i < input_ids.size(); ++i) {
        const auto spec = find_value(input_ids[i]);
        runtime::NDArray array =
            runtime::NDArray::Empty(spec.shape(), spec->dtype, spec->device);
        Require(array.NBytes() == payloads[i].size(),
                "host payload byte count must match the input spec");
        array.CopyFromBytes(payloads[i].data(), payloads[i].size());
        inputs.push_back(std::move(array));
    }
    return inputs;
}

std::vector<uint8_t> ReadBytesOutput(const kxc::runtime::NDArray& array) {
    std::vector<uint8_t> bytes(array.NBytes(), 0);
    array.CopyToBytes(bytes.data(), bytes.size());
    return bytes;
}

std::vector<uint8_t> FloatBytes(const std::vector<float>& values) {
    std::vector<uint8_t> bytes(values.size() * sizeof(float), 0);
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

std::vector<float> BytesAsFloat(const std::vector<uint8_t>& bytes) {
    std::vector<float> values(bytes.size() / sizeof(float), 0.0f);
    std::memcpy(values.data(), bytes.data(), values.size() * sizeof(float));
    return values;
}

/*! \brief 真实 Compiler::Compile 的 LLVM Where fixture 运行证据：
 *  数值不变，且同一 run_id 下出现 runtime_session_run/kernel_submit/
 *  kernel_exec/alloc，内核与分配事件的 parent_span_id 指向 run span。
 *  构图方式与 op_numeric_llvm_test.cpp 的 TestWhere 一致（矩阵归属）。
 *  payloads 按计划输入顺序给出主机字节；reference 为期望的输出字节。
 *  返回观测运行的原始输出字节，供调用方做数值断言。 */
std::vector<uint8_t> RunWhereViaCompiler(
    const std::string& bundle_name, const kxc::Function& func,
    const std::vector<std::vector<uint8_t>>& payloads,
    const std::vector<uint8_t>& reference) {
    using namespace kxc;
    profiling::ProfileOptions options;
    options.enabled = true;
    options.ir_capture_mode = profiling::IRCaptureMode::kDisabled;
    options.bundle_dir = (std::filesystem::current_path() /
                          "runtime_profiling_output" / bundle_name)
                             .string();
    const std::string bundle_dir = options.bundle_dir;

    // 关闭观测的同构运行用于逐位对照。
    std::vector<uint8_t> unobserved;
    {
        auto plain_config = api::CompileConfig::Create(BuildTarget(Device::CPU()), 0);
        auto plain_compiled = api::Compiler::Compile(func, plain_config);
        Require(plain_compiled.module().IsReady(),
                   "Where LLVM module should be ready");
        runtime::RuntimeSession plain_session(plain_compiled.module(),
                                              plain_compiled.plan());
        const auto outputs = plain_session.Run(FillPlanInputs(plain_compiled.plan(),
                                                              payloads));
        Require(outputs.size() == 1, "Where produces one output");
        unobserved = ReadBytesOutput(outputs[0]);
        Require(unobserved == reference,
                   "the unobserved run must match the reference");
    }

    std::vector<uint8_t> observed;
    {
        auto config =
            api::CompileConfig::Create(BuildTarget(Device::CPU()), 0, options);
        auto compiled = api::Compiler::Compile(func, config);
        Require(compiled.module().IsReady() && compiled.plan().defined(),
                   "Where LLVM module should be ready");
        runtime::RuntimeSession session(compiled.module(), compiled.plan());
        const auto outputs =
            session.Run(FillPlanInputs(compiled.plan(), payloads));
        observed = ReadBytesOutput(outputs[0]);
        Require(observed == unobserved,
                   "observed and unobserved runs must be bitwise identical");
        // 模块与 session 析构会触发其 ProfileContext 的最后一次 Flush。
    }

    const std::vector<EventLine> events = LoadEvents(bundle_dir);
    const std::vector<const EventLine*> runs = Filter(events, "runtime_session_run");
    Require(runs.size() == 1 && runs[0]->StringValue("status") == "ok",
               "the LLVM run records one successful run span");
    const std::string run_id = runs[0]->StringValue("run_id");
    const std::string span_id = runs[0]->StringValue("span_id");
    Require(!run_id.empty() && !span_id.empty(),
               "the LLVM run span carries correlation ids");
    int submits = 0;
    int execs = 0;
    int allocs = 0;
    for (const EventLine* submit : Filter(events, "kernel_submit")) {
        if (submit->StringValue("run_id") != run_id) continue;
        ++submits;
        Require(submit->StringValue("parent_span_id") == span_id,
                   "LLVM kernel_submit parents to the run span");
        Require(submit->raw.find("\"timing\":\"host_submit\"") !=
                       std::string::npos,
                   "LLVM kernel_submit timing is host_submit");
    }
    for (const EventLine* exec : Filter(events, "kernel_exec")) {
        if (exec->StringValue("run_id") != run_id) continue;
        ++execs;
        Require(exec->StringValue("parent_span_id") == span_id,
                   "LLVM kernel_exec parents to the run span");
        Require(exec->raw.find("\"timing\":\"host_execute\"") !=
                       std::string::npos,
                   "LLVM kernel_exec closes at host-observed completion");
        Require(!exec->StringValue("kernel_symbol").empty(),
                   "LLVM kernel_exec carries the real kernel symbol");
    }
    for (const EventLine* alloc : Filter(events, "alloc")) {
        if (alloc->StringValue("run_id") != run_id) continue;
        ++allocs;
        Require(alloc->StringValue("parent_span_id") == span_id,
                   "LLVM alloc parents to the run span");
    }
    Require(submits == 1 && execs == 1 && allocs >= 1,
               "one LLVM run submits one kernel and allocates its outputs");
    Require(!Filter(events, "run_pass").empty(),
               "the same bundle also carries the compile-stage events");
    return observed;
}

/*! \brief Where 数值证据 + 常量快照复制证据。 */
bool TestWhereRuntimeBundleEvidence() {
    using namespace kxc;
    // 矩阵归属的 Where：condition {2,1} 广播选择 x 标量或 y {1,3}。
    const std::vector<uint8_t> condition_data = {1, 0};
    Var condition("condition", TensorType({2, 1}, "bool"));
    Var x("x", TensorType({}, "float32"));
    Var y("y", TensorType({1, 3}, "float32"));
    Function func({condition, x, y},
                  Call(relay::Op::Get("where"), {condition, x, y}));
    const std::vector<uint8_t> observed = RunWhereViaCompiler(
        "where_runtime", func,
        {condition_data, FloatBytes({10}), FloatBytes({1, 2, 3})},
        FloatBytes({10, 10, 10, 1, 2, 3}));
    TEST_CHECK(BytesAsFloat(observed) == std::vector<float>({10, 10, 10, 1, 2, 3}),
               "Where result must match the reference output");

    // bool 常量变体：额外证明模块构建期的常量快照复制进入 bundle。
    runtime::NDArray constant_condition = runtime::NDArray::Empty(
        {2, 1}, runtime::DataTypeFromString("bool"), Device::CPU());
    constant_condition.CopyFromBytes(condition_data.data(), condition_data.size());
    Var bool_x("bool_x", TensorType({1, 3}, "bool"));
    Var bool_y("bool_y", TensorType({2, 1}, "bool"));
    Function bool_func({bool_x, bool_y},
                       Call(relay::Op::Get("where"),
                            {Constant(constant_condition), bool_x, bool_y}));
    const std::vector<uint8_t> bool_out = RunWhereViaCompiler(
        "where_bool_constant_runtime", bool_func, {{1, 0, 1}, {0, 1}},
        {1, 0, 1, 1, 1, 1});
    TEST_CHECK(bool_out == std::vector<uint8_t>({1, 0, 1, 1, 1, 1}),
               "Where bool constant variant must match the reference output");

    const std::vector<EventLine> events = LoadEvents(
        (std::filesystem::current_path() / "runtime_profiling_output" /
         "where_bool_constant_runtime")
            .string());
    bool saw_snapshot_copy = false;
    for (const EventLine* copy : Filter(events, "copy")) {
        if (copy->raw.find("\"copy_kind\":\"constant_snapshot\"") !=
            std::string::npos) {
            saw_snapshot_copy = true;
            TEST_CHECK(copy->NumberValue("bytes") == 2,
                       "the bool constant snapshot copies 2 bytes");
        }
    }
    TEST_CHECK(saw_snapshot_copy,
               "the module constant snapshot copy must appear in the bundle");
    return true;
}
#endif  // KXC_USE_LLVM

/*! \brief LLVM 构建下真实 Where fixture 的运行证据；无 LLVM 时显式跳过。 */
bool TestWhereRuntimeBundleEvidenceEntry() {
#if KXC_USE_LLVM
    return TestWhereRuntimeBundleEvidence();
#else
    std::cout << "[SKIP] where_runtime_bundle_evidence: KXC_USE_LLVM=0\n";
    return true;
#endif
}

}  // namespace

/*! \brief 顺序执行执行观测契约用例，并将任一失败转换为非零退出码。 */
int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"bundle_correlation_across_runs_and_sessions",
         TestBundleCorrelationAcrossRunsAndSessions},
        {"disabled_profiling_keeps_output_identical",
         TestDisabledProfilingKeepsOutputIdentical},
        {"throwing_observer_does_not_change_execution",
         TestThrowingObserverDoesNotChangeExecution},
        {"validation_error_records_error_run", TestValidationErrorRecordsErrorRun},
        {"launcher_failure_records_error_run", TestLauncherFailureRecordsErrorRun},
        {"copy_accounting_on_run_path", TestCopyAccountingOnRunPath},
        {"state_alias_and_construction_accounting",
         TestStateAliasAndConstructionAccounting},
        {"planned_reuse_accounting", TestPlannedReuseAccounting},
        {"completion_settles_exactly_once", TestCompletionSettlesExactlyOnce},
        {"record_path_does_not_recurse", TestRecordPathDoesNotRecurse},
        {"where_runtime_bundle_evidence", TestWhereRuntimeBundleEvidenceEntry},
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
