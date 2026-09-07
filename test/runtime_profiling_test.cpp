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

#include "kxc/profiling/profiling.h"
#include "kxc/runtime/compiled_module.h"
#include "kxc/runtime/session.h"
#include "../src/runtime/internal/compiled_module_node.h"

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
