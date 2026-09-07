/*! \file test/g1_combined_profiling_test.cpp
 * \brief 第一波 G1 组合验收：ONNX 导入的 Equal→Where 组合图在开启执行
 * 观测下经 LLVM Compiler 与 RuntimeSession 运行，导出真实 runtime
 * bundle；bundle 内 runtime_session_run/kernel_submit/kernel_exec/alloc
 * 以同一 run_id 关联，内核与分配事件的 parent_span_id 指向 run span；
 * 同一 bundle 也携带编译期事件；观测开启与关闭的输出逐位一致。
 *
 * 该用例在独立进程内只创建一个 ProfileContext，bundle 目录由 CMake
 * 注入（build 目录内），避免多上下文共用目录时最后一次 flush 覆盖
 * 前序事件。数值参考与 onnx_importer_test.cpp 的 Equal-Where 用例一致：
 * bit-exact（容差 0）。
 */

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "kxc/compiler/compiler.h"
#include "kxc/frontend/onnx_importer.h"
#include "kxc/relay/op.h"
#include "kxc/relay/transforms/infer_type.h"
#include "kxc/relay/transforms/pipeline.h"
#include "kxc/runtime/session.h"

#ifndef KXC_ONNX_EQUAL_WHERE_JSON_PATH
#define KXC_ONNX_EQUAL_WHERE_JSON_PATH "equal_where.import.json"
#endif

#ifndef KXC_ONNX_EQUAL_WHERE_PARAMS_PATH
#define KXC_ONNX_EQUAL_WHERE_PARAMS_PATH "equal_where.params.bin"
#endif

#ifndef KXC_G1_BUNDLE_ROOT
#define KXC_G1_BUNDLE_ROOT "g1_bundles"
#endif

#ifndef KXC_USE_LLVM
#define KXC_USE_LLVM 0
#endif

namespace {

#define TEST_CHECK(condition, message)                                           \
    do {                                                                          \
        if (!(condition)) {                                                       \
            throw std::runtime_error(std::string("[FAIL] ") +                     \
                                     std::string(__FUNCTION__) + ": " +           \
                                     (message));                                  \
        }                                                                         \
    } while (0)

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

kxc::Function PrepareJitRelayFunction(kxc::Function func) {
    func = kxc::relay::InferTypePass(func);
    func = kxc::relay::RunRelayPassPipeline(
        func, {kxc::String("fold_constant"), kxc::String("simplify_expr")});
    return kxc::relay::InferTypePass(func);
}

std::filesystem::path BundleRoot() {
    std::filesystem::path root = KXC_G1_BUNDLE_ROOT;
    if (root.is_relative()) {
        root = std::filesystem::current_path() / root;
    }
    return root;
}

/*! \brief 运行 ONNX Equal→Where 组合图；observe 决定是否装配执行观测。
 *  返回输出元素；模块与 session 在函数返回时析构并 flush bundle。 */
std::vector<float> RunEqualWhereCombined(bool observe,
                                         const std::string& bundle_name) {
    using namespace kxc;
    kxc::frontend::ImportedONNXModel imported =
        kxc::frontend::LoadONNXImportSpec(
            KXC_ONNX_EQUAL_WHERE_JSON_PATH,
            KXC_ONNX_EQUAL_WHERE_PARAMS_PATH);
    TEST_CHECK(imported.function.defined() &&
                   imported.function->params.size() == 2 &&
                   imported.input_names.size() == 2 &&
                   imported.output_names.size() == 1,
               "Equal-Where fixture must import with a and b as inputs");
    const kxc::Function prepared = PrepareJitRelayFunction(imported.function);

    api::CompileConfig config = [&] {
        if (!observe) {
            return api::CompileConfig::Create(BuildTarget(Device::CPU()), 1);
        }
        profiling::ProfileOptions options;
        options.enabled = true;
        options.ir_capture_mode = profiling::IRCaptureMode::kDisabled;
        options.bundle_dir = (BundleRoot() / bundle_name).string();
        return api::CompileConfig::Create(BuildTarget(Device::CPU()), 1, options);
    }();

    const auto compiled = api::Compiler::Compile(prepared, config);
    TEST_CHECK(compiled.module().IsReady() && compiled.plan().calls().size() == 2,
               "Equal-Where combined graph must compile to two LLVM units");

    const std::vector<float> a_values = {1, 2, 3, 4, 5, 6};
    const std::vector<float> b_values = {1, 0, 3};
    runtime::NDArray a = runtime::NDArray::Empty(
        {2, 3}, runtime::DataTypeFromString("float32"), Device::CPU());
    a.CopyFromBytes(a_values.data(), a.NBytes());
    runtime::NDArray b = runtime::NDArray::Empty(
        {3}, runtime::DataTypeFromString("float32"), Device::CPU());
    b.CopyFromBytes(b_values.data(), b.NBytes());

    runtime::RuntimeSession session(compiled.module(), compiled.plan());
    const kxc::Array<runtime::NDArray> outputs = session.Run({a, b});
    TEST_CHECK(outputs.size() == 1 && outputs[0]->shape_storage.size() == 2 &&
                   outputs[0]->shape_storage[0] == 2 &&
                   outputs[0]->shape_storage[1] == 3,
               "Equal-Where output must be [2,3]");
    std::vector<float> actual(6, 0.0f);
    outputs[0].CopyToBytes(actual.data(), outputs[0].NBytes());
    return actual;
}

bool TestCombinedOnnxGraphWithRuntimeBundle() {
    using namespace kxc;
    const std::vector<float> expected = {10, -1, 30, -1, -1, -1};

    const std::vector<float> unobserved = RunEqualWhereCombined(false, "");
    TEST_CHECK(unobserved == expected,
               "unobserved Equal-Where run must match the reference bit-exactly");

    const std::string bundle_name = "equal_where_combined";
    const std::string bundle_dir = (BundleRoot() / bundle_name).string();
    const std::vector<float> observed =
        RunEqualWhereCombined(true, bundle_name);
    TEST_CHECK(observed == unobserved,
               "observed run must be bitwise identical to the unobserved run");
    TEST_CHECK(observed == expected,
               "observed Equal-Where run must match the reference bit-exactly");

    const std::vector<EventLine> events = LoadEvents(bundle_dir);
    const std::vector<const EventLine*> runs = Filter(events, "runtime_session_run");
    TEST_CHECK(runs.size() == 1 && runs[0]->StringValue("status") == "ok",
               "the combined ONNX run records one successful run span");
    const std::string run_id = runs[0]->StringValue("run_id");
    const std::string span_id = runs[0]->StringValue("span_id");
    TEST_CHECK(!run_id.empty() && !span_id.empty(),
               "the combined run span carries correlation ids");
    TEST_CHECK(runs[0]->StringValue("component") == "runtime_session" &&
                   runs[0]->raw.find("\"timing\":\"host_execute\"") !=
                       std::string::npos,
               "the combined run span is host_execute on runtime_session");

    int submits = 0;
    for (const EventLine* submit : Filter(events, "kernel_submit")) {
        if (submit->StringValue("run_id") != run_id) continue;
        ++submits;
        TEST_CHECK(submit->StringValue("phase") == "submit" &&
                       submit->StringValue("parent_span_id") == span_id &&
                       submit->raw.find("\"timing\":\"host_submit\"") !=
                           std::string::npos,
                   "combined kernel_submit must parent to the run span");
    }
    int execs = 0;
    for (const EventLine* exec : Filter(events, "kernel_exec")) {
        if (exec->StringValue("run_id") != run_id) continue;
        ++execs;
        TEST_CHECK(exec->StringValue("parent_span_id") == span_id &&
                       !exec->StringValue("kernel_symbol").empty() &&
                       exec->raw.find("\"timing\":\"host_execute\"") !=
                           std::string::npos,
                   "combined kernel_exec must parent to the run span");
    }
    int allocs = 0;
    for (const EventLine* alloc : Filter(events, "alloc")) {
        if (alloc->StringValue("run_id") != run_id) continue;
        ++allocs;
        TEST_CHECK(alloc->StringValue("parent_span_id") == span_id,
                   "combined alloc must parent to the run span");
    }
    TEST_CHECK(submits == 2 && execs == 2 && allocs >= 1,
               "the combined graph submits two kernels (equal, where) and "
               "allocates its outputs");

    bool relay_pipeline_seen = false;
    for (const auto& event : events) {
        if (event.StringValue("component") == "relay_pipeline") {
            relay_pipeline_seen = true;
        }
    }
    TEST_CHECK(!Filter(events, "run_pass").empty() && relay_pipeline_seen,
               "the same bundle carries the ONNX-imported compile events");
    return true;
}

}  // namespace

int main() {
    try {
        if (TestCombinedOnnxGraphWithRuntimeBundle()) {
            std::cout << "[PASS] g1_combined_profiling_test" << std::endl;
            return 0;
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
    }
    std::cerr << "[FAIL] g1_combined_profiling_test" << std::endl;
    return 1;
}
