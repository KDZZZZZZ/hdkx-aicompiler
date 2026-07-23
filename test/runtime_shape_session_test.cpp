#include <atomic>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "kxc/runtime/runtime_shape_session.h"

namespace {
using namespace kxc::runtime;

#define CHECK(condition, message) do { \
    if (!(condition)) { std::cerr << "[FAIL] " << __FUNCTION__ << ": " << message << "\n"; return false; } \
} while (0)

bool Throws(const std::function<void()>& body) {
    try { body(); } catch (const std::exception&) { return true; }
    return false;
}

RuntimeShapePlan MakePlan(RuntimeShapeBoundLauncher launcher,
                          std::size_t budget = 4096,
                          std::size_t max_bytes = 4096,
                          std::size_t alignment = 64) {
    const auto s = RuntimeShapeExpr::InputAxis(0, 0);
    const auto two_s = RuntimeShapeExpr::Mul(RuntimeShapeExpr::Const(2), s);
    RuntimeShapeTensorContract output;
    output.dtype = "float32";
    output.logical = {two_s};
    output.physical = {two_s};
    output.valid = {s};
    output.alignment = alignment;
    output.max_bytes = max_bytes;
    RuntimeShapePlanSpec spec;
    spec.inputs = {{"float32", 1, "CPU:0", 1}};
    spec.outputs = {output};
    spec.entry = {"local-fake", "twice", 1, true, std::move(launcher), {}};
    spec.run_byte_budget = budget;
    return RuntimeShapePlan(std::move(spec));
}

RuntimeShapeInput Input(RuntimeShapeExtent s = 3) {
    return RuntimeShapeInput{{s}, "float32", "CPU:0", 1};
}

bool Has(const RuntimeShapeAsyncResult& result, RuntimeShapeEventKind kind) {
    for (const auto& event : result.events()) if (event.kind == kind) return true;
    return false;
}

bool TestTwoSContractsAlignmentAndEvents() {
    int launches = 0;
    RuntimeShapeSession session(MakePlan([&](const RuntimeShapeLaunchArgs& args) {
        ++launches;
        return args.outputs.size() == 1 ? RuntimeShapeLaunchResult{}
                                        : RuntimeShapeLaunchResult{false, "bad output count", {}};
    }));
    const auto result = session.Run({Input(3)});
    CHECK(result.ok(), result.failure_reason());
    CHECK(launches == 1, "kernel launched once");
    CHECK(result.outputs().size() == 1, "one dynamic output");
    const auto& output = result.outputs()[0];
    CHECK(output.logical == std::vector<RuntimeShapeExtent>{6}, "2*S logical shape");
    CHECK(output.physical == std::vector<RuntimeShapeExtent>{6}, "2*S physical shape");
    CHECK(output.valid == std::vector<RuntimeShapeExtent>{3}, "valid shape");
    CHECK(output.valid[0] <= output.logical[0] && output.logical[0] <= output.physical[0],
          "valid <= logical <= physical");
    CHECK(reinterpret_cast<std::uintptr_t>(output.data) % 64 == 0, "output alignment");
    CHECK(Has(result, RuntimeShapeEventKind::kShapeEval) &&
          Has(result, RuntimeShapeEventKind::kAllocate) && Has(result, RuntimeShapeEventKind::kKernel),
          "ShapeEval -> Allocate -> Kernel events");
    return true;
}

bool TestZeroOverflowAndInputMismatch() {
    RuntimeShapeSession session(MakePlan([](const RuntimeShapeLaunchArgs&) {
        return RuntimeShapeLaunchResult{};
    }));
    CHECK(!session.Run({Input(0)}).ok(), "zero extent rejected");
    CHECK(!session.Run({Input(std::numeric_limits<RuntimeShapeExtent>::max())}).ok(),
          "checked 2*S overflow rejected");
    auto rank = Input(); rank.shape.push_back(1);
    auto dtype = Input(); dtype.dtype = "int32";
    auto device = Input(); device.device = "CUDA:0";
    auto abi = Input(); abi.abi_version = 2;
    CHECK(!session.Run({rank}).ok(), "rank mismatch rejected");
    CHECK(!session.Run({dtype}).ok(), "dtype mismatch rejected");
    CHECK(!session.Run({device}).ok(), "device mismatch rejected");
    CHECK(!session.Run({abi}).ok(), "ABI mismatch rejected");
    return true;
}

bool TestContractAndBudgetFailuresBeforeLaunch() {
    int launches = 0;
    RuntimeShapeSession budget(MakePlan([&](const RuntimeShapeLaunchArgs&) {
        ++launches; return RuntimeShapeLaunchResult{};
    }, 8));
    const auto over_budget = budget.Run({Input(3)});  // 6 float32 elements.
    CHECK(!over_budget.ok() && launches == 0, "budget failure precedes launch");
    CHECK(!Has(over_budget, RuntimeShapeEventKind::kAllocate) &&
          Has(over_budget, RuntimeShapeEventKind::kFailure), "budget failure event");

    RuntimeShapeSession max_bytes(MakePlan([&](const RuntimeShapeLaunchArgs&) {
        ++launches; return RuntimeShapeLaunchResult{};
    }, 4096, 8));
    const auto oom_guard = max_bytes.Run({Input(3)});
    CHECK(!oom_guard.ok() && launches == 0, "max-byte OOM guard precedes launch");

    auto bad = RuntimeShapeExpr::InputAxis(0, 0);
    RuntimeShapeTensorContract contract;
    contract.dtype = "float32";
    contract.logical = {bad}; contract.physical = {bad};
    contract.valid = {RuntimeShapeExpr::Mul(RuntimeShapeExpr::Const(2), bad)};
    contract.max_bytes = 4096;
    RuntimeShapePlanSpec spec;
    spec.inputs = {{"float32", 1, "CPU:0", 1}};
    spec.outputs = {contract};
    spec.entry = {"local-fake", "bad", 1, true,
                  [](const RuntimeShapeLaunchArgs&) { return RuntimeShapeLaunchResult{}; }, {}};
    spec.run_byte_budget = 4096;
    CHECK(!RuntimeShapeSession(RuntimeShapePlan(std::move(spec))).Run({Input(2)}).ok(),
          "valid <= logical <= physical enforced");
    return true;
}

bool TestLaunchFailureAndLeaseRetention() {
    RuntimeShapeSession failed(MakePlan([](const RuntimeShapeLaunchArgs&) {
        return RuntimeShapeLaunchResult{false, "injected launch failure", {}};
    }));
    const auto failure = failed.Run({Input()});
    CHECK(!failure.ok() && failure.outputs().empty(), "launch failure cleans unpublished outputs");
    CHECK(Has(failure, RuntimeShapeEventKind::kAllocate) &&
          Has(failure, RuntimeShapeEventKind::kFailure), "allocation and failure events retained");

    auto completion = FakeRuntimeShapeCompletion::Pending();
    auto lease = std::make_shared<int>(7);
    std::weak_ptr<int> weak = lease;
    {
        RuntimeShapeSession pending(MakePlan([completion](const RuntimeShapeLaunchArgs&) {
            return RuntimeShapeLaunchResult{true, "", completion};
        }));
        const auto result = pending.RunAsync({Input()}, lease);
        lease.reset();
        CHECK(result.ok() && !result.IsReady() && !weak.expired(), "result retains caller lease while fake pending");
        result.Wait();
        CHECK(result.IsReady(), "fake completion is deterministic");
    }
    CHECK(weak.expired(), "lease released after result cleanup");
    return true;
}

bool TestConcurrentVariantsAreIsolated() {
    std::atomic<int> launches{0};
    std::mutex mutex;
    std::vector<void*> addresses;
    RuntimeShapeSession session(MakePlan([&](const RuntimeShapeLaunchArgs& args) {
        launches.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(mutex);
        addresses.push_back(args.outputs[0].data);
        return RuntimeShapeLaunchResult{};
    }));
    std::shared_ptr<RuntimeShapeAsyncResult> first;
    std::shared_ptr<RuntimeShapeAsyncResult> second;
    std::thread first_thread([&] {
        first = std::make_shared<RuntimeShapeAsyncResult>(session.RunAsync({Input(2)}));
    });
    std::thread second_thread([&] {
        second = std::make_shared<RuntimeShapeAsyncResult>(session.RunAsync({Input(5)}));
    });
    first_thread.join();
    second_thread.join();
    CHECK(first->ok() && second->ok(), "concurrent variants launch");
    CHECK(first->outputs()[0].logical[0] == 4 && second->outputs()[0].logical[0] == 10,
          "each run has private evaluated shape state");
    CHECK(first->outputs()[0].data != second->outputs()[0].data && launches == 2,
          "no allocation reuse across variants");
    return true;
}

}  // namespace

int main() {
#if !KXC_ENABLE_RUNTIME_SHAPE_TASKS
    std::cerr << "runtime shape test must be built with gate on\n";
    return 1;
#else
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"two_s", TestTwoSContractsAlignmentAndEvents},
        {"validation", TestZeroOverflowAndInputMismatch},
        {"budget", TestContractAndBudgetFailuresBeforeLaunch},
        {"failure_lease", TestLaunchFailureAndLeaseRetention},
        {"concurrency", TestConcurrentVariantsAreIsolated},
    };
    for (const auto& test : tests) if (!test.second()) return 1;
    std::cout << "runtime_shape_session_test: PASS\n";
    return 0;
#endif
}
