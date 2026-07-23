#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
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
                          std::size_t alignment = 64,
                          std::shared_ptr<void> module_lease = {}) {
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
    spec.entry.module_label = "local-fake";
    spec.entry.entry_symbol = "twice";
    spec.entry.abi_version = 1;
    spec.entry.ready = true;
    spec.entry.exact_abi_fingerprint =
        RuntimeShapePlan::ExactAbiFingerprint(spec.inputs, spec.outputs);
    spec.entry.launcher = std::move(launcher);
    spec.entry.module_lease = std::move(module_lease);
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

bool TestTwoSContractsExactAbiAndEvents() {
    int launches = 0;
    const std::string expected = RuntimeShapePlan::ExactAbiFingerprint(
        {{"float32", 1, "CPU:0", 1}},
        [] {
            const auto s = RuntimeShapeExpr::InputAxis(0, 0);
            RuntimeShapeTensorContract output;
            output.dtype = "float32";
            output.logical = {RuntimeShapeExpr::Mul(RuntimeShapeExpr::Const(2), s)};
            output.physical = output.logical;
            output.valid = {s};
            output.alignment = 64;
            output.max_bytes = 4096;
            return std::vector<RuntimeShapeTensorContract>{output};
        }());
    RuntimeShapeSession session(MakePlan([&](const RuntimeShapeLaunchArgs& args) {
        ++launches;
        return args.outputs.size() == 1 && args.exact_abi_fingerprint == expected
                   ? RuntimeShapeLaunchResult{}
                   : RuntimeShapeLaunchResult{false, "incorrect exact ABI", {}};
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
          "ShapeEval -> Allocate -> accepted Kernel events");
    return true;
}

bool TestZeroExtentsAndValidation() {
    int zero_launches = 0;
    RuntimeShapeSession session(MakePlan([&](const RuntimeShapeLaunchArgs& args) {
        ++zero_launches;
        return args.outputs[0].bytes == 0 && args.outputs[0].data == nullptr
                   ? RuntimeShapeLaunchResult{}
                   : RuntimeShapeLaunchResult{false, "zero byte output contract broken", {}};
    }));
    const auto zero = session.Run({Input(0)});
    CHECK(zero.ok() && zero_launches == 1, "zero extent synchronously launches");
    CHECK(zero.outputs()[0].bytes == 0 && zero.outputs()[0].data == nullptr,
          "zero extent produces zero bytes without allocation");

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

    auto expression = RuntimeShapeExpr::Const(1);
    for (std::size_t depth = 1; depth < 64; ++depth) {
        expression = RuntimeShapeExpr::Add(std::move(expression), RuntimeShapeExpr::Const(1));
    }
    CHECK(Throws([&] {
        (void)RuntimeShapeExpr::Add(std::move(expression), RuntimeShapeExpr::Const(1));
    }), "expression depth is bounded");
    return true;
}

bool TestPlanContractsAndExactAbiMismatch() {
    RuntimeShapeSession budget(MakePlan([](const RuntimeShapeLaunchArgs&) {
        return RuntimeShapeLaunchResult{};
    }, 8));
    const auto over_budget = budget.Run({Input(3)});  // 6 float32 elements.
    CHECK(!over_budget.ok() && !Has(over_budget, RuntimeShapeEventKind::kAllocate),
          "budget failure precedes allocation and launch");
    RuntimeShapeSession max_bytes(MakePlan([](const RuntimeShapeLaunchArgs&) {
        return RuntimeShapeLaunchResult{};
    }, 4096, 8));
    const auto over_max = max_bytes.Run({Input(3)});
    CHECK(!over_max.ok() && !Has(over_max, RuntimeShapeEventKind::kAllocate),
          "max-byte failure precedes allocation and launch");

    RuntimeShapePlanSpec spec;
    spec.inputs = {{"float32", 1, "CPU:0", 1}};
    RuntimeShapeTensorContract output;
    output.dtype = "float32";
    output.logical = {RuntimeShapeExpr::Const(1)};
    output.physical = output.logical;
    output.valid = output.logical;
    output.max_bytes = 4;
    spec.outputs = {output};
    spec.entry.module_label = "local-fake";
    spec.entry.entry_symbol = "entry";
    spec.entry.ready = true;
    spec.entry.exact_abi_fingerprint = "not the canonical contract";
    spec.entry.launcher = [](const RuntimeShapeLaunchArgs&) { return RuntimeShapeLaunchResult{}; };
    spec.run_byte_budget = 4;
    CHECK(Throws([&] { (void)RuntimeShapePlan(spec); }), "exact ABI mismatch rejected");

    RuntimeShapePlanSpec invalid_bounds = spec;
    invalid_bounds.outputs[0].valid = {RuntimeShapeExpr::Const(2)};
    invalid_bounds.entry.exact_abi_fingerprint =
        RuntimeShapePlan::ExactAbiFingerprint(invalid_bounds.inputs, invalid_bounds.outputs);
    RuntimeShapeSession invalid_bound_session(RuntimeShapePlan(std::move(invalid_bounds)));
    CHECK(!invalid_bound_session.Run({Input()}).ok(), "valid <= logical <= physical is enforced");

    spec.entry.exact_abi_fingerprint = RuntimeShapePlan::ExactAbiFingerprint(spec.inputs, spec.outputs);
    spec.outputs[0].layout = "contiguous";
    spec.entry.exact_abi_fingerprint = RuntimeShapePlan::ExactAbiFingerprint(spec.inputs, spec.outputs);
    CHECK(Throws([&] { (void)RuntimeShapePlan(spec); }), "only contiguous.row_major is accepted");
    spec.outputs[0].layout = "contiguous.row_major";
    spec.outputs[0].scope = "shared";
    spec.entry.exact_abi_fingerprint = RuntimeShapePlan::ExactAbiFingerprint(spec.inputs, spec.outputs);
    CHECK(Throws([&] { (void)RuntimeShapePlan(spec); }), "only global scope is accepted");
    return true;
}

bool TestScalarAbiAndRequiredInputData() {
    RuntimeShapeInputContract input;
    input.dtype = "float32";
    input.rank = 1;
    input.requires_data = true;
    input.axis_guards = {{0, 0, 8, 2, std::nullopt, std::nullopt}};
    RuntimeShapeTensorContract output;
    output.dtype = "float32";
    output.logical = {RuntimeShapeExpr::Const(1)};
    output.physical = output.logical;
    output.valid = output.logical;
    output.max_bytes = 4;
    RuntimeShapePlanSpec spec;
    spec.inputs = {input};
    spec.outputs = {output};
    spec.runtime_extent_abi = {{0, "extent", "n", 0, 0, 0, 8, 2}};
    spec.entry.module_label = "local-fake";
    spec.entry.entry_symbol = "scalar";
    spec.entry.ready = true;
    spec.entry.launcher = [](const RuntimeShapeLaunchArgs& args) {
        return args.runtime_extent_values == std::vector<RuntimeShapeExtent>{4}
            ? RuntimeShapeLaunchResult{} : RuntimeShapeLaunchResult{false, "wrong scalar", {}};
    };
    spec.entry.exact_abi_fingerprint = RuntimeShapePlan::ExactAbiFingerprint(
        spec.inputs, spec.outputs, spec.runtime_extent_abi);
    spec.run_byte_budget = 4;
    float values[4]{};
    RuntimeShapeInput valid = Input(4);
    valid.data = values;
    valid.bytes = sizeof(values);
    CHECK(RuntimeShapeSession(RuntimeShapePlan(spec)).Run({valid}).ok(),
          "evaluated scalar and exact required input bytes reach launcher");
    RuntimeShapeInput wrong_bytes = valid;
    wrong_bytes.bytes -= sizeof(float);
    CHECK(!RuntimeShapeSession(RuntimeShapePlan(spec)).Run({wrong_bytes}).ok(),
          "required input byte size fails before ShapeEval");
    RuntimeShapePlanSpec gap = spec;
    gap.runtime_extent_abi[0].ordinal = 1;
    gap.entry.exact_abi_fingerprint = RuntimeShapePlan::ExactAbiFingerprint(
        gap.inputs, gap.outputs, gap.runtime_extent_abi);
    CHECK(Throws([&] { (void)RuntimeShapePlan(gap); }),
          "scalar ABI ordinal gaps are rejected");
    RuntimeShapePlanSpec duplicate = spec;
    duplicate.runtime_extent_abi.push_back({1, "extent2", "n2", 0, 0, 0, 8, 2});
    duplicate.entry.exact_abi_fingerprint = RuntimeShapePlan::ExactAbiFingerprint(
        duplicate.inputs, duplicate.outputs, duplicate.runtime_extent_abi);
    CHECK(Throws([&] { (void)RuntimeShapePlan(duplicate); }),
          "scalar ABI duplicate input-axis mappings are rejected");
    return true;
}

bool TestCallbackFailureAndOwnerTransfer() {
    RuntimeShapeSession rejected(MakePlan([](const RuntimeShapeLaunchArgs&) {
        return RuntimeShapeLaunchResult{false, "injected launch failure", {}};
    }));
    const auto rejection = rejected.Run({Input()});
    CHECK(!rejection.ok() && rejection.outputs().empty(), "launch rejection cleans unpublished outputs");
    CHECK(!Has(rejection, RuntimeShapeEventKind::kKernel), "kernel event follows acceptance only");

    RuntimeShapeSession std_throw(MakePlan([](const RuntimeShapeLaunchArgs&) -> RuntimeShapeLaunchResult {
        throw std::runtime_error("injected std exception");
    }));
    CHECK(!std_throw.Run({Input()}).ok(), "standard callback exception is contained");
    RuntimeShapeSession nonstd_throw(MakePlan([](const RuntimeShapeLaunchArgs&) -> RuntimeShapeLaunchResult {
        throw 7;
    }));
    CHECK(!nonstd_throw.Run({Input()}).ok(), "non-standard callback exception is contained");

    int launches = 0;
    RuntimeShapeSession transfer(MakePlan([&](const RuntimeShapeLaunchArgs&) {
        ++launches;
        return RuntimeShapeLaunchResult{};
    }));
    detail::FailNextRuntimeShapeOwnerTransferForTest();
    const auto failed_transfer = transfer.Run({Input()});
    CHECK(!failed_transfer.ok() && failed_transfer.outputs().empty() && launches == 0,
          "RAII owner survives deterministic transfer-failure proxy");
    return true;
}

bool TestSynchronousFakeRetentionAndLeases() {
    auto completion = FakeRuntimeShapeCompletion::Pending();
    auto module = std::make_shared<int>(1);
    auto caller = std::make_shared<int>(2);
    std::weak_ptr<int> weak_module = module;
    std::weak_ptr<int> weak_caller = caller;
    std::shared_ptr<RuntimeShapeAsyncResult> result;
    bool launched = false;
    {
        RuntimeShapeSession session(MakePlan(
            [&completion, &launched](const RuntimeShapeLaunchArgs&) {
                launched = true;
                return RuntimeShapeLaunchResult{true, "", completion};
            }, 4096, 4096, 64, module));
        const auto pending = session.RunAsync({Input()}, caller);
        CHECK(launched, "RunAsync invokes the trusted launcher before returning");
        CHECK(pending.ok() && !pending.IsReady(), "pending token is fake retention simulation only");
        CHECK(pending.outputs().size() == 1 && pending.outputs()[0].data != nullptr,
              "result retains its output allocation while fake pending");
        *static_cast<float*>(pending.outputs()[0].data) = 3.0F;
        CHECK(*static_cast<float*>(pending.outputs()[0].data) == 3.0F,
              "output allocation remains usable while result is retained");
        result = std::make_shared<RuntimeShapeAsyncResult>(pending);
    }
    module.reset();
    caller.reset();
    CHECK(!weak_module.expired() && !weak_caller.expired(),
          "result retains module and caller leases after session destruction");
    result->Wait();
    CHECK(result->IsReady(), "Wait completes the fake token, not device work");
    result.reset();
    CHECK(weak_module.expired() && weak_caller.expired(),
          "module and caller leases release with result and output lease");
    return true;
}

bool TestConcurrentLaunchesAreSerialized() {
    std::atomic<int> launches{0};
    std::atomic<int> active{0};
    std::atomic<bool> overlap{false};
    RuntimeShapeSession session(MakePlan([&](const RuntimeShapeLaunchArgs&) {
        if (active.fetch_add(1, std::memory_order_acq_rel) != 0) overlap.store(true);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        active.fetch_sub(1, std::memory_order_acq_rel);
        launches.fetch_add(1, std::memory_order_relaxed);
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
    CHECK(first->ok() && second->ok() && launches == 2 && !overlap.load(),
          "shared trusted launcher is serialized");
    CHECK(first->outputs()[0].data != second->outputs()[0].data,
          "concurrent variants retain independent output allocations");
    return true;
}

}  // namespace

int main() {
#if !KXC_ENABLE_RUNTIME_SHAPE_TASKS
    std::cerr << "runtime shape test must be built with gate on\n";
    return 1;
#else
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"two_s", TestTwoSContractsExactAbiAndEvents},
        {"zero_validation", TestZeroExtentsAndValidation},
        {"plan_contracts", TestPlanContractsAndExactAbiMismatch},
        {"scalar_data", TestScalarAbiAndRequiredInputData},
        {"callback_raii", TestCallbackFailureAndOwnerTransfer},
        {"leases", TestSynchronousFakeRetentionAndLeases},
        {"concurrency", TestConcurrentLaunchesAreSerialized},
    };
    for (const auto& test : tests) if (!test.second()) return 1;
    std::cout << "runtime_shape_session_test: PASS\n";
    return 0;
#endif
}
