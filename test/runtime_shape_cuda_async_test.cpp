#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "kxc/runtime/runtime_shape_session.h"

namespace {
using namespace kxc;
using namespace kxc::runtime;

#define CHECK(condition, message) do { \
    if (!(condition)) { std::cerr << "[FAIL] " << message << "\n"; return false; } \
} while (0)

static_assert(!std::is_constructible<RuntimeShapeCudaLaunchResult, bool, std::string,
                                     AsyncOperation>::value,
              "CUDA callbacks must not be able to forge runtime completion events");

bool CudaOk(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << "\n";
    return false;
}

RuntimeShapeCudaLaunchResult Accepted() { return {true, {}}; }
RuntimeShapeCudaLaunchResult Rejected(std::string reason) { return {false, std::move(reason)}; }

void CUDART_CB SleepBlocker(void*) {
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
}

struct DelayedSignal {
    std::atomic<bool> done{false};
};

void CUDART_CB SleepAndSignal(void* raw) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    static_cast<DelayedSignal*>(raw)->done.store(true, std::memory_order_release);
}

RuntimeShapePlan MakePlan(RuntimeShapeCudaBoundLauncher launcher, std::size_t budget = 64,
                          std::shared_ptr<void> module = {}) {
    const auto n = RuntimeShapeExpr::InputAxis(0, 0);
    RuntimeShapeTensorContract output;
    output.dtype = "float32";
    output.logical = {n};
    output.physical = {n};
    output.valid = {n};
    output.max_bytes = 64;
    output.device = "CUDA:0";
    RuntimeShapePlanSpec spec;
    spec.inputs = {{"float32", 1, "CUDA:0", 1}};
    spec.outputs = {output};
    spec.entry.module_label = "cuda-runtime-shape-test";
    spec.entry.entry_symbol = "blocked";
    spec.entry.ready = true;
    spec.entry.execution_kind = RuntimeShapeExecutionKind::kCudaAsync;
    spec.entry.device_contract = "CUDA:0";
    spec.entry.cuda_launcher = std::move(launcher);
    spec.entry.module_lease = std::move(module);
    spec.entry.exact_abi_fingerprint = RuntimeShapePlan::ExactAbiFingerprint(
        spec.inputs, spec.outputs, {}, {}, {}, spec.entry.execution_kind,
        spec.entry.device_contract);
    spec.run_byte_budget = budget;
    return RuntimeShapePlan(std::move(spec));
}

RuntimeShapeInput Input(Storage storage, RuntimeShapeExtent n = 4) {
    RuntimeShapeInput input{{n}, "float32", "CUDA:0", 1};
    input.storage = std::move(storage);
    input.data = input.storage.data();
    input.bytes = static_cast<std::size_t>(n) * sizeof(float);
    return input;
}

bool Has(const RuntimeShapeAsyncResult& result, RuntimeShapeEventKind kind) {
    for (const auto& event : result.events()) if (event.kind == kind) return true;
    return false;
}

bool TestRuntimeOwnedPendingRetentionAndCompletion() {
    const Device device = Device::CUDA(0);
    DeviceStream stream = DeviceStream::Create(device);
    cudaEvent_t blocker = nullptr;
    cudaStream_t blocker_stream = nullptr;
    CHECK(CudaOk(cudaStreamCreateWithFlags(&blocker_stream, cudaStreamNonBlocking),
                 "cudaStreamCreate(blocker)"), "create blocker stream");
    CHECK(CudaOk(cudaEventCreateWithFlags(&blocker, cudaEventDisableTiming), "cudaEventCreate"),
          "create blocker event");
    CHECK(CudaOk(cudaLaunchHostFunc(blocker_stream, SleepBlocker, nullptr), "cudaLaunchHostFunc"),
          "enqueue blocker");
    CHECK(CudaOk(cudaEventRecord(blocker, blocker_stream), "cudaEventRecord(blocker)"),
          "record blocker completion");

    auto module = std::make_shared<int>(1);
    auto caller = std::make_shared<int>(2);
    std::weak_ptr<int> weak_module = module;
    std::weak_ptr<int> weak_caller = caller;
    auto input_storage = Storage::Alloc(device, 4 * sizeof(float));
    RuntimeShapeInput input = Input(input_storage);
    std::shared_ptr<RuntimeShapeAsyncResult> result;
    {
        RuntimeShapeSession session(MakePlan([blocker](const RuntimeShapeCudaLaunchArgs& args) {
            const auto native = reinterpret_cast<cudaStream_t>(args.stream->backend_handle);
            return cudaStreamWaitEvent(native, blocker, 0) == cudaSuccess
                ? Accepted() : Rejected("cudaStreamWaitEvent failed");
        }, 64, module));
        const auto pending = session.RunAsync({input}, stream, caller);
        CHECK(pending.ok() && !pending.IsReady(),
              "runtime-owned event recorded after callback work remains pending");
        CHECK(pending.outputs().size() == 1 && pending.outputs()[0].data != nullptr,
              "pending result owns CUDA output");
        CHECK(pending.retained_device_bytes() == 4 * sizeof(float),
              "per-result output-byte accounting is exact");
        result = std::make_shared<RuntimeShapeAsyncResult>(pending);
    }
    module.reset();
    caller.reset();
    input_storage = Storage{};
    input.storage = Storage{};
    CHECK(!weak_module.expired() && !weak_caller.expired(),
          "runtime completion retains plan/module/caller/input state after session destruction");
    result->Wait();
    CHECK(result->IsReady() && result->ok(), "Wait observes runtime-owned CUDA completion");
    CHECK(Has(*result, RuntimeShapeEventKind::kCompletion) &&
          Has(*result, RuntimeShapeEventKind::kRetire),
          "completion telemetry is synthesized from immutable result state");
    result.reset();
    CHECK(weak_module.expired() && weak_caller.expired(), "retention drops with result/completion");
    CHECK(CudaOk(cudaEventDestroy(blocker), "cudaEventDestroy"), "destroy blocker event");
    CHECK(CudaOk(cudaStreamDestroy(blocker_stream), "cudaStreamDestroy"), "destroy blocker stream");
    return true;
}

bool TestPostSubmitFailureSafety() {
    const Device device = Device::CUDA(0);
    DeviceStream stream = DeviceStream::Create(device);
    auto input = Storage::Alloc(device, 4 * sizeof(float));

    DelayedSignal retention_signal;
    RuntimeShapeSession retention_failure(MakePlan([&retention_signal](const RuntimeShapeCudaLaunchArgs& args) {
        const auto native = reinterpret_cast<cudaStream_t>(args.stream->backend_handle);
        return cudaLaunchHostFunc(native, SleepAndSignal, &retention_signal) == cudaSuccess
            ? Accepted() : Rejected("cudaLaunchHostFunc failed");
    }));
    detail::FailNextRuntimeShapeCudaRetentionForTest();
    const auto retention = retention_failure.RunAsync({Input(input)}, stream);
    CHECK(!retention.ok() && retention.failure_kind() == RuntimeShapeFailureKind::kSubmissionFailed &&
          retention.outputs().empty() && retention.retained_device_bytes() == 0,
          "retention failure synchronizes before clearing outputs");
    CHECK(retention_signal.done.load(std::memory_order_acquire) &&
          Has(retention, RuntimeShapeEventKind::kCompletion) &&
          Has(retention, RuntimeShapeEventKind::kRetire),
          "retention failure has a runtime-owned completion proof");

    DelayedSignal record_signal;
    RuntimeShapeSession record_failure(MakePlan([&record_signal](const RuntimeShapeCudaLaunchArgs& args) {
        const auto native = reinterpret_cast<cudaStream_t>(args.stream->backend_handle);
        return cudaLaunchHostFunc(native, SleepAndSignal, &record_signal) == cudaSuccess
            ? Accepted() : Rejected("cudaLaunchHostFunc failed");
    }));
    detail::FailNextRuntimeShapeCudaEventRecordForTest();
    const auto record = record_failure.RunAsync({Input(input)}, stream);
    CHECK(!record.ok() && record.failure_kind() == RuntimeShapeFailureKind::kSubmissionFailed &&
          record.outputs().empty() && record.retained_device_bytes() == 0 &&
          record_signal.done.load(std::memory_order_acquire),
          "event-record failure synchronizes submitted work before cleanup");

    DelayedSignal throw_signal;
    RuntimeShapeSession callback_throw(MakePlan([&throw_signal](const RuntimeShapeCudaLaunchArgs& args)
                                                   -> RuntimeShapeCudaLaunchResult {
        const auto native = reinterpret_cast<cudaStream_t>(args.stream->backend_handle);
        if (cudaLaunchHostFunc(native, SleepAndSignal, &throw_signal) != cudaSuccess) {
            return Rejected("cudaLaunchHostFunc failed");
        }
        throw std::runtime_error("submitted then threw");
    }));
    const auto thrown = callback_throw.RunAsync({Input(input)}, stream);
    CHECK(!thrown.ok() && thrown.failure_kind() == RuntimeShapeFailureKind::kSubmissionFailed &&
          thrown.outputs().empty() && thrown.retained_device_bytes() == 0 &&
          throw_signal.done.load(std::memory_order_acquire) &&
          Has(thrown, RuntimeShapeEventKind::kCompletion),
          "throw-after-submit is fenced by the runtime event before cleanup");
    return true;
}

bool TestClosedFailures() {
    const Device device = Device::CUDA(0);
    DeviceStream stream = DeviceStream::Create(device);
    auto storage = Storage::Alloc(device, 20 * sizeof(float));
    int launches = 0;
    RuntimeShapeSession budgeted(MakePlan([&launches](const RuntimeShapeCudaLaunchArgs&) {
        ++launches; return Accepted();
    }, 8));
    const auto budget = budgeted.RunAsync({Input(storage)}, stream);
    CHECK(!budget.ok() && budget.failure_kind() == RuntimeShapeFailureKind::kResourceExhausted &&
          budget.outputs().empty() && budget.retained_device_bytes() == 0 && launches == 0,
          "budget failure precedes submission and publishes nothing");

    RuntimeShapeSession rejected(MakePlan([](const RuntimeShapeCudaLaunchArgs&) {
        return Rejected("test rejection");
    }));
    const auto rejection = rejected.RunAsync({Input(storage)}, stream);
    CHECK(!rejection.ok() && rejection.failure_kind() == RuntimeShapeFailureKind::kLaunchRejected &&
          rejection.outputs().empty() && rejection.retained_device_bytes() == 0,
          "explicit CUDA rejection safely releases unsubmitted outputs");
    return true;
}
}  // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    if (!TestRuntimeOwnedPendingRetentionAndCompletion() || !TestPostSubmitFailureSafety() ||
        !TestClosedFailures()) return 1;
    std::cout << "runtime_shape_cuda_async_test: PASS\n";
    return 0;
}
