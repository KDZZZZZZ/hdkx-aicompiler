#include <cuda_runtime.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <thread>
#include <string>
#include <vector>

#include "kxc/runtime/runtime_shape_session.h"

namespace {
using namespace kxc;
using namespace kxc::runtime;

#define CHECK(condition, message) do { \
    if (!(condition)) { std::cerr << "[FAIL] " << message << "\n"; return false; } \
} while (0)

bool CudaOk(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << "\n";
    return false;
}

void CUDART_CB SleepBlocker(void*) {
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
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

bool TestPendingRetentionAndCompletion() {
    const Device device = Device::CUDA(0);
    DeviceStream stream = DeviceStream::Create(device);
    cudaEvent_t blocker = nullptr;
    cudaStream_t blocker_stream = nullptr;
    CHECK(CudaOk(cudaStreamCreateWithFlags(&blocker_stream, cudaStreamNonBlocking),
                 "cudaStreamCreate(blocker)"), "create blocker stream");
    CHECK(CudaOk(cudaEventCreateWithFlags(&blocker, cudaEventDisableTiming), "cudaEventCreate"),
          "create blocker");
    CHECK(CudaOk(cudaLaunchHostFunc(blocker_stream, SleepBlocker, nullptr), "cudaLaunchHostFunc"),
          "enqueue real pending blocker");
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
            cudaEvent_t done = nullptr;
            if (cudaEventCreateWithFlags(&done, cudaEventDisableTiming) != cudaSuccess ||
                cudaStreamWaitEvent(native, blocker, 0) != cudaSuccess ||
                cudaEventRecord(done, native) != cudaSuccess) {
                if (done) cudaEventDestroy(done);
                return AsyncOperation{};
            }
            return AsyncOperation::Pending(args.stream, done, {});
        }, 64, module));
        const auto pending = session.RunAsync({input}, stream, caller);
        CHECK(pending.ok() && !pending.IsReady(), "real blocked CUDA event remains pending");
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
          "completion retains plan/module/caller/input state after session destruction");
    result->Wait();
    CHECK(result->IsReady() && result->ok(), "Wait delegates actual CUDA completion");
    CHECK(result->retained_device_bytes() == 4 * sizeof(float),
          "completion does not claim physical output release while result owns it");
    result.reset();
    CHECK(weak_module.expired() && weak_caller.expired(), "retention drops with result/completion");
    CHECK(CudaOk(cudaEventDestroy(blocker), "cudaEventDestroy"), "destroy blocker");
    CHECK(CudaOk(cudaStreamDestroy(blocker_stream), "cudaStreamDestroy"), "destroy blocker stream");
    return true;
}

bool TestClosedFailures() {
    const Device device = Device::CUDA(0);
    DeviceStream stream = DeviceStream::Create(device);
    auto storage = Storage::Alloc(device, 20 * sizeof(float));
    int launches = 0;
    RuntimeShapeSession budgeted(MakePlan([&launches](const RuntimeShapeCudaLaunchArgs&) {
        ++launches; return AsyncOperation{};
    }, 8));
    const auto budget = budgeted.RunAsync({Input(storage)}, stream);
    CHECK(!budget.ok() && budget.failure_kind() == RuntimeShapeFailureKind::kResourceExhausted &&
          budget.outputs().empty() && budget.retained_device_bytes() == 0 && launches == 0,
          "budget failure precedes submission and publishes nothing");
    RuntimeShapeSession guarded(MakePlan([&launches](const RuntimeShapeCudaLaunchArgs&) {
        ++launches; return AsyncOperation{};
    }));
    RuntimeShapeInput guarded_input = Input(storage, 20);  // exceeds output max bytes.
    const auto resource = guarded.RunAsync({guarded_input}, stream);
    CHECK(!resource.ok() && resource.failure_kind() == RuntimeShapeFailureKind::kResourceExhausted &&
          resource.outputs().empty() && resource.retained_device_bytes() == 0 && launches == 0,
          "resource failure precedes submission and publishes nothing");

    RuntimeShapeInputContract contract{"float32", 1, "CUDA:0", 1};
    contract.axis_guards = {{0, 1, 8, 1, std::nullopt, std::nullopt}};
    RuntimeShapeTensorContract output;
    output.dtype = "float32"; output.logical = {RuntimeShapeExpr::Const(1)};
    output.physical = output.logical; output.valid = output.logical; output.max_bytes = 4;
    output.device = "CUDA:0";
    RuntimeShapePlanSpec spec;
    spec.inputs = {contract}; spec.outputs = {output}; spec.run_byte_budget = 4;
    spec.entry.module_label = "cuda"; spec.entry.entry_symbol = "guard"; spec.entry.ready = true;
    spec.entry.execution_kind = RuntimeShapeExecutionKind::kCudaAsync;
    spec.entry.device_contract = "CUDA:0";
    spec.entry.cuda_launcher = [&launches](const RuntimeShapeCudaLaunchArgs&) { ++launches; return AsyncOperation{}; };
    spec.entry.exact_abi_fingerprint = RuntimeShapePlan::ExactAbiFingerprint(
        spec.inputs, spec.outputs, {}, {}, {}, spec.entry.execution_kind, spec.entry.device_contract);
    RuntimeShapeSession guard_session(RuntimeShapePlan(std::move(spec)));
    const auto guard = guard_session.RunAsync({Input(storage, 0)}, stream);
    CHECK(!guard.ok() && guard.failure_kind() == RuntimeShapeFailureKind::kApplicabilityMiss &&
          guard.outputs().empty() && guard.retained_device_bytes() == 0 && launches == 0,
          "guard miss precedes submission and publishes nothing");

    RuntimeShapeSession rejected(MakePlan([](const RuntimeShapeCudaLaunchArgs&) { return AsyncOperation{}; }));
    const auto rejection = rejected.RunAsync({Input(storage)}, stream);
    CHECK(!rejection.ok() && rejection.failure_kind() == RuntimeShapeFailureKind::kLaunchRejected,
          "undefined async completion is an explicit rejection");
    RuntimeShapeSession wrong(MakePlan([](const RuntimeShapeCudaLaunchArgs&) {
        return AsyncOperation::Completed(DeviceStream::Default(Device::CPU()));
    }));
    const auto wrong_completion = wrong.RunAsync({Input(storage)}, stream);
    CHECK(!wrong_completion.ok() &&
          wrong_completion.failure_kind() == RuntimeShapeFailureKind::kSubmissionFailed,
          "wrong-device completion is rejected");
    RuntimeShapeSession fake(MakePlan([device](const RuntimeShapeCudaLaunchArgs&) {
        return AsyncOperation::Completed(DeviceStream::Default(device));
    }));
    const auto fake_completion = fake.RunAsync({Input(storage)}, stream);
    CHECK(!fake_completion.ok() &&
          fake_completion.failure_kind() == RuntimeShapeFailureKind::kSubmissionFailed,
          "completed/fake CUDA completion is rejected");
    return true;
}
}  // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    if (!TestPendingRetentionAndCompletion() || !TestClosedFailures()) return 1;
    std::cout << "runtime_shape_cuda_async_test: PASS\n";
    return 0;
}
