// Production copy -> CUDA kernel -> copy, plus genuinely pending DMA lifetime.
// The Python bundle consumer verifies exact copy_id/CUPTI joins independently.
#include "kxc/compiler/compiler.h"
#include "kxc/profiling/runtime_observer.h"
#include "kxc/relay/op.h"
#include "kxc/runtime/device_api.h"
#include "kxc/runtime/session.h"

#include <cuda_runtime_api.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace kxc;
constexpr size_t kCount = 257;
constexpr size_t kBytes = kCount * sizeof(float);

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename F>
void Reject(F&& function) {
    bool rejected = false;
    try { function(); } catch (const std::exception&) { rejected = true; }
    Require(rejected, "invalid copy was accepted");
}

profiling::ProfileOptions Options(const std::filesystem::path& path, bool cupti = true) {
    profiling::ProfileOptions options;
    options.enabled = true;
    options.enable_cupti = cupti;
    options.ir_capture_mode = profiling::IRCaptureMode::kDisabled;
    options.bundle_dir = path.string();
    return options;
}

profiling::EventSpec Caller() {
    profiling::EventSpec spec;
    spec.component = "test";
    spec.event_type = "copy_test";
    return spec;
}

// Test-only stream gate. The CUDA callback uses only host synchronization and
// never calls CUDA or waits for device work. Its timeout makes failure bounded.
class StreamGate {
public:
    explicit StreamGate(DeviceStream stream) : stream_(std::move(stream)) {
        Require(cudaLaunchHostFunc(static_cast<cudaStream_t>(stream_->backend_handle),
                                   Callback, this) == cudaSuccess, "cannot enqueue host gate");
    }
    ~StreamGate() {
        Release();
        try { stream_.Sync(); } catch (...) {}
    }
    void Release() {
        std::lock_guard<std::mutex> lock(mutex_);
        released_ = true;
        cv_.notify_all();
    }
    bool Started() {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, std::chrono::seconds(5), [&] { return started_; });
    }
    bool TimedOut() {
        std::lock_guard<std::mutex> lock(mutex_);
        return timed_out_;
    }
private:
    static void CUDART_CB Callback(void* pointer) {
        auto& gate = *static_cast<StreamGate*>(pointer);
        std::unique_lock<std::mutex> lock(gate.mutex_);
        gate.started_ = true;
        gate.cv_.notify_all();
        gate.timed_out_ = !gate.cv_.wait_for(lock, std::chrono::seconds(10),
                                           [&] { return gate.released_; });
    }
    DeviceStream stream_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool released_{false};
    bool started_{false};
    bool timed_out_{false};
};

Storage TrackedStorage(std::atomic<int>* frees) {
    const auto device = Device::CUDA();
    void* data = DeviceAlloc(device, kBytes);
    return Storage::FromExternal(device, data, kBytes, [](void* pointer, void* context) {
        DeviceFree(Device::CUDA(), pointer);
        ++*static_cast<std::atomic<int>*>(context);
    }, frees);
}

void PendingLifetime(const std::filesystem::path& root, const std::string& mode) {
    const auto device = Device::CUDA();
    const auto stream = DeviceStream::Create(device);
    auto profile = profiling::ProfileContext::Create(Options(root / mode));
    auto observer = profiling::MakeRuntimeExecutionObserver(profile);
    std::weak_ptr<profiling::ProfileContext> weak_profile = profile;
    const auto foreign = profiling::ProfileContext::Create(Options(root / (mode + "_foreign"), false));
    std::atomic<int> frees{0}, callbacks{0}, throwing_callbacks{0};
    std::exception_ptr callback_error;
    std::vector<float> expected(kCount);
    for (size_t i = 0; i < kCount; ++i) expected[i] = static_cast<float>(3 * i + 1);
    auto source = TrackedStorage(&frees);
    auto destination = TrackedStorage(&frees);
    DeviceCopySync(Device::CPU(), expected.data(), 0, device, source.data(), 0, kBytes);
    void* result_pointer = destination.data();
    // Declared before the gate so exception unwind releases the gate first.
    AsyncOperation operation;
    StreamGate gate(stream);
    {
        const profiling::ActivationScope active(profile, "copy_" + mode);
        const profiling::ScopedSpan caller(profile, Caller());
        const runtime::ExecutionObservationScope observed(observer.get(), {});
        operation = StorageCopyAsync(source, 0, destination, 0, kBytes, stream);
    }
    operation.ObserveCompletion([&](bool) {
        ++throwing_callbacks;
        throw std::runtime_error("deliberate observation failure");
    });
    operation.ObserveCompletion([&](bool at_registration) {
        ++callbacks;
        try {
            Require(!at_registration, "pending callback reported immediate completion");
            std::vector<float> output(kCount);
            DeviceCopySync(device, result_pointer, 0, Device::CPU(), output.data(), 0, kBytes);
            Require(output == expected, "pending D2D numerical mismatch");
        } catch (...) { callback_error = std::current_exception(); }
    });
    source = {}; destination = {}; observer.reset(); profile.reset();
    Require(gate.Started() && !gate.TimedOut(), "host gate did not start promptly");
    Require(!operation.IsReady(), "copy completed before independent host release");
    Require(frees == 0 && callbacks == 0 && throwing_callbacks == 0 && !weak_profile.expired(),
            "pending operation released storage/context or settled observation early");
    auto worker = std::async(std::launch::async,
        [&, operation = std::move(operation)]() mutable {
            // Completing on a different thread/profile must keep submission IDs.
            const profiling::ActivationScope active(foreign, "foreign_run", "foreign_parent");
            if (mode == "query") {
                stream.Sync();
                Require(operation.IsReady(), "completed stream event remained pending");
            } else if (mode == "wait") {
                operation.Wait();
            }
            if (mode != "destroy") {
                operation.Wait();
                Require(operation.IsReady() && callbacks == 1 && throwing_callbacks == 1,
                        "repeated observation settled callbacks twice");
                Require(frees == 0 && weak_profile.expired(),
                        "completed handle lost storage or retained profiling callback");
                operation.ObserveCompletion([&](bool at_registration) {
                    if (!at_registration) callback_error = std::make_exception_ptr(
                        std::runtime_error("late callback was not immediate"));
                    ++callbacks;
                });
            }
            operation = {};
            Require(profiling::CurrentContext() == foreign &&
                    profiling::CurrentRunId() == "foreign_run" &&
                    profiling::CurrentSpanId() == "foreign_parent", "completion changed worker TLS");
        });
    const bool blocked = worker.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout;
    gate.Release();
    worker.get();
    if (callback_error) std::rethrow_exception(callback_error);
    Require(blocked && !gate.TimedOut(), "completion did not await independent host release");
    Require(frees == 2 && weak_profile.expired() && throwing_callbacks == 1 &&
            callbacks == (mode == "destroy" ? 1 : 2), "final lifetime/callback counts differ");
    foreign->Flush();
    std::cout << "[INFO] pending " << mode << ": false readiness, two frees, exact 257 floats, captured profile released\n";
}

void CopyChain(const std::filesystem::path& root) {
    using runtime::NDArray;
    const auto device = Device::CUDA();
    const auto dtype = runtime::DataTypeFromString("float32");
    const auto options = Options(root / "chain");
    const auto profile = profiling::ProfileContext::Create(options);
    const auto observer = profiling::MakeRuntimeExecutionObserver(profile);
    const auto graph = [&] {
        const profiling::ActivationScope scope(profile, "prepare");
        const Var data("data", TensorType({kCount}, "float32"));
        return api::Compiler::Compile(Function({data}, Call(relay::Op::Get("nn_relu"), {data})),
            api::CompileConfig::Create(BuildTarget(device), 3, options));
    }();
    const runtime::RuntimeSession session(graph.module(), graph.plan());
    const auto producer = DeviceStream::Create(device);
    const auto consumer = DeviceStream::Create(device);
    auto source = NDArray::Empty({kCount + 2}, dtype, Device::CPU());
    auto staging = NDArray::Empty({kCount + 2}, dtype, device);
    auto source_view = source.CreateView({kCount}, {1}, sizeof(float));
    auto staging_view = staging.CreateView({kCount}, {1}, sizeof(float));
    const auto input = NDArray::Empty({kCount}, dtype, device);
    const auto output = NDArray::Empty({kCount + 2}, dtype, Device::CPU());
    const auto output_view = output.CreateView({kCount}, {1}, sizeof(float));
    std::vector<float> values(kCount + 2, -999), actual(kCount + 2), guards(kCount + 2, -999);
    for (size_t i = 1; i <= kCount; ++i) values[i] = static_cast<float>(i) - 129;
    source.CopyFromBytes(values.data(), source.NBytes());
    staging.CopyFromBytes(guards.data(), staging.NBytes());
    output.CopyFromBytes(guards.data(), output.NBytes());
    {
        const profiling::ActivationScope active(profile, "copy_chain");
        const profiling::ScopedSpan caller(profile, Caller());
        const runtime::ExecutionObservationScope observed(observer.get(), {});
        Reject([&] { input.CopyFromAsync(source, producer); });
        Reject([&] { input.CopyFromAsync(source_view, DeviceStream::Default(Device::CPU())); });
        Reject([&] { StorageCopyAsync(staging.storage(), 0, staging.storage(), 4, kBytes, producer); });
        Reject([&] { StorageCopySync(staging.storage(), 0, staging.storage(), 4, kBytes); });
        Reject([&] { StorageCopyAsync(source.storage(), source.NBytes(), input.storage(), 0, kBytes, producer); });
        auto noop = input.CopyFromAsync(input, producer);
        Require(noop.IsReady(), "exact same-storage no-op must already be complete");
        auto upload = staging_view.CopyFromAsync(source_view, producer);
        source = {}; source_view = {};
        // Different streams have no implicit dependency. Explicitly join H2D.
        upload.Wait();
        auto transfer = input.CopyFromAsync(staging_view, consumer);
        auto run = session.RunAsync({input}, consumer, {{"model", "copy_chain"}, {"stage", "copy_event"}});
        auto download = output_view.CopyFromAsync(run.outputs[0], consumer);
        run.outputs = {}; staging_view = {};
        download.Wait(); transfer.Wait(); run.completion.Wait();
        // Exercise synchronous copy receipts and verify both untouched GPU guards.
        const auto readback = NDArray::Empty({kCount + 2}, dtype, Device::CPU());
        readback.CopyFrom(staging);
        readback.CopyToBytes(actual.data(), readback.NBytes());
        Require(actual.front() == -999 && actual.back() == -999, "offset H2D damaged GPU guards");
        Require(profiling::CurrentSpanId() == caller.span_id(), "copy scope leaked its parent");
    }
    output.CopyToBytes(actual.data(), output.NBytes());
    Require(actual.front() == -999 && actual.back() == -999, "offset D2H damaged CPU guards");
    for (size_t i = 1; i <= kCount; ++i)
        Require(actual[i] == (values[i] > 0 ? values[i] : 0), "copy/CUDA ReLU/copy numerical mismatch");
    profile->Flush();
    std::cout << "[INFO] chain: H2D/D2D/ReLU/D2H, explicit cross-stream wait, 257 exact values, guards, five rejections, no-op\n";
}
}  // namespace

int main(int argc, char** argv) {
    try {
        Require(argc == 2, "provide an unused bundle root directory");
        if (!kxc::CollectDeviceAttributes(kxc::Device::CUDA()).exists) {
            std::cout << "[SKIP] no CUDA device\n";
            return 77;
        }
        CopyChain(argv[1]);
        for (const auto* mode : {"wait", "query", "destroy"}) PendingLifetime(argv[1], mode);
        std::cout << "[PASS] cuda_copy_profiling_chain_and_three_pending_lifetimes_1028_elements\n";
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] cuda_copy_profiling: " << error.what() << '\n';
        return 1;
    }
}
