// Real production launches across profiles, threads, streams and repeated runs.
// cuda_profile_bundle_test.py validates the resulting CUPTI correlation graph.
#include "kxc/compiler/compiler.h"
#include "kxc/profiling/profiling.h"
#include "kxc/relay/op.h"
#include "kxc/runtime/device_api.h"
#include "kxc/runtime/session.h"

#include <filesystem>
#include <future>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void Run(const std::filesystem::path& root) {
    using namespace kxc;
    const auto device = Device::CUDA(0);
    profiling::ProfileOptions options;
    options.enabled = options.enable_cupti = true;
    options.ir_capture_mode = profiling::IRCaptureMode::kDisabled;
    std::vector<std::shared_ptr<profiling::ProfileContext>> profiles;
    std::vector<std::future<void>> workers;
    std::promise<void> ready;
    auto start = ready.get_future().share();
    for (int model = 0; model < 2; ++model) {
        const auto name = "model_" + std::to_string(model);
        options.bundle_dir = (root / name).string();
        const auto profile = profiling::ProfileContext::Create(options);
        profiles.push_back(profile);
        Var x("x", TensorType({257}, "float32"));
        Var y("y", TensorType({257}, "float32"));
        const auto add = relay::Op::Get("add");
        const Function function({x, y}, Call(add, {Call(add, {x, x}), Call(add, {y, y})}));
        const auto graph = [&] {
            const profiling::ActivationScope scope(profile, "prepare");
            return api::Compiler::Compile(function, api::CompileConfig::Create(BuildTarget(device), 3, options));
        }();
        Require(graph.plan().calls().size() == 3, "expected three production CUDA calls");
        for (int worker = 0; worker < 2; ++worker) {
            workers.push_back(std::async(std::launch::async, [=] {
                const profiling::ActivationScope caller(profile, "caller", "caller_parent");
                runtime::RuntimeSession session(graph.module(), graph.plan());
                const auto stream = DeviceStream::Create(device);
                auto a = runtime::NDArray::Empty({257}, runtime::DataTypeFromString("float32"), device);
                auto b = runtime::NDArray::Empty({257}, runtime::DataTypeFromString("float32"), device);
                std::vector<float> left(257), right(257), output(257);
                for (size_t i = 0; i < left.size(); ++i) {
                    left[i] = static_cast<float>(i + worker + model);
                    right[i] = static_cast<float>(3 * i + 2 * worker + model);
                }
                a.CopyFromBytes(left.data(), a.NBytes());
                b.CopyFromBytes(right.data(), b.NBytes());
                {
                    // Explicitly unobserved work must mask the outer CUPTI IDs.
                    const profiling::ActivationScope disabled(nullptr, "", "");
                    a.CopyFromBytes(left.data(), a.NBytes());
                }
                start.wait();
                // Preflight failure must submit nothing and leave caller TLS intact.
                bool rejected = false;
                try { (void)session.RunAsync({a}, stream); }
                catch (const std::exception&) { rejected = true; }
                Require(rejected, "bad input count was accepted");
                for (int repeat = 0; repeat < 3; ++repeat) {
                    const auto result = [&] {
                        profiling::EventSpec spec;
                        spec.component = "test";
                        spec.event_type = "moved_caller_scope";
                        profiling::ScopedSpan original(profile, spec);
                        auto moved = std::move(original);
                        return session.RunAsync({a, b}, stream,
                            {{"model", name}, {"worker", std::to_string(worker)},
                             {"repeat", std::to_string(repeat)}, {"stage", "correlation_test"}});
                    }();
                    Require(profiling::CurrentContext() == profile && profiling::CurrentRunId() == "caller" &&
                            profiling::CurrentSpanId() == "caller_parent", "runtime leaked its profiling scope");
                    result.completion.Wait();
                    Require(result.outputs.size() == 1, "unexpected output count");
                    result.outputs[0].CopyToBytes(output.data(), output.size() * sizeof(float));
                    for (size_t i = 0; i < output.size(); ++i)
                        Require(output[i] == 2 * left[i] + 2 * right[i], "concurrent CUDA numerical mismatch");
                }
            }));
        }
    }
    ready.set_value();
    for (auto& worker : workers) worker.get();
    for (auto& profile : profiles) profile->Flush();
    std::cout << "[PASS] cuda_runtime_profiling_two_profiles_four_threads_12_runs_3084_elements\n";
}
}  // namespace

int main(int argc, char** argv) {
    try {
        Require(argc == 2, "provide an unused bundle root directory");
        if (!kxc::CollectDeviceAttributes(kxc::Device::CUDA()).exists) {
            std::cout << "[SKIP] no CUDA device\n";
            return 77;
        }
        Run(argv[1]);
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] cuda_runtime_profiling: " << error.what() << '\n';
        return 1;
    }
}
