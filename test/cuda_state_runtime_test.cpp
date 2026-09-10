// Real CUDA capacity state: outer dimensions, commit order and owned storage.
#include "kxc/compiler/compiler.h"
#include "kxc/profiling/profiling.h"
#include "kxc/relay/op.h"
#include "kxc/runtime/device_api.h"
#include "kxc/runtime/session.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace kxc;
using runtime::NDArray;

void Check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template <typename F>
void Reject(F&& action) {
    bool rejected = false;
    try { action(); } catch (const std::exception&) { rejected = true; }
    Check(rejected, "invalid state operation was accepted");
}
NDArray Tensor(Array<int64_t> shape, const std::vector<float>& values, Device device) {
    auto result = NDArray::Empty(shape, runtime::DataTypeFromString("float32"), device);
    result.CopyFromBytes(values.data(), values.size() * sizeof(float));
    return result;
}
std::vector<float> Read(const NDArray& value) {
    std::vector<float> result(value.NBytes() / sizeof(float));
    value.CopyToBytes(result.data(), value.NBytes());
    return result;
}

void Run(const std::filesystem::path& bundle) {
    const auto device = Device::CUDA();
    profiling::ProfileOptions options;
    options.enabled = options.enable_cupti = true;
    options.ir_capture_mode = profiling::IRCaptureMode::kDisabled;
    options.bundle_dir = bundle.string();
    const auto profile = profiling::ProfileContext::Create(options);
    std::vector<float> expected_state(120, 7), expected_output(120), other_state(120, 7);
    NDArray retained_state, retained_output;
    {
        const Var past("past", TensorType({2, 3, 4, 5}, "float32"));
        const Var token("token", TensorType({2, 3, 1, 5}, "float32"));
        const auto present = Call(relay::Op::Get("concatenate"), {past, token},
                                  relay::ConcatenateAttrs::Create(2));
        // This read must see the old cache even if the present kernel ran first.
        const auto read = Call(relay::Op::Get("add"), {past, token});
        const auto graph = [&] {
            const profiling::ActivationScope activation(profile, "prepare");
            return api::Compiler::Compile(Function({past, token}, Tuple({present, read})),
                api::CompileConfig::Create(BuildTarget(device), 3, options));
        }();
        const auto state_id = graph.plan().input_value_ids()[0];
        const auto source_id = graph.plan().output_value_ids()[0];
        const auto plan = graph.plan().BindStateOutputs({{state_id, source_id, 2, 4, 1}}, 7.0);
        Check(plan.input_value_ids().size() == 1 && plan.output_value_ids().size() == 1,
              "state binding leaked past/present into caller arguments");
        const runtime::RuntimeSession session(graph.module(), plan), other(graph.module(), plan);
        const std::vector<DeviceStream> streams{DeviceStream::Create(device), DeviceStream::Create(device)};
        const void* address = session.StateValue(state_id).storage().data();
        Check(address != other.StateValue(state_id).storage().data(), "sessions shared mutable state");
        std::vector<float> seed(30), other_seed(30), tokens(30);
        for (size_t i = 0; i < seed.size(); ++i) {
            seed[i] = static_cast<float>(i) - 20;
            other_seed[i] = static_cast<float>(i) - 500;
            expected_state[(i / 5) * 20 + i % 5] = seed[i];
            other_state[(i / 5) * 20 + i % 5] = other_seed[i];
        }
        auto initial = Tensor({2, 3, 1, 5}, seed, device);
        Reject([&] { session.InitializeState(state_id, initial, 5); });
        Reject([&] { session.InitializeState(state_id, Tensor({2, 3, 1, 5}, seed, Device::CPU()), 1); });
        Check(session.StateExtent(state_id) == 0 && Read(session.StateValue(state_id)) == std::vector<float>(120, 7),
              "initialization rejection changed state");
        session.InitializeState(state_id, initial, 1);
        Check(session.StateValue(state_id).storage().get() != initial.storage().get(),
              "initialization borrowed the source allocation");
        initial = {};
        other.InitializeState(state_id, Tensor({2, 3, 1, 5}, other_seed, device), 1);
        Reject([&] { session.InitializeState(state_id, Tensor({2, 3, 1, 5}, seed, device), 1); });
        Reject([&] { session.RunAsync({Tensor({2, 3, 2, 5}, std::vector<float>(60), device)}, streams[0]); });
        Reject([&] { session.RunAsync({Tensor({2, 3, 1, 5}, seed, device)}, DeviceStream::Default(Device::CPU())); });
        Check(session.StateExtent(state_id) == 1 && Read(session.StateValue(state_id)) == expected_state,
              "preflight rejection changed initialized state");
        for (int extent = 1; extent < 4; ++extent) {
            for (size_t i = 0; i < tokens.size(); ++i) tokens[i] = static_cast<float>(extent * 10 + i);
            for (size_t outer = 0; outer < 6; ++outer) {
                for (size_t slot = 0; slot < 4; ++slot)
                    for (size_t lane = 0; lane < 5; ++lane) {
                        const auto index = (outer * 4 + slot) * 5 + lane;
                        expected_output[index] = expected_state[index] + tokens[outer * 5 + lane];
                    }
                for (size_t lane = 0; lane < 5; ++lane)
                    expected_state[(outer * 4 + extent) * 5 + lane] = tokens[outer * 5 + lane];
            }
            const auto result = session.RunAsync({Tensor({2, 3, 1, 5}, tokens, device)}, streams[extent % 2],
                {{"model", "cuda_state_contract"}, {"stage", "append"},
                 {"state_extent", std::to_string(extent)}});
            Check(result.completion.IsReady() && session.StateExtent(state_id) == extent + 1,
                  "RunAsync returned before state commit");
            Check(Read(result.outputs[0]) == expected_output, "kernel saw a prematurely committed cache");
            Check(Read(session.StateValue(state_id)) == expected_state, "axis-2 append changed the wrong slots");
            Check(session.StateValue(state_id).storage().data() == address, "state storage moved between runs");
            retained_output = result.outputs[0];
        }
        Reject([&] { session.RunAsync({Tensor({2, 3, 1, 5}, tokens, device)}, streams[0]); });
        Check(session.StateExtent(state_id) == 4 && Read(session.StateValue(state_id)) == expected_state,
              "capacity rejection changed committed state");
        Check(other.StateExtent(state_id) == 1 && Read(other.StateValue(state_id)) == other_state,
              "one session changed another session's cache");
        retained_state = session.StateValue(state_id);
    }
    Check(Read(retained_state) == expected_state && Read(retained_output) == expected_output,
          "session/module destruction invalidated retained arrays");
    profile->Flush();
    std::cout << "[PASS] cuda_state_external_axis2_capacity_and_ownership: three appends, 360 output elements, 360 cache elements, six rejections\n";
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (!CollectDeviceAttributes(Device::CUDA()).exists) {
            std::cout << "[SKIP] no CUDA device\n";
            return 77;
        }
        Check(argc == 2, "provide an unused profiling bundle directory");
        Run(argv[1]);
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] cuda_state_runtime: " << error.what() << '\n';
        return 1;
    }
}
