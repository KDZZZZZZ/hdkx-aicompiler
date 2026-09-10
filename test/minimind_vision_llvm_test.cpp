// Full MiniMind-V SigLIP2 + projector, through the production static CPU chain.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "kxc/compiler/compiler.h"
#include "kxc/frontend/onnx_importer.h"
#include "kxc/runtime/session.h"

namespace {
using namespace kxc;
using runtime::NDArray;
void Check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
std::vector<float> Read(const std::filesystem::path& path, size_t elements) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    Check(file.good() && file.tellg() == std::streamoff(elements * sizeof(float)),
          "missing or wrongly sized fixture: " + path.string());
    std::vector<float> values(elements);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(float));
    Check(file.good(), "failed to read fixture: " + path.string());
    return values;
}
size_t Submits(const std::shared_ptr<profiling::ProfileContext>& profile) {
    profile->Flush();
    std::ifstream file(std::filesystem::path(profile->bundle_dir()) / "events.jsonl");
    Check(file.good(), "missing vision profile events");
    size_t count = 0;
    std::string line;
    while (std::getline(file, line)) count += line.find("\"event_type\":\"kernel_submit\"") != std::string::npos;
    return count;
}
}  // namespace

int main() {
    try {
#if KXC_USE_LLVM
        const char* directory = std::getenv("KXC_MINIMIND_VISION_DIR");
        if (!directory) {
            std::cout << "[SKIP] full L2 vision: set KXC_MINIMIND_VISION_DIR\n";
            return 0;
        }
        const std::filesystem::path root(directory);
        profiling::ProfileOptions options;
        options.enabled = true;
        options.ir_capture_mode = profiling::IRCaptureMode::kDisabled;
        options.record_pass_ir = false;
        options.bundle_dir = (std::filesystem::current_path() / "out" / "minimind_vision_profile").string();
        const auto profile = profiling::ProfileContext::Create(options);
        const profiling::ActivationScope activation(profile, "minimind_v_vision");
        auto imported = frontend::LoadONNXImportSpec((root / "vision.json").string(),
                                                    (root / "vision.params").string());
        Check(imported.input_names.size() == 1 && imported.input_names[0] == "pixel_values",
              "vision input ABI changed");
        Check(imported.output_names.size() == 2 && imported.output_names[0] == "vision_features" &&
                  imported.output_names[1] == "visual_tokens", "vision output order changed");
        const auto config = api::CompileConfig::Create(BuildTarget(Device::CPU()), 2, options);
        const auto start = std::chrono::steady_clock::now();
        const auto graph = api::Compiler::Compile(imported.function, config);
        Check(graph.module().IsReady(), "vision LLVM module is not ready");
        std::cout << "vision compiled: " << graph.plan().calls().size() << " kernels, "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()
                  << " seconds\n" << std::flush;
        const runtime::RuntimeSession session(graph.module(), graph.plan());
        for (int case_index = 0; case_index < 2; ++case_index) {
            const auto values = Read(root / ("input_" + std::to_string(case_index) + ".bin"), 3*256*256);
            NDArray input = NDArray::Empty({1,3,256,256}, runtime::DataTypeFromString("float32"), Device::CPU());
            input.CopyFromBytes(values.data(), input.NBytes());
            const auto before = Submits(profile);
            const auto run_start = std::chrono::steady_clock::now();
            const auto outputs = session.Run({input}, {{"model", "minimind-v"}, {"stage", "vision"},
                                                       {"case", std::to_string(case_index)}});
            input = {};
            Check(outputs.size() == 2, "vision must return both encoder and projector outputs");
            Check(Submits(profile) - before == graph.plan().calls().size(), "vision did not execute the complete plan");
            for (size_t index = 0; index < 2; ++index) {
                const auto shape = outputs[index].shape();
                Check(shape.size() == 3 && shape[0] == 1 && shape[1] == 64 && shape[2] == 768,
                      "vision output must be [1,64,768]");
                const std::string name = imported.output_names[index];
                const auto expected = Read(root / ("ref_" + std::to_string(case_index) + "_" + name + ".bin"), 64*768);
                std::vector<float> actual(expected.size());
                outputs[index].CopyToBytes(actual.data(), actual.size()*sizeof(float));
                double worst = 0;
                for (size_t i = 0; i < actual.size(); ++i) {
                    const double error = std::abs(double(actual[i]) - expected[i]);
                    Check(std::isfinite(actual[i]) && std::isfinite(expected[i]) &&
                              error <= 2e-4 + 2e-4*std::abs(expected[i]),
                          name + " numeric mismatch at " + std::to_string(i) + ": " + std::to_string(error));
                    worst = std::max(worst, error);
                }
                std::cout << "case " << case_index << " " << name << " max_abs_error=" << worst << '\n';
            }
            std::cout << "case " << case_index << " host_run_seconds="
                      << std::chrono::duration<double>(std::chrono::steady_clock::now()-run_start).count()
                      << '\n' << std::flush;
        }
        const auto before = Submits(profile);
        for (const bool wrong_dtype : {false, true}) {
            bool rejected = false;
            try {
                (void)session.Run({NDArray::Empty(wrong_dtype ? Array<int64_t>{1,3,256,256} : Array<int64_t>{1,3,255,256},
                    runtime::DataTypeFromString(wrong_dtype ? "float64" : "float32"), Device::CPU())});
            } catch (const std::exception&) { rejected = true; }
            Check(rejected && Submits(profile) == before, "invalid vision input reached a kernel");
        }
        std::cout << "full vision numeric and pre-launch input rejection passed; submits=" << before << '\n';
#else
        std::cout << "[SKIP] full L2 vision requires LLVM\n";
#endif
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
