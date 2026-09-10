// Full eight-layer static MiniMind prefill through Compiler/CUDA/RuntimeSession.
// Generate the existing decode-loop fixture and set KXC_MINIMIND_CUDA_PREFILL_DIR.
// All logits and 16 returned K/V tensors use independent ONNX reference files.
#include "../src/compiler/internal/primitive_cache.h"
#include "kxc/compiler/compiler.h"
#include "kxc/frontend/onnx_importer.h"
#include "kxc/relay/transforms/infer_type.h"
#include "kxc/relay/transforms/pipeline.h"
#include "kxc/runtime/session.h"
#include "kxc/runtime/device_api.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Scalar>
std::vector<Scalar> ReadValues(const std::string& path, size_t count) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    Require(file && file.tellg() == static_cast<std::streamoff>(count * sizeof(Scalar)),
            "missing or wrong-sized fixture: " + path);
    file.seekg(0);
    std::vector<Scalar> values(count);
    Require(static_cast<bool>(file.read(reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(count * sizeof(Scalar)))), "cannot read fixture: " + path);
    return values;
}

bool SameStats(const kxc::api::internal::PrimitiveCacheStats& a,
               const kxc::api::internal::PrimitiveCacheStats& b) {
    return a.hits == b.hits && a.misses == b.misses && a.entries == b.entries &&
        a.accounted_bytes == b.accounted_bytes && a.evictions == b.evictions &&
        a.in_flight == b.in_flight && a.merged_waiters == b.merged_waiters &&
        a.failures == b.failures && a.rejections == b.rejections && a.active_pins == b.active_pins;
}

void Run(const std::string& root) {
    using namespace kxc;
    std::cout << "[INFO] loading_full_minimind_cuda_prefill fixture=" << root << std::endl;
    int64_t batch = 0, heads = 0, head_dim = 0, layers = 0, extent = 0;
    std::ifstream layout(root + "/layout.txt"), seed(root + "/seed_extent.txt");
    Require(static_cast<bool>(layout >> batch >> heads >> head_dim >> layers) &&
            static_cast<bool>(seed >> extent) && batch == 1 && heads == 4 &&
            head_dim == 96 && layers == 8 && extent == 16, "expected full static B1/S16 eight-layer fixture");
    std::string receipt;
    std::ifstream receipt_file(root + "/prefill_export_receipt.txt");
    Require(static_cast<bool>(std::getline(receipt_file, receipt)) &&
            receipt.rfind("sha256:minimind_prefill_static.onnx:", 0) == 0, "missing ONNX export hash");
    const auto imported = frontend::LoadONNXImportSpec(root + "/prefill.json", root + "/prefill.params");
    Require(imported.input_names == std::vector<std::string>{"input_ids"} &&
            imported.output_names.size() == 17, "prefill must consume tokens and return logits plus all K/V");
    auto function = relay::InferTypePass(imported.function);
    function = relay::RunRelayPassPipeline(function, {"fold_constant", "simplify_expr"});
    function = relay::InferTypePass(function);
    const auto device = Device::CUDA(0);
    Require(CollectDeviceAttributes(device).exists != 0, "CUDA fixture was requested but the device is unavailable");
    std::cout << "[INFO] compiling_full_minimind_cuda_prefill" << std::endl;
    profiling::ProfileOptions profile;
    profile.enabled = true;
    profile.enable_cupti = true;
    profile.ir_capture_mode = profiling::IRCaptureMode::kDisabled;
    profile.bundle_dir = root + "/cuda_prefill_profile_bundle";
    const auto compile_start = std::chrono::steady_clock::now();
    const auto graph = api::Compiler::Compile(function, api::CompileConfig::Create(BuildTarget(device), 3, profile));
    Require(graph.module().IsReady() && graph.plan().calls().size() == 650 &&
            graph.artifact_pins().size() == graph.plan().calls().size(),
            "full prefill must compile all 650 production primitives");
    std::cout << "[INFO] full_minimind_cuda_prefill kernels=" << graph.plan().calls().size()
              << " compile_ms=" << std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - compile_start).count()
              << " export_receipt=" << receipt << std::endl;
    const auto tokens = ReadValues<int64_t>(root + "/prefill_input_ids.bin", 16);
    auto input = runtime::NDArray::Empty({1, 16}, runtime::DataTypeFromString("int64"), device);
    input.CopyFromBytes(tokens.data(), input.NBytes());
    runtime::RuntimeSession session(graph.module(), graph.plan());
    const auto stream = DeviceStream::Create(device);
    const auto before = api::internal::GetPrimitiveCacheStats();
    Array<runtime::NDArray> first_outputs;
    for (int repeat = 0; repeat < 2; ++repeat) {
        const auto run_start = std::chrono::steady_clock::now();
        const auto run = session.RunAsync({input}, stream, {{"model", "minimind"},
            {"stage", "prefill"}, {"export_receipt", receipt}, {"sequence_length", "16"},
            {"repeat", std::to_string(repeat)}});
        run.completion.Wait();
        const double run_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - run_start).count();
        Require(run.outputs.size() == 17 && SameStats(before, api::internal::GetPrimitiveCacheStats()),
                "RunAsync must return every tensor without compiling or consulting the primitive cache");
        for (size_t index = 0; index < run.outputs.size(); ++index) {
            const bool logits = index == 0;
            const std::string name = logits ? "logits" :
                std::string(index % 2 ? "k_" : "v_") + std::to_string((index - 1) / 2);
            const auto expected = ReadValues<float>(root + (logits ? "/prefill_reference_logits.bin" :
                "/seed_present_" + name + ".bin"), logits ? 102400 : 6144);
            const auto shape = logits ? Array<int64_t>{1, 16, 6400} : Array<int64_t>{1, 16, 4, 96};
            const auto& output = run.outputs[index];
            Require(output.device() == device && output.shape().size() == shape.size() &&
                    std::equal(shape.begin(), shape.end(), output.shape().begin()) &&
                    output.dtype().code == kDLFloat && output.dtype().bits == 32 &&
                    output.NBytes() == expected.size() * sizeof(float), "output contract mismatch: " + name);
            if (repeat) Require(output.storage().get() != first_outputs[index].storage().get(),
                                "separate prefill runs must own separate output storage");
            std::vector<float> actual(expected.size());
            output.CopyToBytes(actual.data(), output.NBytes());
            double worst = 0;
            for (size_t i = 0; i < actual.size(); ++i) {
                Require(std::isfinite(actual[i]) && std::isfinite(expected[i]),
                        "non-finite full-model output: " + name);
                const double error = std::fabs(static_cast<double>(actual[i]) - expected[i]);
                if (error > 1e-3) throw std::runtime_error(name + " mismatch at " + std::to_string(i) +
                    ": actual=" + std::to_string(actual[i]) + " expected=" + std::to_string(expected[i]));
                worst = std::max(worst, error);
            }
            std::cout << "[NUMERIC] minimind_prefill_" << name << "_repeat" << repeat
                      << " elements=" << actual.size() << " max_abs_error=" << std::setprecision(10) << worst << '\n';
        }
        if (!repeat) first_outputs = run.outputs;
        std::cout << "[INFO] completed_prefill repeat=" << repeat << " observed_run_ms=" << run_ms << std::endl;
    }
    std::cout << "[PASS] full_minimind_static_cuda_prefill_and_all_kv_no_runtime_compile\n";
}
}  // namespace

int main() {
    const char* root = std::getenv("KXC_MINIMIND_CUDA_PREFILL_DIR");
    if (!root || !*root) {
        std::cout << "[SKIP] set KXC_MINIMIND_CUDA_PREFILL_DIR to the complete static fixture\n";
        return 77;
    }
    try { Run(root); }
    catch (const std::exception& error) {
        std::cerr << "[FAIL] full_minimind_cuda_prefill: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
