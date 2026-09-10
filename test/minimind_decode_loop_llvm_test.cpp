/*! \file test/minimind_decode_loop_llvm_test.cpp
 * \brief G2 第 2 项：定容 KV 的多步 decode——同一产物、同一份 cache 存储，
 *        prefill 之后连续追加多步，逐步与 ONNX 参考比较，并由 host 做 greedy argmax。
 *
 * 为什么需要定容图：E0 的静态 decode 导出把 past 长度烤进形状（past 16 → 17、
 * past 17 → 18），每步是一个不同的编译产物，与「同一产物、runtime 期间不发生
 * 隐式编译」直接冲突。定容图把历史长度从形状里移出去，改由运行时的 `position`
 * 与 `attention_mask` 决定，因此一个产物服务任意步。
 *
 * 无效容量区填有限哨兵而不是 1e30 量级的值：掩码是加性的 `(1 - mask) * -1e9`，
 * 只能压住 1e9 量级的分数。实测填充值 <= 1e6 时输出与精确长度 past 逐位相同，
 * >= 1e9 时掩码失效、连有效位都会被一起压掉。取非零哨兵是为了让泄漏可被发现。
 *
 * 产物不入库。`python/tools/make_minimind_decode_loop_fixture.py` 生成后用
 * `KXC_MINIMIND_DECODE_LOOP_DIR` 指向；未设置时跳过。
 */

#include "../src/compiler/internal/primitive_cache.h"
#include "kxc/compiler/compiler.h"
#include "kxc/compiler/experimental_identity.h"
#include "kxc/frontend/onnx_importer.h"
#include "kxc/profiling/profiling.h"
#include "kxc/relay/transforms/infer_type.h"
#include "kxc/relay/transforms/pipeline.h"
#include "kxc/runtime/session.h"
#include "kxc/runtime/device_api.h"

#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace compiler_internal = kxc::api::internal;

#define CHECK(condition, message)                                             \
    do {                                                                      \
        if (!(condition)) {                                                   \
            std::cerr << "[FAIL] " << (message) << '\n';                      \
            return false;                                                     \
        }                                                                     \
    } while (false)

struct Step final {
    int index = 0;
    int64_t token = 0;
    int64_t extent_before = 0;
};

int64_t ArgmaxLastToken(const std::vector<float>& logits, int64_t vocab) {
    if (vocab <= 0 || logits.size() != static_cast<size_t>(vocab)) {
        throw std::invalid_argument("greedy logits row has an invalid vocabulary size");
    }
    size_t best = 0;
    for (size_t index = 1; index < logits.size(); ++index) {
        if (logits[index] > logits[best]) best = index;
    }
    return static_cast<int64_t>(best);
}

std::string ReadLine(const std::string& path, bool* ok) {
    std::ifstream stream(path);
    std::string line;
    *ok = *ok && static_cast<bool>(std::getline(stream, line));
    return line;
}

std::vector<char> ReadFile(const std::string& path, bool* ok) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        *ok = false;
        return {};
    }
    const std::streamsize size = stream.tellg();
    stream.seekg(0);
    std::vector<char> bytes(static_cast<size_t>(size));
    *ok = *ok && static_cast<bool>(stream.read(bytes.data(), size));
    return bytes;
}

bool SameStats(const compiler_internal::PrimitiveCacheStats& before,
               const compiler_internal::PrimitiveCacheStats& after) {
    return before.hits == after.hits && before.misses == after.misses &&
           before.entries == after.entries && before.evictions == after.evictions &&
           before.in_flight == after.in_flight && before.failures == after.failures &&
           before.accounted_bytes == after.accounted_bytes &&
           before.merged_waiters == after.merged_waiters &&
           before.rejections == after.rejections && before.active_pins == after.active_pins;
}

kxc::Function PrepareJitRelayFunction(kxc::Function function) {
    function = kxc::relay::InferTypePass(function);
    function = kxc::relay::RunRelayPassPipeline(
        function, {kxc::String("fold_constant"), kxc::String("simplify_expr")});
    return kxc::relay::InferTypePass(function);
}

kxc::api::CompiledGraph CompileForTarget(kxc::Function function, kxc::Device device,
                                         kxc::profiling::ProfileOptions options) {
    const bool cuda = device.device_type() == kxc::kCUDA;
    // Two modules need independent bundles even under a global profiling env.
    std::optional<kxc::profiling::ActivationScope> activation;
    if (cuda) activation.emplace(kxc::profiling::ProfileContext::Create(options), "prepare");
    return kxc::api::Compiler::Compile(PrepareJitRelayFunction(function),
        kxc::api::CompileConfig::Create(kxc::BuildTarget(device), cuda ? 3 : 0, options));
}

std::string PlanAbi(const kxc::api::CompiledGraph& compiled,
                    const kxc::runtime::ExecutablePlan& plan) {
    const auto calls = plan.calls();
    const auto& pins = compiled.artifact_pins();
    if (pins.size() != calls.size()) {
        throw std::invalid_argument("stateful plan must preserve compiled artifact order");
    }
    std::vector<kxc::api::OrderedArtifactIdentity> artifacts;
    for (size_t index = 0; index < calls.size(); ++index) {
        artifacts.push_back({index, std::string(calls[index]->symbol),
                             pins[index].record().artifact_key});
    }
    return kxc::api::BuildPlanAbiFingerprint(compiled.module(), plan, artifacts).digest();
}

bool TestCapacityDecodeLoop(const std::string& root, bool vision, bool cuda) {
#if !KXC_USE_LLVM && !KXC_USE_CUDA
    std::cout << "[SKIP] minimind decode loop: no compiled backend\n";
    return true;
#else
#if !KXC_USE_LLVM
    if (!cuda) {
        std::cout << "[SKIP] minimind decode loop: KXC_USE_LLVM=0\n";
        return true;
    }
#endif
    const auto device = cuda ? kxc::Device::CUDA() : kxc::Device::CPU();
    const std::vector<kxc::DeviceStream> streams{
        cuda ? kxc::DeviceStream::Create(device) : kxc::DeviceStream::Default(device),
        cuda ? kxc::DeviceStream::Create(device) : kxc::DeviceStream::Default(device)};
    bool ok = true;
    CHECK(ReadLine(root + "/graph.txt", &ok) == "decode_capacity" && ok,
          "the fixture must be a capacity-shaped decode graph");
    const int64_t capacity = std::stoll(ReadLine(root + "/capacity.txt", &ok));
    const float sentinel = std::stof(ReadLine(root + "/sentinel.txt", &ok));
    int64_t extent = std::stoll(ReadLine(root + "/seed_extent.txt", &ok));
    CHECK(ok, "capacity.txt / sentinel.txt / seed_extent.txt must be readable");
    CHECK(std::fabs(sentinel) <= 1e6f,
          "the invalid-region sentinel must stay within what the additive mask can suppress");

    int64_t batch = 0, kv_heads = 0, head_dim = 0, layers = 0;
    {
        std::istringstream layout(ReadLine(root + "/layout.txt", &ok));
        CHECK(ok && (layout >> batch >> kv_heads >> head_dim >> layers),
              "layout.txt must hold 'batch kv_heads head_dim layers'");
    }

    std::vector<Step> steps;
    {
        std::ifstream stream(root + "/steps.txt");
        std::string line;
        while (std::getline(stream, line)) {
            if (line.empty()) continue;
            std::istringstream fields(line);
            Step step;
            CHECK(fields >> step.index >> step.token >> step.extent_before,
                  "each steps.txt line must be 'step token extent'");
            steps.push_back(step);
        }
    }
    CHECK(steps.size() >= 3, "G2 item 2 requires at least three decode steps");
    CHECK(batch == 1, "the minimal greedy fixture requires batch=1");

    const std::string sampling = ReadLine(root + "/sampling.txt", &ok);
    const std::string export_receipt = ReadLine(root + "/export_receipt.txt", &ok);
    CHECK(ok && sampling == "greedy_argmax" && !export_receipt.empty(),
          "the fixture must declare greedy_argmax and an export receipt");

    // 编译一次；后面每一步都复用这一个产物。
    const kxc::frontend::ImportedONNXModel imported =
        kxc::frontend::LoadONNXImportSpec(root + "/decode_capacity.json",
                                          root + "/decode_capacity.params");
    const size_t kv_inputs = static_cast<size_t>(layers) * 2;
    CHECK(imported.input_names.size() == kv_inputs + 3,
          "the capacity graph takes input_ids, position, attention_mask and the KV pairs");
    CHECK(imported.input_names[0] == "input_ids" &&
              imported.input_names[1] == "position" &&
              imported.input_names[2] == "attention_mask",
          "the runtime-driven inputs must lead the signature");
    for (int64_t layer = 0; layer < layers; ++layer) {
        const std::string suffix = std::to_string(layer);
        CHECK(imported.input_names[3 + layer * 2] == "past_k_" + suffix &&
                  imported.input_names[4 + layer * 2] == "past_v_" + suffix,
              "past tensors must stay interleaved per layer");
    }

    kxc::profiling::ProfileOptions profile_options;
    profile_options.enabled = true;
    profile_options.enable_cupti = cuda;
    profile_options.ir_capture_mode = kxc::profiling::IRCaptureMode::kDisabled;
    const char* cuda_bundle = cuda ? std::getenv("KXC_MINIMIND_CUDA_STATE_BUNDLE_DIR") : nullptr;
    profile_options.bundle_dir = cuda
        ? (cuda_bundle ? std::string(cuda_bundle) : root + "/cuda_profile_bundle")
        : root + "/profile_bundle";
    const auto compiled = CompileForTarget(imported.function, device, profile_options);
    CHECK(compiled.module().IsReady(), "the capacity decode graph must compile");
    std::cout << "[INFO] decode loop: " << compiled.plan().calls().size()
              << " kernel calls, capacity " << capacity << ", seed extent " << extent
              << ", " << steps.size() << " steps\n";

    // Prefill 也走真实 Compiler；其输出通过初始化合同交给 decode 的
    // RuntimeSession。ONNX 文件只提供独立参考，不能替代被测的 K/V 生产者。
    const auto prefill_imported = kxc::frontend::LoadONNXImportSpec(
        root + "/prefill.json", root + "/prefill.params");
    auto prefill_profile = profile_options;
    prefill_profile.bundle_dir = cuda ? profile_options.bundle_dir + "/prefill"
                                      : root + "/prefill_profile_bundle";
    const auto prefill_compiled = CompileForTarget(prefill_imported.function, device, prefill_profile);
    kxc::runtime::RuntimeSession prefill_session(
        prefill_compiled.module(), prefill_compiled.plan());
    const auto prefill_input_bytes = ReadFile(root + "/prefill_input_ids.bin", &ok);
    CHECK(ok && prefill_input_bytes.size() ==
                    static_cast<size_t>(batch * extent) * sizeof(int64_t),
          "prefill input must match the declared seed extent");
    const auto prefill_input = kxc::runtime::NDArray::Empty(
        {batch, extent}, kxc::runtime::DataTypeFromString("int64"), device);
    prefill_input.CopyFromBytes(prefill_input_bytes.data(), prefill_input_bytes.size());
    kxc::Array<kxc::runtime::NDArray> prefill_inputs{prefill_input};
    if (vision) {
        CHECK(layers == 8 && kv_heads == 4 && head_dim == 96,
              "the VLM fixture must retain the full locked language architecture");
        CHECK(prefill_imported.input_names.size() == 2 &&
                  prefill_imported.input_names[0] == "input_ids" &&
                  prefill_imported.input_names[1] == "pixel_values",
              "VLM prefill must consume tokens and pixels in the declared order");
        std::vector<int64_t> image_fields;
        {
            std::istringstream profile(ReadLine(root + "/image_profile.txt", &ok));
            int64_t field = 0;
            while (profile >> field) image_fields.push_back(field);
            CHECK(profile.eof(), "the image profile must contain only integer fields");
        }
        CHECK(ok && (image_fields.size() == 4 || image_fields.size() == 6),
              "image_profile.txt must hold the legacy 4-field or current 6-field profile");
        const int64_t image_marker = image_fields[0];
        const int64_t image_start = image_fields[1];
        const int64_t image_tokens = image_fields[2];
        const int64_t image_count = image_fields.size() == 4 ? 1 : image_fields[3];
        const int64_t vocab = image_fields.size() == 4 ? image_fields[3] : image_fields[4];
        const int64_t profile_extent = image_fields.size() == 4 ? extent : image_fields[5];
        CHECK(image_marker == 12 && image_start == 2 && image_tokens == 64 &&
                  image_count > 0 && image_count <= 30 && vocab == 6400 &&
                  profile_extent == extent && extent == 2 + image_count * 66 &&
                  capacity == extent + 12,
              "the export must retain the complete fixed image marker and slot profile");
        std::vector<int64_t> token_ids(static_cast<size_t>(extent));
        std::memcpy(token_ids.data(), prefill_input_bytes.data(), prefill_input_bytes.size());
        const auto valid_image_layout = [=](const std::vector<int64_t>& values) {
            if (values.size() != static_cast<size_t>(extent)) return false;
            for (size_t index = 0; index < values.size(); ++index) {
                bool image_slot = false;
                for (int64_t image_index = 0; image_index < image_count; ++image_index) {
                    const int64_t start = image_start + image_index * (image_tokens + 2);
                    image_slot = image_slot ||
                        (static_cast<int64_t>(index) >= start &&
                         static_cast<int64_t>(index) < start + image_tokens);
                }
                if (values[index] < 0 || values[index] >= vocab ||
                    ((values[index] == image_marker) != image_slot)) return false;
            }
            return true;
        };
        CHECK(valid_image_layout(token_ids),
              "input tokens violate the explicit ONNX image-slot specialization");
        for (int64_t image_index = 0; image_index < image_count; ++image_index) {
            auto wrong_layout = token_ids;
            const int64_t start = image_start + image_index * (image_tokens + 2);
            wrong_layout[static_cast<size_t>(start)] = image_marker == 0 ? 1 : 0;
            CHECK(!valid_image_layout(wrong_layout), "missing image marker must fail preflight");
        }
        auto wrong_layout = token_ids;
        wrong_layout[0] = image_marker;
        CHECK(!valid_image_layout(wrong_layout), "an image marker outside the slot must fail preflight");
        const auto pixels = ReadFile(root + "/pixel_values.bin", &ok);
        const size_t pixels_per_image = 3 * 256 * 256 * sizeof(float);
        CHECK(ok && pixels.size() == static_cast<size_t>(image_count) * pixels_per_image,
              "VLM pixels must contain one complete float32 [1,3,256,256] image per slot");
        const auto image = image_count == 1
            ? kxc::runtime::NDArray::Empty({1, 3, 256, 256},
                  kxc::runtime::DataTypeFromString("float32"), device)
            : kxc::runtime::NDArray::Empty({1, image_count, 3, 256, 256},
                  kxc::runtime::DataTypeFromString("float32"), device);
        image.CopyFromBytes(pixels.data(), pixels.size());
        prefill_inputs.push_back(image);
    }
    const std::string prefill_receipt =
        ReadLine(root + "/prefill_export_receipt.txt", &ok);
    CHECK(ok && !prefill_receipt.empty(), "prefill export receipt must be present");
    auto prefill_result = prefill_session.RunAsync(
        prefill_inputs, streams[0],
        {{"export_receipt", prefill_receipt}, {"stage", "prefill"},
         {"model", vision ? "minimind-v" : "minimind"},
         {"generation", "1"}, {"plan_abi", PlanAbi(prefill_compiled, prefill_compiled.plan())},
         {"state_extent", std::to_string(extent)}, {"state_version", "1"}});
    prefill_result.completion.Wait();
    auto prefill_outputs = std::move(prefill_result.outputs);
    CHECK(prefill_outputs.size() == kv_inputs + 1 &&
              prefill_imported.output_names.size() == kv_inputs + 1,
          "compiled prefill must produce logits and every layer's K/V pair");
    const auto prefill_expected_bytes =
        ReadFile(root + "/prefill_reference_logits.bin", &ok);
    CHECK(ok && prefill_outputs[0].NBytes() == prefill_expected_bytes.size(),
          "compiled prefill logits shape must match the reference");
    std::vector<float> prefill_produced(prefill_expected_bytes.size() / sizeof(float));
    std::vector<float> prefill_expected(prefill_produced.size());
    prefill_outputs[0].CopyToBytes(prefill_produced.data(), prefill_outputs[0].NBytes());
    std::memcpy(prefill_expected.data(), prefill_expected_bytes.data(),
                prefill_expected_bytes.size());
    double prefill_worst = 0.0;
    for (size_t index = 0; index < prefill_produced.size(); ++index) {
        CHECK(std::isfinite(prefill_produced[index]) && std::isfinite(prefill_expected[index]),
              "prefill logits and their reference must be finite");
        prefill_worst = std::max(prefill_worst,
            std::fabs(static_cast<double>(prefill_produced[index]) - prefill_expected[index]));
    }

    std::vector<kxc::runtime::StateOutputBinding> state_bindings;
    const auto input_ids = compiled.plan().input_value_ids();
    const auto output_ids = compiled.plan().output_value_ids();
    CHECK(input_ids.size() == kv_inputs + 3 && output_ids.size() == kv_inputs + 1,
          "compiled graph must preserve the capacity past/present interface");
    for (size_t index = 0; index < kv_inputs; ++index) {
        state_bindings.push_back(kxc::runtime::StateOutputBinding{
            input_ids[index + 3], output_ids[index + 1], 1, capacity, 1});
    }
    const kxc::runtime::ExecutablePlan state_plan =
        compiled.plan().BindStateOutputs(state_bindings, sentinel);
    const std::string state_plan_abi = PlanAbi(compiled, state_plan);
    CHECK(state_plan_abi != PlanAbi(compiled, compiled.plan()),
          "state binding must change the callable plan identity");
    kxc::runtime::RuntimeSession session(compiled.module(), state_plan);
    CHECK(state_plan.input_value_ids().size() == 3 &&
              state_plan.output_value_ids().size() == 1,
          "state bindings must leave only ids/position/mask inputs and logits output");

    // Cache 由 RuntimeSession 持有：只初始化有效前缀，其余保留 plan 的哨兵。
    const auto f32 = kxc::runtime::DataTypeFromString("float32");
    const size_t slot = static_cast<size_t>(kv_heads * head_dim);
    const size_t cache_elements = static_cast<size_t>(batch * capacity) * slot;
    std::vector<const void*> cache_addresses;
    size_t state_index = 0;
    for (int64_t layer = 0; layer < layers; ++layer) {
        for (const char* kind : {"k", "v"}) {
            const std::string name =
                std::string("present_") + kind + "_" + std::to_string(layer);
            CHECK(prefill_imported.output_names[state_index + 1] == name,
                  "prefill K/V output order must match the locked decode layout");
            const std::vector<char> seed = ReadFile(root + "/seed_" + name + ".bin", &ok);
            CHECK(ok && seed.size() == static_cast<size_t>(batch * extent) * slot * sizeof(float),
                  "seed_" + name + ".bin is missing or sized wrong");
            const auto& tensor = prefill_outputs[state_index + 1];
            CHECK(tensor.NBytes() == seed.size(),
                  "compiled prefill K/V output must match the reference shape");
            std::vector<float> produced(seed.size() / sizeof(float));
            std::vector<float> expected(produced.size());
            tensor.CopyToBytes(produced.data(), tensor.NBytes());
            std::memcpy(expected.data(), seed.data(), seed.size());
            for (size_t index = 0; index < produced.size(); ++index) {
                CHECK(std::isfinite(produced[index]) && std::isfinite(expected[index]),
                      "prefill K/V and their reference must be finite");
                prefill_worst = std::max(prefill_worst,
                    std::fabs(static_cast<double>(produced[index]) - expected[index]));
            }
            const int64_t state_id = state_bindings[state_index++].state_value_id;
            session.InitializeState(state_id, tensor, extent);
            CHECK(session.StateExtent(state_id) == extent,
                  "prefill initialization must commit the valid state prefix");
            CHECK(session.StateValue(state_id).storage().get() != tensor.storage().get(),
                  "session must own an independent state allocation");
            cache_addresses.push_back(session.StateValue(state_id).storage().data());
        }
    }
    prefill_outputs = {};
    prefill_result = {};
    CHECK(prefill_worst <= 1e-4, "compiled prefill logits and all K/V must match ONNX");
    std::cout << "[INFO] compiled prefill: " << prefill_compiled.plan().calls().size()
              << " kernels, worst difference " << prefill_worst << '\n';

    const compiler_internal::PrimitiveCacheStats before_steps =
        compiler_internal::GetPrimitiveCacheStats();

    const size_t logits_elements =
        static_cast<size_t>(batch) *
        static_cast<size_t>(ReadFile(root + "/ref_step0_logits.bin", &ok).size() /
                            (sizeof(float) * static_cast<size_t>(batch)));
    CHECK(ok, "ref_step0_logits.bin must be readable");
    CHECK(logits_elements % static_cast<size_t>(batch) == 0,
          "logits element count must divide evenly by batch");
    const int64_t vocab = static_cast<int64_t>(logits_elements / static_cast<size_t>(batch));
    const std::vector<char> prefill_reference =
        ReadFile(root + "/prefill_logits.bin", &ok);
    CHECK(ok && prefill_reference.size() == static_cast<size_t>(vocab) * sizeof(float),
          "prefill_logits.bin must contain one [1,1,vocab] last-position row");
    std::vector<float> prefill_logits(static_cast<size_t>(vocab), 0.0f);
    CHECK(prefill_produced.size() >= static_cast<size_t>(vocab),
          "compiled prefill must provide a complete final logits row");
    std::copy(prefill_produced.end() - vocab, prefill_produced.end(), prefill_logits.begin());
    int64_t next_token = ArgmaxLastToken(prefill_logits, vocab);

    double worst = 0.0;
    double worst_state = 0.0;
    for (const Step& step : steps) {
        CHECK(step.extent_before == extent,
              "step " + std::to_string(step.index) + " must start at the tracked extent");
        CHECK(step.token == next_token,
              "steps.txt token must equal host greedy argmax from the preceding logits");

        const kxc::runtime::NDArray ids = kxc::runtime::NDArray::Empty(
            {batch, 1}, kxc::runtime::DataTypeFromString("int64"), device);
        std::vector<int64_t> token_values(static_cast<size_t>(batch), step.token);
        ids.CopyFromBytes(token_values.data(), token_values.size() * sizeof(int64_t));

        const kxc::runtime::NDArray position = kxc::runtime::NDArray::Empty(
            {1}, kxc::runtime::DataTypeFromString("int64"), device);
        position.CopyFromBytes(&extent, sizeof(int64_t));

        // 掩码开有效历史 [0, extent) 与新 token（恒在下标 capacity），其余关闭。
        std::vector<float> mask(static_cast<size_t>(batch * (capacity + 1)), 0.0f);
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t i = 0; i < extent; ++i) mask[static_cast<size_t>(b * (capacity + 1) + i)] = 1.0f;
            mask[static_cast<size_t>(b * (capacity + 1) + capacity)] = 1.0f;
        }
        const kxc::runtime::NDArray attention_mask = kxc::runtime::NDArray::Empty(
            {batch, capacity + 1}, f32, device);
        attention_mask.CopyFromBytes(mask.data(), mask.size() * sizeof(float));

        kxc::Array<kxc::runtime::NDArray> inputs;
        inputs.push_back(ids);
        inputs.push_back(position);
        inputs.push_back(attention_mask);

        const kxc::runtime::ExecutionMetadata metadata{
            {"export_receipt", export_receipt},
            {"stage", "decode"},
            {"model", vision ? "minimind-v" : "minimind"},
            {"generation", "1"},
            {"plan_abi", state_plan_abi},
            {"state_extent", std::to_string(extent)},
            {"state_version", "1"},
            {"token_index", std::to_string(step.index)},
            {"token_id", std::to_string(step.token)},
        };
        const auto completed = session.RunAsync(inputs, streams[step.index % streams.size()], metadata);
        CHECK(completed.completion.IsReady(), "external state commit must finish before RunAsync returns");
        const auto& outputs = completed.outputs;
        CHECK(outputs.size() == 1,
              "stateful capacity plan must keep present tensors private");

        const std::vector<char> reference = ReadFile(
            root + "/ref_step" + std::to_string(step.index) + "_logits.bin", &ok);
        CHECK(ok && reference.size() == logits_elements * sizeof(float),
              "reference logits for step " + std::to_string(step.index) + " are missing");
        std::vector<float> produced(logits_elements, 0.0f);
        outputs[0].CopyToBytes(produced.data(), produced.size() * sizeof(float));
        std::vector<float> expected(logits_elements);
        std::memcpy(expected.data(), reference.data(), reference.size());
        for (size_t i = 0; i < logits_elements; ++i) {
            CHECK(std::isfinite(produced[i]) && std::isfinite(expected[i]),
                  "decode logits and their reference must be finite");
            worst = std::max(worst, std::fabs(static_cast<double>(produced[i]) -
                                              static_cast<double>(expected[i])));
        }
        next_token = ArgmaxLastToken(produced, vocab);

        // cache 地址在整个循环中不变。
        for (size_t i = 0; i < state_bindings.size(); ++i) {
            const int64_t state_id = state_bindings[i].state_value_id;
            CHECK(session.StateValue(state_id).storage().data() == cache_addresses[i],
                  "cache storage must not move across decode steps");
            CHECK(session.StateExtent(state_id) == extent + 1,
                  "RuntimeSession must commit each new KV slot and extent");
            std::vector<float> stored(cache_elements, 0.0f);
            session.StateValue(state_id).CopyToBytes(
                stored.data(), stored.size() * sizeof(float));
            if (vision || cuda) {
                const auto state_reference = ReadFile(
                    root + "/ref_step" + std::to_string(step.index) + "_state_" +
                        std::to_string(i) + ".bin", &ok);
                CHECK(ok && state_reference.size() == stored.size() * sizeof(float),
                      "every decode step needs all 16 complete cache references");
                std::vector<float> expected_state(stored.size());
                std::memcpy(expected_state.data(), state_reference.data(), state_reference.size());
                for (size_t element = 0; element < stored.size(); ++element) {
                    CHECK(std::isfinite(stored[element]) && std::isfinite(expected_state[element]),
                          "cache and reference must be finite");
                    worst_state = std::max(worst_state,
                        std::fabs(static_cast<double>(stored[element]) - expected_state[element]));
                }
            }
            for (int64_t s = extent + 1; s < capacity; ++s) {
                for (size_t element = 0; element < slot; ++element) {
                    CHECK(stored[static_cast<size_t>(s) * slot + element] == sentinel,
                          "a state append must preserve the invalid-region fill");
                }
            }
        }
        // runtime 期间不得发生隐式编译。
        CHECK(SameStats(before_steps, compiler_internal::GetPrimitiveCacheStats()),
              "no decode step may compile: the artifact is fixed");

        ++extent;
    }

    // 直接证据：把无效容量区换成另一个值，同一步的 logits 必须逐位相同。
    // 这比「匹配参考值」更强——它单独钉住「无效区不参与计算」，而不是靠整体
    // 一致来间接推断。
    {
        const Step& last = steps.back();
        const int64_t replay_extent = last.extent_before;
        const auto run_with_fill = [&](float fill) {
            std::vector<float> logits(logits_elements, 0.0f);
            const auto replay_plan =
                compiled.plan().BindStateOutputs(state_bindings, fill);
            kxc::runtime::RuntimeSession replay_session(compiled.module(), replay_plan);
            for (const auto& binding : state_bindings) {
                replay_session.InitializeState(
                    binding.state_value_id, session.StateValue(binding.state_value_id),
                    replay_extent);
            }
            const kxc::runtime::NDArray ids = kxc::runtime::NDArray::Empty(
                {batch, 1}, kxc::runtime::DataTypeFromString("int64"), device);
            std::vector<int64_t> token_values(static_cast<size_t>(batch), last.token);
            ids.CopyFromBytes(token_values.data(), token_values.size() * sizeof(int64_t));
            const kxc::runtime::NDArray position = kxc::runtime::NDArray::Empty(
                {1}, kxc::runtime::DataTypeFromString("int64"), device);
            position.CopyFromBytes(&replay_extent, sizeof(int64_t));
            std::vector<float> mask(static_cast<size_t>(batch * (capacity + 1)), 0.0f);
            for (int64_t b = 0; b < batch; ++b) {
                for (int64_t i = 0; i < replay_extent; ++i) {
                    mask[static_cast<size_t>(b * (capacity + 1) + i)] = 1.0f;
                }
                mask[static_cast<size_t>(b * (capacity + 1) + capacity)] = 1.0f;
            }
            const kxc::runtime::NDArray attention_mask = kxc::runtime::NDArray::Empty(
                {batch, capacity + 1}, f32, device);
            attention_mask.CopyFromBytes(mask.data(), mask.size() * sizeof(float));
            kxc::Array<kxc::runtime::NDArray> inputs;
            inputs.push_back(ids);
            inputs.push_back(position);
            inputs.push_back(attention_mask);
            const kxc::runtime::ExecutionMetadata metadata{
                {"export_receipt", export_receipt},
                {"stage", "decode_replay"},
                {"model", vision ? "minimind-v" : "minimind"},
                {"generation", "1"},
                {"plan_abi", PlanAbi(compiled, replay_plan)},
                {"state_extent", std::to_string(replay_extent)},
                {"state_version", "1"},
                {"token_index", std::to_string(last.index)},
                {"token_id", std::to_string(last.token)},
            };
            const auto replay = replay_session.RunAsync(inputs, streams[0], metadata);
            if (!replay.completion.IsReady())
                throw std::runtime_error("replay state commit is still pending");
            replay.outputs[0].CopyToBytes(logits.data(), logits.size() * sizeof(float));
            return logits;
        };
        const std::vector<float> quiet = run_with_fill(0.0f);
        const std::vector<float> noisy = run_with_fill(-1234.5f);
        CHECK(quiet == noisy,
              "logits must be bit-identical no matter what the invalid capacity region holds");
        std::cout << "[INFO] invalid-region fill 0.0 vs -1234.5: logits bit-identical\n";
    }
    CHECK(SameStats(before_steps, compiler_internal::GetPrimitiveCacheStats()),
          "decode and replay must preserve all ten primitive cache counters");

    std::cout << "[INFO] decode loop worst logits difference " << worst
              << " over " << steps.size() << " steps, extent "
              << steps.front().extent_before << " -> " << extent << '\n';
    CHECK(worst <= 1e-4,
          "every decode step must match the ONNX reference within 1e-4");
    if (vision || cuda) {
        CHECK(worst_state <= 1e-4, "all 16 caches must match the independent reference at every step");
        std::cout << "[INFO] complete cache worst difference " << worst_state << '\n';
    }
    std::cout << "[INFO] host sampling=greedy_argmax, export receipt "
              << export_receipt << ", profile bundle "
              << std::string(compiled.module().GetProfileBundlePath()) << '\n';
    return true;
#endif
}

}  // namespace

int main(int argc, char** argv) {
    std::cout << std::setprecision(12);
    const bool cuda = argc == 2 && std::string(argv[1]) == "--cuda";
    const bool vision = argc == 3 && std::string(argv[1]) == "--vlm";
    if (argc != 1 && !cuda && (!vision || (std::string(argv[2]) != "0" &&
                                  std::string(argv[2]) != "1" && std::string(argv[2]) != "2"))) {
        std::cerr << "usage: minimind_decode_loop_test [--vlm 0|1|2 | --cuda]\n";
        return 1;
    }
    if (cuda && !kxc::CollectDeviceAttributes(kxc::Device::CUDA()).exists) {
        std::cout << "[SKIP] no CUDA device\n";
        return 77;
    }
    const char* variable = cuda ? "KXC_MINIMIND_CUDA_STATE_DIR"
                               : vision ? "KXC_MINIMIND_VLM_DIR" : "KXC_MINIMIND_DECODE_LOOP_DIR";
    const char* directory = std::getenv(variable);
    if (!directory) {
        std::cout << "[SKIP] set " << variable << '\n';
        return cuda ? 77 : 0;
    }
    const std::string root = std::string(directory) + (vision ? "/case" + std::string(argv[2]) : "");
    try {
        if (!TestCapacityDecodeLoop(root, vision, cuda)) return 1;
        std::cout << "[PASS] " << (cuda ? "cuda_" : vision ? "vlm_" : "") << "capacity_decode_loop " << root << '\n';
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] capacity_decode_loop: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
