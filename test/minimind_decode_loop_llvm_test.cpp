/*! \file test/minimind_decode_loop_llvm_test.cpp
 * \brief G2 第 2 项：定容 KV 的多步 decode——同一产物、同一份 cache 存储，
 *        prefill 之后连续追加多步，逐步与 ONNX 参考比较。
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
#include "kxc/frontend/onnx_importer.h"
#include "kxc/relay/transforms/infer_type.h"
#include "kxc/relay/transforms/pipeline.h"
#include "kxc/runtime/session.h"

#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
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
           before.in_flight == after.in_flight && before.failures == after.failures;
}

kxc::Function PrepareJitRelayFunction(kxc::Function function) {
    function = kxc::relay::InferTypePass(function);
    function = kxc::relay::RunRelayPassPipeline(
        function, {kxc::String("fold_constant"), kxc::String("simplify_expr")});
    return kxc::relay::InferTypePass(function);
}

bool TestCapacityDecodeLoop() {
#if !KXC_USE_LLVM
    std::cout << "[SKIP] minimind decode loop: KXC_USE_LLVM=0\n";
    return true;
#else
    const char* directory = std::getenv("KXC_MINIMIND_DECODE_LOOP_DIR");
    if (!directory) {
        std::cout << "[SKIP] minimind decode loop: set KXC_MINIMIND_DECODE_LOOP_DIR\n";
        return true;
    }
    const std::string root(directory);

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

    const kxc::api::CompiledGraph compiled = kxc::api::Compiler::Compile(
        PrepareJitRelayFunction(imported.function),
        kxc::api::CompileConfig::Create(kxc::BuildTarget(kxc::Device::CPU()), 0));
    CHECK(compiled.module().IsReady(), "the capacity decode graph must compile");
    std::cout << "[INFO] decode loop: " << compiled.plan().calls().size()
              << " kernel calls, capacity " << capacity << ", seed extent " << extent
              << ", " << steps.size() << " steps\n";

    // 定容 cache：有效区来自 prefill 种子，其余填哨兵。整个循环复用这批张量。
    const auto f32 = kxc::runtime::DataTypeFromString("float32");
    const size_t slot = static_cast<size_t>(kv_heads * head_dim);
    const size_t cache_elements = static_cast<size_t>(batch * capacity) * slot;
    std::vector<kxc::runtime::NDArray> cache;
    std::vector<const void*> cache_addresses;
    for (int64_t layer = 0; layer < layers; ++layer) {
        for (const char* kind : {"k", "v"}) {
            const std::string name =
                std::string("present_") + kind + "_" + std::to_string(layer);
            const std::vector<char> seed = ReadFile(root + "/seed_" + name + ".bin", &ok);
            CHECK(ok && seed.size() == static_cast<size_t>(batch * extent) * slot * sizeof(float),
                  "seed_" + name + ".bin is missing or sized wrong");
            std::vector<float> values(cache_elements, sentinel);
            std::memcpy(values.data(), seed.data(), seed.size());
            const kxc::runtime::NDArray tensor = kxc::runtime::NDArray::Empty(
                {batch, capacity, kv_heads, head_dim}, f32, kxc::Device::CPU());
            tensor.CopyFromBytes(values.data(), values.size() * sizeof(float));
            cache_addresses.push_back(tensor.storage().data());
            cache.push_back(tensor);
        }
    }

    kxc::runtime::RuntimeSession session(compiled.module(), compiled.plan());
    const compiler_internal::PrimitiveCacheStats before_steps =
        compiler_internal::GetPrimitiveCacheStats();

    const size_t logits_elements =
        static_cast<size_t>(batch) *
        static_cast<size_t>(ReadFile(root + "/ref_step0_logits.bin", &ok).size() /
                            (sizeof(float) * static_cast<size_t>(batch)));
    CHECK(ok, "ref_step0_logits.bin must be readable");

    double worst = 0.0;
    for (const Step& step : steps) {
        CHECK(step.extent_before == extent,
              "step " + std::to_string(step.index) + " must start at the tracked extent");

        const kxc::runtime::NDArray ids = kxc::runtime::NDArray::Empty(
            {batch, 1}, kxc::runtime::DataTypeFromString("int64"), kxc::Device::CPU());
        std::vector<int64_t> token_values(static_cast<size_t>(batch), step.token);
        ids.CopyFromBytes(token_values.data(), token_values.size() * sizeof(int64_t));

        const kxc::runtime::NDArray position = kxc::runtime::NDArray::Empty(
            {1}, kxc::runtime::DataTypeFromString("int64"), kxc::Device::CPU());
        position.CopyFromBytes(&extent, sizeof(int64_t));

        // 掩码开有效历史 [0, extent) 与新 token（恒在下标 capacity），其余关闭。
        std::vector<float> mask(static_cast<size_t>(batch * (capacity + 1)), 0.0f);
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t i = 0; i < extent; ++i) mask[static_cast<size_t>(b * (capacity + 1) + i)] = 1.0f;
            mask[static_cast<size_t>(b * (capacity + 1) + capacity)] = 1.0f;
        }
        const kxc::runtime::NDArray attention_mask = kxc::runtime::NDArray::Empty(
            {batch, capacity + 1}, f32, kxc::Device::CPU());
        attention_mask.CopyFromBytes(mask.data(), mask.size() * sizeof(float));

        kxc::Array<kxc::runtime::NDArray> inputs;
        inputs.push_back(ids);
        inputs.push_back(position);
        inputs.push_back(attention_mask);
        for (const kxc::runtime::NDArray& tensor : cache) inputs.push_back(tensor);

        const kxc::Array<kxc::runtime::NDArray> outputs = session.Run(inputs);
        CHECK(outputs.size() == kv_inputs + 1,
              "the plan must return logits plus one present per KV input");

        const std::vector<char> reference = ReadFile(
            root + "/ref_step" + std::to_string(step.index) + "_logits.bin", &ok);
        CHECK(ok && reference.size() == logits_elements * sizeof(float),
              "reference logits for step " + std::to_string(step.index) + " are missing");
        std::vector<float> produced(logits_elements, 0.0f);
        outputs[0].CopyToBytes(produced.data(), produced.size() * sizeof(float));
        const float* expected = reinterpret_cast<const float*>(reference.data());
        for (size_t i = 0; i < logits_elements; ++i) {
            worst = std::max(worst, std::fabs(static_cast<double>(produced[i]) -
                                              static_cast<double>(expected[i])));
        }

        // cache 地址在整个循环中不变。
        for (size_t i = 0; i < cache.size(); ++i) {
            CHECK(cache[i].storage().data() == cache_addresses[i],
                  "cache storage must not move across decode steps");
        }
        // runtime 期间不得发生隐式编译。
        CHECK(SameStats(before_steps, compiler_internal::GetPrimitiveCacheStats()),
              "no decode step may compile: the artifact is fixed");

        // 新 K/V 恒在 present 的下标 capacity；写回 cache 的 extent 槽位。
        for (size_t i = 0; i < cache.size(); ++i) {
            std::vector<float> present(
                static_cast<size_t>(batch * (capacity + 1)) * slot, 0.0f);
            outputs[i + 1].CopyToBytes(present.data(), present.size() * sizeof(float));
            std::vector<float> stored(cache_elements, 0.0f);
            cache[i].CopyToBytes(stored.data(), stored.size() * sizeof(float));
            for (int64_t b = 0; b < batch; ++b) {
                const size_t from =
                    (static_cast<size_t>(b) * static_cast<size_t>(capacity + 1) +
                     static_cast<size_t>(capacity)) * slot;
                const size_t to =
                    (static_cast<size_t>(b) * static_cast<size_t>(capacity) +
                     static_cast<size_t>(extent)) * slot;
                std::memcpy(stored.data() + to, present.data() + from, slot * sizeof(float));
            }
            cache[i].CopyFromBytes(stored.data(), stored.size() * sizeof(float));
        }
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
            std::vector<kxc::runtime::NDArray> replay;
            for (size_t i = 0; i < cache.size(); ++i) {
                std::vector<float> stored(cache_elements, 0.0f);
                cache[i].CopyToBytes(stored.data(), stored.size() * sizeof(float));
                for (int64_t b = 0; b < batch; ++b) {
                    for (int64_t s = replay_extent; s < capacity; ++s) {
                        const size_t base =
                            (static_cast<size_t>(b) * static_cast<size_t>(capacity) +
                             static_cast<size_t>(s)) * slot;
                        std::fill_n(stored.begin() + static_cast<long>(base), slot, fill);
                    }
                }
                const kxc::runtime::NDArray tensor = kxc::runtime::NDArray::Empty(
                    {batch, capacity, kv_heads, head_dim}, f32, kxc::Device::CPU());
                tensor.CopyFromBytes(stored.data(), stored.size() * sizeof(float));
                replay.push_back(tensor);
            }
            const kxc::runtime::NDArray ids = kxc::runtime::NDArray::Empty(
                {batch, 1}, kxc::runtime::DataTypeFromString("int64"), kxc::Device::CPU());
            std::vector<int64_t> token_values(static_cast<size_t>(batch), last.token);
            ids.CopyFromBytes(token_values.data(), token_values.size() * sizeof(int64_t));
            const kxc::runtime::NDArray position = kxc::runtime::NDArray::Empty(
                {1}, kxc::runtime::DataTypeFromString("int64"), kxc::Device::CPU());
            position.CopyFromBytes(&replay_extent, sizeof(int64_t));
            std::vector<float> mask(static_cast<size_t>(batch * (capacity + 1)), 0.0f);
            for (int64_t b = 0; b < batch; ++b) {
                for (int64_t i = 0; i < replay_extent; ++i) {
                    mask[static_cast<size_t>(b * (capacity + 1) + i)] = 1.0f;
                }
                mask[static_cast<size_t>(b * (capacity + 1) + capacity)] = 1.0f;
            }
            const kxc::runtime::NDArray attention_mask = kxc::runtime::NDArray::Empty(
                {batch, capacity + 1}, f32, kxc::Device::CPU());
            attention_mask.CopyFromBytes(mask.data(), mask.size() * sizeof(float));
            kxc::Array<kxc::runtime::NDArray> inputs;
            inputs.push_back(ids);
            inputs.push_back(position);
            inputs.push_back(attention_mask);
            for (const kxc::runtime::NDArray& tensor : replay) inputs.push_back(tensor);
            session.Run(inputs)[0].CopyToBytes(logits.data(), logits.size() * sizeof(float));
            return logits;
        };
        const std::vector<float> quiet = run_with_fill(0.0f);
        const std::vector<float> noisy = run_with_fill(-1234.5f);
        CHECK(quiet == noisy,
              "logits must be bit-identical no matter what the invalid capacity region holds");
        std::cout << "[INFO] invalid-region fill 0.0 vs -1234.5: logits bit-identical\n";
    }

    std::cout << "[INFO] decode loop worst logits difference " << worst
              << " over " << steps.size() << " steps, extent "
              << steps.front().extent_before << " -> " << extent << '\n';
    CHECK(worst <= 1e-4,
          "every decode step must match the ONNX reference within 1e-4");
    return true;
#endif
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"capacity_decode_loop", TestCapacityDecodeLoop},
    };
    for (const auto& [name, test] : tests) {
        try {
            if (!test()) return 1;
            std::cout << "[PASS] " << name << '\n';
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
            return 1;
        }
    }
    return 0;
}
