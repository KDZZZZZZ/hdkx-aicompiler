/*! \file test/minimind_l1a_llvm_test.cpp
 * \brief L1a/L1b 验收：E0 锁定的真实 MiniMind 静态图经真实 importer → Relay →
 *        LLVM → RuntimeSession 运行，与 ONNX 参考实现逐元素比较。
 *
 * 两张图共用同一条链路：prefill（1 输入 / 17 输出）与 decode（17 输入 /
 * 17 输出）。签名不写死在代码里，而是从 fixture 的 inputs.txt / outputs.txt
 * 读出后**逐项校验结构**——这样同一个测试能覆盖两张图，签名漂移仍然必然失败。
 *
 * 产物不入库（权重数百 MB）。由 `python/tools/make_minimind_l1a_fixture.py`
 * 生成后，用 `KXC_MINIMIND_IMPORT_DIR` 指向该目录才会真正执行；未设置时跳过，
 * 以免把大产物变成构建硬依赖。
 *
 * 本测试始终走标准 Relay 流水线。不要加绕过流水线的开关：绕过之后测到的
 * 规模数据不完整（会漏掉 mutator 一侧的行为），这个坑踩过一次。
 */

#include "kxc/compiler/compiler.h"
#include "kxc/frontend/onnx_importer.h"
#include "kxc/relay/transforms/infer_type.h"
#include "kxc/relay/transforms/pipeline.h"
#include "kxc/runtime/session.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

#define CHECK(condition, message)                                             \
    do {                                                                      \
        if (!(condition)) {                                                   \
            std::cerr << "[FAIL] " << (message) << '\n';                      \
            return false;                                                     \
        }                                                                     \
    } while (false)

struct TensorEntry final {
    std::string name;
    std::string dtype;  // 输出清单省略 dtype，一律 float32
    std::vector<int64_t> shape;
};

// 读清单：每行 "name [dtype] d0 d1 ..."。dtype 只出现在 inputs.txt。
std::vector<TensorEntry> ReadManifest(const std::string& path, bool with_dtype,
                                      bool* ok) {
    std::ifstream stream(path);
    std::vector<TensorEntry> entries;
    if (!stream) {
        *ok = false;
        return entries;
    }
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty()) continue;
        std::istringstream fields(line);
        TensorEntry entry;
        entry.dtype = "float32";
        if (!(fields >> entry.name)) {
            *ok = false;
            return entries;
        }
        if (with_dtype && !(fields >> entry.dtype)) {
            *ok = false;
            return entries;
        }
        int64_t dim = 0;
        while (fields >> dim) entry.shape.push_back(dim);
        if (entry.shape.empty()) {
            *ok = false;
            return entries;
        }
        entries.push_back(std::move(entry));
    }
    *ok = !entries.empty();
    return entries;
}

std::string ReadLine(const std::string& path, bool* ok) {
    std::ifstream stream(path);
    std::string line;
    *ok = static_cast<bool>(std::getline(stream, line));
    return line;
}

// 输出签名：logits 打头，随后按层交错 present_k_i / present_v_i。两张图相同。
bool OutputSignatureIsWellFormed(const std::vector<TensorEntry>& outputs) {
    if (outputs.empty() || outputs.front().name != "logits") return false;
    if ((outputs.size() - 1) % 2 != 0) return false;
    for (size_t layer = 0; layer * 2 + 1 < outputs.size(); ++layer) {
        const std::string suffix = std::to_string(layer);
        if (outputs[layer * 2 + 1].name != "present_k_" + suffix) return false;
        if (outputs[layer * 2 + 2].name != "present_v_" + suffix) return false;
    }
    return true;
}

// 输入签名：prefill 只有 input_ids；decode 是 input_ids 后按层交错的
// past_k_i / past_v_i。past 的层序错了必须被发现，否则会静默算错。
bool InputSignatureIsWellFormed(const std::vector<TensorEntry>& inputs,
                                const std::string& graph) {
    if (inputs.empty() || inputs.front().name != "input_ids" ||
        inputs.front().dtype != "int64") {
        return false;
    }
    if (graph == "prefill") return inputs.size() == 1;
    if ((inputs.size() - 1) % 2 != 0) return false;
    for (size_t layer = 0; layer * 2 + 1 < inputs.size(); ++layer) {
        const std::string suffix = std::to_string(layer);
        if (inputs[layer * 2 + 1].name != "past_k_" + suffix) return false;
        if (inputs[layer * 2 + 2].name != "past_v_" + suffix) return false;
    }
    return true;
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
    *ok = static_cast<bool>(stream.read(bytes.data(), size));
    return bytes;
}

int64_t ElementCount(const std::vector<int64_t>& shape) {
    int64_t count = 1;
    for (int64_t dim : shape) count *= dim;
    return count;
}

kxc::Function PrepareJitRelayFunction(kxc::Function function) {
    function = kxc::relay::InferTypePass(function);
    function = kxc::relay::RunRelayPassPipeline(
        function, {kxc::String("fold_constant"), kxc::String("simplify_expr")});
    return kxc::relay::InferTypePass(function);
}

bool RunLockedGraph() {
#if !KXC_USE_LLVM
    std::cout << "[SKIP] minimind: KXC_USE_LLVM=0\n";
    return true;
#else
    const char* directory = std::getenv("KXC_MINIMIND_IMPORT_DIR");
    if (!directory) {
        std::cout << "[SKIP] minimind: set KXC_MINIMIND_IMPORT_DIR to the "
                     "generated fixture directory\n";
        return true;
    }
    const std::string root(directory);

    bool ok = true;
    const std::string graph = ReadLine(root + "/graph.txt", &ok);
    CHECK(ok && (graph == "prefill" || graph == "decode"),
          "graph.txt must name the fixture's graph: prefill or decode");

    const kxc::frontend::ImportedONNXModel imported =
        kxc::frontend::LoadONNXImportSpec(root + "/" + graph + ".json",
                                          root + "/" + graph + ".params");

    const std::vector<TensorEntry> declared_inputs =
        ReadManifest(root + "/inputs.txt", /*with_dtype=*/true, &ok);
    CHECK(ok, "inputs.txt must list the fixture's input signature");
    const std::vector<TensorEntry> declared_outputs =
        ReadManifest(root + "/outputs.txt", /*with_dtype=*/false, &ok);
    CHECK(ok, "outputs.txt must list the fixture's output signature");

    CHECK(InputSignatureIsWellFormed(declared_inputs, graph),
          "the input signature must be input_ids, then per-layer past_k/past_v");
    CHECK(OutputSignatureIsWellFormed(declared_outputs),
          "the output signature must be logits, then per-layer present_k/present_v");

    CHECK(imported.input_names.size() == declared_inputs.size(),
          "the imported graph must keep the fixture's input arity");
    for (size_t i = 0; i < declared_inputs.size(); ++i) {
        CHECK(imported.input_names[i] == declared_inputs[i].name,
              "input " + std::to_string(i) + " must stay '" +
                  declared_inputs[i].name + "'");
    }
    CHECK(imported.output_names.size() == declared_outputs.size(),
          "the imported graph must keep the fixture's output arity");
    for (size_t i = 0; i < declared_outputs.size(); ++i) {
        CHECK(imported.output_names[i] == declared_outputs[i].name,
              "output " + std::to_string(i) + " must stay '" +
                  declared_outputs[i].name + "'");
    }

    const kxc::Function prepared = PrepareJitRelayFunction(imported.function);
    const auto config = kxc::api::CompileConfig::Create(
        kxc::BuildTarget(kxc::Device::CPU()), 0);
    const kxc::api::CompiledGraph compiled =
        kxc::api::Compiler::Compile(prepared, config);
    CHECK(compiled.module().IsReady(),
          "the real MiniMind " + graph + " graph must compile to a ready LLVM module");
    std::cout << "[INFO] minimind " << graph << ": "
              << compiled.plan().calls().size() << " kernel calls, "
              << compiled.module().constants().size() << " constants\n";

    // 按声明顺序构造全部输入；decode 的 16 个 past 也从这里进来。
    kxc::Array<kxc::runtime::NDArray> inputs;
    for (const TensorEntry& declared : declared_inputs) {
        const std::vector<char> bytes =
            ReadFile(root + "/in_" + declared.name + ".bin", &ok);
        const size_t width = declared.dtype == "int64" ? sizeof(int64_t) : sizeof(float);
        const size_t elements = static_cast<size_t>(ElementCount(declared.shape));
        CHECK(ok && bytes.size() == elements * width,
              "in_" + declared.name + ".bin is missing or sized wrong");
        const kxc::runtime::NDArray tensor = kxc::runtime::NDArray::Empty(
            kxc::Array<int64_t>(declared.shape),
            kxc::runtime::DataTypeFromString(declared.dtype), kxc::Device::CPU());
        tensor.CopyFromBytes(bytes.data(), bytes.size());
        inputs.push_back(tensor);
    }

    kxc::runtime::RuntimeSession session(compiled.module(), compiled.plan());
    const kxc::Array<kxc::runtime::NDArray> outputs = session.Run(inputs);
    CHECK(outputs.size() == declared_outputs.size(),
          "the runtime plan must return one tensor per declared output");

    // 逐个输出比较，不只比 logits：KV 错而 logits 对的情况是存在的。
    double worst = 0.0;
    std::string worst_name;
    for (size_t i = 0; i < declared_outputs.size(); ++i) {
        const TensorEntry& declared = declared_outputs[i];
        const kxc::runtime::NDArray actual = outputs[i];
        CHECK(actual.shape().size() == declared.shape.size(),
              declared.name + " rank must match the locked signature");
        for (size_t axis = 0; axis < declared.shape.size(); ++axis) {
            CHECK(actual.shape()[axis] == declared.shape[axis],
                  declared.name + " axis " + std::to_string(axis) +
                      " must match the locked signature");
        }
        const size_t elements = static_cast<size_t>(ElementCount(declared.shape));
        const std::vector<char> reference_bytes =
            ReadFile(root + "/ref_" + declared.name + ".bin", &ok);
        CHECK(ok && reference_bytes.size() == elements * sizeof(float),
              "reference bytes for " + declared.name + " are missing or sized wrong");
        std::vector<float> produced(elements, 0.0f);
        actual.CopyToBytes(produced.data(), produced.size() * sizeof(float));
        const float* reference = reinterpret_cast<const float*>(reference_bytes.data());
        for (size_t index = 0; index < elements; ++index) {
            const double difference =
                std::fabs(static_cast<double>(produced[index]) -
                          static_cast<double>(reference[index]));
            if (difference > worst) {
                worst = difference;
                worst_name = declared.name;
            }
        }
    }
    std::cout << "[INFO] minimind " << graph << " worst absolute difference "
              << worst << " at '" << worst_name << "'\n";
    // 实测最坏误差随层数平稳增长：prefill 1/2/3/4/8 层 = 4.91 / 6.93 / 8.17 /
    // 8.70 / 8.82e-06，就是 float32 舍入噪声（差异只来自 MatMul 的归约顺序）。
    // 阈值取 1e-4，比实测高一个数量级，够容纳更深的模型，又不至于放过真实的
    // 数值退化。decode 若超出这个阈值，那是真实信号——先查清楚，不要为了让它
    // 通过而放宽 prefill 也在用的这个值。
    CHECK(worst <= 1e-4,
          "every output must match the ONNX reference element-wise within 1e-4");
    return true;
#endif
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"minimind_locked_graph", RunLockedGraph},
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
