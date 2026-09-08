/*! \file test/minimind_l1a_llvm_test.cpp
 * \brief L1a 验收：E0 锁定的真实 MiniMind 静态 prefill 图经真实 importer →
 *        Relay → LLVM → RuntimeSession 运行，与 ONNX 参考实现逐元素比较。
 *
 * 产物不入库（权重 275 MB）。由 `python/tools/make_minimind_l1a_fixture.py`
 * 生成后，用 `KXC_MINIMIND_IMPORT_DIR` 指向该目录才会真正执行；未设置时跳过，
 * 以免把大产物变成构建硬依赖。
 *
 * 输出签名（17 个，顺序固定）直接钉在本文件里：它就是 M9 E2 审计确定、M2/M3
 * 依赖的那份 ABI，写死等于让签名漂移必然触发失败。
 */

#include "kxc/compiler/compiler.h"
#include "kxc/frontend/onnx_importer.h"
#include "kxc/relay/transforms/infer_type.h"
#include "kxc/relay/transforms/pipeline.h"
#include "kxc/runtime/session.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
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

struct ExpectedOutput final {
    std::string name;
    std::vector<int64_t> shape;
};

// 从 fixture 的 outputs.txt 读出签名（每行 "name d0 d1 ..."）。签名由 E0 导出
// 决定，本测试只负责钉住它：顺序、命名与形状任何漂移都会失败。
std::vector<ExpectedOutput> ReadExpectedOutputs(const std::string& path, bool* ok) {
    std::ifstream stream(path);
    std::vector<ExpectedOutput> outputs;
    if (!stream) {
        *ok = false;
        return outputs;
    }
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty()) continue;
        std::istringstream fields(line);
        ExpectedOutput output;
        if (!(fields >> output.name)) {
            *ok = false;
            return outputs;
        }
        int64_t dim = 0;
        while (fields >> dim) output.shape.push_back(dim);
        if (output.shape.empty()) {
            *ok = false;
            return outputs;
        }
        outputs.push_back(std::move(output));
    }
    *ok = !outputs.empty();
    return outputs;
}

// 签名结构：logits 打头，随后按层交错 present_k_i / present_v_i。
bool SignatureIsWellFormed(const std::vector<ExpectedOutput>& outputs) {
    if (outputs.empty() || outputs.front().name != "logits") return false;
    if ((outputs.size() - 1) % 2 != 0) return false;
    for (size_t layer = 0; layer * 2 + 1 < outputs.size(); ++layer) {
        const std::string suffix = std::to_string(layer);
        if (outputs[layer * 2 + 1].name != "present_k_" + suffix) return false;
        if (outputs[layer * 2 + 2].name != "present_v_" + suffix) return false;
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

bool TestMiniMindL1aPrefill() {
#if !KXC_USE_LLVM
    std::cout << "[SKIP] minimind L1a: KXC_USE_LLVM=0\n";
    return true;
#else
    const char* directory = std::getenv("KXC_MINIMIND_IMPORT_DIR");
    if (!directory) {
        std::cout << "[SKIP] minimind L1a: set KXC_MINIMIND_IMPORT_DIR to the "
                     "generated fixture directory\n";
        return true;
    }
    const std::string root(directory);

    
    const kxc::frontend::ImportedONNXModel imported =
        kxc::frontend::LoadONNXImportSpec(root + "/prefill.json",
                                          root + "/prefill.params");
    
    bool ok = true;
    const std::vector<ExpectedOutput> expected =
        ReadExpectedOutputs(root + "/outputs.txt", &ok);
    CHECK(ok, "outputs.txt must list the fixture's output signature");
    CHECK(SignatureIsWellFormed(expected),
          "the signature must be logits followed by per-layer present_k/present_v");
    CHECK(imported.output_names.size() == expected.size(),
          "the imported prefill graph must keep the fixture's output signature");
    for (size_t i = 0; i < expected.size(); ++i) {
        CHECK(imported.output_names[i] == expected[i].name,
              "output " + std::to_string(i) + " must stay '" + expected[i].name + "'");
    }
    CHECK(imported.input_names.size() == 1 &&
              imported.input_names[0] == "input_ids",
          "the prefill graph takes exactly one int64 input_ids tensor");

    
    const kxc::Function prepared = PrepareJitRelayFunction(imported.function);
    
    const auto config = kxc::api::CompileConfig::Create(
        kxc::BuildTarget(kxc::Device::CPU()), 0);
    
    const kxc::api::CompiledGraph compiled =
        kxc::api::Compiler::Compile(prepared, config);
    
    CHECK(compiled.module().IsReady(),
          "the real MiniMind prefill graph must compile to a ready LLVM module");
    std::cout << "[INFO] minimind L1a: " << compiled.plan().calls().size()
              << " kernel calls, " << compiled.module().constants().size()
              << " constants\n";

    const std::vector<char> input_bytes = ReadFile(root + "/input_ids.bin", &ok);
    const int64_t batch = expected.front().shape[0];
    const int64_t seq = expected.front().shape[1];
    CHECK(ok && input_bytes.size() == static_cast<size_t>(batch * seq) * sizeof(int64_t),
          "input_ids.bin must hold the fixture's int64[batch, seq] token ids");
    const kxc::runtime::NDArray input = kxc::runtime::NDArray::Empty(
        {batch, seq}, kxc::runtime::DataTypeFromString("int64"), kxc::Device::CPU());
    input.CopyFromBytes(input_bytes.data(), input_bytes.size());

    kxc::runtime::RuntimeSession session(compiled.module(), compiled.plan());
    const kxc::Array<kxc::runtime::NDArray> outputs = session.Run({input});
    CHECK(outputs.size() == expected.size(),
          "the runtime plan must return one tensor per declared output");

    double worst = 0.0;
    std::string worst_name;
    for (size_t i = 0; i < expected.size(); ++i) {
        const ExpectedOutput& declared = expected[i];
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
    std::cout << "[INFO] minimind L1a worst absolute difference " << worst
              << " at '" << worst_name << "'\n";
    // 实测最坏误差随层数平稳增长：1/2/3/4/8 层 = 4.91 / 6.93 / 8.17 / 8.70 /
    // 8.82e-06，就是 float32 舍入噪声（差异只来自 MatMul 的归约顺序）。阈值取
    // 1e-4，比实测高一个数量级，够容纳更深的模型，又不至于放过真实的数值退化。
    CHECK(worst <= 1e-4,
          "every output must match the ONNX reference element-wise within 1e-4");
    return true;
#endif
}

}  // namespace

int main() {
    const std::vector<std::pair<const char*, bool (*)()>> tests = {
        {"minimind_l1a_prefill", TestMiniMindL1aPrefill},
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
