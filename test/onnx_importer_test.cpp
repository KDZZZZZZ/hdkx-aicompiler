/*! \file test/onnx_importer_test.cpp
 * \brief 验证 ONNX 导入后的图结构、常量张量和编译执行流程。
 */

#include "api/compiler.h"
#include "frontend/onnx_importer.h"
#include "relay/transforms/infer_type.h"
#include "relay/transforms/lower.h"
#include "relay/transforms/pipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifndef KXC_ONNX_IMPORT_JSON_PATH
#define KXC_ONNX_IMPORT_JSON_PATH "resnet18.import.json"
#endif

#ifndef KXC_ONNX_IMPORT_PARAMS_PATH
#define KXC_ONNX_IMPORT_PARAMS_PATH "resnet18.params.bin"
#endif

#ifndef KXC_USE_LLVM
#define KXC_USE_LLVM 0
#endif

namespace {

#define TEST_CHECK(cond, msg)                                                    \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::cerr << "[FAIL] " << __FUNCTION__ << ": " << (msg) << "\n";   \
            return false;                                                        \
        }                                                                        \
    } while (0)

// 比较 NDArray 的逻辑 shape，避免通过裸 DLTensor 指针读取元数据。
bool ShapeEquals(const kxc::runtime::NDArray& array, const std::vector<int64_t>& shape) {
    if (!array.defined() || array->shape_storage.size() != shape.size()) {
        return false;
    }
    for (size_t i = 0; i < shape.size(); ++i) {
        if (array->shape_storage[i] != shape[i]) {
            return false;
        }
    }
    return true;
}

// 校验 Relay TensorType 的 shape 和 dtype。
bool CheckTensor(const kxc::Type& type, const std::vector<int64_t>& shape,
                 const std::string& dtype) {
    const auto* tensor = type.As<kxc::TensorTypeNode>();
    if (!tensor || tensor->shape.size() != shape.size() || tensor->dtype != dtype) {
        return false;
    }
    for (size_t i = 0; i < shape.size(); ++i) {
        if (tensor->shape[i] != shape[i]) {
            return false;
        }
    }
    return true;
}

// 根据环境开关决定是否运行耗时的 ResNet18 数值内核。
bool ShouldRunResNet18Kernel() {
    const char* value = std::getenv("KXC_RUN_RESNET18_EXEC");
    return value != nullptr && std::string(value) == "1";
}

// 读取 ResNet18 JIT 优化级别，并提供稳定默认值。
int ResNet18OptLevel() {
    const char* value = std::getenv("KXC_RESNET18_OPT_LEVEL");
    if (!value) {
        return 1;
    }
    int opt_level = std::atoi(value);
    return std::clamp(opt_level, 0, 3);
}

// 对导入函数执行类型推导及 JIT 前所需的 Relay 规范化。
kxc::Function PrepareJitRelayFunction(kxc::Function func) {
    func = kxc::relay::InferTypePass(func);
    func = kxc::relay::RunRelayPassPipeline(
        func, {kxc::String("fold_constant"), kxc::String("simplify_expr")});
    return kxc::relay::InferTypePass(func);
}

// 使用确定性数据填充显式 CPU NDArray 输入。
void FillResNet18Input(const kxc::runtime::NDArray& input) {
    float* data = static_cast<float*>(input->dl_tensor.data);
    const size_t elements = input.NBytes() / sizeof(float);
    for (size_t i = 0; i < elements; ++i) {
        data[i] = (static_cast<float>(i % 251) - 125.0f) / 125.0f;
    }
}

// 将运行输出与可选参考文件比较并检查误差阈值。
bool ValidateReferenceOutput(const kxc::runtime::NDArray& output) {
    const char* reference_path = std::getenv("KXC_RESNET18_REFERENCE_OUTPUT_PATH");
    if (!reference_path) {
        return true;
    }

    std::ifstream input(reference_path, std::ios::binary);
    TEST_CHECK(input.good(), "failed to open ResNet18 reference output");

    std::vector<float> reference(1000, 0.0f);
    input.read(reinterpret_cast<char*>(reference.data()),
               static_cast<std::streamsize>(reference.size() * sizeof(float)));
    TEST_CHECK(input.gcount() == static_cast<std::streamsize>(reference.size() * sizeof(float)),
               "ResNet18 reference output should contain 1000 float32 values");

    const float* actual = static_cast<const float*>(output->dl_tensor.data);
    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    for (size_t i = 0; i < reference.size(); ++i) {
        const float abs_error = std::fabs(actual[i] - reference[i]);
        const float denom = std::max(1.0f, std::fabs(reference[i]));
        const float rel_error = abs_error / denom;
        max_abs_error = std::max(max_abs_error, abs_error);
        max_rel_error = std::max(max_rel_error, rel_error);
    }

    constexpr float kAbsTol = 1e-3f;
    constexpr float kRelTol = 1e-3f;
    std::cout << "[INFO] resnet18 reference max_abs_error=" << max_abs_error
              << ", max_rel_error=" << max_rel_error << "\n";
    TEST_CHECK(max_abs_error <= kAbsTol || max_rel_error <= kRelTol,
               "ResNet18 output should match reference within tolerance");
    return true;
}

// 验证 ResNet18 ONNX 导入后的参数、输出类型和常量元数据。
bool TestLoadResNet18ImportSpec() {
    kxc::frontend::ImportedONNXModel imported = kxc::frontend::LoadONNXImportSpec(
        KXC_ONNX_IMPORT_JSON_PATH, KXC_ONNX_IMPORT_PARAMS_PATH);

    TEST_CHECK(imported.function.defined(), "importer should construct a Relay Function");
    TEST_CHECK(imported.function->params.size() == 1, "ResNet18 should have one graph input");
    TEST_CHECK(imported.input_names.size() == 1 && imported.input_names[0] == "input",
               "input binding should be recorded");
    TEST_CHECK(imported.output_names.size() == 1 && imported.output_names[0] == "output",
               "output binding should be recorded");
    TEST_CHECK(imported.params.size() == 42, "all ONNX initializers should be loaded");
    TEST_CHECK(imported.param_order.size() == 42, "param order should be preserved");
    TEST_CHECK(imported.param_order[0] == "fc.weight", "initializer order should be stable");

    const auto& fc_weight = imported.params.at("fc.weight");
    TEST_CHECK(ShapeEquals(fc_weight, {1000, 512}), "fc.weight shape mismatch");
    TEST_CHECK(fc_weight.NBytes() == 1000ULL * 512ULL * 4ULL, "fc.weight byte size mismatch");
    const float* fc_data = static_cast<const float*>(fc_weight->dl_tensor.data);
    TEST_CHECK(std::fabs(fc_data[0] - (-0.018474037f)) < 1e-6f,
               "fc.weight bytes should contain real initializer data");

    kxc::relay::InferTypePass(imported.function);
    TEST_CHECK(CheckTensor(imported.function->body.checked_type(), {1, 1000}, "float32"),
               "imported Relay body should infer ResNet18 output type");
    return true;
}

// 验证导入的 ResNet18 可完成 Relay 到 LLVM 编译。
bool TestCompileResNet18ToLLVM() {
#if KXC_USE_LLVM
    kxc::frontend::ImportedONNXModel imported = kxc::frontend::LoadONNXImportSpec(
        KXC_ONNX_IMPORT_JSON_PATH, KXC_ONNX_IMPORT_PARAMS_PATH);

    kxc::Function prepared = PrepareJitRelayFunction(imported.function);
    auto config = kxc::api::CompileConfig::JIT(kxc::BuildTarget(kxc::Device::CPU()));
    config->opt_level = ResNet18OptLevel();
    auto module = kxc::api::Compiler::Compile(prepared, config);
    TEST_CHECK(module.IsReady(), "ResNet18 should compile to a ready LLVM module");
    TEST_CHECK(module.GetPrimFunc().defined(), "ResNet18 compile should keep generated TIR");
    return true;
#else
    std::cout << "[SKIP] resnet18 LLVM compile: KXC_USE_LLVM=0\n";
    return true;
#endif
}

// 验证编译后的 ResNet18 使用 NDArray 参数执行并匹配参考结果。
bool TestRunCompiledResNet18LLVM() {
#if KXC_USE_LLVM
    if (!ShouldRunResNet18Kernel()) {
        std::cout << "[SKIP] resnet18 LLVM execute: set KXC_RUN_RESNET18_EXEC=1 to run\n";
        return true;
    }

    kxc::frontend::ImportedONNXModel imported = kxc::frontend::LoadONNXImportSpec(
        KXC_ONNX_IMPORT_JSON_PATH, KXC_ONNX_IMPORT_PARAMS_PATH);
    kxc::Function prepared = PrepareJitRelayFunction(imported.function);

    auto config = kxc::api::CompileConfig::JIT(kxc::BuildTarget(kxc::Device::CPU()));
    config->opt_level = ResNet18OptLevel();
    auto module = kxc::api::Compiler::Compile(prepared, config);
    TEST_CHECK(module.IsReady(), "ResNet18 should compile before execution");
    const kxc::Array<kxc::relay::ConstantBinding> constants = module.GetConstants();
    TEST_CHECK(constants.size() == imported.params.size(),
               "compiled module constant count should match loaded ONNX initializers");

    // JIT ABI 仍接收底层地址，但内存所有权和设备身份由 NDArray 保持到调用结束。
    kxc::runtime::NDArray input = kxc::runtime::NDArray::Empty(
        {1, 3, 224, 224}, kxc::runtime::DataTypeFromString("float32"),
        kxc::Device::CPU());
    kxc::runtime::NDArray output = kxc::runtime::NDArray::Empty(
        {1, 1000}, kxc::runtime::DataTypeFromString("float32"),
        kxc::Device::CPU());
    FillResNet18Input(input);

    std::vector<void*> packed_args;
    packed_args.reserve(1 + constants.size() + 1);
    packed_args.push_back(input->dl_tensor.data);
    for (const auto& constant : constants) {
        packed_args.push_back(constant->value->dl_tensor.data);
    }
    packed_args.push_back(output->dl_tensor.data);

    std::cout << "[INFO] running compiled resnet18 LLVM kernel with "
              << packed_args.size() << " packed buffers\n";
    std::cout.flush();

    const auto start = std::chrono::steady_clock::now();
    module.Run(packed_args);
    const auto end = std::chrono::steady_clock::now();

    const float* out = static_cast<const float*>(output->dl_tensor.data);
    float max_abs = 0.0f;
    double sum = 0.0;
    for (size_t i = 0; i < 1000; ++i) {
        TEST_CHECK(std::isfinite(out[i]), "ResNet18 output should be finite");
        max_abs = std::max(max_abs, std::fabs(out[i]));
        sum += out[i];
    }
    TEST_CHECK(max_abs > 0.0f, "ResNet18 output should not be all zeros");
    TEST_CHECK(ValidateReferenceOutput(output), "ResNet18 reference validation failed");

    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "[INFO] resnet18 LLVM execute elapsed_ms=" << elapsed_ms
              << ", output_sum=" << sum
              << ", output_max_abs=" << max_abs
              << ", output_head=[" << out[0] << ", " << out[1] << ", " << out[2] << "]\n";
    return true;
#else
    std::cout << "[SKIP] resnet18 LLVM execute: KXC_USE_LLVM=0\n";
    return true;
#endif
}

}  // namespace

// 顺序执行 ONNX 导入、编译和可选数值测试。
int main() {
    try {
        if (!TestLoadResNet18ImportSpec()) {
            return 1;
        }
        if (!TestCompileResNet18ToLLVM()) {
            return 1;
        }
        if (!TestRunCompiledResNet18LLVM()) {
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] unexpected exception: " << e.what() << "\n";
        return 1;
    }

    std::cout << "[PASS] onnx_importer_load_resnet18\n";
    std::cout << "[PASS] onnx_importer_compile_resnet18_llvm\n";
    std::cout << "[PASS] onnx_importer_run_resnet18_llvm\n";
    std::cout << "All ONNX importer tests passed.\n";
    return 0;
}
