/*! \file test/onnx_importer_test.cpp
 * \brief 验证 ONNX 导入后的图结构、常量张量和编译执行流程。
 */

#include "kxc/compiler/compiler.h"
#include "kxc/frontend/onnx_importer.h"
#include "kxc/relay/op.h"
#include "kxc/relay/transforms/infer_type.h"
#include "kxc/relay/transforms/pipeline.h"
#include "kxc/runtime/session.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#ifndef KXC_ONNX_IMPORT_JSON_PATH
#define KXC_ONNX_IMPORT_JSON_PATH "resnet18.import.json"
#endif

#ifndef KXC_ONNX_IMPORT_PARAMS_PATH
#define KXC_ONNX_IMPORT_PARAMS_PATH "resnet18.params.bin"
#endif

#ifndef KXC_ONNX_TRANSFORMER_JSON_PATH
#define KXC_ONNX_TRANSFORMER_JSON_PATH "exact_transformer.import.json"
#endif

#ifndef KXC_ONNX_TRANSFORMER_PARAMS_PATH
#define KXC_ONNX_TRANSFORMER_PARAMS_PATH "exact_transformer.params.bin"
#endif

#ifndef KXC_ONNX_STATIC_S1_JSON_PATH
#define KXC_ONNX_STATIC_S1_JSON_PATH "static_s1.import.json"
#endif

#ifndef KXC_ONNX_STATIC_S1_PARAMS_PATH
#define KXC_ONNX_STATIC_S1_PARAMS_PATH "static_s1.params.bin"
#endif

#ifndef KXC_ONNX_EQUAL_WHERE_JSON_PATH
#define KXC_ONNX_EQUAL_WHERE_JSON_PATH "equal_where.import.json"
#endif

#ifndef KXC_ONNX_EQUAL_WHERE_PARAMS_PATH
#define KXC_ONNX_EQUAL_WHERE_PARAMS_PATH "equal_where.params.bin"
#endif

#ifndef KXC_ONNX_M4M5_OPS_JSON_PATH
#define KXC_ONNX_M4M5_OPS_JSON_PATH "m4m5_ops.import.json"
#endif

#ifndef KXC_ONNX_M4M5_OPS_PARAMS_PATH
#define KXC_ONNX_M4M5_OPS_PARAMS_PATH "m4m5_ops.params.bin"
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
    const size_t elements = input.NBytes() / sizeof(float);
    std::vector<float> data(elements);
    for (size_t i = 0; i < elements; ++i) {
        data[i] = (static_cast<float>(i % 251) - 125.0f) / 125.0f;
    }
    input.CopyFromBytes(data.data(), input.NBytes());
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

    std::vector<float> actual(1000);
    output.CopyToBytes(actual.data(), output.NBytes());
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

// 从真实 ONNX protobuf 生成的 spec 必须经 reifier、LLVM Compiler 和
// RuntimeSession 完成 exact Transformer 数值链路；该小 fixture 不使用跳过开关。
bool TestRunExactTransformerProtobufLLVM() {
    kxc::frontend::ImportedONNXModel imported =
        kxc::frontend::LoadONNXImportSpec(
            KXC_ONNX_TRANSFORMER_JSON_PATH,
            KXC_ONNX_TRANSFORMER_PARAMS_PATH);
    TEST_CHECK(imported.function.defined() && imported.function->params.size() == 1 &&
                   imported.params.size() == 9 &&
                   imported.input_names.size() == 1 &&
                   imported.input_names[0] == "embedding_table" &&
                   imported.output_names.size() == 1 &&
                   imported.output_names[0] == "context",
               "protobuf Transformer fixture must preserve importer/reifier bindings");
    TEST_CHECK(CheckTensor(imported.function->body.checked_type(), {3, 2}, "float32"),
               "protobuf Transformer fixture output contract mismatch");
#if KXC_USE_LLVM
    const kxc::Function prepared = PrepareJitRelayFunction(imported.function);
    const auto compiled = kxc::api::Compiler::Compile(
        prepared, kxc::api::CompileConfig::Create(
                      kxc::BuildTarget(kxc::Device::CPU()), 1));
    TEST_CHECK(compiled.module().IsReady() && compiled.plan().calls().size() == 9,
               "protobuf Transformer fixture must compile to nine LLVM units");

    const std::vector<float> table_values = {1, 3, 2, 2, 4, 0, 0, 4};
    kxc::runtime::NDArray table = kxc::runtime::NDArray::Empty(
        {4, 2}, kxc::runtime::DataTypeFromString("float32"),
        kxc::Device::CPU());
    table.CopyFromBytes(table_values.data(), table.NBytes());
    kxc::runtime::RuntimeSession session(compiled.module(), compiled.plan());
    const kxc::Array<kxc::runtime::NDArray> outputs = session.Run({table});
    TEST_CHECK(outputs.size() == 1 && ShapeEquals(outputs[0], {3, 2}),
               "protobuf Transformer RuntimeSession output shape mismatch");

    std::vector<float> actual(6, 0.0f);
    outputs[0].CopyToBytes(actual.data(), outputs[0].NBytes());
    constexpr float epsilon = 1e-5f;
    const float a = static_cast<float>(1.0 / std::sqrt(1.0 + epsilon));
    const std::vector<float> sequence = {-a, a, -a, a, 0, 0};
    std::vector<float> expected(6, 0.0f);
    for (size_t row = 0; row < 3; ++row) {
        float scores[3]{};
        float maximum = -std::numeric_limits<float>::infinity();
        for (size_t column = 0; column < 3; ++column) {
            scores[column] =
                sequence[row * 2] * sequence[column * 2] +
                sequence[row * 2 + 1] * sequence[column * 2 + 1];
            maximum = std::max(maximum, scores[column]);
        }
        float denominator = 0.0f;
        float probabilities[3]{};
        for (size_t column = 0; column < 3; ++column) {
            probabilities[column] = std::exp(scores[column] - maximum);
            denominator += probabilities[column];
        }
        for (size_t column = 0; column < 3; ++column) {
            const float probability = probabilities[column] / denominator;
            expected[row * 2] += probability * sequence[column * 2];
            expected[row * 2 + 1] += probability * sequence[column * 2 + 1];
        }
    }
    for (size_t index = 0; index < actual.size(); ++index) {
        TEST_CHECK(std::isfinite(actual[index]) &&
                       std::fabs(actual[index] - expected[index]) <= 3e-4f,
                   "protobuf Transformer RuntimeSession numeric mismatch at " +
                       std::to_string(index));
    }
#endif
    return true;
}

// 静态 S1 组合 fixture：真实 protobuf 导入后必须经过 LLVM Compiler 与
// RuntimeSession，并与独立手写参考逐元素比较（绝对误差 1e-5）。
// 图：Mul(x,const[4]) → Sub(-0.25) → Div(/2) → Sqrt → Cast(idx int64→f32)
//     → Mul(*3) → ReduceMean(axes=[2],keepdims=1) → Reshape([6])。
bool TestRunStaticS1ProtobufLLVM() {
    kxc::frontend::ImportedONNXModel imported =
        kxc::frontend::LoadONNXImportSpec(
            KXC_ONNX_STATIC_S1_JSON_PATH,
            KXC_ONNX_STATIC_S1_PARAMS_PATH);
    TEST_CHECK(imported.function.defined() && imported.function->params.size() == 2 &&
                   imported.params.size() == 4 &&
                   imported.input_names.size() == 2 &&
                   imported.input_names[0] == "x" &&
                   imported.input_names[1] == "idx" &&
                   imported.output_names.size() == 1 &&
                   imported.output_names[0] == "out",
               "static S1 fixture must preserve importer/reifier bindings");
    TEST_CHECK(CheckTensor(imported.function->body.checked_type(), {6}, "float32"),
               "static S1 fixture output contract mismatch");
#if KXC_USE_LLVM
    const kxc::Function prepared = PrepareJitRelayFunction(imported.function);
    const auto compiled = kxc::api::Compiler::Compile(
        prepared, kxc::api::CompileConfig::Create(
                      kxc::BuildTarget(kxc::Device::CPU()), 1));
    TEST_CHECK(compiled.module().IsReady() && compiled.plan().calls().size() == 8,
               "static S1 fixture must compile to eight LLVM units");

    // 独立参考：不经过 importer/Relay，直接按 ONNX 语义手写计算。
    const std::vector<float> scale = {1.0f, 2.0f, 0.5f, 4.0f};
    std::vector<float> x_values(24);
    for (size_t i = 0; i < x_values.size(); ++i) {
        x_values[i] = 1.0f + static_cast<float>(i % 7) * 0.25f;
    }
    std::vector<float> expected(6, 0.0f);
    for (int b = 0; b < 2; ++b) {
        for (int r = 0; r < 3; ++r) {
            float sum = 0.0f;
            for (int c = 0; c < 4; ++c) {
                const size_t index = static_cast<size_t>((b * 3 + r) * 4 + c);
                sum += std::sqrt((x_values[index] * scale[c] - 0.25f) / 2.0f) * 3.0f;
            }
            expected[static_cast<size_t>(b * 3 + r)] = sum / 4.0f;
        }
    }

    kxc::runtime::NDArray x = kxc::runtime::NDArray::Empty(
        {2, 3, 4}, kxc::runtime::DataTypeFromString("float32"), kxc::Device::CPU());
    x.CopyFromBytes(x_values.data(), x.NBytes());
    const int64_t idx_value = 3;
    kxc::runtime::NDArray idx = kxc::runtime::NDArray::Empty(
        {1}, kxc::runtime::DataTypeFromString("int64"), kxc::Device::CPU());
    idx.CopyFromBytes(&idx_value, idx.NBytes());
    kxc::runtime::RuntimeSession session(compiled.module(), compiled.plan());
    const kxc::Array<kxc::runtime::NDArray> outputs = session.Run({x, idx});
    TEST_CHECK(outputs.size() == 1 && ShapeEquals(outputs[0], {6}),
               "static S1 RuntimeSession output shape mismatch");

    std::vector<float> actual(6, 0.0f);
    outputs[0].CopyToBytes(actual.data(), outputs[0].NBytes());
    float max_abs_error = 0.0f;
    for (size_t index = 0; index < actual.size(); ++index) {
        TEST_CHECK(std::isfinite(actual[index]),
                   "static S1 output must be finite at " + std::to_string(index));
        max_abs_error = std::max(max_abs_error, std::fabs(actual[index] - expected[index]));
    }
    std::cout << "[INFO] static_s1 max_abs_error=" << max_abs_error << "\n";
    TEST_CHECK(max_abs_error <= 1e-5f,
               "static S1 RuntimeSession numeric mismatch beyond 1e-5 absolute "
               "tolerance");
#endif
    return true;
}

// M4 接线验证：真实 protobuf 经 Python importer 序列化后，C++ reifier 重建
// Equal→Where 组合图（Equal 的 bool 输出直接作 Where condition），通过 LLVM
// Compiler 与 RuntimeSession 执行，并与独立 NumPy 参考逐元素 bit-exact 比较。
// 图：Equal(a[2,3], b[3]) → cond[2,3] bool → Where(cond, x[1,3], y[]) → out[2,3]。
// x/y 是 initializer 常量参数；Where 只做分支选择且比较值均可精确表示，
// 因此期望结果为 bit-exact（容差 0）。
bool TestRunEqualWhereProtobufLLVM() {
    kxc::frontend::ImportedONNXModel imported =
        kxc::frontend::LoadONNXImportSpec(
            KXC_ONNX_EQUAL_WHERE_JSON_PATH,
            KXC_ONNX_EQUAL_WHERE_PARAMS_PATH);
    TEST_CHECK(imported.function.defined() && imported.function->params.size() == 2 &&
                   imported.params.size() == 2 &&
                   imported.input_names.size() == 2 &&
                   imported.input_names[0] == "a" &&
                   imported.input_names[1] == "b" &&
                   imported.output_names.size() == 1 &&
                   imported.output_names[0] == "out",
               "Equal-Where fixture must preserve importer/reifier bindings");
    TEST_CHECK(CheckTensor(imported.function->body.checked_type(), {2, 3}, "float32"),
               "Equal-Where fixture output contract mismatch");
#if KXC_USE_LLVM
    const kxc::Function prepared = PrepareJitRelayFunction(imported.function);
    const auto compiled = kxc::api::Compiler::Compile(
        prepared, kxc::api::CompileConfig::Create(
                      kxc::BuildTarget(kxc::Device::CPU()), 1));
    TEST_CHECK(compiled.module().IsReady() && compiled.plan().calls().size() == 2,
               "Equal-Where fixture must compile to two LLVM units");

    const std::vector<float> a_values = {1, 2, 3, 4, 5, 6};
    const std::vector<float> b_values = {1, 0, 3};
    // 独立参考：Equal 的 NumPy 多向广播 + Where 逐元素选择，手工展开。
    const std::vector<float> expected = {10, -1, 30, -1, -1, -1};

    kxc::runtime::NDArray a = kxc::runtime::NDArray::Empty(
        {2, 3}, kxc::runtime::DataTypeFromString("float32"), kxc::Device::CPU());
    a.CopyFromBytes(a_values.data(), a.NBytes());
    kxc::runtime::NDArray b = kxc::runtime::NDArray::Empty(
        {3}, kxc::runtime::DataTypeFromString("float32"), kxc::Device::CPU());
    b.CopyFromBytes(b_values.data(), b.NBytes());
    kxc::runtime::RuntimeSession session(compiled.module(), compiled.plan());
    const kxc::Array<kxc::runtime::NDArray> outputs = session.Run({a, b});
    TEST_CHECK(outputs.size() == 1 && ShapeEquals(outputs[0], {2, 3}),
               "Equal-Where RuntimeSession output shape mismatch");

    std::vector<float> actual(6, 0.0f);
    outputs[0].CopyToBytes(actual.data(), outputs[0].NBytes());
    for (size_t index = 0; index < actual.size(); ++index) {
        TEST_CHECK(actual[index] == expected[index],
                   "Equal-Where RuntimeSession must match the NumPy reference "
                   "bit-exactly at " +
                       std::to_string(index) + ": got " +
                       std::to_string(actual[index]));
    }
#endif
    return true;
}

// M4/M5 五算子接线验证：真实 protobuf 经 Python importer 序列化后，C++
// reifier 重建 Neg → Pow(x²) → Sigmoid → Unsqueeze(axes=[0]→reshape) →
// Expand([1,2,4]→[3,2,4]) 图，通过 LLVM Compiler 与 RuntimeSession 执行，
// 并与独立手写参考逐元素比较（绝对误差 2e-5，覆盖 float32 sigmoid/pow）。
bool TestRunM4M5OpsProtobufLLVM() {
    kxc::frontend::ImportedONNXModel imported =
        kxc::frontend::LoadONNXImportSpec(
            KXC_ONNX_M4M5_OPS_JSON_PATH,
            KXC_ONNX_M4M5_OPS_PARAMS_PATH);
    TEST_CHECK(imported.function.defined() && imported.function->params.size() == 1 &&
                   imported.params.size() == 3 &&
                   imported.input_names.size() == 1 &&
                   imported.input_names[0] == "x" &&
                   imported.output_names.size() == 1 &&
                   imported.output_names[0] == "out",
               "M4/M5 ops fixture must preserve importer/reifier bindings");
    TEST_CHECK(CheckTensor(imported.function->body.checked_type(), {3, 2, 4}, "float32"),
               "M4/M5 ops fixture output contract mismatch");
#if KXC_USE_LLVM
    const kxc::Function prepared = PrepareJitRelayFunction(imported.function);
    const auto compiled = kxc::api::Compiler::Compile(
        prepared, kxc::api::CompileConfig::Create(
                      kxc::BuildTarget(kxc::Device::CPU()), 1));
    TEST_CHECK(compiled.module().IsReady() && compiled.plan().calls().size() == 5,
               "M4/M5 ops fixture must compile to five LLVM units");

    // 独立参考：不经过 importer/Relay，直接按 ONNX 语义手写计算。
    const float x_values[8] = {1.0f, -2.0f, 0.5f, 3.0f, -0.25f, 2.0f, -4.0f, 0.75f};
    std::vector<float> expected(24, 0.0f);
    for (int copy = 0; copy < 3; ++copy) {
        for (int row = 0; row < 2; ++row) {
            for (int column = 0; column < 4; ++column) {
                const double value = static_cast<double>(x_values[row * 4 + column]);
                const double squared = std::pow(-value, 2.0);
                expected[static_cast<size_t>((copy * 2 + row) * 4 + column)] =
                    static_cast<float>(1.0 / (1.0 + std::exp(-squared)));
            }
        }
    }

    kxc::runtime::NDArray x = kxc::runtime::NDArray::Empty(
        {2, 4}, kxc::runtime::DataTypeFromString("float32"), kxc::Device::CPU());
    x.CopyFromBytes(x_values, x.NBytes());
    kxc::runtime::RuntimeSession session(compiled.module(), compiled.plan());
    const kxc::Array<kxc::runtime::NDArray> outputs = session.Run({x});
    TEST_CHECK(outputs.size() == 1 && ShapeEquals(outputs[0], {3, 2, 4}),
               "M4/M5 ops RuntimeSession output shape mismatch");

    std::vector<float> actual(24, 0.0f);
    outputs[0].CopyToBytes(actual.data(), outputs[0].NBytes());
    float max_abs_error = 0.0f;
    for (size_t index = 0; index < actual.size(); ++index) {
        TEST_CHECK(std::isfinite(actual[index]),
                   "M4/M5 ops output must be finite at " + std::to_string(index));
        max_abs_error = std::max(max_abs_error, std::fabs(actual[index] - expected[index]));
    }
    std::cout << "[INFO] m4m5_ops max_abs_error=" << max_abs_error << "\n";
    TEST_CHECK(max_abs_error <= 2e-5f,
               "M4/M5 ops RuntimeSession numeric mismatch beyond 2e-5 absolute "
               "tolerance");
#endif
    return true;
}

// Split 三路 fixture 走完整 JSON spec reifier、Relay/LLVM 编译和 RuntimeSession，
// 用独立切片参考检查输出顺序与多输出 ABI。fixture 不依赖外部 ONNX/Python 运行时。
bool TestRunSplitMultiOutputProtobufLLVM() {
    const std::string json_path = "/tmp/kxc_split_multi_output.import.json";
    const std::string params_path = "/tmp/kxc_split_multi_output.params.bin";
    const char* json = R"json({
  "format": "kxc.onnx_import.v1",
  "function": {
    "inputs": [{"name":"data","shape":[2,6],"dtype":"float32"}],
    "outputs": [
      {"name":"left","shape":[2,1],"dtype":"float32"},
      {"name":"middle","shape":[2,3],"dtype":"float32"},
      {"name":"right","shape":[2,2],"dtype":"float32"}
    ],
    "nodes": [{
      "name":"split",
      "op_name":"split",
      "inputs":["data"],
      "outputs":["left","middle","right"],
      "attrs":{"axis":1,"sections":[1,3,2]}
    }]
  },
  "params": [],
  "param_order": []
})json";
    {
        std::ofstream output(json_path, std::ios::trunc);
        TEST_CHECK(output.good(), "split fixture JSON should be writable");
        output << json;
    }
    {
        std::ofstream output(params_path, std::ios::binary | std::ios::trunc);
        TEST_CHECK(output.good(), "split fixture params should be writable");
    }

    const auto cleanup = [&] {
        std::remove(json_path.c_str());
        std::remove(params_path.c_str());
    };
    try {
        kxc::frontend::ImportedONNXModel imported =
            kxc::frontend::LoadONNXImportSpec(json_path, params_path);
        TEST_CHECK(imported.function.defined() && imported.function->params.size() == 1 &&
                       imported.output_names.size() == 3 &&
                       imported.output_names[0] == "left" &&
                       imported.output_names[1] == "middle" &&
                       imported.output_names[2] == "right",
                   "Split fixture must preserve three output bindings and order");
        const auto* output_tuple = imported.function->body.checked_type().As<kxc::TupleTypeNode>();
        TEST_CHECK(output_tuple && output_tuple->fields.size() == 3 &&
                       CheckTensor(output_tuple->fields[0], {2, 1}, "float32") &&
                       CheckTensor(output_tuple->fields[1], {2, 3}, "float32") &&
                       CheckTensor(output_tuple->fields[2], {2, 2}, "float32"),
                   "Split fixture output tuple type mismatch");
#if KXC_USE_LLVM
        const kxc::Function prepared = PrepareJitRelayFunction(imported.function);
        const auto compiled = kxc::api::Compiler::Compile(
            prepared, kxc::api::CompileConfig::Create(
                          kxc::BuildTarget(kxc::Device::CPU()), 1));
        TEST_CHECK(compiled.module().IsReady() && compiled.plan().calls().size() == 1,
                   "Split fixture must compile to one LLVM multi-output unit");
        kxc::runtime::NDArray input = kxc::runtime::NDArray::Empty(
            {2, 6}, kxc::runtime::DataTypeFromString("float32"), kxc::Device::CPU());
        const std::vector<float> values{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        input.CopyFromBytes(values.data(), input.NBytes());
        kxc::runtime::RuntimeSession session(compiled.module(), compiled.plan());
        const kxc::Array<kxc::runtime::NDArray> outputs = session.Run({input});
        TEST_CHECK(outputs.size() == 3 && ShapeEquals(outputs[0], {2, 1}) &&
                       ShapeEquals(outputs[1], {2, 3}) &&
                       ShapeEquals(outputs[2], {2, 2}),
                   "Split RuntimeSession output shapes must preserve order");
        const std::vector<float> expected_left{0, 6};
        const std::vector<float> expected_middle{1, 2, 3, 7, 8, 9};
        const std::vector<float> expected_right{4, 5, 10, 11};
        std::vector<float> actual_left(expected_left.size());
        std::vector<float> actual_middle(expected_middle.size());
        std::vector<float> actual_right(expected_right.size());
        outputs[0].CopyToBytes(actual_left.data(), outputs[0].NBytes());
        outputs[1].CopyToBytes(actual_middle.data(), outputs[1].NBytes());
        outputs[2].CopyToBytes(actual_right.data(), outputs[2].NBytes());
        TEST_CHECK(actual_left == expected_left && actual_middle == expected_middle &&
                       actual_right == expected_right,
                   "Split RuntimeSession values must match independent slicing reference");
#endif
    } catch (...) {
        cleanup();
        throw;
    }
    cleanup();
    return true;
}

// 验证导入的 ResNet18 可完成 Relay 到 LLVM 编译。
bool TestCompileResNet18ToLLVM() {
#if KXC_USE_LLVM
    kxc::frontend::ImportedONNXModel imported = kxc::frontend::LoadONNXImportSpec(
        KXC_ONNX_IMPORT_JSON_PATH, KXC_ONNX_IMPORT_PARAMS_PATH);

    kxc::Function prepared = PrepareJitRelayFunction(imported.function);
    const auto config = kxc::api::CompileConfig::Create(
        kxc::BuildTarget(kxc::Device::CPU()), ResNet18OptLevel());
    auto compiled = kxc::api::Compiler::Compile(prepared, config);
    TEST_CHECK(compiled.module().IsReady(),
               "ResNet18 should compile to a ready LLVM module");
    TEST_CHECK(compiled.module().entry_count() == compiled.plan().calls().size() &&
                   compiled.module().entry_count() > 1,
               "ResNet18 should compile to one entry per operator call");
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

    const auto config = kxc::api::CompileConfig::Create(
        kxc::BuildTarget(kxc::Device::CPU()), ResNet18OptLevel());
    auto compiled = kxc::api::Compiler::Compile(prepared, config);
    TEST_CHECK(compiled.module().IsReady(), "ResNet18 should compile before execution");
    const kxc::Map<kxc::String, kxc::runtime::NDArray> constants =
        compiled.module().constants();
    TEST_CHECK(constants.size() == imported.params.size(),
               "compiled module constant count should match loaded ONNX initializers");

    // 输入、常量和输出全部作为 NDArray 句柄进入统一 Launch 契约。
    kxc::runtime::NDArray input = kxc::runtime::NDArray::Empty(
        {1, 3, 224, 224}, kxc::runtime::DataTypeFromString("float32"),
        kxc::Device::CPU());
    FillResNet18Input(input);

    std::cout << "[INFO] running compiled resnet18 LLVM kernel with "
              << compiled.plan().calls().size() << " kernel calls\n";
    std::cout.flush();

    const auto start = std::chrono::steady_clock::now();
    kxc::runtime::RuntimeSession session(compiled.module(), compiled.plan());
    const kxc::Array<kxc::runtime::NDArray> outputs = session.Run({input});
    const auto end = std::chrono::steady_clock::now();
    TEST_CHECK(outputs.size() == 1,
               "ResNet18 graph plan should return one runtime output");
    const kxc::runtime::NDArray output = outputs[0];

    std::vector<float> out(1000);
    output.CopyToBytes(out.data(), output.NBytes());
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
        if (!TestRunExactTransformerProtobufLLVM()) {
            return 1;
        }
        if (!TestRunStaticS1ProtobufLLVM()) {
            return 1;
        }
        if (!TestRunEqualWhereProtobufLLVM()) {
            return 1;
        }
        if (!TestRunM4M5OpsProtobufLLVM()) {
            return 1;
        }
        if (!TestRunSplitMultiOutputProtobufLLVM()) {
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
#if KXC_USE_LLVM
    std::cout << "[PASS] onnx_transformer_protobuf_reifier_llvm_runtime\n";
    std::cout << "[PASS] onnx_static_s1_protobuf_reifier_llvm_runtime\n";
    std::cout << "[PASS] onnx_equal_where_protobuf_reifier_llvm_runtime\n";
    std::cout << "[PASS] onnx_m4m5_ops_protobuf_reifier_llvm_runtime\n";
    std::cout << "[PASS] onnx_split_multi_output_protobuf_reifier_llvm_runtime\n";
    std::cout << "[PASS] onnx_importer_compile_resnet18_llvm\n";
    if (ShouldRunResNet18Kernel()) {
        std::cout << "[PASS] onnx_importer_run_resnet18_llvm\n";
    }
#else
    std::cout << "[SKIP] onnx_transformer_protobuf_reifier_llvm_runtime: KXC_USE_LLVM=0\n";
    std::cout << "[SKIP] onnx_static_s1_protobuf_reifier_llvm_runtime: KXC_USE_LLVM=0\n";
    std::cout << "[SKIP] onnx_equal_where_protobuf_reifier_llvm_runtime: KXC_USE_LLVM=0\n";
    std::cout << "[SKIP] onnx_m4m5_ops_protobuf_reifier_llvm_runtime: KXC_USE_LLVM=0\n";
    std::cout << "[SKIP] onnx_split_multi_output_protobuf_reifier_llvm_runtime: KXC_USE_LLVM=0\n";
#endif
    std::cout << "All available ONNX importer tests passed.\n";
    return 0;
}
