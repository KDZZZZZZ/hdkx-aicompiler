#include "api/compiler.h"
#include "frontend/onnx_importer.h"
#include "relay/transforms/infer_type.h"
#include "relay/transforms/pipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
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

bool ShapeEquals(const kxc::runtime::NDArray& array, const std::vector<int64_t>& shape) {
    if (!array.defined() || array->shape.size() != shape.size()) {
        return false;
    }
    for (size_t i = 0; i < shape.size(); ++i) {
        if (array->shape[i] != shape[i]) {
            return false;
        }
    }
    return true;
}

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

bool ShouldRunResNet18Kernel() {
    const char* value = std::getenv("KXC_RUN_RESNET18_EXEC");
    return value != nullptr && std::string(value) == "1";
}

int ResNet18OptLevel() {
    const char* value = std::getenv("KXC_RESNET18_OPT_LEVEL");
    if (!value) {
        return 1;
    }
    int opt_level = std::atoi(value);
    return std::clamp(opt_level, 0, 3);
}

void CollectConstantsInLoweringOrder(const kxc::Expr& expr,
                                     std::vector<kxc::runtime::NDArray>* constants,
                                     std::unordered_set<const kxc::Object*>* visited) {
    if (!expr.defined() || visited->count(expr.get()) != 0) {
        return;
    }
    visited->insert(expr.get());

    if (const auto* op = expr.As<kxc::ConstantNode>()) {
        constants->push_back(op->data);
        return;
    }
    if (const auto* op = expr.As<kxc::CallNode>()) {
        for (const auto& arg : op->args) {
            CollectConstantsInLoweringOrder(arg, constants, visited);
        }
        return;
    }
    if (const auto* op = expr.As<kxc::FunctionNode>()) {
        CollectConstantsInLoweringOrder(op->body, constants, visited);
        return;
    }
    if (const auto* op = expr.As<kxc::TupleNode>()) {
        for (const auto& field : op->fields) {
            CollectConstantsInLoweringOrder(field, constants, visited);
        }
        return;
    }
    if (const auto* op = expr.As<kxc::TupleGetItemNode>()) {
        CollectConstantsInLoweringOrder(op->tuple, constants, visited);
        return;
    }
    if (const auto* op = expr.As<kxc::LetNode>()) {
        CollectConstantsInLoweringOrder(op->value, constants, visited);
        CollectConstantsInLoweringOrder(op->body, constants, visited);
        return;
    }
    if (const auto* op = expr.As<kxc::IfNode>()) {
        CollectConstantsInLoweringOrder(op->cond, constants, visited);
        CollectConstantsInLoweringOrder(op->true_branch, constants, visited);
        CollectConstantsInLoweringOrder(op->false_branch, constants, visited);
        return;
    }
}

std::vector<kxc::runtime::NDArray> ConstantsInLoweringOrder(const kxc::Function& func) {
    std::vector<kxc::runtime::NDArray> constants;
    std::unordered_set<const kxc::Object*> visited;
    CollectConstantsInLoweringOrder(func->body, &constants, &visited);
    return constants;
}

kxc::Function PrepareJitRelayFunction(kxc::Function func) {
    func = kxc::relay::InferTypePass(func);
    func = kxc::relay::RunRelayPassPipeline(
        func, {kxc::String("fold_constant"), kxc::String("simplify_expr")});
    return kxc::relay::InferTypePass(func);
}

void FillResNet18Input(const kxc::runtime::NDArray& input) {
    float* data = static_cast<float*>(input->dl_tensor.data);
    const size_t elements = input.NBytes() / sizeof(float);
    for (size_t i = 0; i < elements; ++i) {
        data[i] = (static_cast<float>(i % 251) - 125.0f) / 125.0f;
    }
}

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

bool TestCompileResNet18ToLLVM() {
#if KXC_USE_LLVM
    kxc::frontend::ImportedONNXModel imported = kxc::frontend::LoadONNXImportSpec(
        KXC_ONNX_IMPORT_JSON_PATH, KXC_ONNX_IMPORT_PARAMS_PATH);

    kxc::Function prepared = PrepareJitRelayFunction(imported.function);
    auto config = kxc::api::CompileConfig::JIT(kxc::BuildTarget(kxc::kCPU));
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

bool TestRunCompiledResNet18LLVM() {
#if KXC_USE_LLVM
    if (!ShouldRunResNet18Kernel()) {
        std::cout << "[SKIP] resnet18 LLVM execute: set KXC_RUN_RESNET18_EXEC=1 to run\n";
        return true;
    }

    kxc::frontend::ImportedONNXModel imported = kxc::frontend::LoadONNXImportSpec(
        KXC_ONNX_IMPORT_JSON_PATH, KXC_ONNX_IMPORT_PARAMS_PATH);
    kxc::Function prepared = PrepareJitRelayFunction(imported.function);
    std::vector<kxc::runtime::NDArray> constants = ConstantsInLoweringOrder(prepared);
    TEST_CHECK(constants.size() == imported.params.size(),
               "collected constant count should match loaded ONNX initializers");

    auto config = kxc::api::CompileConfig::JIT(kxc::BuildTarget(kxc::kCPU));
    config->opt_level = ResNet18OptLevel();
    auto module = kxc::api::Compiler::Compile(prepared, config);
    TEST_CHECK(module.IsReady(), "ResNet18 should compile before execution");

    kxc::runtime::NDArray input({1, 3, 224, 224}, "float32");
    kxc::runtime::NDArray output({1, 1000}, "float32");
    FillResNet18Input(input);

    std::vector<void*> packed_args;
    packed_args.reserve(1 + constants.size() + 1);
    packed_args.push_back(input->dl_tensor.data);
    for (const auto& constant : constants) {
        packed_args.push_back(constant->dl_tensor.data);
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
