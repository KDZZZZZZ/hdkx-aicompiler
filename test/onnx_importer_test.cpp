#include "frontend/onnx_importer.h"
#include "relay/transforms/infer_type.h"

#include <cmath>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#ifndef KXC_ONNX_IMPORT_JSON_PATH
#define KXC_ONNX_IMPORT_JSON_PATH "resnet18.import.json"
#endif

#ifndef KXC_ONNX_IMPORT_PARAMS_PATH
#define KXC_ONNX_IMPORT_PARAMS_PATH "resnet18.params.bin"
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

}  // namespace

int main() {
    try {
        if (!TestLoadResNet18ImportSpec()) {
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] unexpected exception: " << e.what() << "\n";
        return 1;
    }

    std::cout << "[PASS] onnx_importer_load_resnet18\n";
    std::cout << "All ONNX importer tests passed.\n";
    return 0;
}
