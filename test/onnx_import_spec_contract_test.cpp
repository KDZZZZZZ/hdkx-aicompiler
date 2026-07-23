/*! \file test/onnx_import_spec_contract_test.cpp
 * \brief Verifies the dependency-free C++ ONNX import-spec reifier contract.
 */

#include "kxc/frontend/onnx_importer.h"

#include <chrono>
#include <exception>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

#define TEST_CHECK(cond, msg)                                                    \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::cerr << "[FAIL] " << __FUNCTION__ << ": " << (msg) << "\n"; \
            return false;                                                        \
        }                                                                        \
    } while (0)

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path() /
                ("kxc_onnx_import_spec_contract_" + std::to_string(stamp));
        std::filesystem::create_directories(path_);
    }

    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

bool ShapeEquals(const kxc::TensorTypeNode* type, const std::vector<int64_t>& shape,
                 const std::string& dtype) {
    if (!type || type->dtype != dtype || type->shape.size() != shape.size()) {
        return false;
    }
    for (size_t axis = 0; axis < shape.size(); ++axis) {
        if (type->shape[axis] != shape[axis]) return false;
    }
    return true;
}

bool Throws(const std::function<void()>& action) {
    try {
        action();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

void WriteFixture(const TemporaryDirectory& directory, const std::string& input_shape,
                  const std::string& output_shape,
                  const std::string& output_dtype = "float32") {
    const std::string json = R"json({
  "format": "kxc.onnx_import.v1",
  "function": {
    "inputs": [
      {"name": "a", "shape": )json" + input_shape + R"json(, "dtype": "float32"},
      {"name": "b", "shape": [1, 3, 4], "dtype": "float32"}
    ],
    "outputs": [
      {"name": "out", "shape": )json" + output_shape + R"json(, "dtype": ")json" +
      output_dtype + R"json("}
    ],
    "nodes": [
      {"name": "matmul", "op_name": "matmul", "inputs": ["a", "b"], "outputs": ["scores"], "attrs": {}},
      {"name": "softmax", "op_name": "softmax", "inputs": ["scores"], "outputs": ["weights"], "attrs": {"axis": -1}},
      {"name": "transpose", "op_name": "transpose", "inputs": ["weights"], "outputs": ["out"], "attrs": {"perm": [0, 2, 1]}}
    ]
  },
  "params": [],
  "param_order": []
})json";
    std::ofstream(directory.path() / "model.json") << json;
    std::ofstream(directory.path() / "params.bin", std::ios::binary);
}

bool TestValidStaticMatMulSoftmaxTranspose() {
    TemporaryDirectory directory;
    WriteFixture(directory, "[2, 2, 3]", "[2, 4, 2]");

    const auto imported = kxc::frontend::LoadONNXImportSpec(
        (directory.path() / "model.json").string(),
        (directory.path() / "params.bin").string());
    TEST_CHECK(imported.function.defined(), "valid static ONNX import spec should reify");
    TEST_CHECK(ShapeEquals(imported.function->body.checked_type().As<kxc::TensorTypeNode>(),
                           {2, 4, 2}, "float32"),
               "reified output should preserve broadcast shape and inferred dtype");
    return true;
}

bool TestNegativeInputDimensionIsRejected() {
    TemporaryDirectory directory;
    WriteFixture(directory, "[-1, 2, 3]", "[1, 4, 2]");

    TEST_CHECK(Throws([&] {
                   kxc::frontend::LoadONNXImportSpec(
                       (directory.path() / "model.json").string(),
                       (directory.path() / "params.bin").string());
               }),
               "negative input dimension must fail closed");
    return true;
}

bool TestDeclaredOutputShapeMismatchIsRejected() {
    TemporaryDirectory directory;
    WriteFixture(directory, "[1, 2, 3]", "[1, 2, 4]");

    TEST_CHECK(Throws([&] {
                   kxc::frontend::LoadONNXImportSpec(
                       (directory.path() / "model.json").string(),
                       (directory.path() / "params.bin").string());
               }),
               "declared output shape must match the inferred output");
    return true;
}

bool TestDeclaredOutputDTypeMismatchIsRejected() {
    TemporaryDirectory directory;
    WriteFixture(directory, "[1, 2, 3]", "[1, 4, 2]", "float64");

    TEST_CHECK(Throws([&] {
                   kxc::frontend::LoadONNXImportSpec(
                       (directory.path() / "model.json").string(),
                       (directory.path() / "params.bin").string());
               }),
               "declared output dtype must match the inferred output");
    return true;
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, bool (*)()>> tests = {
        {"valid_static_matmul_softmax_transpose", TestValidStaticMatMulSoftmaxTranspose},
        {"negative_input_dimension", TestNegativeInputDimensionIsRejected},
        {"declared_output_shape_mismatch", TestDeclaredOutputShapeMismatchIsRejected},
        {"declared_output_dtype_mismatch", TestDeclaredOutputDTypeMismatchIsRejected},
    };

    for (const auto& test : tests) {
        try {
            if (!test.second()) return 1;
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << test.first << ": unexpected exception: "
                      << error.what() << "\n";
            return 1;
        }
        std::cout << "[PASS] " << test.first << "\n";
    }
    return 0;
}
