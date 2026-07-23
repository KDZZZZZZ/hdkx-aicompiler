/*! \file test/onnx_import_spec_contract_test.cpp
 * \brief Verifies the dependency-free C++ ONNX import-spec reifier contract.
 */

#include "kxc/frontend/onnx_importer.h"
#include "kxc/compiler/lowering/relay_to_tir.h"

#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iostream>
#include <limits>
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

void WriteGatherFixture(const TemporaryDirectory& directory,
                        const std::string& output_shape,
                        const std::string& attrs = R"json({"axis": 1})json",
                        const std::vector<int64_t>& indices = {0, -3},
                        bool constant_indices = true) {
    const std::string index_input = constant_indices
        ? ""
        : R"json(,
      {"name": "indices", "shape": [2], "dtype": "int64"})json";
    const std::string params = constant_indices
        ? R"json([{"name": "indices", "shape": [2], "dtype": "int64", "offset": 0, "nbytes": 16}])json"
        : "[]";
    const std::string param_order = constant_indices ? R"json(["indices"])json" : "[]";
    const std::string json = R"json({
  "format": "kxc.onnx_import.v1",
  "function": {
    "inputs": [
      {"name": "data", "shape": [2, 3, 4], "dtype": "float32"})json" + index_input + R"json(
    ],
    "outputs": [
      {"name": "out", "shape": )json" + output_shape + R"json(, "dtype": "float32"}
    ],
    "nodes": [
      {"name": "gather", "op_name": "gather", "inputs": ["data", "indices"], "outputs": ["out"], "attrs": )json" + attrs + R"json(}
    ]
  },
  "params": )json" + params + R"json(,
  "param_order": )json" + param_order + R"json(
})json";
    std::ofstream(directory.path() / "model.json") << json;
    std::ofstream binary(directory.path() / "params.bin", std::ios::binary);
    if (constant_indices) {
        binary.write(reinterpret_cast<const char*>(indices.data()),
                     static_cast<std::streamsize>(indices.size() * sizeof(int64_t)));
    }
}

void WriteConcatenateFixture(const TemporaryDirectory& directory,
                             const std::string& output_shape,
                             const std::string& attrs = R"json({"axis": -1})json") {
    const std::string json = R"json({
  "format": "kxc.onnx_import.v1",
  "function": {
    "inputs": [
      {"name": "lhs", "shape": [2, 2], "dtype": "float32"},
      {"name": "rhs", "shape": [2, 3], "dtype": "float32"}
    ],
    "outputs": [
      {"name": "out", "shape": )json" + output_shape + R"json(, "dtype": "float32"}
    ],
    "nodes": [
      {"name": "concatenate", "op_name": "concatenate", "inputs": ["lhs", "rhs"], "outputs": ["out"], "attrs": )json" + attrs + R"json(}
    ]
  },
  "params": [],
  "param_order": []
})json";
    std::ofstream(directory.path() / "model.json") << json;
    std::ofstream(directory.path() / "params.bin", std::ios::binary);
}

void WriteSliceFixture(const TemporaryDirectory& directory, const std::string& output_shape,
                       const std::string& attrs =
                           R"json({"starts": [-2], "ends": [99], "axes": [-1], "steps": [1]})json") {
    const std::string json = R"json({
  "format": "kxc.onnx_import.v1",
  "function": {
    "inputs": [{"name": "data", "shape": [2, 3], "dtype": "float32"}],
    "outputs": [{"name": "out", "shape": )json" + output_shape + R"json(, "dtype": "float32"}],
    "nodes": [{"name": "slice", "op_name": "slice", "inputs": ["data"], "outputs": ["out"], "attrs": )json" + attrs + R"json(}]
  },
  "params": [], "param_order": []
})json";
    std::ofstream(directory.path() / "model.json") << json;
    std::ofstream(directory.path() / "params.bin", std::ios::binary);
}

void WriteWhereFixture(const TemporaryDirectory& directory,
                       const std::string& output_shape,
                       const std::string& branch_dtype = "float32",
                       const std::string& attrs = R"json({})json") {
    const std::string json = R"json({
  "format": "kxc.onnx_import.v1",
  "function": {
    "inputs": [
      {"name": "condition", "shape": [2, 1], "dtype": "bool"},
      {"name": "x", "shape": [], "dtype": ")json" + branch_dtype + R"json("},
      {"name": "y", "shape": [1, 3], "dtype": ")json" + branch_dtype + R"json("}
    ],
    "outputs": [
      {"name": "out", "shape": )json" + output_shape + R"json(, "dtype": ")json" + branch_dtype + R"json("}
    ],
    "nodes": [
      {"name": "where", "op_name": "where", "inputs": ["condition", "x", "y"], "outputs": ["out"], "attrs": )json" + attrs + R"json(}
    ]
  },
  "params": [],
  "param_order": []
})json";
    std::ofstream(directory.path() / "model.json") << json;
    std::ofstream(directory.path() / "params.bin", std::ios::binary);
}

void WriteLayerNormFixture(const TemporaryDirectory& directory,
                           const std::string& output_shape,
                           const std::string& attrs =
                               R"json({"axis": 1, "epsilon": 0.00001, "accumulation_dtype": "float64"})json") {
    const std::string json = R"json({
  "format": "kxc.onnx_import.v1",
  "function": {
    "inputs": [
      {"name": "data", "shape": [2, 3, 4], "dtype": "float32"},
      {"name": "scale", "shape": [3, 4], "dtype": "float32"},
      {"name": "bias", "shape": [3, 4], "dtype": "float32"}
    ],
    "outputs": [
      {"name": "out", "shape": )json" + output_shape + R"json(, "dtype": "float32"}
    ],
    "nodes": [
      {"name": "layer_norm", "op_name": "nn_layer_norm", "inputs": ["data", "scale", "bias"], "outputs": ["out"], "attrs": )json" + attrs + R"json(}
    ]
  },
  "params": [],
  "param_order": []
})json";
    std::ofstream(directory.path() / "model.json") << json;
    std::ofstream(directory.path() / "params.bin", std::ios::binary);
}

void WriteExactTransformerOperatorSliceFixture(const TemporaryDirectory& directory) {
    const std::string json = R"json({
  "format": "kxc.onnx_import.v1",
  "function": {
    "inputs": [
      {"name": "embedding_table", "shape": [4, 2], "dtype": "float32"}
    ],
    "outputs": [
      {"name": "context", "shape": [3, 2], "dtype": "float32"}
    ],
    "nodes": [
      {"name": "embedding", "op_name": "gather", "inputs": ["embedding_table", "token_ids"], "outputs": ["embedded"], "attrs": {"axis": 0}},
      {"name": "norm", "op_name": "nn_layer_norm", "inputs": ["embedded", "scale", "bias"], "outputs": ["normalized"], "attrs": {"axis": -1, "epsilon": 0.00001, "accumulation_dtype": "float64"}},
      {"name": "select", "op_name": "where", "inputs": ["condition", "normalized", "fallback"], "outputs": ["selected"], "attrs": {}},
      {"name": "prefix", "op_name": "slice", "inputs": ["selected"], "outputs": ["prefix_value"], "attrs": {"starts": [0], "ends": [1], "axes": [0], "steps": [1]}},
      {"name": "sequence", "op_name": "concatenate", "inputs": ["prefix_value", "selected"], "outputs": ["sequence_value"], "attrs": {"axis": 0}},
      {"name": "keys", "op_name": "transpose", "inputs": ["sequence_value"], "outputs": ["keys_value"], "attrs": {"perm": [1, 0]}},
      {"name": "scores", "op_name": "matmul", "inputs": ["sequence_value", "keys_value"], "outputs": ["scores_value"], "attrs": {}},
      {"name": "weights", "op_name": "softmax", "inputs": ["scores_value"], "outputs": ["weights_value"], "attrs": {"axis": -1}},
      {"name": "context_node", "op_name": "matmul", "inputs": ["weights_value", "sequence_value"], "outputs": ["context"], "attrs": {}}
    ]
  },
  "params": [
    {"name": "token_ids", "shape": [2], "dtype": "int64", "offset": 0, "nbytes": 16},
    {"name": "condition", "shape": [2, 1], "dtype": "bool", "offset": 16, "nbytes": 2},
    {"name": "fallback", "shape": [1, 2], "dtype": "float32", "offset": 18, "nbytes": 8},
    {"name": "scale", "shape": [2], "dtype": "float32", "offset": 26, "nbytes": 8},
    {"name": "bias", "shape": [2], "dtype": "float32", "offset": 34, "nbytes": 8}
  ],
  "param_order": ["token_ids", "condition", "fallback", "scale", "bias"]
})json";
    std::ofstream(directory.path() / "model.json") << json;
    std::ofstream binary(directory.path() / "params.bin", std::ios::binary);
    const auto write = [&binary](const auto& values) {
        binary.write(reinterpret_cast<const char*>(values.data()),
                     static_cast<std::streamsize>(values.size() * sizeof(values[0])));
    };
    write(std::vector<int64_t>{0, 2});
    write(std::vector<uint8_t>{1, 0});
    write(std::vector<float>{0.0f, 0.0f});
    write(std::vector<float>{1.0f, 1.0f});
    write(std::vector<float>{0.0f, 0.0f});
}

bool TestExactTransformerOperatorSliceReifier() {
    TemporaryDirectory directory;
    WriteExactTransformerOperatorSliceFixture(directory);

    const auto imported = kxc::frontend::LoadONNXImportSpec(
        (directory.path() / "model.json").string(),
        (directory.path() / "params.bin").string());
    TEST_CHECK(imported.function.defined() && imported.params.size() == 5,
               "exact Transformer operator slice must reify initializer-backed values");
    TEST_CHECK(ShapeEquals(imported.function->body.checked_type().As<kxc::TensorTypeNode>(),
                           {3, 2}, "float32"),
               "exact Transformer operator slice must preserve its declared output contract");
    TEST_CHECK(kxc::relay::LowerToTIR(imported.function)->prim_func.defined(),
               "reified bool Constant and exact Transformer operator slice must lower to TIR");
    return true;
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

bool TestValidStaticGather() {
    TemporaryDirectory directory;
    WriteGatherFixture(directory, "[2, 2, 4]");

    const auto imported = kxc::frontend::LoadONNXImportSpec(
        (directory.path() / "model.json").string(),
        (directory.path() / "params.bin").string());
    TEST_CHECK(imported.function.defined(), "valid static Gather import spec should reify");
    TEST_CHECK(ShapeEquals(imported.function->body.checked_type().As<kxc::TensorTypeNode>(),
                           {2, 2, 4}, "float32"),
               "reified Gather output should insert indices shape at axis");
    return true;
}

bool TestGatherAttrsAreStrict() {
    TemporaryDirectory directory;
    WriteGatherFixture(directory, "[2, 2, 4]",
                       R"json({"axis": 1, "unknown": 0})json");

    TEST_CHECK(Throws([&] {
                   kxc::frontend::LoadONNXImportSpec(
                       (directory.path() / "model.json").string(),
                       (directory.path() / "params.bin").string());
               }),
               "Gather reifier must reject attrs outside the exact canonical set");
    return true;
}

bool TestGatherDeclaredOutputMismatchIsRejected() {
    TemporaryDirectory directory;
    WriteGatherFixture(directory, "[2, 1, 4]");

    TEST_CHECK(Throws([&] {
                   kxc::frontend::LoadONNXImportSpec(
                       (directory.path() / "model.json").string(),
                       (directory.path() / "params.bin").string());
               }),
               "declared Gather output shape must match inferred output");
    return true;
}

bool TestGatherDynamicIndicesAreRejected() {
    TemporaryDirectory directory;
    WriteGatherFixture(directory, "[2, 2, 4]", R"json({"axis": 1})json",
                       {0, -3}, false);
    TEST_CHECK(Throws([&] {
                   kxc::frontend::LoadONNXImportSpec(
                       (directory.path() / "model.json").string(),
                       (directory.path() / "params.bin").string());
               }),
               "Gather reifier must reject dynamic indices even in hand-written specs");
    return true;
}

bool TestGatherOutOfDomainConstantIndicesAreRejected() {
    for (const std::vector<int64_t>& indices :
         {std::vector<int64_t>{3, 0},
          std::vector<int64_t>{std::numeric_limits<int64_t>::min(), 0}}) {
        TemporaryDirectory directory;
        WriteGatherFixture(directory, "[2, 2, 4]", R"json({"axis": 1})json",
                           indices);
        TEST_CHECK(Throws([&] {
                       kxc::frontend::LoadONNXImportSpec(
                           (directory.path() / "model.json").string(),
                           (directory.path() / "params.bin").string());
                   }),
                   "Gather reifier must validate every constant index against ONNX bounds");
    }
    return true;
}

bool TestValidStaticConcatenate() {
    TemporaryDirectory directory;
    WriteConcatenateFixture(directory, "[2, 5]");

    const auto imported = kxc::frontend::LoadONNXImportSpec(
        (directory.path() / "model.json").string(),
        (directory.path() / "params.bin").string());
    TEST_CHECK(imported.function.defined(), "valid static Concatenate import spec should reify");
    TEST_CHECK(ShapeEquals(imported.function->body.checked_type().As<kxc::TensorTypeNode>(),
                           {2, 5}, "float32"),
               "reified Concatenate output should normalize axis and sum extents");
    return true;
}

bool TestConcatenateDeclaredOutputMismatchIsRejected() {
    TemporaryDirectory directory;
    WriteConcatenateFixture(directory, "[2, 4]");
    TEST_CHECK(Throws([&] {
                   kxc::frontend::LoadONNXImportSpec(
                       (directory.path() / "model.json").string(),
                       (directory.path() / "params.bin").string());
               }), "declared Concatenate output shape must match inferred output");
    return true;
}

bool TestConcatenateAttrsAreStrict() {
    TemporaryDirectory directory;
    WriteConcatenateFixture(directory, "[2, 5]", R"json({})json");
    TEST_CHECK(Throws([&] {
                   kxc::frontend::LoadONNXImportSpec(
                       (directory.path() / "model.json").string(),
                       (directory.path() / "params.bin").string());
               }), "Concatenate reifier must construct and require strong axis attrs");
    return true;
}

bool TestValidStaticSlice() {
    TemporaryDirectory directory;
    WriteSliceFixture(directory, "[2, 2]");
    const auto imported = kxc::frontend::LoadONNXImportSpec(
        (directory.path() / "model.json").string(),
        (directory.path() / "params.bin").string());
    TEST_CHECK(imported.function.defined() &&
                   ShapeEquals(imported.function->body.checked_type().As<kxc::TensorTypeNode>(),
                               {2, 2}, "float32"),
               "valid static Slice import spec should reify canonical strong attrs");
    return true;
}

bool TestSliceDeclaredOutputMismatchIsRejected() {
    TemporaryDirectory directory;
    WriteSliceFixture(directory, "[2, 3]");
    TEST_CHECK(Throws([&] { kxc::frontend::LoadONNXImportSpec(
                   (directory.path() / "model.json").string(),
                   (directory.path() / "params.bin").string()); }),
               "declared Slice output shape must match inferred output");
    return true;
}

bool TestSliceAttrsAreStrict() {
    TemporaryDirectory directory;
    WriteSliceFixture(directory, "[2, 2]", R"json({"starts": [0]})json");
    TEST_CHECK(Throws([&] { kxc::frontend::LoadONNXImportSpec(
                   (directory.path() / "model.json").string(),
                   (directory.path() / "params.bin").string()); }),
               "Slice reifier must require all canonical strong attrs");
    return true;
}

bool TestValidStaticWhere() {
    TemporaryDirectory directory;
    WriteWhereFixture(directory, "[2, 3]");

    const auto imported = kxc::frontend::LoadONNXImportSpec(
        (directory.path() / "model.json").string(),
        (directory.path() / "params.bin").string());
    TEST_CHECK(imported.function.defined(), "valid static Where import spec should reify");
    TEST_CHECK(ShapeEquals(imported.function->body.checked_type().As<kxc::TensorTypeNode>(),
                           {2, 3}, "float32"),
               "reified Where output should use joint broadcast shape and x/y dtype");
    return true;
}

bool TestWhereAttrsAreStrict() {
    TemporaryDirectory directory;
    WriteWhereFixture(directory, "[2, 3]", "float32", R"json({"axis": 0})json");

    TEST_CHECK(Throws([&] {
                   kxc::frontend::LoadONNXImportSpec(
                       (directory.path() / "model.json").string(),
                       (directory.path() / "params.bin").string());
               }),
               "Where reifier must reject all noncanonical attrs");
    return true;
}

bool TestWhereDeclaredOutputMismatchIsRejected() {
    TemporaryDirectory directory;
    WriteWhereFixture(directory, "[2, 2]");

    TEST_CHECK(Throws([&] {
                   kxc::frontend::LoadONNXImportSpec(
                       (directory.path() / "model.json").string(),
                       (directory.path() / "params.bin").string());
               }),
               "declared Where output shape must match inferred output");
    return true;
}

bool TestWhereUnsupportedBranchDTypeIsRejected() {
    TemporaryDirectory directory;
    WriteWhereFixture(directory, "[2, 3]", "float16");

    TEST_CHECK(Throws([&] {
                   kxc::frontend::LoadONNXImportSpec(
                       (directory.path() / "model.json").string(),
                       (directory.path() / "params.bin").string());
               }),
               "Where branch dtype outside the supported ABI set must fail during reification");
    return true;
}

bool TestValidStaticLayerNorm() {
    TemporaryDirectory directory;
    WriteLayerNormFixture(directory, "[2, 3, 4]");

    const auto imported = kxc::frontend::LoadONNXImportSpec(
        (directory.path() / "model.json").string(),
        (directory.path() / "params.bin").string());
    TEST_CHECK(imported.function.defined(), "valid static LayerNorm import spec should reify");
    TEST_CHECK(ShapeEquals(imported.function->body.checked_type().As<kxc::TensorTypeNode>(),
                           {2, 3, 4}, "float32"),
               "reified LayerNorm output should preserve data shape and dtype");
    return true;
}

bool TestLayerNormDeclaredOutputMismatchIsRejected() {
    TemporaryDirectory directory;
    WriteLayerNormFixture(directory, "[2, 3, 5]");

    TEST_CHECK(Throws([&] {
                   kxc::frontend::LoadONNXImportSpec(
                       (directory.path() / "model.json").string(),
                       (directory.path() / "params.bin").string());
               }),
               "declared LayerNorm output shape must match inferred output");
    return true;
}

bool TestLayerNormUnsupportedAttrsAreRejected() {
    TemporaryDirectory directory;
    WriteLayerNormFixture(
        directory, "[2, 3, 4]",
        R"json({"axis": 1, "epsilon": 0.00001, "accumulation_dtype": "float64", "stash_type": 1})json");

    TEST_CHECK(Throws([&] {
                   kxc::frontend::LoadONNXImportSpec(
                       (directory.path() / "model.json").string(),
                       (directory.path() / "params.bin").string());
               }),
               "LayerNorm import spec must reject attrs outside the exact canonical set");
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
        {"exact_transformer_operator_slice_reifier", TestExactTransformerOperatorSliceReifier},
        {"valid_static_matmul_softmax_transpose", TestValidStaticMatMulSoftmaxTranspose},
        {"valid_static_gather", TestValidStaticGather},
        {"gather_strict_attrs", TestGatherAttrsAreStrict},
        {"gather_declared_output_mismatch", TestGatherDeclaredOutputMismatchIsRejected},
        {"gather_dynamic_indices_rejected", TestGatherDynamicIndicesAreRejected},
        {"gather_oob_constant_indices_rejected",
         TestGatherOutOfDomainConstantIndicesAreRejected},
        {"valid_static_concatenate", TestValidStaticConcatenate},
        {"concatenate_declared_output_mismatch", TestConcatenateDeclaredOutputMismatchIsRejected},
        {"concatenate_strict_attrs", TestConcatenateAttrsAreStrict},
        {"valid_static_slice", TestValidStaticSlice},
        {"slice_declared_output_mismatch", TestSliceDeclaredOutputMismatchIsRejected},
        {"slice_strict_attrs", TestSliceAttrsAreStrict},
        {"valid_static_where", TestValidStaticWhere},
        {"where_strict_attrs", TestWhereAttrsAreStrict},
        {"where_declared_output_mismatch", TestWhereDeclaredOutputMismatchIsRejected},
        {"where_unsupported_branch_dtype", TestWhereUnsupportedBranchDTypeIsRejected},
        {"valid_static_layer_norm", TestValidStaticLayerNorm},
        {"layer_norm_declared_output_mismatch", TestLayerNormDeclaredOutputMismatchIsRejected},
        {"layer_norm_unsupported_attrs", TestLayerNormUnsupportedAttrsAreRejected},
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
