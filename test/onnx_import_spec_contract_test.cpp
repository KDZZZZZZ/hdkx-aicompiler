/*! \file test/onnx_import_spec_contract_test.cpp
 * \brief Verifies the dependency-free C++ ONNX import-spec reifier contract.
 */

#include "kxc/frontend/onnx_importer.h"
#include "support/primitive_lowering.h"

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

// 与 Throws 相同，但额外要求诊断文本包含 needle（例如节点名）。
bool ThrowsWithMessage(const std::function<void()>& action, const std::string& needle,
                       std::string* message = nullptr) {
    try {
        action();
    } catch (const std::exception& error) {
        if (message != nullptr) *message = error.what();
        return error.what() != nullptr && std::string(error.what()).find(needle) !=
                                              std::string::npos;
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
                        bool constant_indices = true,
                        const std::string& index_dtype = "int64") {
    const std::string index_input = constant_indices
        ? ""
        : R"json(,
      {"name": "indices", "shape": [2], "dtype": ")json" + index_dtype + R"json("})json";
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

void WriteEqualFixture(const TemporaryDirectory& directory,
                       const std::string& a_dtype, const std::string& b_dtype,
                       const std::string& output_shape,
                       const std::string& output_dtype = "bool",
                       const std::string& attrs = R"json({})json") {
    const std::string json = R"json({
  "format": "kxc.onnx_import.v1",
  "function": {
    "inputs": [
      {"name": "a", "shape": [2, 3], "dtype": ")json" + a_dtype + R"json("},
      {"name": "b", "shape": [3], "dtype": ")json" + b_dtype + R"json("}
    ],
    "outputs": [
      {"name": "out", "shape": )json" + output_shape + R"json(, "dtype": ")json" +
      output_dtype + R"json("}
    ],
    "nodes": [
      {"name": "equal_node", "op_name": "equal", "inputs": ["a", "b"], "outputs": ["out"], "attrs": )json" + attrs + R"json(}
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

void WriteArithmeticFixture(const TemporaryDirectory& directory,
                            const std::string& output_shape,
                            const std::string& op_name = "mul",
                            const std::string& attrs = R"json({})json") {
    const std::string json = R"json({
  "format": "kxc.onnx_import.v1",
  "function": {
    "inputs": [
      {"name": "lhs", "shape": [2, 1, 3], "dtype": "float32"},
      {"name": "rhs", "shape": [1, 4, 1], "dtype": "float32"}
    ],
    "outputs": [
      {"name": "out", "shape": )json" + output_shape + R"json(, "dtype": "float32"}
    ],
    "nodes": [
      {"name": "arith_node", "op_name": ")json" + op_name + R"json(", "inputs": ["lhs", "rhs"], "outputs": ["out"], "attrs": )json" + attrs + R"json(}
    ]
  },
  "params": [],
  "param_order": []
})json";
    std::ofstream(directory.path() / "model.json") << json;
    std::ofstream(directory.path() / "params.bin", std::ios::binary);
}

void WriteSqrtFixture(const TemporaryDirectory& directory,
                      const std::string& output_shape,
                      const std::string& attrs = R"json({})json") {
    const std::string json = R"json({
  "format": "kxc.onnx_import.v1",
  "function": {
    "inputs": [{"name": "data", "shape": [2, 3], "dtype": "float32"}],
    "outputs": [{"name": "out", "shape": )json" + output_shape + R"json(, "dtype": "float32"}],
    "nodes": [
      {"name": "sqrt_node", "op_name": "sqrt", "inputs": ["data"], "outputs": ["out"], "attrs": )json" + attrs + R"json(}
    ]
  },
  "params": [],
  "param_order": []
})json";
    std::ofstream(directory.path() / "model.json") << json;
    std::ofstream(directory.path() / "params.bin", std::ios::binary);
}

// M4/M5 fieldless float32 一元算子的 reifier 合同：合法 spec 重建为 shape 保持
// 的 Relay 调用；attrs 非空、dtype 越界、输出声明失配都以节点名失败。
void WriteUnaryMathFixture(const TemporaryDirectory& directory, const std::string& op_name,
                           const std::string& dtype, const std::string& output_shape,
                           const std::string& output_dtype = "float32",
                           const std::string& attrs = R"json({})json") {
    const std::string json = R"json({
  "format": "kxc.onnx_import.v1",
  "function": {
    "inputs": [
      {"name": "a", "shape": [2, 3], "dtype": ")json" + dtype + R"json("}
    ],
    "outputs": [
      {"name": "out", "shape": )json" + output_shape + R"json(, "dtype": ")json" +
      output_dtype + R"json("}
    ],
    "nodes": [
      {"name": "s1_neg", "op_name": ")json" + op_name + R"json(", "inputs": ["a"], "outputs": ["out"], "attrs": )json" + attrs + R"json(}
    ]
  },
  "params": [],
  "param_order": []
})json";
    std::ofstream(directory.path() / "model.json") << json;
    std::ofstream(directory.path() / "params.bin", std::ios::binary);
}

// M5 S2 Pow 的 reifier 合同：合法广播 spec 重建；attrs 非空、dtype 越界/失配、
// 不可广播与输出声明失配都以节点名失败。
void WritePowFixture(const TemporaryDirectory& directory, const std::string& a_shape,
                     const std::string& b_shape, const std::string& output_shape,
                     const std::string& a_dtype = "float32",
                     const std::string& b_dtype = "float32",
                     const std::string& attrs = R"json({})json") {
    const std::string json = R"json({
  "format": "kxc.onnx_import.v1",
  "function": {
    "inputs": [
      {"name": "a", "shape": )json" + a_shape + R"json(, "dtype": ")json" + a_dtype + R"json("},
      {"name": "b", "shape": )json" + b_shape + R"json(, "dtype": ")json" + b_dtype + R"json("}
    ],
    "outputs": [
      {"name": "out", "shape": )json" + output_shape + R"json(, "dtype": "float32"}
    ],
    "nodes": [
      {"name": "s2_pow", "op_name": "pow", "inputs": ["a", "b"], "outputs": ["out"], "attrs": )json" + attrs + R"json(}
    ]
  },
  "params": [],
  "param_order": []
})json";
    std::ofstream(directory.path() / "model.json") << json;
    std::ofstream(directory.path() / "params.bin", std::ios::binary);
}

// M4/M5 Expand 的 reifier 合同：canonical attrs 携带已解析的静态目标形状；
// attrs 键集、负维度、rank 越界、维度不兼容与输出声明失配都以节点名失败。
void WriteExpandFixture(const TemporaryDirectory& directory,
                        const std::string& data_shape,
                        const std::string& target_shape,
                        const std::string& output_shape,
                        const std::string& attrs_prefix = "") {
    const std::string attrs = attrs_prefix.empty()
        ? R"json({"target_shape": )json" + target_shape + R"json(})json"
        : attrs_prefix;
    const std::string json = R"json({
  "format": "kxc.onnx_import.v1",
  "function": {
    "inputs": [
      {"name": "a", "shape": )json" + data_shape + R"json(, "dtype": "float32"}
    ],
    "outputs": [
      {"name": "out", "shape": )json" + output_shape + R"json(, "dtype": "float32"}
    ],
    "nodes": [
      {"name": "s1_expand", "op_name": "expand", "inputs": ["a"], "outputs": ["out"], "attrs": )json" + attrs + R"json(}
    ]
  },
  "params": [],
  "param_order": []
})json";
    std::ofstream(directory.path() / "model.json") << json;
    std::ofstream(directory.path() / "params.bin", std::ios::binary);
}

// M4 S2 Unsqueeze 规范化的 C++ 侧证据：规范化产物是 reshape 节点（无新算子
// 表面），reifier 沿既有 reshape 合同校验；手写非法 spec 的失配诊断仍能通过
// producer 节点名定位到规范化的 Unsqueeze 节点。
void WriteUnsqueezeNormalizedFixture(const TemporaryDirectory& directory,
                                     const std::string& newshape,
                                     const std::string& output_shape) {
    const std::string json = R"json({
  "format": "kxc.onnx_import.v1",
  "function": {
    "inputs": [
      {"name": "a", "shape": [2, 4], "dtype": "float32"}
    ],
    "outputs": [
      {"name": "out", "shape": )json" + output_shape + R"json(, "dtype": "float32"}
    ],
    "nodes": [
      {"name": "s2_unsqueeze", "op_name": "reshape", "inputs": ["a"], "outputs": ["out"],
       "attrs": {"newshape": )json" + newshape + R"json(, "allowzero": 0}}
    ]
  },
  "params": [],
  "param_order": []
})json";
    std::ofstream(directory.path() / "model.json") << json;
    std::ofstream(directory.path() / "params.bin", std::ios::binary);
}

bool TestUnsqueezeNormalizationReifier() {
    // 合法规范化：Unsqueeze([2,4], axes=[0]) → reshape [1,2,4]。
    TemporaryDirectory directory;
    WriteUnsqueezeNormalizedFixture(directory, "[1, 2, 4]", "[1, 2, 4]");
    const auto imported = kxc::frontend::LoadONNXImportSpec(
        (directory.path() / "model.json").string(),
        (directory.path() / "params.bin").string());
    TEST_CHECK(imported.function.defined(),
               "a valid Unsqueeze-normalized reshape spec should reify");
    TEST_CHECK(ShapeEquals(imported.function->body.checked_type().As<kxc::TensorTypeNode>(),
                           {1, 2, 4}, "float32"),
               "the normalized reshape output should carry the proven target shape");

    // 手写非法 spec：newshape 与声明的规范化输出不一致（元素数相同但 shape
    // 不同）→ 输出契约失配，诊断含 producer 节点名。
    TemporaryDirectory bad_directory;
    WriteUnsqueezeNormalizedFixture(bad_directory, "[4, 2]", "[1, 2, 4]");
    std::string message;
    TEST_CHECK(ThrowsWithMessage(
                   [&] {
                       kxc::frontend::LoadONNXImportSpec(
                           (bad_directory.path() / "model.json").string(),
                           (bad_directory.path() / "params.bin").string());
                   },
                   "s2_unsqueeze", &message),
               "a hand-written normalized-reshape spec must fail with the node name");
    TEST_CHECK(message.find("output contract mismatch for 'out'") != std::string::npos,
               "the normalized-reshape diagnostic should name the output value");
    return true;
}

bool TestValidStaticExpand() {
    TemporaryDirectory directory;
    WriteExpandFixture(directory, "[2, 1]", "[2, 3]", "[2, 3]");

    const auto imported = kxc::frontend::LoadONNXImportSpec(
        (directory.path() / "model.json").string(),
        (directory.path() / "params.bin").string());
    TEST_CHECK(imported.function.defined(), "valid static Expand import spec should reify");
    TEST_CHECK(ShapeEquals(imported.function->body.checked_type().As<kxc::TensorTypeNode>(),
                           {2, 3}, "float32"),
               "reified Expand output should use the target shape");
    return true;
}

bool TestExpandAttrsAreStrict() {
    // 键集多余或缺失都必须失败，且诊断携带节点名。
    for (const std::string& attrs :
         {std::string(R"json({"target_shape": [2, 3], "axis": 0})json"),
          std::string(R"json({})json"),
          std::string(R"json({"newshape": [2, 3]})json")}) {
        TemporaryDirectory directory;
        WriteExpandFixture(directory, "[2, 1]", "[2, 3]", "[2, 3]", attrs);
        std::string message;
        TEST_CHECK(ThrowsWithMessage(
                       [&] {
                           kxc::frontend::LoadONNXImportSpec(
                               (directory.path() / "model.json").string(),
                               (directory.path() / "params.bin").string());
                       },
                       "s1_expand", &message),
                   "Expand reifier must reject noncanonical attrs with the node name");
    }
    return true;
}

bool TestExpandNegativeTargetDimensionIsRejected() {
    TemporaryDirectory directory;
    WriteExpandFixture(directory, "[2, 1]", "[-2, 3]", "[2, 3]");
    std::string message;
    TEST_CHECK(ThrowsWithMessage(
                   [&] {
                       kxc::frontend::LoadONNXImportSpec(
                           (directory.path() / "model.json").string(),
                           (directory.path() / "params.bin").string());
                   },
                   "s1_expand", &message),
               "Expand import target_shape must reject negative dimensions");
    return true;
}

bool TestExpandIncompatibleContractIsRejected() {
    // 手写 spec 的数据维度既不等于目标维度也不是 1：reifier 必须拒绝。
    TemporaryDirectory directory;
    WriteExpandFixture(directory, "[3]", "[2, 4]", "[2, 4]");
    std::string message;
    TEST_CHECK(ThrowsWithMessage(
                   [&] {
                       kxc::frontend::LoadONNXImportSpec(
                           (directory.path() / "model.json").string(),
                           (directory.path() / "params.bin").string());
                   },
                   "s1_expand", &message),
               "Expand reifier must reject incompatible data dimensions");
    TEST_CHECK(message.find("must be 1 or equal to the target dimension") !=
                   std::string::npos,
               "Expand dimension diagnostic should name the rule");

    // rank 超过目标 rank 也要失败。
    TemporaryDirectory rank_directory;
    WriteExpandFixture(rank_directory, "[2, 3, 4]", "[3, 4]", "[3, 4]");
    TEST_CHECK(Throws([&] {
                   kxc::frontend::LoadONNXImportSpec(
                       (rank_directory.path() / "model.json").string(),
                       (rank_directory.path() / "params.bin").string());
               }),
               "Expand reifier must reject data rank above the target rank");
    return true;
}

bool TestExpandDeclaredOutputMismatchIsRejected() {
    TemporaryDirectory directory;
    WriteExpandFixture(directory, "[2, 1]", "[2, 3]", "[2, 4]");
    std::string message;
    TEST_CHECK(ThrowsWithMessage(
                   [&] {
                       kxc::frontend::LoadONNXImportSpec(
                           (directory.path() / "model.json").string(),
                           (directory.path() / "params.bin").string());
                   },
                   "s1_expand", &message),
               "declared Expand output shape must match the target shape");
    return true;
}

bool TestValidStaticPow() {
    TemporaryDirectory directory;
    WritePowFixture(directory, "[2, 1]", "[1, 3]", "[2, 3]");

    const auto imported = kxc::frontend::LoadONNXImportSpec(
        (directory.path() / "model.json").string(),
        (directory.path() / "params.bin").string());
    TEST_CHECK(imported.function.defined(), "valid static Pow import spec should reify");
    TEST_CHECK(ShapeEquals(imported.function->body.checked_type().As<kxc::TensorTypeNode>(),
                           {2, 3}, "float32"),
               "reified Pow output should use the broadcast shape");
    return true;
}

bool TestPowAttrsAreStrict() {
    TemporaryDirectory directory;
    WritePowFixture(directory, "[2, 3]", "[2, 3]", "[2, 3]", "float32", "float32",
                    R"json({"axis": 0})json");
    std::string message;
    TEST_CHECK(ThrowsWithMessage(
                   [&] {
                       kxc::frontend::LoadONNXImportSpec(
                           (directory.path() / "model.json").string(),
                           (directory.path() / "params.bin").string());
                   },
                   "s2_pow", &message),
               "Pow reifier must reject noncanonical attrs with the node name");
    return true;
}

bool TestPowInvalidDTypesAreRejected() {
    // 手写 spec 不能依赖 Python 已校验的假设：非 float32 与 dtype 失配都必须失败。
    for (auto [a_dtype, b_dtype] :
         {std::pair<std::string, std::string>{"int32", "int32"},
          {"float64", "float64"},
          {"float32", "int64"}}) {
        TemporaryDirectory directory;
        WritePowFixture(directory, "[2, 3]", "[2, 3]", "[2, 3]", a_dtype, b_dtype);
        std::string message;
        TEST_CHECK(ThrowsWithMessage(
                       [&] {
                           kxc::frontend::LoadONNXImportSpec(
                               (directory.path() / "model.json").string(),
                               (directory.path() / "params.bin").string());
                       },
                       "s2_pow", &message),
                   "Pow reifier must reject dtypes outside the M4/M5 float32 subset");
        TEST_CHECK(message.find("M4/M5 static subset") != std::string::npos,
                   "Pow dtype diagnostic should name the subset");
    }
    return true;
}

bool TestPowIncompatibleBroadcastIsRejected() {
    TemporaryDirectory directory;
    WritePowFixture(directory, "[2, 3]", "[2, 4]", "[2, 4]");
    std::string message;
    TEST_CHECK(ThrowsWithMessage(
                   [&] {
                       kxc::frontend::LoadONNXImportSpec(
                           (directory.path() / "model.json").string(),
                           (directory.path() / "params.bin").string());
                   },
                   "s2_pow", &message),
               "Pow reifier must reject incompatible broadcast shapes");
    return true;
}

bool TestPowDeclaredOutputMismatchIsRejected() {
    TemporaryDirectory directory;
    WritePowFixture(directory, "[2, 3]", "[3]", "[2, 4]");
    std::string message;
    TEST_CHECK(ThrowsWithMessage(
                   [&] {
                       kxc::frontend::LoadONNXImportSpec(
                           (directory.path() / "model.json").string(),
                           (directory.path() / "params.bin").string());
                   },
                   "s2_pow", &message),
               "declared Pow output shape must match the inferred broadcast");
    return true;
}

bool TestValidStaticUnaryMath() {
    for (const std::string& op_name : {"neg", "sigmoid"}) {
        TemporaryDirectory directory;
        WriteUnaryMathFixture(directory, op_name, "float32", "[2, 3]");
        const auto imported = kxc::frontend::LoadONNXImportSpec(
            (directory.path() / "model.json").string(),
            (directory.path() / "params.bin").string());
        TEST_CHECK(imported.function.defined(),
                   "valid static " + op_name + " import spec should reify");
        TEST_CHECK(ShapeEquals(
                       imported.function->body.checked_type().As<kxc::TensorTypeNode>(),
                       {2, 3}, "float32"),
                   "reified " + op_name + " output should preserve shape and dtype");
    }
    return true;
}

bool TestUnaryMathAttrsAreStrict() {
    for (const std::string& op_name : {"neg", "sigmoid"}) {
        TemporaryDirectory directory;
        WriteUnaryMathFixture(directory, op_name, "float32", "[2, 3]", "float32",
                              R"json({"axis": 0})json");
        std::string message;
        TEST_CHECK(ThrowsWithMessage(
                       [&] {
                           kxc::frontend::LoadONNXImportSpec(
                               (directory.path() / "model.json").string(),
                               (directory.path() / "params.bin").string());
                       },
                       "s1_neg", &message),
                   op_name + " reifier must reject noncanonical attrs with the node name");
        TEST_CHECK(message.find("(" + op_name + ")") != std::string::npos,
                   op_name + " attr diagnostics should name the canonical op");
    }
    return true;
}

bool TestUnaryMathNonFloat32InputsAreRejected() {
    for (const std::string& dtype : {"int64", "float64"}) {
        for (const std::string& op_name : {"neg", "sigmoid"}) {
            TemporaryDirectory directory;
            WriteUnaryMathFixture(directory, op_name, dtype, "[2, 3]", dtype);
            std::string message;
            TEST_CHECK(ThrowsWithMessage(
                           [&] {
                               kxc::frontend::LoadONNXImportSpec(
                                   (directory.path() / "model.json").string(),
                                   (directory.path() / "params.bin").string());
                           },
                           "s1_neg", &message),
                       op_name + " reifier must reject dtypes outside the M4/M5 float32 subset");
            TEST_CHECK(message.find("M4/M5 static subset") != std::string::npos,
                       op_name + " dtype diagnostic should name the subset");
        }
    }
    return true;
}

bool TestUnaryMathDeclaredOutputMismatchIsRejected() {
    TemporaryDirectory directory;
    WriteUnaryMathFixture(directory, "neg", "float32", "[2, 4]");
    std::string message;
    TEST_CHECK(ThrowsWithMessage(
                   [&] {
                       kxc::frontend::LoadONNXImportSpec(
                           (directory.path() / "model.json").string(),
                           (directory.path() / "params.bin").string());
                   },
                   "s1_neg", &message),
               "declared neg output shape must match the inferred output");
    return true;
}

bool TestValidStaticArithmetic() {
    for (const std::string& op_name : {"mul", "subtract", "divide"}) {
        TemporaryDirectory directory;
        WriteArithmeticFixture(directory, "[2, 4, 3]", op_name);
        const auto imported = kxc::frontend::LoadONNXImportSpec(
            (directory.path() / "model.json").string(),
            (directory.path() / "params.bin").string());
        TEST_CHECK(imported.function.defined(),
                   "valid static " + op_name + " import spec should reify");
        TEST_CHECK(ShapeEquals(
                       imported.function->body.checked_type().As<kxc::TensorTypeNode>(),
                       {2, 4, 3}, "float32"),
                   "reified " + op_name + " output should use the broadcast shape");
    }
    return true;
}

bool TestArithmeticAttrsAreStrict() {
    TemporaryDirectory directory;
    WriteArithmeticFixture(directory, "[2, 4, 3]", "mul", R"json({"axis": 0})json");
    std::string message;
    TEST_CHECK(ThrowsWithMessage(
                   [&] {
                       kxc::frontend::LoadONNXImportSpec(
                           (directory.path() / "model.json").string(),
                           (directory.path() / "params.bin").string());
                   },
                   "arith_node", &message),
               "arithmetic reifier must reject noncanonical attrs with the node name");
    TEST_CHECK(message.find("mul") != std::string::npos,
               "arithmetic attr diagnostics should name the canonical op");

    TemporaryDirectory sqrt_directory;
    WriteSqrtFixture(sqrt_directory, "[2, 3]", R"json({"axis": 0})json");
    message.clear();
    TEST_CHECK(ThrowsWithMessage(
                   [&] {
                       kxc::frontend::LoadONNXImportSpec(
                           (sqrt_directory.path() / "model.json").string(),
                           (sqrt_directory.path() / "params.bin").string());
                   },
                   "sqrt_node", &message),
               "Sqrt reifier must reject noncanonical attrs with the node name");
    return true;
}

bool TestArithmeticNonFloat32InputsAreRejected() {
    TemporaryDirectory directory;
    const std::string json = R"json({
  "format": "kxc.onnx_import.v1",
  "function": {
    "inputs": [
      {"name": "lhs", "shape": [2, 3], "dtype": "int32"},
      {"name": "rhs", "shape": [2, 3], "dtype": "int32"}
    ],
    "outputs": [{"name": "out", "shape": [2, 3], "dtype": "int32"}],
    "nodes": [
      {"name": "arith_node", "op_name": "divide", "inputs": ["lhs", "rhs"], "outputs": ["out"], "attrs": {}}
    ]
  },
  "params": [],
  "param_order": []
})json";
    std::ofstream(directory.path() / "model.json") << json;
    std::ofstream(directory.path() / "params.bin", std::ios::binary);
    std::string message;
    TEST_CHECK(ThrowsWithMessage(
                   [&] {
                       kxc::frontend::LoadONNXImportSpec(
                           (directory.path() / "model.json").string(),
                           (directory.path() / "params.bin").string());
                   },
                   "arith_node", &message),
               "hand-written specs cannot smuggle integer division past the reifier");
    return true;
}

bool TestValidStaticSqrt() {
    TemporaryDirectory directory;
    WriteSqrtFixture(directory, "[2, 3]");

    const auto imported = kxc::frontend::LoadONNXImportSpec(
        (directory.path() / "model.json").string(),
        (directory.path() / "params.bin").string());
    TEST_CHECK(imported.function.defined(), "valid static Sqrt import spec should reify");
    TEST_CHECK(ShapeEquals(imported.function->body.checked_type().As<kxc::TensorTypeNode>(),
                           {2, 3}, "float32"),
               "reified Sqrt output should preserve shape and dtype");
    return true;
}

bool TestParamNameConflictsAreRejected() {
    TemporaryDirectory directory;
    const std::string json = R"json({
  "format": "kxc.onnx_import.v1",
  "function": {
    "inputs": [{"name": "data", "shape": [2, 3], "dtype": "float32"}],
    "outputs": [{"name": "out", "shape": [2, 3], "dtype": "float32"}],
    "nodes": [
      {"name": "sqrt_node", "op_name": "sqrt", "inputs": ["data"], "outputs": ["out"], "attrs": {}}
    ]
  },
  "params": [
    {"name": "data", "shape": [2, 3], "dtype": "float32", "offset": 0, "nbytes": 24}
  ],
  "param_order": ["data"]
})json";
    std::ofstream(directory.path() / "model.json") << json;
    std::ofstream binary(directory.path() / "params.bin", std::ios::binary);
    binary.write("012345678901234567890123", 24);
    TEST_CHECK(Throws([&] {
                   kxc::frontend::LoadONNXImportSpec(
                       (directory.path() / "model.json").string(),
                       (directory.path() / "params.bin").string());
               }),
               "params must not shadow graph inputs");
    return true;
}

bool TestNodeOutputNameConflictsWithValueAreRejected() {
    TemporaryDirectory directory;
    const std::string json = R"json({
  "format": "kxc.onnx_import.v1",
  "function": {
    "inputs": [{"name": "data", "shape": [2, 3], "dtype": "float32"}],
    "outputs": [{"name": "out", "shape": [2, 3], "dtype": "float32"}],
    "nodes": [
      {"name": "first", "op_name": "sqrt", "inputs": ["data"], "outputs": ["mid"], "attrs": {}},
      {"name": "second", "op_name": "sqrt", "inputs": ["data"], "outputs": ["mid"], "attrs": {}}
    ]
  },
  "params": [],
  "param_order": []
})json";
    std::ofstream(directory.path() / "model.json") << json;
    std::ofstream(directory.path() / "params.bin", std::ios::binary);
    TEST_CHECK(Throws([&] {
                   kxc::frontend::LoadONNXImportSpec(
                       (directory.path() / "model.json").string(),
                       (directory.path() / "params.bin").string());
               }),
               "duplicate node output names must be rejected");
    return true;
}

void WriteCastFixture(const TemporaryDirectory& directory,
                      const std::string& input_dtype, const std::string& output_dtype,
                      const std::string& attrs = R"json({"to": 0})json") {
    const std::string json = R"json({
  "format": "kxc.onnx_import.v1",
  "function": {
    "inputs": [{"name": "data", "shape": [2, 3], "dtype": ")json" + input_dtype + R"json("}],
    "outputs": [{"name": "out", "shape": [2, 3], "dtype": ")json" + output_dtype + R"json("}],
    "nodes": [
      {"name": "cast_node", "op_name": "cast", "inputs": ["data"], "outputs": ["out"], "attrs": )json" + attrs + R"json(}
    ]
  },
  "params": [],
  "param_order": []
})json";
    std::ofstream(directory.path() / "model.json") << json;
    std::ofstream(directory.path() / "params.bin", std::ios::binary);
}

void WriteReduceMeanFixture(const TemporaryDirectory& directory,
                            const std::string& output_shape,
                            const std::string& attrs =
                                R"json({"axes": [1], "keepdims": 1})json") {
    const std::string json = R"json({
  "format": "kxc.onnx_import.v1",
  "function": {
    "inputs": [{"name": "data", "shape": [2, 3, 4], "dtype": "float32"}],
    "outputs": [{"name": "out", "shape": )json" + output_shape + R"json(, "dtype": "float32"}],
    "nodes": [
      {"name": "reduce_node", "op_name": "reduce_mean", "inputs": ["data"], "outputs": ["out"], "attrs": )json" + attrs + R"json(}
    ]
  },
  "params": [],
  "param_order": []
})json";
    std::ofstream(directory.path() / "model.json") << json;
    std::ofstream(directory.path() / "params.bin", std::ios::binary);
}

void WriteReshapeFixture(const TemporaryDirectory& directory,
                         const std::string& output_shape,
                         const std::string& attrs =
                             R"json({"newshape": [6, 4], "allowzero": 0})json") {
    const std::string json = R"json({
  "format": "kxc.onnx_import.v1",
  "function": {
    "inputs": [{"name": "data", "shape": [2, 3, 4], "dtype": "float32"}],
    "outputs": [{"name": "out", "shape": )json" + output_shape + R"json(, "dtype": "float32"}],
    "nodes": [
      {"name": "reshape_node", "op_name": "reshape", "inputs": ["data"], "outputs": ["out"], "attrs": )json" + attrs + R"json(}
    ]
  },
  "params": [],
  "param_order": []
})json";
    std::ofstream(directory.path() / "model.json") << json;
    std::ofstream(directory.path() / "params.bin", std::ios::binary);
}

bool TestValidStaticCast() {
    TemporaryDirectory directory;
    WriteCastFixture(directory, "int64", "float32");

    const auto imported = kxc::frontend::LoadONNXImportSpec(
        (directory.path() / "model.json").string(),
        (directory.path() / "params.bin").string());
    TEST_CHECK(imported.function.defined(), "valid static Cast import spec should reify");
    TEST_CHECK(ShapeEquals(imported.function->body.checked_type().As<kxc::TensorTypeNode>(),
                           {2, 3}, "float32"),
               "reified Cast output should carry the target dtype");
    return true;
}

bool TestCastSubsetIsEnforced() {
    TemporaryDirectory wrong_target;
    WriteCastFixture(wrong_target, "int64", "int32", R"json({"to": 1})json");
    std::string message;
    TEST_CHECK(ThrowsWithMessage(
                   [&] {
                       kxc::frontend::LoadONNXImportSpec(
                           (wrong_target.path() / "model.json").string(),
                           (wrong_target.path() / "params.bin").string());
                   },
                   "cast_node", &message),
               "hand-written Cast specs cannot target non-float32 dtypes");

    // float32 源是恒等转换：MiniMind 的 RMSNorm 用 `.float()` 提升精度，在已是
    // float32 的图上导出成恒等 Cast。Relay cast 对 dtype 不设限，同 dtype 经
    // topi 落成一次拷贝，因此这条路径被接受而不是拒绝。
    TemporaryDirectory identity_source;
    WriteCastFixture(identity_source, "float32", "float32");
    const auto identity = kxc::frontend::LoadONNXImportSpec(
        (identity_source.path() / "model.json").string(),
        (identity_source.path() / "params.bin").string());
    TEST_CHECK(identity.function.defined(),
               "an identity float32 Cast should reify");

    // 未支持的源 dtype 仍然拒绝：合同只放开恒等这一格。
    TemporaryDirectory wrong_source;
    WriteCastFixture(wrong_source, "bool", "float32");
    TEST_CHECK(ThrowsWithMessage(
                   [&] {
                       kxc::frontend::LoadONNXImportSpec(
                           (wrong_source.path() / "model.json").string(),
                           (wrong_source.path() / "params.bin").string());
                   },
                   "cast_node", &message),
               "hand-written Cast specs cannot use unsupported source dtypes");
    return true;
}

bool TestCastAttrsAreStrict() {
    TemporaryDirectory directory;
    WriteCastFixture(directory, "int64", "float32", R"json({"to": 0, "extra": 1})json");
    TEST_CHECK(Throws([&] {
                   kxc::frontend::LoadONNXImportSpec(
                       (directory.path() / "model.json").string(),
                       (directory.path() / "params.bin").string());
               }),
               "Cast reifier must reject attrs outside the exact canonical set");
    return true;
}

bool TestValidStaticReduceMean() {
    TemporaryDirectory directory;
    WriteReduceMeanFixture(directory, "[2, 1, 4]");

    const auto imported = kxc::frontend::LoadONNXImportSpec(
        (directory.path() / "model.json").string(),
        (directory.path() / "params.bin").string());
    TEST_CHECK(imported.function.defined(),
               "valid static ReduceMean import spec should reify");
    TEST_CHECK(ShapeEquals(imported.function->body.checked_type().As<kxc::TensorTypeNode>(),
                           {2, 1, 4}, "float32"),
               "reified ReduceMean output should apply keepdims");
    return true;
}

bool TestReduceMeanAttrsAreStrict() {
    TemporaryDirectory directory;
    WriteReduceMeanFixture(directory, "[2, 1, 4]", R"json({"axes": [1]})json");
    TEST_CHECK(Throws([&] {
                   kxc::frontend::LoadONNXImportSpec(
                       (directory.path() / "model.json").string(),
                       (directory.path() / "params.bin").string());
               }),
               "ReduceMean reifier must require axes and keepdims");

    TemporaryDirectory bad_keepdims;
    WriteReduceMeanFixture(bad_keepdims, "[2, 1, 4]",
                           R"json({"axes": [1], "keepdims": 2})json");
    std::string message;
    TEST_CHECK(ThrowsWithMessage(
                   [&] {
                       kxc::frontend::LoadONNXImportSpec(
                           (bad_keepdims.path() / "model.json").string(),
                           (bad_keepdims.path() / "params.bin").string());
                   },
                   "reduce_node", &message),
               "ReduceMean reifier must reject keepdims outside 0/1");
    return true;
}

bool TestReduceMeanZeroExtentAxisIsRejected() {
    TemporaryDirectory directory;
    const std::string json = R"json({
  "format": "kxc.onnx_import.v1",
  "function": {
    "inputs": [{"name": "data", "shape": [2, 0], "dtype": "float32"}],
    "outputs": [{"name": "out", "shape": [2, 1], "dtype": "float32"}],
    "nodes": [
      {"name": "reduce_node", "op_name": "reduce_mean", "inputs": ["data"], "outputs": ["out"], "attrs": {"axes": [1], "keepdims": 1}}
    ]
  },
  "params": [],
  "param_order": []
})json";
    std::ofstream(directory.path() / "model.json") << json;
    std::ofstream(directory.path() / "params.bin", std::ios::binary);
    std::string message;
    TEST_CHECK(ThrowsWithMessage(
                   [&] {
                       kxc::frontend::LoadONNXImportSpec(
                           (directory.path() / "model.json").string(),
                           (directory.path() / "params.bin").string());
                   },
                   "reduce_node", &message),
               "reducing over a zero-extent axis must fail closed");
    return true;
}

bool TestReduceMeanDeclaredOutputMismatchIsRejected() {
    TemporaryDirectory directory;
    WriteReduceMeanFixture(directory, "[2, 3, 4]");
    TEST_CHECK(Throws([&] {
                   kxc::frontend::LoadONNXImportSpec(
                       (directory.path() / "model.json").string(),
                       (directory.path() / "params.bin").string());
               }),
               "declared ReduceMean output shape must match inferred output");
    return true;
}

bool TestValidStaticReshape() {
    TemporaryDirectory directory;
    WriteReshapeFixture(directory, "[6, 4]");

    const auto imported = kxc::frontend::LoadONNXImportSpec(
        (directory.path() / "model.json").string(),
        (directory.path() / "params.bin").string());
    TEST_CHECK(imported.function.defined(), "valid static Reshape import spec should reify");
    TEST_CHECK(ShapeEquals(imported.function->body.checked_type().As<kxc::TensorTypeNode>(),
                           {6, 4}, "float32"),
               "reified Reshape output should use the resolved target shape");
    return true;
}

bool TestReshapeNegativeNewshapeIsRejected() {
    TemporaryDirectory directory;
    WriteReshapeFixture(directory, "[6, 4]", R"json({"newshape": [-1, 4], "allowzero": 0})json");
    std::string message;
    TEST_CHECK(ThrowsWithMessage(
                   [&] {
                       kxc::frontend::LoadONNXImportSpec(
                           (directory.path() / "model.json").string(),
                           (directory.path() / "params.bin").string());
                   },
                   "reshape_node", &message),
               "hand-written Reshape specs must carry the fully resolved newshape");
    return true;
}

bool TestReshapeAllowzeroIsRejected() {
    TemporaryDirectory directory;
    WriteReshapeFixture(directory, "[2, 3, 4]",
                        R"json({"newshape": [2, 3, 4], "allowzero": 1})json");
    TEST_CHECK(Throws([&] {
                   kxc::frontend::LoadONNXImportSpec(
                       (directory.path() / "model.json").string(),
                       (directory.path() / "params.bin").string());
               }),
               "Reshape import requires allowzero=0");
    return true;
}

bool TestReshapeDeclaredOutputMismatchIsRejected() {
    TemporaryDirectory directory;
    WriteReshapeFixture(directory, "[6, 5]");
    TEST_CHECK(Throws([&] {
                   kxc::frontend::LoadONNXImportSpec(
                       (directory.path() / "model.json").string(),
                       (directory.path() / "params.bin").string());
               }),
               "declared Reshape output shape must match inferred output");
    return true;
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
    TEST_CHECK(kxc::test_support::LowerFirstPrimitive(imported.function)->prim_func.defined(),
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

// 运行时索引是 embedding 查表的形态：值到 launch 才存在，导入期只能固定
// 形状/dtype/axis 合同。值域由已 lower 的 GatherCompute 守卫承担——负索引按
// ONNX 语义折回，越界经 Select 取零且不形成越界 Load。
bool TestGatherRuntimeIndicesAreAccepted() {
    TemporaryDirectory directory;
    WriteGatherFixture(directory, "[2, 2, 4]", R"json({"axis": 1})json",
                       {0, -3}, false);
    const auto imported = kxc::frontend::LoadONNXImportSpec(
        (directory.path() / "model.json").string(),
        (directory.path() / "params.bin").string());
    TEST_CHECK(imported.function.defined(),
               "Gather with a runtime index tensor should reify");
    TEST_CHECK(ShapeEquals(imported.function->body.checked_type().As<kxc::TensorTypeNode>(),
                           {2, 2, 4}, "float32"),
               "runtime-index Gather keeps the declared output contract");
    TEST_CHECK(imported.input_names.size() == 2 &&
                   imported.input_names[1] == "indices",
               "the index tensor stays a real graph input");
    return true;
}

// 运行时索引仍须是整数张量：dtype 合同在导入期就要成立。
bool TestGatherRuntimeIndicesRejectNonIntegerDType() {
    TemporaryDirectory directory;
    WriteGatherFixture(directory, "[2, 2, 4]", R"json({"axis": 1})json",
                       {0, -3}, false, "float32");
    TEST_CHECK(Throws([&] {
                   kxc::frontend::LoadONNXImportSpec(
                       (directory.path() / "model.json").string(),
                       (directory.path() / "params.bin").string());
               }),
               "Gather reifier must reject a non-integer runtime index tensor");
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

bool TestValidStaticEqual() {
    TemporaryDirectory directory;
    WriteEqualFixture(directory, "float32", "float32", "[2, 3]");

    const auto imported = kxc::frontend::LoadONNXImportSpec(
        (directory.path() / "model.json").string(),
        (directory.path() / "params.bin").string());
    TEST_CHECK(imported.function.defined(), "valid static Equal import spec should reify");
    TEST_CHECK(ShapeEquals(imported.function->body.checked_type().As<kxc::TensorTypeNode>(),
                           {2, 3}, "bool"),
               "reified Equal output should use the broadcast shape and bool dtype");
    return true;
}

bool TestEqualAttrsAreStrict() {
    TemporaryDirectory directory;
    WriteEqualFixture(directory, "float32", "float32", "[2, 3]", "bool",
                      R"json({"axis": 0})json");
    std::string message;
    TEST_CHECK(ThrowsWithMessage(
                   [&] {
                       kxc::frontend::LoadONNXImportSpec(
                           (directory.path() / "model.json").string(),
                           (directory.path() / "params.bin").string());
                   },
                   "equal_node", &message),
               "Equal reifier must reject noncanonical attrs with the node name");
    return true;
}

bool TestEqualMismatchedDTypesAreRejected() {
    TemporaryDirectory directory;
    WriteEqualFixture(directory, "int32", "float32", "[2, 3]");
    std::string message;
    TEST_CHECK(ThrowsWithMessage(
                   [&] {
                       kxc::frontend::LoadONNXImportSpec(
                           (directory.path() / "model.json").string(),
                           (directory.path() / "params.bin").string());
                   },
                   "equal_node", &message),
               "hand-written Equal specs cannot compare mixed dtypes");
    TEST_CHECK(message.find("matching input dtypes") != std::string::npos,
               "Equal dtype-mismatch diagnostic should name the rule");
    return true;
}

bool TestEqualUnsupportedDTypesAreRejected() {
    for (const std::string& dtype : {"float64", "bool"}) {
        TemporaryDirectory directory;
        WriteEqualFixture(directory, dtype, dtype, "[2, 3]");
        std::string message;
        TEST_CHECK(ThrowsWithMessage(
                       [&] {
                           kxc::frontend::LoadONNXImportSpec(
                               (directory.path() / "model.json").string(),
                               (directory.path() / "params.bin").string());
                       },
                       "equal_node", &message),
                       "Equal reifier must reject dtypes outside int32/int64/float32");
        TEST_CHECK(message.find("same-dtype int32, int64, or float32") !=
                       std::string::npos,
                   "Equal unsupported-dtype diagnostic should name the subset");
    }
    return true;
}

bool TestEqualIncompatibleBroadcastIsRejected() {
    // 手写 spec 把 b 声明为 [4]：广播不可行，必须连同节点名一起失败。
    TemporaryDirectory directory;
    const std::string json = R"json({
  "format": "kxc.onnx_import.v1",
  "function": {
    "inputs": [
      {"name": "a", "shape": [2, 3], "dtype": "float32"},
      {"name": "b", "shape": [4], "dtype": "float32"}
    ],
    "outputs": [{"name": "out", "shape": [2, 3], "dtype": "bool"}],
    "nodes": [
      {"name": "equal_node", "op_name": "equal", "inputs": ["a", "b"], "outputs": ["out"], "attrs": {}}
    ]
  },
  "params": [],
  "param_order": []
})json";
    std::ofstream(directory.path() / "model.json") << json;
    std::ofstream(directory.path() / "params.bin", std::ios::binary);
    std::string message;
    TEST_CHECK(ThrowsWithMessage(
                   [&] {
                       kxc::frontend::LoadONNXImportSpec(
                           (directory.path() / "model.json").string(),
                           (directory.path() / "params.bin").string());
                   },
                   "equal_node", &message),
               "Equal reifier must reject incompatible broadcast shapes");
    return true;
}

bool TestEqualNonBoolDeclaredOutputIsRejected() {
    TemporaryDirectory directory;
    WriteEqualFixture(directory, "float32", "float32", "[2, 3]", "float32");

    std::string message;
    TEST_CHECK(ThrowsWithMessage(
                   [&] {
                       kxc::frontend::LoadONNXImportSpec(
                           (directory.path() / "model.json").string(),
                           (directory.path() / "params.bin").string());
                   },
                   "equal_node", &message),
               "declared Equal output must be bool to match the inferred output");
    TEST_CHECK(message.find("output contract mismatch for 'out'") != std::string::npos,
               "Equal declared-output diagnostic should name the output value");
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
        {"gather_runtime_indices_accepted", TestGatherRuntimeIndicesAreAccepted},
        {"gather_runtime_indices_reject_non_integer_dtype",
         TestGatherRuntimeIndicesRejectNonIntegerDType},
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
        {"valid_static_equal", TestValidStaticEqual},
        {"equal_strict_attrs", TestEqualAttrsAreStrict},
        {"equal_mismatched_dtypes_rejected", TestEqualMismatchedDTypesAreRejected},
        {"equal_unsupported_dtypes_rejected", TestEqualUnsupportedDTypesAreRejected},
        {"equal_incompatible_broadcast_rejected", TestEqualIncompatibleBroadcastIsRejected},
        {"equal_non_bool_declared_output_rejected",
         TestEqualNonBoolDeclaredOutputIsRejected},
        {"valid_static_layer_norm", TestValidStaticLayerNorm},
        {"layer_norm_declared_output_mismatch", TestLayerNormDeclaredOutputMismatchIsRejected},
        {"layer_norm_unsupported_attrs", TestLayerNormUnsupportedAttrsAreRejected},
        {"unsqueeze_normalization_reifier", TestUnsqueezeNormalizationReifier},
        {"valid_static_expand", TestValidStaticExpand},
        {"expand_attrs_are_strict", TestExpandAttrsAreStrict},
        {"expand_negative_target_dimension_rejected",
         TestExpandNegativeTargetDimensionIsRejected},
        {"expand_incompatible_contract_rejected", TestExpandIncompatibleContractIsRejected},
        {"expand_declared_output_mismatch", TestExpandDeclaredOutputMismatchIsRejected},
        {"valid_static_pow", TestValidStaticPow},
        {"pow_attrs_are_strict", TestPowAttrsAreStrict},
        {"pow_invalid_dtypes_rejected", TestPowInvalidDTypesAreRejected},
        {"pow_incompatible_broadcast_rejected", TestPowIncompatibleBroadcastIsRejected},
        {"pow_declared_output_mismatch", TestPowDeclaredOutputMismatchIsRejected},
        {"valid_static_unary_math", TestValidStaticUnaryMath},
        {"unary_math_attrs_are_strict", TestUnaryMathAttrsAreStrict},
        {"unary_math_non_float32_inputs_rejected", TestUnaryMathNonFloat32InputsAreRejected},
        {"unary_math_declared_output_mismatch", TestUnaryMathDeclaredOutputMismatchIsRejected},
        {"valid_static_arithmetic", TestValidStaticArithmetic},
        {"arithmetic_attrs_are_strict", TestArithmeticAttrsAreStrict},
        {"arithmetic_non_float32_inputs_rejected", TestArithmeticNonFloat32InputsAreRejected},
        {"valid_static_sqrt", TestValidStaticSqrt},
        {"param_name_conflicts_are_rejected", TestParamNameConflictsAreRejected},
        {"node_output_name_conflicts_are_rejected",
         TestNodeOutputNameConflictsWithValueAreRejected},
        {"valid_static_cast", TestValidStaticCast},
        {"cast_subset_is_enforced", TestCastSubsetIsEnforced},
        {"cast_attrs_are_strict", TestCastAttrsAreStrict},
        {"valid_static_reduce_mean", TestValidStaticReduceMean},
        {"reduce_mean_attrs_are_strict", TestReduceMeanAttrsAreStrict},
        {"reduce_mean_zero_extent_axis_rejected", TestReduceMeanZeroExtentAxisIsRejected},
        {"reduce_mean_declared_output_mismatch", TestReduceMeanDeclaredOutputMismatchIsRejected},
        {"valid_static_reshape", TestValidStaticReshape},
        {"reshape_negative_newshape_rejected", TestReshapeNegativeNewshapeIsRejected},
        {"reshape_allowzero_rejected", TestReshapeAllowzeroIsRejected},
        {"reshape_declared_output_mismatch", TestReshapeDeclaredOutputMismatchIsRejected},
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
