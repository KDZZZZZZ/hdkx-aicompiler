/*! \file test/op_numeric_llvm_test.cpp
 * \brief 验证 Relay 算子在 LLVM 后端上的数值正确性。
 */

#include "kxc/compiler/compiler.h"
#include "kxc/relay/op.h"
#include "kxc/runtime/session.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef KXC_USE_LLVM
#define KXC_USE_LLVM 0
#endif

namespace {

// 将失败条件转换为带上下文的测试异常。
void Check(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// 按容差比较浮点结果向量。
void ExpectNear(const std::vector<float>& actual, const std::vector<float>& expected,
                float tolerance = 1e-4f) {
    Check(actual.size() == expected.size(), "result size mismatch");
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
            throw std::runtime_error("non-finite value at " + std::to_string(i));
        }
        const float diff = std::fabs(actual[i] - expected[i]);
        if (diff > tolerance) {
            throw std::runtime_error("value mismatch at " + std::to_string(i) +
                                     ": actual=" + std::to_string(actual[i]) +
                                     ", expected=" + std::to_string(expected[i]));
        }
    }
}

std::vector<float> LayerNormReference(
    const std::vector<float>& data, size_t outer_size, size_t normalized_size,
    const std::vector<float>& scale, const std::vector<float>& bias,
    float epsilon) {
    Check(data.size() == outer_size * normalized_size &&
              scale.size() == normalized_size && bias.size() == normalized_size,
          "LayerNorm reference shape mismatch");
    std::vector<float> result(data.size(), 0.0f);
    for (size_t outer = 0; outer < outer_size; ++outer) {
        const size_t offset = outer * normalized_size;
        double mean = 0.0;
        for (size_t inner = 0; inner < normalized_size; ++inner) {
            mean += static_cast<double>(data[offset + inner]);
        }
        mean /= static_cast<double>(normalized_size);
        double variance = 0.0;
        for (size_t inner = 0; inner < normalized_size; ++inner) {
            const double centered =
                static_cast<double>(data[offset + inner]) - mean;
            variance += centered * centered;
        }
        variance /= static_cast<double>(normalized_size);
        const double inverse_stddev =
            1.0 / std::sqrt(variance + static_cast<double>(epsilon));
        for (size_t inner = 0; inner < normalized_size; ++inner) {
            result[offset + inner] = static_cast<float>(
                (static_cast<double>(data[offset + inner]) - mean) *
                    inverse_stddev * static_cast<double>(scale[inner]) +
                static_cast<double>(bias[inner]));
        }
    }
    return result;
}

// 精确比较整数结果向量。
void ExpectEqual(const std::vector<int32_t>& actual, const std::vector<int32_t>& expected) {
    Check(actual == expected, "integer result mismatch");
}

/*! \brief 描述主机测试向量与单个 NDArray 参数之间的上传或回读动作。 */
struct HostArgument {
    size_t bytes{0};
    std::function<void(const kxc::runtime::NDArray&)> before_launch;
    std::function<void(const kxc::runtime::NDArray&)> after_launch;
};

// 把只读主机向量声明为输入；实际 NDArray shape/dtype 由 KernelSignature 决定。
template <typename T>
HostArgument Input(const std::vector<T>& values) {
    return HostArgument{
        values.size() * sizeof(T),
        [&values](const kxc::runtime::NDArray& array) {
            array.CopyFromBytes(values.data(), array.NBytes());
        },
        {},
    };
}

// 把可写主机向量声明为输出；内核完成后再从 NDArray 回读。
template <typename T>
HostArgument Output(std::vector<T>& values) {
    return HostArgument{
        values.size() * sizeof(T),
        {},
        [&values](const kxc::runtime::NDArray& array) {
            array.CopyToBytes(values.data(), array.NBytes());
        },
    };
}

// 编译 Relay 函数，按签名分配 NDArray，并通过显式 CPU stream 执行 LLVM 内核。
void CompileAndRun(const std::string& op_name, kxc::Function func,
                   const std::vector<HostArgument>& host_arguments) {
#if KXC_USE_LLVM
    auto config = kxc::api::CompileConfig::Create(
        kxc::BuildTarget(kxc::Device::CPU()), 0);
    auto compiled = kxc::api::Compiler::Compile(func, config);
    Check(compiled.module().IsReady(), op_name + " LLVM module should be ready");
    const auto values = compiled.plan().values();
    const auto find_value = [&](int64_t value_id) {
        for (const auto& value : values) {
            if (value->value_id == value_id) return value;
        }
        throw std::runtime_error(op_name + " plan references an unknown value");
    };
    const auto input_ids = compiled.plan().input_value_ids();
    const auto output_ids = compiled.plan().output_value_ids();
    Check(input_ids.size() + output_ids.size() == host_arguments.size(),
          op_name + " host argument count does not match graph ABI");

    kxc::Array<kxc::runtime::NDArray> inputs;
    for (size_t i = 0; i < input_ids.size(); ++i) {
        const auto spec = find_value(input_ids[i]);
        const auto& host = host_arguments[i];
        kxc::runtime::NDArray array = kxc::runtime::NDArray::Empty(
            spec.shape(), spec->dtype, spec->device);
        Check(array.NBytes() == host.bytes,
              op_name + " host byte count does not match input " +
                  std::to_string(i));
        if (host.before_launch) host.before_launch(array);
        inputs.push_back(std::move(array));
    }
    kxc::runtime::RuntimeSession session(compiled.module(), compiled.plan());
    const auto outputs = session.Run(inputs);
    Check(outputs.size() == output_ids.size(),
          op_name + " runtime output count does not match graph ABI");
    for (size_t i = 0; i < outputs.size(); ++i) {
        const auto& host = host_arguments[input_ids.size() + i];
        Check(outputs[i].NBytes() == host.bytes,
              op_name + " host byte count does not match output " +
                  std::to_string(i));
        if (host.after_launch) host.after_launch(outputs[i]);
    }
#else
    (void)op_name;
    (void)func;
    (void)host_arguments;
    std::cout << "[SKIP] op numeric LLVM test: KXC_USE_LLVM=0\n";
#endif
}

// 验证逐元素 Add 的 LLVM 数值结果。
void TestAdd() {
    kxc::Var x("x", kxc::TensorType({2, 3}, "float32"));
    kxc::Var y("y", kxc::TensorType({3}, "float32"));
    kxc::Call call(kxc::relay::Op::Get("add"), {x, y});
    kxc::Function func({x, y}, call);

    std::vector<float> x_data = {1, 2, 3, 4, 5, 6};
    std::vector<float> y_data = {10, 20, 30};
    std::vector<float> out(6, 0.0f);
    CompileAndRun("add", func, {Input(x_data), Input(y_data), Output(out)});
    ExpectNear(out, {11, 22, 33, 14, 25, 36});
}

// 验证逐元素 Subtract 的 LLVM 数值结果。
void TestSubtract() {
    kxc::Var x("x", kxc::TensorType({2, 3}, "float32"));
    kxc::Var y("y", kxc::TensorType({3}, "float32"));
    kxc::Call call(kxc::relay::Op::Get("subtract"), {x, y});
    kxc::Function func({x, y}, call);

    std::vector<float> x_data = {1, 2, 3, 4, 5, 6};
    std::vector<float> y_data = {10, 20, 30};
    std::vector<float> out(6, 0.0f);
    CompileAndRun("subtract", func, {Input(x_data), Input(y_data), Output(out)});
    ExpectNear(out, {-9, -18, -27, -6, -15, -24});
}

// 验证逐元素 Mul 的 LLVM 数值结果。
void TestMul() {
    kxc::Var x("x", kxc::TensorType({2, 3}, "float32"));
    kxc::Var y("y", kxc::TensorType({3}, "float32"));
    kxc::Call call(kxc::relay::Op::Get("mul"), {x, y});
    kxc::Function func({x, y}, call);

    std::vector<float> x_data = {1, 2, 3, 4, 5, 6};
    std::vector<float> y_data = {10, 20, 30};
    std::vector<float> out(6, 0.0f);
    CompileAndRun("mul", func, {Input(x_data), Input(y_data), Output(out)});
    ExpectNear(out, {10, 40, 90, 40, 100, 180});
}

// 验证逐元素 Divide 的 LLVM 数值结果。
void TestDivide() {
    kxc::Var x("x", kxc::TensorType({2, 3}, "float32"));
    kxc::Var y("y", kxc::TensorType({3}, "float32"));
    kxc::Call call(kxc::relay::Op::Get("divide"), {x, y});
    kxc::Function func({x, y}, call);

    std::vector<float> x_data = {1, 2, 3, 4, 5, 6};
    std::vector<float> y_data = {10, 20, 30};
    std::vector<float> out(6, 0.0f);
    CompileAndRun("divide", func, {Input(x_data), Input(y_data), Output(out)});
    ExpectNear(out, {0.1f, 0.1f, 0.1f, 0.4f, 0.25f, 0.2f});
}

// 验证逐元素 Sqrt 的 LLVM 数值结果。
void TestSqrt() {
    kxc::Var x("x", kxc::TensorType({4}, "float32"));
    kxc::Call call(kxc::relay::Op::Get("sqrt"), {x});
    kxc::Function func({x}, call);

    std::vector<float> x_data = {1, 4, 9, 16};
    std::vector<float> out(4, 0.0f);
    CompileAndRun("sqrt", func, {Input(x_data), Output(out)});
    ExpectNear(out, {1, 2, 3, 4});
}

// 验证二维 Matmul 的 LLVM 数值结果。
void TestMatmul() {
    kxc::Var a("a", kxc::TensorType({2, 3}, "float32"));
    kxc::Var b("b", kxc::TensorType({3, 2}, "float32"));
    kxc::Call call(kxc::relay::Op::Get("matmul"), {a, b});
    kxc::Function func({a, b}, call);

    std::vector<float> a_data = {1, 2, 3, 4, 5, 6};
    std::vector<float> b_data = {1, 2, 3, 4, 5, 6};
    std::vector<float> out(4, 0.0f);
    CompileAndRun("matmul", func, {Input(a_data), Input(b_data), Output(out)});
    ExpectNear(out, {22, 28, 49, 64});
}

// 验证 batched MatMul 的 leading batch broadcast 和 LLVM 数值结果。
void TestBatchedMatmul() {
    kxc::Var a("a", kxc::TensorType({2, 2, 3}, "float32"));
    kxc::Var b("b", kxc::TensorType({1, 3, 2}, "float32"));
    kxc::Call call(kxc::relay::Op::Get("matmul"), {a, b});
    kxc::Function func({a, b}, call);

    std::vector<float> a_data = {1, 2, 3, 4, 5, 6, 1, 0, 1, 0, 1, 0};
    std::vector<float> b_data = {1, 2, 3, 4, 5, 6};
    std::vector<float> out(8, 0.0f);
    CompileAndRun("batched_matmul", func, {Input(a_data), Input(b_data), Output(out)});
    ExpectNear(out, {22, 28, 49, 64, 6, 8, 3, 4});
}

// 验证 Dense 的权重布局和 LLVM 数值结果。
void TestDense() {
    kxc::Var data("data", kxc::TensorType({2, 3}, "float32"));
    kxc::Var weight("weight", kxc::TensorType({2, 3}, "float32"));
    kxc::Call call(kxc::relay::Op::Get("nn_dense"), {data, weight},
                   kxc::relay::DenseAttrs::Create(2, ""));
    kxc::Function func({data, weight}, call);

    std::vector<float> data_buf = {1, 2, 3, 4, 5, 6};
    std::vector<float> weight_buf = {1, 0, 1, 0, 1, 1};
    std::vector<float> out(4, 0.0f);
    CompileAndRun("nn_dense", func,
                  {Input(data_buf), Input(weight_buf), Output(out)});
    ExpectNear(out, {4, 5, 10, 11});
}

// 验证 Gemm 的转置、缩放和偏置语义。
void TestGemm() {
    kxc::Var a("a", kxc::TensorType({2, 3}, "float32"));
    kxc::Var b("b", kxc::TensorType({4, 3}, "float32"));
    kxc::Var c("c", kxc::TensorType({4}, "float32"));
    kxc::Call call(kxc::relay::Op::Get("nn_gemm"), {a, b, c},
                   kxc::relay::GemmAttrs::Create(1.0f, 1.0f, 0, 1));
    kxc::Function func({a, b, c}, call);

    std::vector<float> a_data = {1, 2, 3, 4, 5, 6};
    std::vector<float> b_data = {1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1};
    std::vector<float> c_data = {10, 20, 30, 40};
    std::vector<float> out(8, 0.0f);
    CompileAndRun("nn_gemm", func,
                  {Input(a_data), Input(b_data), Input(c_data), Output(out)});
    ExpectNear(out, {11, 22, 33, 46, 14, 25, 36, 55});
}

// 验证 Relu 对正负输入的数值结果。
void TestRelu() {
    kxc::Var x("x", kxc::TensorType({6}, "float32"));
    kxc::Call call(kxc::relay::Op::Get("nn_relu"), {x}, kxc::relay::ReluAttrs::Create());
    kxc::Function func({x}, call);

    std::vector<float> x_data = {-2, -0.5f, 0, 1, 2, -3};
    std::vector<float> out(6, 0.0f);
    CompileAndRun("nn_relu", func, {Input(x_data), Output(out)});
    ExpectNear(out, {0, 0, 0, 1, 2, 0});
}

// 验证 Conv2D 的布局、步幅和卷积数值结果。
void TestConv2D() {
    kxc::Var data("data", kxc::TensorType({1, 1, 3, 3}, "float32"));
    kxc::Var weight("weight", kxc::TensorType({1, 1, 2, 2}, "float32"));
    auto attrs = kxc::relay::Conv2DAttrs::Create(
        {1, 1}, {0, 0, 0, 0}, {1, 1}, 1, 1, {2, 2}, "NCHW", "OIHW", "", "");
    kxc::Call call(kxc::relay::Op::Get("nn_conv2d"), {data, weight}, attrs);
    kxc::Function func({data, weight}, call);

    std::vector<float> data_buf = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<float> weight_buf = {1, 0, 0, 1};
    std::vector<float> out(4, 0.0f);
    CompileAndRun("nn_conv2d", func,
                  {Input(data_buf), Input(weight_buf), Output(out)});
    ExpectNear(out, {6, 8, 12, 14});
}

// 验证 MaxPool2D 的窗口归约结果。
void TestMaxPool2D() {
    kxc::Var data("data", kxc::TensorType({1, 1, 4, 4}, "float32"));
    auto attrs = kxc::relay::MaxPool2DAttrs::Create(
        {2, 2}, {0, 0, 0, 0}, {1, 1}, {2, 2}, "NCHW", false);
    kxc::Call call(kxc::relay::Op::Get("nn_max_pool2d"), {data}, attrs);
    kxc::Function func({data}, call);

    std::vector<float> data_buf = {1, 2, 3, 4, 5, 6, 7, 8,
                                   9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<float> out(4, 0.0f);
    CompileAndRun("nn_max_pool2d", func, {Input(data_buf), Output(out)});
    ExpectNear(out, {6, 8, 14, 16});
}

void TestDilatedAsymmetricMaxPool2D() {
    kxc::Var data("data", kxc::TensorType({1, 1, 4, 5}, "float32"));
    auto attrs = kxc::relay::MaxPool2DAttrs::Create(
        {1, 2}, {1, 0, 0, 1}, {2, 2}, {2, 2}, "NCHW", false);
    kxc::Call call(kxc::relay::Op::Get("nn_max_pool2d"), {data}, attrs);
    kxc::Function func({data}, call);

    std::vector<float> data_buf = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                   11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    std::vector<float> out(6, 0.0f);
    CompileAndRun("nn_max_pool2d_dilated_asymmetric", func,
                  {Input(data_buf), Output(out)});
    ExpectNear(out, {8, 10, 13, 15, 18, 20});
}

// 验证 AvgPool2D 的窗口平均结果。
void TestAvgPool2D() {
    kxc::Var data("data", kxc::TensorType({1, 1, 4, 4}, "float32"));
    auto attrs = kxc::relay::MaxPool2DAttrs::Create(
        {2, 2}, {0, 0, 0, 0}, {1, 1}, {2, 2}, "NCHW", false);
    kxc::Call call(kxc::relay::Op::Get("nn_avg_pool2d"), {data}, attrs);
    kxc::Function func({data}, call);

    std::vector<float> data_buf = {1, 2, 3, 4, 5, 6, 7, 8,
                                   9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<float> out(4, 0.0f);
    CompileAndRun("nn_avg_pool2d", func, {Input(data_buf), Output(out)});
    ExpectNear(out, {3.5f, 5.5f, 11.5f, 13.5f});
}

// 验证 GlobalAvgPool2D 对空间维的归约结果。
void TestGlobalAvgPool2D() {
    kxc::Var data("data", kxc::TensorType({1, 2, 2, 2}, "float32"));
    kxc::Call call(kxc::relay::Op::Get("nn_global_avg_pool2d"), {data},
                   kxc::relay::GlobalAvgPool2DAttrs::Create());
    kxc::Function func({data}, call);

    std::vector<float> data_buf = {1, 2, 3, 4, 10, 20, 30, 40};
    std::vector<float> out(2, 0.0f);
    CompileAndRun("nn_global_avg_pool2d", func, {Input(data_buf), Output(out)});
    ExpectNear(out, {2.5f, 25.0f});
}

// 验证 Flatten 只改变逻辑 shape 而保持元素顺序。
void TestFlatten() {
    kxc::Var data("data", kxc::TensorType({1, 2, 2, 3}, "float32"));
    kxc::Call call(kxc::relay::Op::Get("nn_flatten"), {data},
                   kxc::relay::FlattenAttrs::Create(1));
    kxc::Function func({data}, call);

    std::vector<float> data_buf = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<float> out(12, 0.0f);
    CompileAndRun("nn_flatten", func, {Input(data_buf), Output(out)});
    ExpectNear(out, data_buf);
}

// 验证 Reshape 保持元素数量和线性顺序。
void TestReshape() {
    kxc::Var data("data", kxc::TensorType({2, 3}, "float32"));
    kxc::Call call(kxc::relay::Op::Get("reshape"), {data},
                   kxc::relay::ReshapeAttrs::Create({3, 2}));
    kxc::Function func({data}, call);

    std::vector<float> data_buf = {1, 2, 3, 4, 5, 6};
    std::vector<float> out(6, 0.0f);
    CompileAndRun("reshape", func, {Input(data_buf), Output(out)});
    ExpectNear(out, data_buf);
}

// 验证 Transpose 按指定轴重排数据。
void TestTranspose() {
    kxc::Var data("data", kxc::TensorType({2, 3}, "float32"));
    kxc::Call call(kxc::relay::Op::Get("transpose"), {data},
                   kxc::relay::TransposeAttrs::Create({1, 0}));
    kxc::Function func({data}, call);

    std::vector<float> data_buf = {1, 2, 3, 4, 5, 6};
    std::vector<float> out(6, 0.0f);
    CompileAndRun("transpose", func, {Input(data_buf), Output(out)});
    ExpectNear(out, {1, 4, 2, 5, 3, 6});
}

// 验证 binary concatenate 的元素顺序，以及空侧仍产生独立的 copy output。
void TestConcatenate() {
    kxc::Var lhs("lhs", kxc::TensorType({2, 2}, "float32"));
    kxc::Var rhs("rhs", kxc::TensorType({2, 1}, "float32"));
    kxc::Call call(kxc::relay::Op::Get("concatenate"), {lhs, rhs},
                   kxc::relay::ConcatenateAttrs::Create(-1));
    kxc::Function function({lhs, rhs}, call);
    std::vector<float> lhs_data = {1, 2, 3, 4};
    std::vector<float> rhs_data = {10, 20};
    std::vector<float> output(6, 0.0f);
    CompileAndRun("concatenate", function,
                  {Input(lhs_data), Input(rhs_data), Output(output)});
    ExpectNear(output, {1, 2, 10, 3, 4, 20});

    kxc::Var empty("empty", kxc::TensorType({2, 0}, "float32"));
    kxc::Var source("source", kxc::TensorType({2, 2}, "float32"));
    kxc::Call empty_side(kxc::relay::Op::Get("concatenate"), {empty, source},
                         kxc::relay::ConcatenateAttrs::Create(1));
    kxc::Function empty_function({empty, source}, empty_side);
    std::vector<float> empty_data;
    std::vector<float> source_data = {7, 8, 9, 10};
    std::vector<float> copied(4, 0.0f);
    CompileAndRun("concatenate_empty_side_fresh_copy", empty_function,
                  {Input(empty_data), Input(source_data), Output(copied)});
    ExpectNear(copied, source_data);

    kxc::Var bool_lhs("bool_lhs", kxc::TensorType({1, 2}, "bool"));
    kxc::Var bool_rhs("bool_rhs", kxc::TensorType({1, 1}, "bool"));
    kxc::Call bool_call(kxc::relay::Op::Get("concatenate"), {bool_lhs, bool_rhs},
                        kxc::relay::ConcatenateAttrs::Create(1));
    kxc::Function bool_function({bool_lhs, bool_rhs}, bool_call);
    const std::vector<uint8_t> bool_lhs_data = {1, 0};
    const std::vector<uint8_t> bool_rhs_data = {1};
    std::vector<uint8_t> bool_output(3, 0);
    CompileAndRun("concatenate_bool", bool_function,
                  {Input(bool_lhs_data), Input(bool_rhs_data), Output(bool_output)});
    Check(bool_output == std::vector<uint8_t>({1, 0, 1}),
          "bool concatenate must preserve byte-backed boolean values");
}

// 验证 Slice 的正步长、identity copy 和空输出数值路径。
void TestSlice() {
    kxc::Var data("data", kxc::TensorType({2, 3}, "float32"));
    kxc::Call call(kxc::relay::Op::Get("slice"), {data},
                   kxc::relay::SliceAttrs::Create({-2}, {100}, {-1}, {1}));
    kxc::Function function({data}, call);
    std::vector<float> source = {1, 2, 3, 4, 5, 6};
    std::vector<float> output(4, 0.0f);
    CompileAndRun("slice", function, {Input(source), Output(output)});
    ExpectNear(output, {2, 3, 5, 6});

    kxc::Call identity(kxc::relay::Op::Get("slice"), {data},
                       kxc::relay::SliceAttrs::Create({0}, {3}, {1}, {1}));
    kxc::Function identity_function({data}, identity);
    std::vector<float> copied(6, 0.0f);
    CompileAndRun("slice_identity_fresh_copy", identity_function,
                  {Input(source), Output(copied)});
    ExpectNear(copied, source);

    kxc::Call empty(kxc::relay::Op::Get("slice"), {data},
                    kxc::relay::SliceAttrs::Create({2}, {1}, {1}, {1}));
    kxc::Function empty_function({data}, empty);
    std::vector<float> empty_output;
    CompileAndRun("slice_empty", empty_function, {Input(source), Output(empty_output)});
}

// 验证 ReduceMean 的轴和 keepdims 语义。
void TestReduceMean() {
    kxc::Var data("data", kxc::TensorType({2, 3}, "float32"));
    kxc::Call call(kxc::relay::Op::Get("reduce_mean"), {data},
                   kxc::relay::ReduceMeanAttrs::Create({1}, 0));
    kxc::Function func({data}, call);

    std::vector<float> data_buf = {1, 2, 3, 4, 5, 6};
    std::vector<float> out(2, 0.0f);
    CompileAndRun("reduce_mean", func, {Input(data_buf), Output(out)});
    ExpectNear(out, {2.0f, 5.0f});
}

// 验证 Softmax 通过 max-subtraction 在极大正负 logits 下保持有限。
void TestSoftmax() {
    kxc::Var data("data", kxc::TensorType({2, 3}, "float32"));
    kxc::Call call(kxc::relay::Op::Get("softmax"), {data},
                   kxc::relay::SoftmaxAttrs::Create(-1));
    kxc::Function func({data}, call);

    std::vector<float> data_buf = {1000, 1001, 999, -1000, -999, -1001};
    std::vector<float> out(6, 0.0f);
    CompileAndRun("softmax", func, {Input(data_buf), Output(out)});

    std::vector<float> expected(6, 0.0f);
    for (size_t row = 0; row < 2; ++row) {
        const auto begin = data_buf.begin() + static_cast<std::ptrdiff_t>(row * 3);
        const float max_value = *std::max_element(begin, begin + 3);
        float denominator = 0.0f;
        for (size_t col = 0; col < 3; ++col) {
            denominator += std::exp(data_buf[row * 3 + col] - max_value);
        }
        for (size_t col = 0; col < 3; ++col) {
            expected[row * 3 + col] =
                std::exp(data_buf[row * 3 + col] - max_value) / denominator;
        }
    }
    ExpectNear(out, expected);
}

// 验证 Gather 的正负合法索引、越界零填充和 INT64_MIN 防溢出语义。
void TestGather() {
    kxc::Var data("data", kxc::TensorType({4}, "float32"));
    kxc::Var indices("indices", kxc::TensorType({5}, "int64"));
    kxc::Call call(kxc::relay::Op::Get("gather"), {data, indices},
                   kxc::relay::GatherAttrs::Create(0));
    kxc::Function func({data, indices}, call);

    std::vector<float> data_buf = {10, 20, 30, 40};
    std::vector<int64_t> indices_buf = {0, -1, 4, -5,
                                        std::numeric_limits<int64_t>::min()};
    std::vector<float> out(5, 0.0f);
    CompileAndRun("gather", func, {Input(data_buf), Input(indices_buf), Output(out)});
    ExpectNear(out, {10, 40, 0, 0, 0});

    kxc::Var empty_data("empty_data", kxc::TensorType({0}, "float32"));
    kxc::Var empty_indices("empty_indices", kxc::TensorType({3}, "int64"));
    kxc::Call empty_call(kxc::relay::Op::Get("gather"),
                         {empty_data, empty_indices},
                         kxc::relay::GatherAttrs::Create(0));
    kxc::Function empty_func({empty_data, empty_indices}, empty_call);
    std::vector<float> empty_data_buf;
    std::vector<int64_t> empty_indices_buf = {0, -1,
                                              std::numeric_limits<int64_t>::min()};
    std::vector<float> empty_out(3, 1.0f);
    CompileAndRun("gather_zero_axis_extent", empty_func,
                  {Input(empty_data_buf), Input(empty_indices_buf), Output(empty_out)});
    ExpectNear(empty_out, {0, 0, 0});
}

// 验证 ONNX Where 的 byte-backed bool condition、scalar/rank broadcast 和分支选择。
void TestWhere() {
    kxc::Var condition("condition", kxc::TensorType({2, 1}, "bool"));
    kxc::Var x("x", kxc::TensorType({}, "float32"));
    kxc::Var y("y", kxc::TensorType({1, 3}, "float32"));
    kxc::Call call(kxc::relay::Op::Get("where"), {condition, x, y});
    kxc::Function func({condition, x, y}, call);

    const std::vector<uint8_t> condition_data = {1, 0};
    const std::vector<float> x_data = {10};
    const std::vector<float> y_data = {1, 2, 3};
    std::vector<float> out(6, 0.0f);
    CompileAndRun("where", func,
                  {Input(condition_data), Input(x_data), Input(y_data), Output(out)});
    ExpectNear(out, {10, 10, 10, 1, 2, 3});

    kxc::runtime::NDArray constant_condition = kxc::runtime::NDArray::Empty(
        {2, 1}, kxc::runtime::DataTypeFromString("bool"), kxc::Device::CPU());
    constant_condition.CopyFromBytes(condition_data.data(), condition_data.size());
    kxc::Var bool_x("bool_x", kxc::TensorType({1, 3}, "bool"));
    kxc::Var bool_y("bool_y", kxc::TensorType({2, 1}, "bool"));
    kxc::Call bool_call(kxc::relay::Op::Get("where"),
                        {kxc::Constant(constant_condition), bool_x, bool_y});
    kxc::Function bool_func({bool_x, bool_y}, bool_call);
    const std::vector<uint8_t> bool_x_data = {1, 0, 1};
    const std::vector<uint8_t> bool_y_data = {0, 1};
    std::vector<uint8_t> bool_out(6, 0);
    CompileAndRun("where_bool_constant", bool_func,
                  {Input(bool_x_data), Input(bool_y_data), Output(bool_out)});
    Check(bool_out == std::vector<uint8_t>({1, 0, 1, 1, 1, 1}),
          "Where bool Constant/branch result mismatch");
}

// 验证 Equal 的 bool 输出、int32/int64/float32 数值语义与合法广播。
// 首版子集：同 dtype int32/int64/float32；NaN 按数值相等语义（NaN != NaN），不能用位相等。
void TestEqual() {
    // M5 文档示例：A=[1,2,3]、B=[1,0,3] → [true,false,true]。
    kxc::Var a("a", kxc::TensorType({3}, "int32"));
    kxc::Var b("b", kxc::TensorType({3}, "int32"));
    kxc::Call call(kxc::relay::Op::Get("equal"), {a, b});
    kxc::Function func({a, b}, call);

    const std::vector<int32_t> a_data = {1, 2, 3};
    const std::vector<int32_t> b_data = {1, 0, 3};
    std::vector<uint8_t> out(3, 0);
    CompileAndRun("equal", func, {Input(a_data), Input(b_data), Output(out)});
    Check(out == std::vector<uint8_t>({1, 0, 1}), "equal int32 doc example mismatch");

    // int64 的 trailing-axis 广播。
    kxc::Var lhs("lhs", kxc::TensorType({2, 3}, "int64"));
    kxc::Var rhs("rhs", kxc::TensorType({3}, "int64"));
    kxc::Call broadcast_call(kxc::relay::Op::Get("equal"), {lhs, rhs});
    kxc::Function broadcast_func({lhs, rhs}, broadcast_call);
    const std::vector<int64_t> lhs_data = {1, 2, 3, 4, 5, 6};
    const std::vector<int64_t> rhs_data = {4, 5, 6};
    std::vector<uint8_t> broadcast_out(6, 1);
    CompileAndRun("equal_int64_broadcast", broadcast_func,
                  {Input(lhs_data), Input(rhs_data), Output(broadcast_out)});
    Check(broadcast_out == std::vector<uint8_t>({0, 0, 0, 1, 1, 1}),
          "equal int64 broadcast mismatch");

    // float32 数值相等：+0 与 -0 相等；NaN 与任何值（含自身）都不相等。
    kxc::Var f_lhs("f_lhs", kxc::TensorType({2, 3}, "float32"));
    kxc::Var f_rhs("f_rhs", kxc::TensorType({3}, "float32"));
    kxc::Call float_call(kxc::relay::Op::Get("equal"), {f_lhs, f_rhs});
    kxc::Function float_func({f_lhs, f_rhs}, float_call);
    const float nan_value = std::numeric_limits<float>::quiet_NaN();
    const std::vector<float> float_lhs_data = {0.0f, -0.0f, nan_value, 1.0f, 0.0f, 5.0f};
    const std::vector<float> float_rhs_data = {-0.0f, 0.0f, nan_value};
    std::vector<uint8_t> float_out(6, 0);
    CompileAndRun("equal_float32_special_values", float_func,
                  {Input(float_lhs_data), Input(float_rhs_data), Output(float_out)});
    // 实际行为：+0 == -0 → true；NaN == NaN → false；数值 == NaN → false。
    Check(float_out == std::vector<uint8_t>({1, 1, 0, 0, 1, 0}),
          "equal float32 NaN and signed-zero semantics mismatch");

    // 标量（rank-0）与空张量的真实 buffer 读写。
    kxc::Var scalar("scalar", kxc::TensorType({}, "float32"));
    kxc::Var row("row", kxc::TensorType({1, 3}, "float32"));
    kxc::Call scalar_call(kxc::relay::Op::Get("equal"), {scalar, row});
    kxc::Function scalar_func({scalar, row}, scalar_call);
    const std::vector<float> scalar_data = {7.0f};
    const std::vector<float> row_data = {7.0f, 8.0f, 7.0f};
    std::vector<uint8_t> scalar_out(3, 0);
    CompileAndRun("equal_scalar_broadcast", scalar_func,
                  {Input(scalar_data), Input(row_data), Output(scalar_out)});
    Check(scalar_out == std::vector<uint8_t>({1, 0, 1}),
          "equal rank-0 broadcast mismatch");

    kxc::Var empty_a("empty_a", kxc::TensorType({0}, "int32"));
    kxc::Var empty_b("empty_b", kxc::TensorType({0}, "int32"));
    kxc::Call empty_call(kxc::relay::Op::Get("equal"), {empty_a, empty_b});
    kxc::Function empty_func({empty_a, empty_b}, empty_call);
    std::vector<uint8_t> empty_out;
    CompileAndRun("equal_empty", empty_func,
                  {Input(std::vector<int32_t>{}), Input(std::vector<int32_t>{}),
                   Output(empty_out)});
    Check(empty_out.empty(), "equal on empty tensors must produce an empty bool buffer");
}

// 验证 Equal 的 bool 结果直接作为 Where 条件的真实 buffer 消费链。
void TestEqualWhereComposition() {
    kxc::Var a("a", kxc::TensorType({2, 3}, "float32"));
    kxc::Var b("b", kxc::TensorType({3}, "float32"));
    kxc::Call equal_call(kxc::relay::Op::Get("equal"), {a, b});
    kxc::Var x("x", kxc::TensorType({1, 3}, "float32"));
    kxc::Var y("y", kxc::TensorType({}, "float32"));
    kxc::Call where_call(kxc::relay::Op::Get("where"), {equal_call, x, y});
    kxc::Function func({a, b, x, y}, where_call);

    const std::vector<float> a_data = {1, 2, 3, 4, 5, 6};
    const std::vector<float> b_data = {1, 0, 3};
    const std::vector<float> x_data = {10, 20, 30};
    const std::vector<float> y_data = {0};
    std::vector<float> out(6, 0.0f);
    CompileAndRun("equal_where_composition", func,
                  {Input(a_data), Input(b_data), Input(x_data), Input(y_data),
                   Output(out)});
    ExpectNear(out, {10, 0, 30, 0, 0, 0});
}

// 验证 LayerNorm 的常量行、微小方差行和仿射参数语义。
void TestLayerNorm() {
    kxc::Var data("data", kxc::TensorType({2, 4}, "float32"));
    kxc::Var scale("scale", kxc::TensorType({4}, "float32"));
    kxc::Var bias("bias", kxc::TensorType({4}, "float32"));
    constexpr float epsilon = 1e-5f;
    kxc::Call call(kxc::relay::Op::Get("nn_layer_norm"), {data, scale, bias},
                   kxc::relay::LayerNormAttrs::Create(-1, epsilon, "float64"));
    kxc::Function func({data, scale, bias}, call);

    const std::vector<float> data_buf = {
        5.0f, 5.0f, 5.0f, 5.0f,
        1.0f, 1.0001f, 0.9999f, 1.0002f,
    };
    const std::vector<float> scale_buf = {1.5f, -2.0f, 0.5f, 3.0f};
    const std::vector<float> bias_buf = {0.25f, -0.5f, 1.0f, 2.0f};
    std::vector<float> out(8, 0.0f);
    CompileAndRun("nn_layer_norm", func,
                  {Input(data_buf), Input(scale_buf), Input(bias_buf), Output(out)});

    ExpectNear(out, LayerNormReference(
                        data_buf, 2, 4, scale_buf, bias_buf, epsilon),
               2e-4f);

    kxc::Var suffix_data("suffix_data", kxc::TensorType({2, 2, 2}, "float32"));
    kxc::Var suffix_scale("suffix_scale", kxc::TensorType({2, 2}, "float32"));
    kxc::Var suffix_bias("suffix_bias", kxc::TensorType({2, 2}, "float32"));
    kxc::Call suffix_call(
        kxc::relay::Op::Get("nn_layer_norm"),
        {suffix_data, suffix_scale, suffix_bias},
        kxc::relay::LayerNormAttrs::Create(1, epsilon, "float64"));
    kxc::Function suffix_func({suffix_data, suffix_scale, suffix_bias}, suffix_call);
    const std::vector<float> suffix_data_buf = {1, 2, 3, 4, 2, 4, 6, 8};
    const std::vector<float> suffix_scale_buf = {1, -2, 0.5f, 3};
    const std::vector<float> suffix_bias_buf = {0.25f, -0.5f, 1, -1};
    std::vector<float> suffix_out(8, 0.0f);
    CompileAndRun("nn_layer_norm_suffix", suffix_func,
                  {Input(suffix_data_buf), Input(suffix_scale_buf),
                   Input(suffix_bias_buf), Output(suffix_out)});
    ExpectNear(suffix_out, LayerNormReference(
                               suffix_data_buf, 2, 4, suffix_scale_buf,
                               suffix_bias_buf, epsilon),
               2e-4f);

    kxc::Var extreme_data("extreme_data", kxc::TensorType({1, 2}, "float32"));
    kxc::Var extreme_scale("extreme_scale", kxc::TensorType({2}, "float32"));
    kxc::Var extreme_bias("extreme_bias", kxc::TensorType({2}, "float32"));
    kxc::Function extreme_function(
        {extreme_data, extreme_scale, extreme_bias},
        kxc::Call(kxc::relay::Op::Get("nn_layer_norm"),
                  {extreme_data, extreme_scale, extreme_bias},
                  kxc::relay::LayerNormAttrs::Create(-1, epsilon, "float64")));
    const float maximum = std::numeric_limits<float>::max();
    const std::vector<float> extreme_values = {maximum, maximum};
    const std::vector<float> extreme_scale_values = {1.0f, 1.0f};
    const std::vector<float> extreme_bias_values = {0.25f, -0.5f};
    std::vector<float> extreme_out(2, 0.0f);
    CompileAndRun("nn_layer_norm_float_max_constant", extreme_function,
                  {Input(extreme_values), Input(extreme_scale_values),
                   Input(extreme_bias_values), Output(extreme_out)});
    ExpectNear(extreme_out, extreme_bias_values, 0.0f);

    const float offset = std::numeric_limits<float>::max() / 8.0f;
    std::vector<float> offset_values(4, offset);
    for (size_t index = 1; index < offset_values.size(); ++index) {
        offset_values[index] = std::nextafter(
            offset_values[index - 1], std::numeric_limits<float>::infinity());
    }
    kxc::Var offset_data("offset_data", kxc::TensorType({1, 4}, "float32"));
    kxc::Var offset_scale("offset_scale", kxc::TensorType({4}, "float32"));
    kxc::Var offset_bias("offset_bias", kxc::TensorType({4}, "float32"));
    kxc::Function offset_function(
        {offset_data, offset_scale, offset_bias},
        kxc::Call(kxc::relay::Op::Get("nn_layer_norm"),
                  {offset_data, offset_scale, offset_bias},
                  kxc::relay::LayerNormAttrs::Create(-1, epsilon, "float64")));
    const std::vector<float> offset_scale_values(4, 1.0f);
    const std::vector<float> offset_bias_values(4, 0.0f);
    std::vector<float> offset_out(4, 0.0f);
    CompileAndRun("nn_layer_norm_huge_offset", offset_function,
                  {Input(offset_values), Input(offset_scale_values),
                   Input(offset_bias_values), Output(offset_out)});
    ExpectNear(offset_out, LayerNormReference(
                               offset_values, 1, 4, offset_scale_values,
                               offset_bias_values, epsilon),
               2e-4f);

    kxc::Var axis_zero_data("axis_zero_data", kxc::TensorType({2, 2}, "float32"));
    kxc::Var axis_zero_scale("axis_zero_scale", kxc::TensorType({2, 2}, "float32"));
    kxc::Var axis_zero_bias("axis_zero_bias", kxc::TensorType({2, 2}, "float32"));
    kxc::Function axis_zero_function(
        {axis_zero_data, axis_zero_scale, axis_zero_bias},
        kxc::Call(kxc::relay::Op::Get("nn_layer_norm"),
                  {axis_zero_data, axis_zero_scale, axis_zero_bias},
                  kxc::relay::LayerNormAttrs::Create(0, epsilon, "float64")));
    const std::vector<float> axis_zero_values = {1, 2, 4, 8};
    const std::vector<float> axis_zero_scale_values = {1, 2, 3, 4};
    const std::vector<float> axis_zero_bias_values = {0.5f, 1, 1.5f, 2};
    std::vector<float> axis_zero_out(4, 0.0f);
    CompileAndRun("nn_layer_norm_axis_zero", axis_zero_function,
                  {Input(axis_zero_values), Input(axis_zero_scale_values),
                   Input(axis_zero_bias_values), Output(axis_zero_out)});
    ExpectNear(axis_zero_out, LayerNormReference(
                                  axis_zero_values, 1, 4,
                                  axis_zero_scale_values, axis_zero_bias_values,
                                  epsilon),
               2e-4f);

    kxc::Var empty_prefix_data("empty_prefix_data",
                               kxc::TensorType({0, 2}, "float32"));
    kxc::Var empty_prefix_scale("empty_prefix_scale",
                                kxc::TensorType({2}, "float32"));
    kxc::Var empty_prefix_bias("empty_prefix_bias",
                               kxc::TensorType({2}, "float32"));
    kxc::Function empty_prefix_function(
        {empty_prefix_data, empty_prefix_scale, empty_prefix_bias},
        kxc::Call(kxc::relay::Op::Get("nn_layer_norm"),
                  {empty_prefix_data, empty_prefix_scale, empty_prefix_bias},
                  kxc::relay::LayerNormAttrs::Create(1, epsilon, "float64")));
    std::vector<float> empty_prefix_out;
    CompileAndRun("nn_layer_norm_empty_prefix", empty_prefix_function,
                  {Input(std::vector<float>{}), Input(extreme_scale_values),
                   Input(std::vector<float>{0.0f, 0.0f}),
                   Output(empty_prefix_out)});
}

// 验证 Cast 的目标 dtype 与数值转换。
void TestCast() {
    kxc::Var data("data", kxc::TensorType({4}, "float32"));
    kxc::Call call(kxc::relay::Op::Get("cast"), {data}, kxc::relay::CastAttrs::Create(1));
    kxc::Function func({data}, call);

    std::vector<float> data_buf = {1.9f, -2.2f, 3.0f, 4.8f};
    std::vector<int32_t> out(4, 0);
    CompileAndRun("cast", func, {Input(data_buf), Output(out)});
    ExpectEqual(out, {1, -2, 3, 4});
}

// 验证多个 Add 串联时中间 NDArray 的执行结果。
void TestModelAddChain() {
    kxc::Var x("x", kxc::TensorType({4}, "float32"));
    kxc::Var y("y", kxc::TensorType({4}, "float32"));
    kxc::Call first(kxc::relay::Op::Get("add"), {x, y});
    kxc::Call second(kxc::relay::Op::Get("add"), {first, y});
    kxc::Function func({x, y}, second);

    std::vector<float> x_data = {1, 2, 3, 4};
    std::vector<float> y_data = {10, 20, 30, 40};
    std::vector<float> out(4, 0.0f);
    CompileAndRun("model_add_chain", func,
                  {Input(x_data), Input(y_data), Output(out)});
    ExpectNear(out, {21, 42, 63, 84});
}

// 验证 Dense/激活组成的 MLP 子图端到端数值结果。
void TestModelMLP() {
    kxc::Var x("x", kxc::TensorType({1, 2}, "float32"));
    kxc::Var w1("w1", kxc::TensorType({3, 2}, "float32"));
    kxc::Var w2("w2", kxc::TensorType({2, 3}, "float32"));
    kxc::Call hidden(kxc::relay::Op::Get("nn_dense"), {x, w1},
                     kxc::relay::DenseAttrs::Create(3, ""));
    kxc::Call relu(kxc::relay::Op::Get("nn_relu"), {hidden},
                   kxc::relay::ReluAttrs::Create());
    kxc::Call out_call(kxc::relay::Op::Get("nn_dense"), {relu, w2},
                       kxc::relay::DenseAttrs::Create(2, ""));
    kxc::Function func({x, w1, w2}, out_call);

    std::vector<float> x_data = {1, 2};
    std::vector<float> w1_data = {1, 0, 0, 1, 1, 1};
    std::vector<float> w2_data = {1, 1, 1, 1, 0, -1};
    std::vector<float> out(2, 0.0f);
    CompileAndRun("model_mlp", func,
                  {Input(x_data), Input(w1_data), Input(w2_data), Output(out)});
    ExpectNear(out, {6, -2});
}

// 验证卷积、池化和分类层组成的 CNN 子图端到端结果。
void TestModelCNN() {
    kxc::Var data("data", kxc::TensorType({1, 1, 3, 3}, "float32"));
    kxc::Var conv_weight("conv_weight", kxc::TensorType({1, 1, 2, 2}, "float32"));
    kxc::Var dense_weight("dense_weight", kxc::TensorType({2, 1}, "float32"));
    auto conv_attrs = kxc::relay::Conv2DAttrs::Create(
        {1, 1}, {0, 0, 0, 0}, {1, 1}, 1, 1, {2, 2}, "NCHW", "OIHW", "", "");
    kxc::Call conv(kxc::relay::Op::Get("nn_conv2d"), {data, conv_weight}, conv_attrs);
    kxc::Call relu(kxc::relay::Op::Get("nn_relu"), {conv},
                   kxc::relay::ReluAttrs::Create());
    kxc::Call pool(kxc::relay::Op::Get("nn_global_avg_pool2d"), {relu},
                   kxc::relay::GlobalAvgPool2DAttrs::Create());
    kxc::Call flatten(kxc::relay::Op::Get("nn_flatten"), {pool},
                      kxc::relay::FlattenAttrs::Create(1));
    kxc::Call logits(kxc::relay::Op::Get("nn_dense"), {flatten, dense_weight},
                     kxc::relay::DenseAttrs::Create(2, ""));
    kxc::Function func({data, conv_weight, dense_weight}, logits);

    std::vector<float> data_buf = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<float> conv_weight_buf = {1, 0, 0, 1};
    std::vector<float> dense_weight_buf = {1, 2};
    std::vector<float> out(2, 0.0f);
    CompileAndRun("model_cnn", func,
                  {Input(data_buf), Input(conv_weight_buf), Input(dense_weight_buf),
                   Output(out)});
    ExpectNear(out, {10, 20});
}

// 最小 exact-static Transformer operator slice：embedding、norm、select、slice/concat、attention。
// 该 fixture 不包含 KV cache、dynamic batching 或动态 shape 语义。
void TestModelExactTransformerOperatorSlice() {
    kxc::Var embedding_table("embedding_table", kxc::TensorType({4, 2}, "float32"));
    kxc::Var token_ids("token_ids", kxc::TensorType({2}, "int64"));
    kxc::Var condition("condition", kxc::TensorType({2, 1}, "bool"));
    kxc::Var fallback("fallback", kxc::TensorType({1, 2}, "float32"));
    kxc::Var scale("scale", kxc::TensorType({2}, "float32"));
    kxc::Var bias("bias", kxc::TensorType({2}, "float32"));

    kxc::Call embedded(kxc::relay::Op::Get("gather"), {embedding_table, token_ids},
                       kxc::relay::GatherAttrs::Create(0));
    constexpr float epsilon = 1e-5f;
    kxc::Call normalized(kxc::relay::Op::Get("nn_layer_norm"), {embedded, scale, bias},
                         kxc::relay::LayerNormAttrs::Create(-1, epsilon, "float64"));
    kxc::Call selected(kxc::relay::Op::Get("where"),
                       {condition, normalized, fallback});
    kxc::Call prefix(kxc::relay::Op::Get("slice"), {selected},
                     kxc::relay::SliceAttrs::Create({0}, {1}, {0}, {1}));
    kxc::Call sequence(kxc::relay::Op::Get("concatenate"), {prefix, selected},
                       kxc::relay::ConcatenateAttrs::Create(0));
    kxc::Call keys(kxc::relay::Op::Get("transpose"), {sequence},
                   kxc::relay::TransposeAttrs::Create({1, 0}));
    kxc::Call scores(kxc::relay::Op::Get("matmul"), {sequence, keys});
    kxc::Call weights(kxc::relay::Op::Get("softmax"), {scores},
                      kxc::relay::SoftmaxAttrs::Create(-1));
    kxc::Call context(kxc::relay::Op::Get("matmul"), {weights, sequence});
    kxc::Function function(
        {embedding_table, token_ids, condition, fallback, scale, bias}, context);

    const std::vector<float> table_data = {1, 3, 2, 2, 4, 0, 0, 4};
    const std::vector<int64_t> token_data = {0, 2};
    const std::vector<uint8_t> condition_data = {1, 0};
    const std::vector<float> fallback_data = {0, 0};
    const std::vector<float> scale_data = {1, 1};
    const std::vector<float> bias_data = {0, 0};
    std::vector<float> out(6, 0.0f);
    CompileAndRun(
        "model_exact_transformer_operator_slice", function,
        {Input(table_data), Input(token_data), Input(condition_data),
         Input(fallback_data), Input(scale_data), Input(bias_data), Output(out)});

    const float a = 1.0f / std::sqrt(1.0f + epsilon);
    const std::vector<float> sequence_data = {-a, a, -a, a, 0, 0};
    std::vector<float> expected(6, 0.0f);
    for (size_t row = 0; row < 3; ++row) {
        float row_max = -std::numeric_limits<float>::infinity();
        float scores_data[3]{};
        for (size_t column = 0; column < 3; ++column) {
            scores_data[column] =
                sequence_data[row * 2] * sequence_data[column * 2] +
                sequence_data[row * 2 + 1] * sequence_data[column * 2 + 1];
            row_max = std::max(row_max, scores_data[column]);
        }
        float denominator = 0.0f;
        float probabilities[3]{};
        for (size_t column = 0; column < 3; ++column) {
            probabilities[column] = std::exp(scores_data[column] - row_max);
            denominator += probabilities[column];
        }
        for (size_t column = 0; column < 3; ++column) {
            probabilities[column] /= denominator;
            expected[row * 2] += probabilities[column] * sequence_data[column * 2];
            expected[row * 2 + 1] +=
                probabilities[column] * sequence_data[column * 2 + 1];
        }
    }
    ExpectNear(out, expected, 3e-4f);
}

// 精确静态 causal prefill attention：batched MatMul、有限加性 mask、稳定 softmax、batched MatMul。
void TestModelPrefillExactAttention() {
    kxc::Var query("query", kxc::TensorType({1, 2, 2}, "float32"));
    kxc::Var key_transposed("key_transposed", kxc::TensorType({1, 2, 2}, "float32"));
    kxc::Var finite_causal_mask("finite_causal_mask", kxc::TensorType({1, 2, 2}, "float32"));
    kxc::Var value("value", kxc::TensorType({1, 2, 2}, "float32"));
    kxc::Call scores(kxc::relay::Op::Get("matmul"), {query, key_transposed});
    kxc::Call masked_scores(kxc::relay::Op::Get("add"), {scores, finite_causal_mask});
    kxc::Call weights(kxc::relay::Op::Get("softmax"), {masked_scores},
                      kxc::relay::SoftmaxAttrs::Create(-1));
    kxc::Call context(kxc::relay::Op::Get("matmul"), {weights, value});
    kxc::Function func({query, key_transposed, finite_causal_mask, value}, context);

    const std::vector<float> query_data = {1, 0, 0, 1};
    const std::vector<float> key_transposed_data = {1, 0, 0, 1};
    const std::vector<float> mask_data = {0, -10000, 0, 0};
    const std::vector<float> value_data = {1, 2, 3, 4};
    std::vector<float> out(4, 0.0f);
    CompileAndRun("model_prefill_exact_attention", func,
                  {Input(query_data), Input(key_transposed_data), Input(mask_data),
                   Input(value_data), Output(out)});
    const float p = std::exp(0.0f) / (std::exp(0.0f) + std::exp(1.0f));
    ExpectNear(out, {1, 2, p * 1 + (1 - p) * 3, p * 2 + (1 - p) * 4});
}

// Exact-static decode attention over externally supplied K/V; this is not a KV-cache update or lifetime test.
void TestModelDecodeExternalKV() {
    kxc::Var query("query", kxc::TensorType({1, 1, 2}, "float32"));
    kxc::Var external_key_transposed("external_key_transposed",
                                    kxc::TensorType({1, 2, 3}, "float32"));
    kxc::Var external_value("external_value", kxc::TensorType({1, 3, 2}, "float32"));
    kxc::Call scores(kxc::relay::Op::Get("matmul"), {query, external_key_transposed});
    kxc::Call weights(kxc::relay::Op::Get("softmax"), {scores},
                      kxc::relay::SoftmaxAttrs::Create(-1));
    kxc::Call context(kxc::relay::Op::Get("matmul"), {weights, external_value});
    kxc::Function func({query, external_key_transposed, external_value}, context);

    const std::vector<float> query_data = {1, 0};
    const std::vector<float> external_key_data = {1, 0, -1, 0, 1, 0};
    const std::vector<float> external_value_data = {1, 2, 3, 4, 5, 6};
    std::vector<float> out(2, 0.0f);
    CompileAndRun("model_decode_external_kv", func,
                  {Input(query_data), Input(external_key_data), Input(external_value_data),
                   Output(out)});
    const float denominator = std::exp(1.0f) + std::exp(0.0f) + std::exp(-1.0f);
    const float w0 = std::exp(1.0f) / denominator;
    const float w1 = 1.0f / denominator;
    const float w2 = std::exp(-1.0f) / denominator;
    ExpectNear(out, {w0 + 3 * w1 + 5 * w2, 2 * w0 + 4 * w1 + 6 * w2});
}

}  // namespace

// 运行全部算子及模型级 LLVM 数值测试。
int main() {
    const std::vector<std::pair<std::string, void (*)()>> tests = {
        {"add", TestAdd},
        {"subtract", TestSubtract},
        {"mul", TestMul},
        {"divide", TestDivide},
        {"sqrt", TestSqrt},
        {"matmul", TestMatmul},
        {"batched_matmul", TestBatchedMatmul},
        {"nn_dense", TestDense},
        {"nn_gemm", TestGemm},
        {"nn_relu", TestRelu},
        {"nn_conv2d", TestConv2D},
        {"nn_max_pool2d", TestMaxPool2D},
        {"nn_max_pool2d_dilated_asymmetric", TestDilatedAsymmetricMaxPool2D},
        {"nn_avg_pool2d", TestAvgPool2D},
        {"nn_global_avg_pool2d", TestGlobalAvgPool2D},
        {"nn_flatten", TestFlatten},
        {"reshape", TestReshape},
        {"transpose", TestTranspose},
        {"concatenate", TestConcatenate},
        {"slice", TestSlice},
        {"reduce_mean", TestReduceMean},
        {"softmax", TestSoftmax},
        {"gather", TestGather},
        {"where", TestWhere},
        {"equal", TestEqual},
        {"equal_where_composition", TestEqualWhereComposition},
        {"nn_layer_norm", TestLayerNorm},
        {"cast", TestCast},
        {"model_add_chain", TestModelAddChain},
        {"model_mlp", TestModelMLP},
        {"model_cnn", TestModelCNN},
        {"model_exact_transformer_operator_slice", TestModelExactTransformerOperatorSlice},
        {"model_prefill_exact_attention", TestModelPrefillExactAttention},
        {"model_decode_external_kv", TestModelDecodeExternalKV},
    };

    for (const auto& test : tests) {
        try {
            test.second();
        } catch (const std::exception& e) {
            std::cerr << "[FAIL] " << test.first << ": " << e.what() << "\n";
            return 1;
        }
        std::cout << "[PASS] " << test.first << "\n";
    }

    std::cout << "All MVP op LLVM numeric tests passed.\n";
    return 0;
}
