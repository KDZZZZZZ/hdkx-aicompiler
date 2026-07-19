/*! \file test/op_numeric_llvm_test.cpp
 * \brief 验证 Relay 算子在 LLVM 后端上的数值正确性。
 */

#include "api/compiler.h"
#include "relay/op.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
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
        const float diff = std::fabs(actual[i] - expected[i]);
        if (diff > tolerance) {
            throw std::runtime_error("value mismatch at " + std::to_string(i) +
                                     ": actual=" + std::to_string(actual[i]) +
                                     ", expected=" + std::to_string(expected[i]));
        }
    }
}

// 精确比较整数结果向量。
void ExpectEqual(const std::vector<int32_t>& actual, const std::vector<int32_t>& expected) {
    Check(actual == expected, "integer result mismatch");
}

// 编译 Relay 函数，并用显式 CPU NDArray 输入输出执行 LLVM 内核。
void CompileAndRun(const std::string& op_name, kxc::Function func,
                   const std::vector<void*>& packed_args) {
#if KXC_USE_LLVM
    auto config = kxc::api::CompileConfig::JIT(kxc::BuildTarget(kxc::Device::CPU()));
    config->opt_level = 0;
    auto module = kxc::api::Compiler::Compile(func, config);
    Check(module.IsReady(), op_name + " LLVM module should be ready");
    module.Run(packed_args);
#else
    (void)op_name;
    (void)func;
    (void)packed_args;
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
    CompileAndRun("add", func, {x_data.data(), y_data.data(), out.data()});
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
    CompileAndRun("subtract", func, {x_data.data(), y_data.data(), out.data()});
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
    CompileAndRun("mul", func, {x_data.data(), y_data.data(), out.data()});
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
    CompileAndRun("divide", func, {x_data.data(), y_data.data(), out.data()});
    ExpectNear(out, {0.1f, 0.1f, 0.1f, 0.4f, 0.25f, 0.2f});
}

// 验证逐元素 Sqrt 的 LLVM 数值结果。
void TestSqrt() {
    kxc::Var x("x", kxc::TensorType({4}, "float32"));
    kxc::Call call(kxc::relay::Op::Get("sqrt"), {x});
    kxc::Function func({x}, call);

    std::vector<float> x_data = {1, 4, 9, 16};
    std::vector<float> out(4, 0.0f);
    CompileAndRun("sqrt", func, {x_data.data(), out.data()});
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
    CompileAndRun("matmul", func, {a_data.data(), b_data.data(), out.data()});
    ExpectNear(out, {22, 28, 49, 64});
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
    CompileAndRun("nn_dense", func, {data_buf.data(), weight_buf.data(), out.data()});
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
    CompileAndRun("nn_gemm", func, {a_data.data(), b_data.data(), c_data.data(), out.data()});
    ExpectNear(out, {11, 22, 33, 46, 14, 25, 36, 55});
}

// 验证 Relu 对正负输入的数值结果。
void TestRelu() {
    kxc::Var x("x", kxc::TensorType({6}, "float32"));
    kxc::Call call(kxc::relay::Op::Get("nn_relu"), {x}, kxc::relay::ReluAttrs::Create());
    kxc::Function func({x}, call);

    std::vector<float> x_data = {-2, -0.5f, 0, 1, 2, -3};
    std::vector<float> out(6, 0.0f);
    CompileAndRun("nn_relu", func, {x_data.data(), out.data()});
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
    CompileAndRun("nn_conv2d", func, {data_buf.data(), weight_buf.data(), out.data()});
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
    CompileAndRun("nn_max_pool2d", func, {data_buf.data(), out.data()});
    ExpectNear(out, {6, 8, 14, 16});
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
    CompileAndRun("nn_avg_pool2d", func, {data_buf.data(), out.data()});
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
    CompileAndRun("nn_global_avg_pool2d", func, {data_buf.data(), out.data()});
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
    CompileAndRun("nn_flatten", func, {data_buf.data(), out.data()});
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
    CompileAndRun("reshape", func, {data_buf.data(), out.data()});
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
    CompileAndRun("transpose", func, {data_buf.data(), out.data()});
    ExpectNear(out, {1, 4, 2, 5, 3, 6});
}

// 验证 ReduceMean 的轴和 keepdims 语义。
void TestReduceMean() {
    kxc::Var data("data", kxc::TensorType({2, 3}, "float32"));
    kxc::Call call(kxc::relay::Op::Get("reduce_mean"), {data},
                   kxc::relay::ReduceMeanAttrs::Create({1}, 0));
    kxc::Function func({data}, call);

    std::vector<float> data_buf = {1, 2, 3, 4, 5, 6};
    std::vector<float> out(2, 0.0f);
    CompileAndRun("reduce_mean", func, {data_buf.data(), out.data()});
    ExpectNear(out, {2.0f, 5.0f});
}

// 验证 Softmax 的归一化轴和数值稳定性。
void TestSoftmax() {
    kxc::Var data("data", kxc::TensorType({2, 3}, "float32"));
    kxc::Call call(kxc::relay::Op::Get("softmax"), {data},
                   kxc::relay::SoftmaxAttrs::Create(1));
    kxc::Function func({data}, call);

    std::vector<float> data_buf = {1, 2, 3, 1, 3, 5};
    std::vector<float> out(6, 0.0f);
    CompileAndRun("softmax", func, {data_buf.data(), out.data()});

    std::vector<float> expected(6, 0.0f);
    for (size_t row = 0; row < 2; ++row) {
        float denom = 0.0f;
        for (size_t col = 0; col < 3; ++col) {
            denom += std::exp(data_buf[row * 3 + col]);
        }
        for (size_t col = 0; col < 3; ++col) {
            expected[row * 3 + col] = std::exp(data_buf[row * 3 + col]) / denom;
        }
    }
    ExpectNear(out, expected);
}

// 验证 Cast 的目标 dtype 与数值转换。
void TestCast() {
    kxc::Var data("data", kxc::TensorType({4}, "float32"));
    kxc::Call call(kxc::relay::Op::Get("cast"), {data}, kxc::relay::CastAttrs::Create(1));
    kxc::Function func({data}, call);

    std::vector<float> data_buf = {1.9f, -2.2f, 3.0f, 4.8f};
    std::vector<int32_t> out(4, 0);
    CompileAndRun("cast", func, {data_buf.data(), out.data()});
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
    CompileAndRun("model_add_chain", func, {x_data.data(), y_data.data(), out.data()});
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
    CompileAndRun("model_mlp", func, {x_data.data(), w1_data.data(), w2_data.data(), out.data()});
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
                  {data_buf.data(), conv_weight_buf.data(), dense_weight_buf.data(), out.data()});
    ExpectNear(out, {10, 20});
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
        {"nn_dense", TestDense},
        {"nn_gemm", TestGemm},
        {"nn_relu", TestRelu},
        {"nn_conv2d", TestConv2D},
        {"nn_max_pool2d", TestMaxPool2D},
        {"nn_avg_pool2d", TestAvgPool2D},
        {"nn_global_avg_pool2d", TestGlobalAvgPool2D},
        {"nn_flatten", TestFlatten},
        {"reshape", TestReshape},
        {"transpose", TestTranspose},
        {"reduce_mean", TestReduceMean},
        {"softmax", TestSoftmax},
        {"cast", TestCast},
        {"model_add_chain", TestModelAddChain},
        {"model_mlp", TestModelMLP},
        {"model_cnn", TestModelCNN},
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
