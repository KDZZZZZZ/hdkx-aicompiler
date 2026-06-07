/*! \file test/infer_type_test.cpp
 * \brief 测试 Relay 类型和 shape 推导。
 */

#include "relay/op.h"
#include "relay/transforms/infer_type.h"
#include "relay/transforms/lower.h"
#include "relay/transforms/pipeline.h"

#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace {

#define TEST_CHECK(cond, msg)                                                     \
    do {                                                                          \
        if (!(cond)) {                                                            \
            std::cerr << "[FAIL] " << __FUNCTION__ << ": " << (msg) << "\n";     \
            return false;                                                         \
        }                                                                         \
    } while (0)

bool ShapeEquals(const kxc::TensorTypeNode* type, const std::vector<int64_t>& shape) {
    if (!type || type->shape.size() != shape.size()) {
        return false;
    }
    for (size_t i = 0; i < shape.size(); ++i) {
        if (type->shape[i] != shape[i]) {
            return false;
        }
    }
    return true;
}

bool CheckTensor(const kxc::Type& type, const std::vector<int64_t>& shape,
                 const std::string& dtype) {
    const auto* tensor = type.As<kxc::TensorTypeNode>();
    return tensor && ShapeEquals(tensor, shape) && tensor->dtype == dtype;
}

bool ExpectThrow(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

bool TestElementwiseBroadcast() {
    kxc::Var x("x", kxc::TensorType({2, 3}, "float32"));
    kxc::Var y("y", kxc::TensorType({3}, "float32"));
    kxc::Call add(kxc::relay::Op::Get("add"), {x, y});
    kxc::Function func({x, y}, add);

    kxc::relay::InferTypePass(func);
    TEST_CHECK(CheckTensor(add.checked_type(), {2, 3}, "float32"),
               "add should infer broadcast output");

    kxc::Var bad("bad", kxc::TensorType({4}, "float32"));
    kxc::Call invalid(kxc::relay::Op::Get("add"), {x, bad});
    kxc::Function invalid_func({x, bad}, invalid);
    TEST_CHECK(ExpectThrow([&] { kxc::relay::InferTypePass(invalid_func); }),
               "incompatible broadcast should fail");
    return true;
}

bool TestMatrixAndDenseOps() {
    kxc::Var a("a", kxc::TensorType({2, 3}, "float32"));
    kxc::Var b("b", kxc::TensorType({3, 4}, "float32"));
    kxc::Call matmul(kxc::relay::Op::Get("matmul"), {a, b});
    kxc::Function matmul_func({a, b}, matmul);
    kxc::relay::InferTypePass(matmul_func);
    TEST_CHECK(CheckTensor(matmul.checked_type(), {2, 4}, "float32"),
               "matmul output shape mismatch");

    kxc::Var batched_a("batched_a", kxc::TensorType({2, 3, 4}, "float32"));
    kxc::Var batched_b("batched_b", kxc::TensorType({2, 4, 5}, "float32"));
    kxc::Call batched_matmul(kxc::relay::Op::Get("matmul"), {batched_a, batched_b});
    kxc::Function batched_matmul_func({batched_a, batched_b}, batched_matmul);
    TEST_CHECK(ExpectThrow([&] { kxc::relay::InferTypePass(batched_matmul_func); }),
               "batched matmul should fail until batched lowering exists");

    kxc::Var weight("weight", kxc::TensorType({5, 3}, "float32"));
    kxc::Call dense(kxc::relay::Op::Get("nn_dense"), {a, weight},
                    kxc::relay::DenseAttrs::Create(5, ""));
    kxc::Function dense_func({a, weight}, dense);
    kxc::relay::InferTypePass(dense_func);
    TEST_CHECK(CheckTensor(dense.checked_type(), {2, 5}, "float32"),
               "dense output shape mismatch");

    kxc::Var gemm_b("gemm_b", kxc::TensorType({4, 3}, "float32"));
    kxc::Var bias("bias", kxc::TensorType({4}, "float32"));
    kxc::Call gemm(kxc::relay::Op::Get("nn_gemm"), {a, gemm_b, bias},
                   kxc::relay::GemmAttrs::Create(1.0f, 1.0f, 0, 1));
    kxc::Function gemm_func({a, gemm_b, bias}, gemm);
    kxc::relay::InferTypePass(gemm_func);
    TEST_CHECK(CheckTensor(gemm.checked_type(), {2, 4}, "float32"),
               "gemm output shape mismatch");
    return true;
}

bool TestConvAndPoolOps() {
    kxc::Var data("data", kxc::TensorType({1, 3, 32, 32}, "float32"));
    kxc::Var weight("weight", kxc::TensorType({8, 3, 3, 3}, "float32"));
    auto conv_attrs = kxc::relay::Conv2DAttrs::Create(
        {2, 2}, {1, 1, 1, 1}, {1, 1}, 1, 8, {3, 3}, "NCHW", "OIHW", "", "");
    kxc::Call conv(kxc::relay::Op::Get("nn_conv2d"), {data, weight}, conv_attrs);
    kxc::Function conv_func({data, weight}, conv);
    kxc::relay::InferTypePass(conv_func);
    TEST_CHECK(CheckTensor(conv.checked_type(), {1, 8, 16, 16}, "float32"),
               "conv2d output shape mismatch");

    auto pool_attrs =
        kxc::relay::MaxPool2DAttrs::Create({2, 2}, {0, 0, 0, 0}, {1, 1}, {2, 2},
                                           "NCHW", false);
    kxc::Call pool(kxc::relay::Op::Get("nn_max_pool2d"), {conv}, pool_attrs);
    kxc::Call global_pool(kxc::relay::Op::Get("nn_global_avg_pool2d"), {conv},
                          kxc::relay::GlobalAvgPool2DAttrs::Create());
    kxc::Call flatten(kxc::relay::Op::Get("nn_flatten"), {global_pool},
                      kxc::relay::FlattenAttrs::Create(1));
    kxc::Function pool_func({data, weight}, kxc::Tuple({pool, flatten}));
    kxc::relay::InferTypePass(pool_func);
    TEST_CHECK(CheckTensor(pool.checked_type(), {1, 8, 8, 8}, "float32"),
               "pool2d output shape mismatch");
    TEST_CHECK(CheckTensor(flatten.checked_type(), {1, 8}, "float32"),
               "flatten after global pool shape mismatch");
    return true;
}

bool TestTransformAndReduceOps() {
    kxc::Var x("x", kxc::TensorType({2, 3, 4}, "float32"));
    kxc::Call reshape(kxc::relay::Op::Get("reshape"), {x},
                      kxc::relay::ReshapeAttrs::Create({0, -1}));
    kxc::Call transpose(kxc::relay::Op::Get("transpose"), {reshape},
                        kxc::relay::TransposeAttrs::Create({1, 0}));
    kxc::Call reduce(kxc::relay::Op::Get("reduce_mean"), {transpose},
                     kxc::relay::ReduceMeanAttrs::Create({1}, 0));
    kxc::Call cast(kxc::relay::Op::Get("cast"), {reduce},
                   kxc::relay::CastAttrs::Create(1));
    kxc::Function func({x}, cast);

    kxc::relay::InferTypePass(func);
    TEST_CHECK(CheckTensor(reshape.checked_type(), {2, 12}, "float32"),
               "reshape output shape mismatch");
    TEST_CHECK(CheckTensor(transpose.checked_type(), {12, 2}, "float32"),
               "transpose output shape mismatch");
    TEST_CHECK(CheckTensor(reduce.checked_type(), {12}, "float32"),
               "reduce_mean output shape mismatch");
    TEST_CHECK(CheckTensor(cast.checked_type(), {12}, "int32"),
               "cast output dtype mismatch");
    return true;
}

bool TestMvpElementwiseLowerToTIR() {
    kxc::Var x("x", kxc::TensorType({2, 3}, "float32"));
    kxc::Var y("y", kxc::TensorType({3}, "float32"));
    kxc::Call add(kxc::relay::Op::Get("add"), {x, y});
    kxc::Call subtract(kxc::relay::Op::Get("subtract"), {add, y});
    kxc::Call mul(kxc::relay::Op::Get("mul"), {subtract, y});
    kxc::Call divide(kxc::relay::Op::Get("divide"), {mul, y});
    kxc::Call sqrt(kxc::relay::Op::Get("sqrt"), {divide});
    kxc::Function func({x, y}, sqrt);

    kxc::tir::PrimFunc lowered = kxc::relay::LowerToTIR(func);
    TEST_CHECK(lowered.defined(), "elementwise MVP ops should lower to TIR");
    return true;
}

bool TestMvpMatrixLowerToTIR() {
    kxc::Var a("a", kxc::TensorType({2, 3}, "float32"));
    kxc::Var b("b", kxc::TensorType({3, 4}, "float32"));
    kxc::Call matmul(kxc::relay::Op::Get("matmul"), {a, b});
    kxc::Function matmul_func({a, b}, matmul);
    TEST_CHECK(kxc::relay::LowerToTIR(matmul_func).defined(), "matmul should lower to TIR");

    kxc::Var dense_weight("dense_weight", kxc::TensorType({5, 3}, "float32"));
    kxc::Call dense(kxc::relay::Op::Get("nn_dense"), {a, dense_weight},
                    kxc::relay::DenseAttrs::Create(5, ""));
    kxc::Function dense_func({a, dense_weight}, dense);
    TEST_CHECK(kxc::relay::LowerToTIR(dense_func).defined(), "nn_dense should lower to TIR");

    kxc::Var gemm_b("gemm_b", kxc::TensorType({4, 3}, "float32"));
    kxc::Var gemm_bias("gemm_bias", kxc::TensorType({4}, "float32"));
    kxc::Call gemm(kxc::relay::Op::Get("nn_gemm"), {a, gemm_b, gemm_bias},
                   kxc::relay::GemmAttrs::Create(1.0f, 1.0f, 0, 1));
    kxc::Function gemm_func({a, gemm_b, gemm_bias}, gemm);
    TEST_CHECK(kxc::relay::LowerToTIR(gemm_func).defined(), "nn_gemm should lower to TIR");
    return true;
}

bool TestMvpNNLowerToTIR() {
    kxc::Var data("data", kxc::TensorType({1, 3, 8, 8}, "float32"));
    kxc::Var weight("weight", kxc::TensorType({4, 3, 3, 3}, "float32"));
    auto conv_attrs = kxc::relay::Conv2DAttrs::Create(
        {1, 1}, {1, 1, 1, 1}, {1, 1}, 1, 4, {3, 3}, "NCHW", "OIHW", "", "");
    kxc::Call conv(kxc::relay::Op::Get("nn_conv2d"), {data, weight}, conv_attrs);
    kxc::Call relu(kxc::relay::Op::Get("nn_relu"), {conv}, kxc::relay::ReluAttrs::Create());
    auto pool_attrs =
        kxc::relay::MaxPool2DAttrs::Create({2, 2}, {0, 0, 0, 0}, {1, 1}, {2, 2},
                                           "NCHW", false);
    kxc::Call max_pool(kxc::relay::Op::Get("nn_max_pool2d"), {relu}, pool_attrs);
    kxc::Call avg_pool(kxc::relay::Op::Get("nn_avg_pool2d"), {relu}, pool_attrs);
    kxc::Call global_pool(kxc::relay::Op::Get("nn_global_avg_pool2d"), {relu},
                          kxc::relay::GlobalAvgPool2DAttrs::Create());
    kxc::Call flatten(kxc::relay::Op::Get("nn_flatten"), {global_pool},
                      kxc::relay::FlattenAttrs::Create(1));
    kxc::Function func({data, weight}, kxc::Tuple({max_pool, avg_pool, flatten}));

    kxc::tir::PrimFunc lowered = kxc::relay::LowerToTIR(func);
    TEST_CHECK(lowered.defined(), "NN MVP ops should lower to TIR");
    return true;
}

bool TestMvpTransformReduceSoftmaxLowerToTIR() {
    kxc::Var x("x", kxc::TensorType({2, 3, 4}, "float32"));
    kxc::Call reshape(kxc::relay::Op::Get("reshape"), {x},
                      kxc::relay::ReshapeAttrs::Create({2, 12}));
    kxc::Call transpose(kxc::relay::Op::Get("transpose"), {reshape},
                        kxc::relay::TransposeAttrs::Create({-1, 0}));
    kxc::Call reduce_mean(kxc::relay::Op::Get("reduce_mean"), {transpose},
                          kxc::relay::ReduceMeanAttrs::Create({1}, 1));
    kxc::Call softmax(kxc::relay::Op::Get("softmax"), {reduce_mean},
                      kxc::relay::SoftmaxAttrs::Create(0));
    kxc::Call cast(kxc::relay::Op::Get("cast"), {softmax}, kxc::relay::CastAttrs::Create(1));
    kxc::Function func({x}, cast);

    kxc::tir::PrimFunc lowered = kxc::relay::LowerToTIR(func);
    TEST_CHECK(lowered.defined(), "transform/reduce/softmax MVP ops should lower to TIR");
    return true;
}

bool TestPipelineAndLoweringIntegration() {
    kxc::Var x("x", kxc::TensorType({4}, "float32"));
    kxc::Var y("y", kxc::TensorType({4}, "float32"));
    kxc::Call add(kxc::relay::Op::Get("add"), {x, y});
    kxc::Function func({x, y}, add);

    kxc::Function optimized =
        kxc::relay::RunRelayPassPipeline(func, {kxc::String("optimize_default")});
    TEST_CHECK(CheckTensor(optimized->body.checked_type(), {4}, "float32"),
               "optimize_default should leave checked_type on body");

    kxc::tir::PrimFunc lowered = kxc::relay::LowerToTIR(func);
    TEST_CHECK(lowered.defined(), "LowerToTIR should infer missing checked_type internally");
    TEST_CHECK(CheckTensor(add.checked_type(), {4}, "float32"),
               "LowerToTIR should write checked_type through InferTypePass");

    kxc::Var untyped("untyped");
    kxc::Function invalid({untyped}, untyped);
    TEST_CHECK(ExpectThrow([&] { kxc::relay::InferTypePass(invalid); }),
               "untyped function parameter should fail");
    return true;
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, bool (*)()>> tests = {
        {"elementwise_broadcast", TestElementwiseBroadcast},
        {"matrix_and_dense_ops", TestMatrixAndDenseOps},
        {"conv_and_pool_ops", TestConvAndPoolOps},
        {"transform_and_reduce_ops", TestTransformAndReduceOps},
        {"mvp_elementwise_lower_to_tir", TestMvpElementwiseLowerToTIR},
        {"mvp_matrix_lower_to_tir", TestMvpMatrixLowerToTIR},
        {"mvp_nn_lower_to_tir", TestMvpNNLowerToTIR},
        {"mvp_transform_reduce_softmax_lower_to_tir", TestMvpTransformReduceSoftmaxLowerToTIR},
        {"pipeline_and_lowering_integration", TestPipelineAndLoweringIntegration},
    };

    for (const auto& test : tests) {
        bool ok = false;
        try {
            ok = test.second();
        } catch (const std::exception& e) {
            std::cerr << "[FAIL] " << test.first << ": unexpected exception: " << e.what()
                      << "\n";
            return 1;
        }
        if (!ok) {
            return 1;
        }
        std::cout << "[PASS] " << test.first << "\n";
    }

    std::cout << "All infer type tests passed.\n";
    return 0;
}
