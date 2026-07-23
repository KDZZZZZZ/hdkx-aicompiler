/*! \file test/infer_type_test.cpp
 * \brief 测试 Relay 类型和 shape 推导。
 */

#include "kxc/relay/op.h"
#include "kxc/relay/transforms/infer_type.h"
#include "kxc/compiler/lowering/relay_to_tir.h"
#include "kxc/relay/transforms/pipeline.h"

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
    kxc::relay::InferTypePass(batched_matmul_func);
    TEST_CHECK(CheckTensor(batched_matmul.checked_type(), {2, 3, 5}, "float32"),
               "batched matmul output shape mismatch");

    kxc::Var broadcast_a("broadcast_a", kxc::TensorType({2, 1, 3, 4}, "float32"));
    kxc::Var broadcast_b("broadcast_b", kxc::TensorType({1, 7, 4, 5}, "float32"));
    kxc::Call broadcast_matmul(kxc::relay::Op::Get("matmul"), {broadcast_a, broadcast_b});
    kxc::Function broadcast_matmul_func({broadcast_a, broadcast_b}, broadcast_matmul);
    kxc::relay::InferTypePass(broadcast_matmul_func);
    TEST_CHECK(CheckTensor(broadcast_matmul.checked_type(), {2, 7, 3, 5}, "float32"),
               "matmul should broadcast leading batch dimensions");

    kxc::Var mixed_b("mixed_b", kxc::TensorType({4, 5}, "float32"));
    kxc::Call mixed_matmul(kxc::relay::Op::Get("matmul"), {batched_a, mixed_b});
    kxc::Function mixed_matmul_func({batched_a, mixed_b}, mixed_matmul);
    kxc::relay::InferTypePass(mixed_matmul_func);
    TEST_CHECK(CheckTensor(mixed_matmul.checked_type(), {2, 3, 5}, "float32"),
               "matmul should broadcast a rank-2 rhs across batches");

    kxc::Var bad_k("bad_k", kxc::TensorType({2, 5, 6}, "float32"));
    kxc::Call invalid_k(kxc::relay::Op::Get("matmul"), {batched_a, bad_k});
    kxc::Function invalid_k_func({batched_a, bad_k}, invalid_k);
    TEST_CHECK(ExpectThrow([&] { kxc::relay::InferTypePass(invalid_k_func); }),
               "matmul incompatible reduction dimensions should fail");

    kxc::Var bad_batch("bad_batch", kxc::TensorType({3, 4, 5}, "float32"));
    kxc::Call invalid_batch(kxc::relay::Op::Get("matmul"), {batched_a, bad_batch});
    kxc::Function invalid_batch_func({batched_a, bad_batch}, invalid_batch);
    TEST_CHECK(ExpectThrow([&] { kxc::relay::InferTypePass(invalid_batch_func); }),
               "matmul incompatible batch dimensions should fail");

    kxc::Var rank_one("rank_one", kxc::TensorType({4}, "float32"));
    kxc::Call invalid_rank(kxc::relay::Op::Get("matmul"), {rank_one, mixed_b});
    kxc::Function invalid_rank_func({rank_one, mixed_b}, invalid_rank);
    TEST_CHECK(ExpectThrow([&] { kxc::relay::InferTypePass(invalid_rank_func); }),
               "matmul rank below two should fail");

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

bool TestGatherInferAndLoweringContract() {
    kxc::Var data("data", kxc::TensorType({2, 3, 4}, "float32"));
    kxc::Var indices("indices", kxc::TensorType({5, 6}, "int64"));
    kxc::Call axis_one(kxc::relay::Op::Get("gather"), {data, indices},
                       kxc::relay::GatherAttrs::Create(1));
    kxc::Function func({data, indices}, axis_one);
    kxc::relay::InferTypePass(func);
    TEST_CHECK(CheckTensor(axis_one.checked_type(), {2, 5, 6, 4}, "float32"),
               "gather must insert indices shape at axis");
    TEST_CHECK(kxc::relay::LowerToTIR(func)->prim_func.defined(),
               "static gather should lower to TIR");

    kxc::Call negative_axis(kxc::relay::Op::Get("gather"), {data, indices},
                            kxc::relay::GatherAttrs::Create(-1));
    kxc::Function negative_axis_func({data, indices}, negative_axis);
    kxc::relay::InferTypePass(negative_axis_func);
    TEST_CHECK(CheckTensor(negative_axis.checked_type(), {2, 3, 5, 6}, "float32"),
               "gather must normalize negative axis");

    kxc::Var bad_indices("bad_indices", kxc::TensorType({1}, "float32"));
    kxc::Call invalid_dtype(kxc::relay::Op::Get("gather"), {data, bad_indices},
                            kxc::relay::GatherAttrs::Create(0));
    TEST_CHECK(ExpectThrow([&] {
                   kxc::relay::InferTypePass(kxc::Function({data, bad_indices}, invalid_dtype));
               }),
               "gather non-integer indices should fail");
    kxc::Call invalid_axis(kxc::relay::Op::Get("gather"), {data, indices},
                           kxc::relay::GatherAttrs::Create(3));
    TEST_CHECK(ExpectThrow([&] {
                   kxc::relay::InferTypePass(kxc::Function({data, indices}, invalid_axis));
               }),
               "gather axis outside data rank should fail");
    kxc::Var scalar("scalar", kxc::TensorType({}, "float32"));
    kxc::Call invalid_rank(kxc::relay::Op::Get("gather"), {scalar, indices},
                           kxc::relay::GatherAttrs::Create(0));
    TEST_CHECK(ExpectThrow([&] {
                   kxc::relay::InferTypePass(kxc::Function({scalar, indices}, invalid_rank));
               }),
               "gather rank-zero data should fail");
    kxc::Var huge("huge", kxc::TensorType({2147483648LL}, "float32"));
    kxc::Var i32("i32", kxc::TensorType({1}, "int32"));
    kxc::Call overflow(kxc::relay::Op::Get("gather"), {huge, i32},
                       kxc::relay::GatherAttrs::Create(0));
    TEST_CHECK(ExpectThrow([&] {
                   kxc::relay::InferTypePass(kxc::Function({huge, i32}, overflow));
               }),
               "gather int32 indices must reject axis extent above INT32_MAX");
    return true;
}

bool TestWhereInferAndLoweringContract() {
    kxc::Var condition("condition", kxc::TensorType({2, 1, 1}, "bool"));
    kxc::Var x("x", kxc::TensorType({}, "float32"));
    kxc::Var y("y", kxc::TensorType({1, 3, 4}, "float32"));
    kxc::Call where(kxc::relay::Op::Get("where"), {condition, x, y});
    kxc::Function func({condition, x, y}, where);
    kxc::relay::InferTypePass(func);
    TEST_CHECK(CheckTensor(where.checked_type(), {2, 3, 4}, "float32"),
               "where must jointly broadcast scalar and rank-misaligned inputs");
    TEST_CHECK(kxc::relay::LowerToTIR(func)->prim_func.defined(),
               "static where should lower to TIR");

    kxc::Var zero_condition("zero_condition", kxc::TensorType({0, 1}, "bool"));
    kxc::Var zero_x("zero_x", kxc::TensorType({1, 3}, "float32"));
    kxc::Var zero_y("zero_y", kxc::TensorType({0, 3}, "float32"));
    kxc::Call zero_where(kxc::relay::Op::Get("where"),
                          {zero_condition, zero_x, zero_y});
    kxc::Function zero_func({zero_condition, zero_x, zero_y}, zero_where);
    kxc::relay::InferTypePass(zero_func);
    TEST_CHECK(CheckTensor(zero_where.checked_type(), {0, 3}, "float32"),
               "where must preserve broadcast zero extents");
    TEST_CHECK(kxc::relay::LowerToTIR(zero_func)->prim_func.defined(),
               "zero-extent where should lower to TIR");

    kxc::Var non_bool("non_bool", kxc::TensorType({2, 1}, "uint8"));
    kxc::Call invalid_condition(kxc::relay::Op::Get("where"), {non_bool, x, y});
    TEST_CHECK(ExpectThrow([&] {
                   kxc::relay::InferTypePass(kxc::Function({non_bool, x, y}, invalid_condition));
               }),
               "where non-bool condition should fail");
    kxc::Var wrong_dtype("wrong_dtype", kxc::TensorType({1, 3, 4}, "int32"));
    kxc::Call invalid_dtype(kxc::relay::Op::Get("where"), {condition, x, wrong_dtype});
    TEST_CHECK(ExpectThrow([&] {
                   kxc::relay::InferTypePass(kxc::Function({condition, x, wrong_dtype}, invalid_dtype));
               }),
               "where mismatched x/y dtypes should fail");
    kxc::Var unsupported_x("unsupported_x", kxc::TensorType({}, "float16"));
    kxc::Var unsupported_y("unsupported_y", kxc::TensorType({1, 3, 4}, "float16"));
    kxc::Call unsupported_dtype(kxc::relay::Op::Get("where"),
                                {condition, unsupported_x, unsupported_y});
    TEST_CHECK(ExpectThrow([&] {
                   kxc::relay::InferTypePass(
                       kxc::Function({condition, unsupported_x, unsupported_y}, unsupported_dtype));
               }),
               "where unsupported but matching branch dtypes should fail during type inference");
    kxc::Var incompatible_condition("incompatible_condition", kxc::TensorType({2, 2}, "bool"));
    kxc::Var incompatible_x("incompatible_x", kxc::TensorType({2, 3}, "float32"));
    kxc::Var incompatible_y("incompatible_y", kxc::TensorType({4}, "float32"));
    kxc::Call incompatible(kxc::relay::Op::Get("where"),
                            {incompatible_condition, incompatible_x, incompatible_y});
    TEST_CHECK(ExpectThrow([&] {
                   kxc::relay::InferTypePass(
                       kxc::Function({incompatible_condition, incompatible_x, incompatible_y},
                                      incompatible));
               }),
               "where incompatibility in any input should fail");
    return true;
}

bool TestSoftmaxInferTypeContract() {
    kxc::Var scalar("scalar", kxc::TensorType({}, "float32"));
    kxc::Call scalar_softmax(kxc::relay::Op::Get("softmax"), {scalar},
                             kxc::relay::SoftmaxAttrs::Create(0));
    kxc::Function scalar_func({scalar}, scalar_softmax);
    TEST_CHECK(ExpectThrow([&] { kxc::relay::InferTypePass(scalar_func); }),
               "softmax rank-0 input should fail type inference");

    kxc::Var integer("integer", kxc::TensorType({2, 3}, "int32"));
    kxc::Call integer_softmax(kxc::relay::Op::Get("softmax"), {integer},
                              kxc::relay::SoftmaxAttrs::Create(-1));
    kxc::Function integer_func({integer}, integer_softmax);
    TEST_CHECK(ExpectThrow([&] { kxc::relay::InferTypePass(integer_func); }),
               "softmax non-floating input should fail type inference");

    kxc::Var data("data", kxc::TensorType({2, 3}, "float64"));
    kxc::Call invalid_axis(kxc::relay::Op::Get("softmax"), {data},
                           kxc::relay::SoftmaxAttrs::Create(2));
    kxc::Function invalid_axis_func({data}, invalid_axis);
    TEST_CHECK(ExpectThrow([&] { kxc::relay::InferTypePass(invalid_axis_func); }),
               "softmax out-of-range axis should fail type inference");
    return true;
}

bool TestNegativeExtentLoweringGates() {
    kxc::Var x("x", kxc::TensorType({-1, 3}, "float32"));
    kxc::Var y("y", kxc::TensorType({-1, 3}, "float32"));
    kxc::Call add(kxc::relay::Op::Get("add"), {x, y});
    kxc::Function func({x, y}, add);

    kxc::relay::InferTypePass(func);
    TEST_CHECK(CheckTensor(add.checked_type(), {-1, 3}, "float32"),
               "negative extent TensorType may type-infer before lowering");
    TEST_CHECK(ExpectThrow([&] { kxc::relay::LowerToTIR(func); }),
               "LowerToTIR must reject negative static extents before producing TIR");
    TEST_CHECK(ExpectThrow([&] { kxc::relay::LowerOperatorCallsToTIR(func); }),
               "LowerOperatorCallsToTIR must reject negative static extents before producing TIR");
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

    kxc::tir::PrimFunc lowered = kxc::relay::LowerToTIR(func)->prim_func;
    TEST_CHECK(lowered.defined(), "elementwise MVP ops should lower to TIR");
    const kxc::Array<kxc::relay::LoweredFunction> units =
        kxc::relay::LowerOperatorCallsToTIR(func);
    TEST_CHECK(units.size() == 5,
               "five checked elementwise Calls must lower to five PrimFuncs");
    for (const auto& unit : units) {
        TEST_CHECK(unit.defined() && unit->prim_func.defined(),
                   "per-operator lowering must preserve inferred type completeness");
    }
    return true;
}

bool TestMvpMatrixLowerToTIR() {
    kxc::Var a("a", kxc::TensorType({2, 3}, "float32"));
    kxc::Var b("b", kxc::TensorType({3, 4}, "float32"));
    kxc::Call matmul(kxc::relay::Op::Get("matmul"), {a, b});
    kxc::Function matmul_func({a, b}, matmul);
    TEST_CHECK(kxc::relay::LowerToTIR(matmul_func)->prim_func.defined(),
               "matmul should lower to TIR");

    kxc::Var batched_a("batched_a", kxc::TensorType({2, 3, 4}, "float32"));
    kxc::Var batched_b("batched_b", kxc::TensorType({1, 4, 5}, "float32"));
    kxc::Call batched_matmul(kxc::relay::Op::Get("matmul"), {batched_a, batched_b});
    kxc::Function batched_matmul_func({batched_a, batched_b}, batched_matmul);
    TEST_CHECK(kxc::relay::LowerToTIR(batched_matmul_func)->prim_func.defined(),
               "batched broadcast matmul should lower to TIR");

    kxc::Var dense_weight("dense_weight", kxc::TensorType({5, 3}, "float32"));
    kxc::Call dense(kxc::relay::Op::Get("nn_dense"), {a, dense_weight},
                    kxc::relay::DenseAttrs::Create(5, ""));
    kxc::Function dense_func({a, dense_weight}, dense);
    TEST_CHECK(kxc::relay::LowerToTIR(dense_func)->prim_func.defined(),
               "nn_dense should lower to TIR");

    kxc::Var gemm_b("gemm_b", kxc::TensorType({4, 3}, "float32"));
    kxc::Var gemm_bias("gemm_bias", kxc::TensorType({4}, "float32"));
    kxc::Call gemm(kxc::relay::Op::Get("nn_gemm"), {a, gemm_b, gemm_bias},
                   kxc::relay::GemmAttrs::Create(1.0f, 1.0f, 0, 1));
    kxc::Function gemm_func({a, gemm_b, gemm_bias}, gemm);
    TEST_CHECK(kxc::relay::LowerToTIR(gemm_func)->prim_func.defined(),
               "nn_gemm should lower to TIR");
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

    kxc::tir::PrimFunc lowered = kxc::relay::LowerToTIR(func)->prim_func;
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

    kxc::tir::PrimFunc lowered = kxc::relay::LowerToTIR(func)->prim_func;
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

    kxc::tir::PrimFunc lowered = kxc::relay::LowerToTIR(func)->prim_func;
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
        {"gather_infer_and_lowering_contract", TestGatherInferAndLoweringContract},
        {"where_infer_and_lowering_contract", TestWhereInferAndLoweringContract},
        {"softmax_infer_type_contract", TestSoftmaxInferTypeContract},
        {"negative_extent_lowering_gates", TestNegativeExtentLoweringGates},
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
