/*! \file test/infer_type_test.cpp
 * \brief 测试 Relay 类型和 shape 推导。
 */

#include "kxc/ffi/registry.h"
#include "kxc/relay/op.h"
#include "kxc/relay/transforms/infer_type.h"
#include "kxc/compiler/lowering/relay_to_tir.h"
#include "kxc/relay/transforms/pipeline.h"
#include "../src/compiler/internal/lowered_graph.h"

#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
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

bool ExpectOverflow(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const std::overflow_error&) {
        return true;
    } catch (const std::exception&) {
    }
    return false;
}

std::vector<kxc::relay::LoweredFunction> LowerUnits(kxc::Function function) {
    function = kxc::relay::InferTypePass(std::move(function));
    const kxc::api::internal::LoweredGraph graph =
        kxc::api::internal::LowerGraph(std::move(function));
    std::vector<kxc::relay::LoweredFunction> result;
    result.reserve(graph.primitives.size());
    for (const auto& primitive : graph.primitives) result.push_back(primitive.lowered);
    return result;
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

bool TestFlattenAndReshapeProductArithmetic() {
    constexpr int64_t maximum = std::numeric_limits<int64_t>::max();

    kxc::Var flatten_overflow_data(
        "flatten_overflow_data", kxc::TensorType({maximum, 2}, "float32"));
    kxc::Call flatten_overflow(
        kxc::relay::Op::Get("nn_flatten"), {flatten_overflow_data},
        kxc::relay::FlattenAttrs::Create(0));
    TEST_CHECK(ExpectOverflow([&] {
                   kxc::relay::InferTypePass(
                       kxc::Function({flatten_overflow_data}, flatten_overflow));
               }),
               "flatten must reject an int64 shape-product overflow during InferTypePass");

    kxc::Var flatten_zero_prefix_data(
        "flatten_zero_prefix_data", kxc::TensorType({0, maximum, 2}, "float32"));
    kxc::Var flatten_zero_suffix_data(
        "flatten_zero_suffix_data", kxc::TensorType({maximum, 2, 0}, "float32"));
    kxc::Var flatten_one_data(
        "flatten_one_data", kxc::TensorType({maximum, 1}, "float32"));
    kxc::Call flatten_zero_prefix(
        kxc::relay::Op::Get("nn_flatten"), {flatten_zero_prefix_data},
        kxc::relay::FlattenAttrs::Create(0));
    kxc::Call flatten_zero_suffix(
        kxc::relay::Op::Get("nn_flatten"), {flatten_zero_suffix_data},
        kxc::relay::FlattenAttrs::Create(0));
    kxc::Call flatten_one(kxc::relay::Op::Get("nn_flatten"), {flatten_one_data},
                          kxc::relay::FlattenAttrs::Create(0));
    kxc::relay::InferTypePass(kxc::Function(
        {flatten_zero_prefix_data, flatten_zero_suffix_data, flatten_one_data},
        kxc::Tuple({flatten_zero_prefix, flatten_zero_suffix, flatten_one})));
    TEST_CHECK(CheckTensor(flatten_zero_prefix.checked_type(), {1, 0}, "float32") &&
                   CheckTensor(flatten_zero_suffix.checked_type(), {1, 0}, "float32") &&
                   CheckTensor(flatten_one.checked_type(), {1, maximum}, "float32"),
               "flatten zero products must be order-independent and INT64_MAX * 1 legal");

    kxc::Var reshape_input_overflow_data(
        "reshape_input_overflow_data", kxc::TensorType({maximum, 2}, "float32"));
    kxc::Call reshape_input_overflow(
        kxc::relay::Op::Get("reshape"), {reshape_input_overflow_data},
        kxc::relay::ReshapeAttrs::Create({-1}));
    TEST_CHECK(ExpectOverflow([&] {
                   kxc::relay::InferTypePass(
                       kxc::Function({reshape_input_overflow_data}, reshape_input_overflow));
               }),
               "reshape must reject input shape-product overflow during InferTypePass");

    kxc::Var reshape_target_overflow_data(
        "reshape_target_overflow_data", kxc::TensorType({1}, "float32"));
    kxc::Call reshape_target_overflow(
        kxc::relay::Op::Get("reshape"), {reshape_target_overflow_data},
        kxc::relay::ReshapeAttrs::Create({maximum, 2}));
    TEST_CHECK(ExpectOverflow([&] {
                   kxc::relay::InferTypePass(
                       kxc::Function({reshape_target_overflow_data}, reshape_target_overflow));
               }),
               "reshape must reject target shape-product overflow during InferTypePass");

    kxc::Var reshape_zero_data("reshape_zero_data", kxc::TensorType({0}, "float32"));
    kxc::Call reshape_zero_prefix(
        kxc::relay::Op::Get("reshape"), {reshape_zero_data},
        kxc::relay::ReshapeAttrs::Create({0, maximum, 2}, 1));
    kxc::Var reshape_copy_zero_data(
        "reshape_copy_zero_data", kxc::TensorType({maximum, 2, 0}, "float32"));
    kxc::Call reshape_zero_suffix(
        kxc::relay::Op::Get("reshape"), {reshape_copy_zero_data},
        kxc::relay::ReshapeAttrs::Create({0, 0, 0}));
    kxc::Var reshape_one_data(
        "reshape_one_data", kxc::TensorType({maximum}, "float32"));
    kxc::Call reshape_one(kxc::relay::Op::Get("reshape"), {reshape_one_data},
                          kxc::relay::ReshapeAttrs::Create({maximum, 1}));
    kxc::relay::InferTypePass(kxc::Function(
        {reshape_zero_data, reshape_copy_zero_data, reshape_one_data},
        kxc::Tuple({reshape_zero_prefix, reshape_zero_suffix, reshape_one})));
    TEST_CHECK(CheckTensor(reshape_zero_prefix.checked_type(), {0, maximum, 2}, "float32") &&
                   CheckTensor(reshape_zero_suffix.checked_type(), {maximum, 2, 0}, "float32") &&
                   CheckTensor(reshape_one.checked_type(), {maximum, 1}, "float32"),
               "reshape zero products must be order-independent and INT64_MAX * 1 legal");

    kxc::Call invalid_negative(
        kxc::relay::Op::Get("reshape"), {reshape_zero_data},
        kxc::relay::ReshapeAttrs::Create({0, -2}, 1));
    kxc::Call duplicate_inferred(
        kxc::relay::Op::Get("reshape"), {reshape_zero_data},
        kxc::relay::ReshapeAttrs::Create({0, -1, -1}, 1));
    kxc::Var legacy_shape_input(
        "legacy_shape_input", kxc::TensorType({1}, "int64"));
    kxc::Call legacy_two_input(
        kxc::relay::Op::Get("reshape"),
        {reshape_zero_data, legacy_shape_input},
        kxc::relay::ReshapeAttrs::Create({0}));
    TEST_CHECK(ExpectThrow([&] {
                   kxc::relay::InferTypePass(
                       kxc::Function({reshape_zero_data}, invalid_negative));
               }) &&
                   ExpectThrow([&] {
                       kxc::relay::InferTypePass(
                           kxc::Function({reshape_zero_data}, duplicate_inferred));
                   }) &&
                   ExpectThrow([&] {
                       kxc::relay::InferTypePass(kxc::Function(
                           {reshape_zero_data, legacy_shape_input},
                           legacy_two_input));
                   }),
               "reshape must reject invalid dimensions and legacy two-input arity");
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

bool TestConcatenateInferAndLoweringContract() {
    kxc::Var lhs("lhs", kxc::TensorType({2, 3}, "float32"));
    kxc::Var rhs("rhs", kxc::TensorType({2, 4}, "float32"));
    kxc::Call concatenate(kxc::relay::Op::Get("concatenate"), {lhs, rhs},
                          kxc::relay::ConcatenateAttrs::Create(-1));
    kxc::Function function({lhs, rhs}, concatenate);
    kxc::relay::InferTypePass(function);
    TEST_CHECK(CheckTensor(concatenate.checked_type(), {2, 7}, "float32"),
               "concatenate must normalize a negative axis and sum its extent");
    TEST_CHECK(kxc::relay::LowerToTIR(function)->prim_func.defined(),
               "static concatenate should lower to TIR");
    TEST_CHECK(LowerUnits(function).size() == 1,
               "one concatenate call must produce one lowering unit");
    TEST_CHECK(kxc::relay::SerializeAttrs(kxc::relay::ConcatenateAttrs::Create(-1)) ==
                   kxc::relay::SerializeAttrs(kxc::relay::ConcatenateAttrs::Create(-1)) &&
                   kxc::relay::SerializeAttrs(kxc::relay::ConcatenateAttrs::Create(-1)) !=
                   kxc::relay::SerializeAttrs(kxc::relay::ConcatenateAttrs::Create(1)),
               "concatenate attrs must canonically preserve axis");
    TEST_CHECK(kxc::Registry::Global().Get("kxc.relay.op._make.concatenate").defined(),
               "canonical concatenate FFI entry must be registered");

    kxc::Var empty_lhs("empty_lhs", kxc::TensorType({2, 0}, "int32"));
    kxc::Var nonempty_rhs("nonempty_rhs", kxc::TensorType({2, 3}, "int32"));
    kxc::Call empty_axis(kxc::relay::Op::Get("concatenate"), {empty_lhs, nonempty_rhs},
                         kxc::relay::ConcatenateAttrs::Create(1));
    kxc::Function empty_function({empty_lhs, nonempty_rhs}, empty_axis);
    kxc::relay::InferTypePass(empty_function);
    TEST_CHECK(CheckTensor(empty_axis.checked_type(), {2, 3}, "int32"),
               "concatenate must accept an empty axis side");
    TEST_CHECK(kxc::relay::LowerToTIR(empty_function)->prim_func.defined(),
               "empty-axis concatenate should lower to a fresh compute");

    kxc::Var bool_lhs("bool_lhs", kxc::TensorType({1, 2}, "bool"));
    kxc::Var bool_rhs("bool_rhs", kxc::TensorType({1, 1}, "bool"));
    kxc::Call bool_concatenate(kxc::relay::Op::Get("concatenate"), {bool_lhs, bool_rhs},
                               kxc::relay::ConcatenateAttrs::Create(1));
    kxc::Function bool_function({bool_lhs, bool_rhs}, bool_concatenate);
    kxc::relay::InferTypePass(bool_function);
    TEST_CHECK(CheckTensor(bool_concatenate.checked_type(), {1, 3}, "bool"),
               "concatenate must preserve bool output type");
    TEST_CHECK(kxc::relay::LowerToTIR(bool_function)->prim_func.defined(),
               "bool concatenate should lower through DataType::Bool");

    kxc::Var scalar("scalar", kxc::TensorType({}, "float32"));
    kxc::Call rank_zero(kxc::relay::Op::Get("concatenate"), {scalar, scalar},
                        kxc::relay::ConcatenateAttrs::Create(0));
    TEST_CHECK(ExpectThrow([&] {
                   kxc::relay::InferTypePass(kxc::Function({scalar}, rank_zero));
               }), "concatenate rank-zero inputs must fail");
    kxc::Var wrong_dtype("wrong_dtype", kxc::TensorType({2, 4}, "float64"));
    kxc::Call dtype(kxc::relay::Op::Get("concatenate"), {lhs, wrong_dtype},
                    kxc::relay::ConcatenateAttrs::Create(1));
    TEST_CHECK(ExpectThrow([&] {
                   kxc::relay::InferTypePass(kxc::Function({lhs, wrong_dtype}, dtype));
               }), "concatenate dtype mismatch must fail");
    kxc::Var unsupported("unsupported", kxc::TensorType({2, 4}, "float16"));
    kxc::Call unsupported_dtype(kxc::relay::Op::Get("concatenate"), {unsupported, unsupported},
                                kxc::relay::ConcatenateAttrs::Create(1));
    TEST_CHECK(ExpectThrow([&] {
                   kxc::relay::InferTypePass(
                       kxc::Function({unsupported}, unsupported_dtype));
               }), "concatenate unsupported dtype must fail");
    kxc::Var rank_three("rank_three", kxc::TensorType({2, 3, 4}, "float32"));
    kxc::Call rank_mismatch(kxc::relay::Op::Get("concatenate"), {lhs, rank_three},
                            kxc::relay::ConcatenateAttrs::Create(1));
    TEST_CHECK(ExpectThrow([&] {
                   kxc::relay::InferTypePass(kxc::Function({lhs, rank_three}, rank_mismatch));
               }), "concatenate rank mismatch must fail");
    kxc::Var nonaxis_rhs("nonaxis_rhs", kxc::TensorType({3, 4}, "float32"));
    kxc::Call nonaxis(kxc::relay::Op::Get("concatenate"), {lhs, nonaxis_rhs},
                      kxc::relay::ConcatenateAttrs::Create(1));
    TEST_CHECK(ExpectThrow([&] {
                   kxc::relay::InferTypePass(kxc::Function({lhs, nonaxis_rhs}, nonaxis));
               }), "concatenate non-axis mismatch must fail");
    kxc::Call bad_axis(kxc::relay::Op::Get("concatenate"), {lhs, rhs},
                       kxc::relay::ConcatenateAttrs::Create(2));
    TEST_CHECK(ExpectThrow([&] {
                   kxc::relay::InferTypePass(kxc::Function({lhs, rhs}, bad_axis));
               }), "concatenate out-of-range axis must fail");
    kxc::Var dynamic("dynamic", kxc::TensorType({2, -1}, "float32"));
    kxc::Call dynamic_extent(kxc::relay::Op::Get("concatenate"), {dynamic, dynamic},
                             kxc::relay::ConcatenateAttrs::Create(1));
    TEST_CHECK(ExpectThrow([&] {
                   kxc::relay::LowerToTIR(kxc::Function({dynamic}, dynamic_extent));
               }), "concatenate dynamic or negative extents must fail lowering");
    const int64_t maximum = std::numeric_limits<int64_t>::max();
    kxc::Var huge_lhs("huge_lhs", kxc::TensorType({maximum}, "int64"));
    kxc::Var huge_rhs("huge_rhs", kxc::TensorType({1}, "int64"));
    kxc::Call overflow(kxc::relay::Op::Get("concatenate"), {huge_lhs, huge_rhs},
                       kxc::relay::ConcatenateAttrs::Create(0));
    TEST_CHECK(ExpectThrow([&] {
                   kxc::relay::LowerToTIR(kxc::Function({huge_lhs, huge_rhs}, overflow));
               }), "concatenate axis sum overflow must fail lowering");
    return true;
}

bool TestSliceInferAndLoweringContract() {
    kxc::Var data("data", kxc::TensorType({2, 3, 4}, "float32"));
    kxc::Call slice(kxc::relay::Op::Get("slice"), {data},
                    kxc::relay::SliceAttrs::Create({-3, -100}, {100, 3}, {-1, 1}, {1, 1}));
    kxc::Function function({data}, slice);
    kxc::relay::InferTypePass(function);
    TEST_CHECK(CheckTensor(slice.checked_type(), {2, 3, 3}, "float32"),
               "slice must normalize negative axes and clamp negative/huge endpoints");
    TEST_CHECK(kxc::relay::LowerToTIR(function)->prim_func.defined() &&
                   LowerUnits(function).size() == 1,
               "slice must lower to one fresh indexed compute");
    TEST_CHECK(kxc::Registry::Global().Get("kxc.relay.op._make.slice").defined(),
               "canonical slice FFI entry must be registered");
    const auto attrs = kxc::relay::SliceAttrs::Create({0}, {4}, {1}, {1});
    const std::string serialized = kxc::relay::SerializeAttrs(attrs);
    TEST_CHECK(serialized.find("SliceAttrsNode") != std::string::npos &&
                   serialized.find("starts") < serialized.find("ends") &&
                   serialized.find("ends") < serialized.find("axes") &&
                   serialized.find("axes") < serialized.find("steps"),
               "slice attrs must serialize in canonical starts/ends/axes/steps order");

    kxc::Call identity(kxc::relay::Op::Get("slice"), {data},
                       kxc::relay::SliceAttrs::Create({0}, {3}, {1}, {1}));
    kxc::Function identity_function({data}, identity);
    kxc::relay::InferTypePass(identity_function);
    TEST_CHECK(kxc::relay::LowerToTIR(identity_function)->prim_func.defined(),
               "identity slice must lower as a fresh compute rather than a view");
    kxc::Call empty(kxc::relay::Op::Get("slice"), {data},
                    kxc::relay::SliceAttrs::Create({3}, {1}, {1}, {1}));
    kxc::Function empty_function({data}, empty);
    kxc::relay::InferTypePass(empty_function);
    TEST_CHECK(CheckTensor(empty.checked_type(), {2, 0, 4}, "float32") &&
                   kxc::relay::LowerToTIR(empty_function)->prim_func.defined(),
               "empty slice output must be legal and lower");

    const auto bad = [&](kxc::Var value, kxc::relay::SliceAttrs attrs) {
        kxc::Call call(kxc::relay::Op::Get("slice"), {value}, attrs);
        return ExpectThrow([&] { kxc::relay::InferTypePass(kxc::Function({value}, call)); });
    };
    TEST_CHECK(bad(kxc::Var("scalar", kxc::TensorType({}, "float32")),
                   kxc::relay::SliceAttrs::Create({0}, {1}, {0}, {1})) &&
                   bad(kxc::Var("half", kxc::TensorType({2}, "float16")),
                       kxc::relay::SliceAttrs::Create({0}, {1}, {0}, {1})) &&
                   bad(data, kxc::relay::SliceAttrs::Create({0}, {1, 2}, {0}, {1})) &&
                   bad(data, kxc::relay::SliceAttrs::Create({0, 1}, {1, 2}, {0, 0}, {1, 1})) &&
                   bad(data, kxc::relay::SliceAttrs::Create({0}, {1}, {3}, {1})) &&
                   bad(data, kxc::relay::SliceAttrs::Create({0}, {1}, {0}, {0})) &&
                   bad(data, kxc::relay::SliceAttrs::Create({0}, {1}, {0}, {-1})) &&
                   bad(data, kxc::relay::SliceAttrs::Create({0}, {1}, {0}, {2})) &&
                   bad(kxc::Var("dynamic", kxc::TensorType({2, -1}, "float32")),
                       kxc::relay::SliceAttrs::Create({0}, {1}, {0}, {1})),
               "slice must reject rank/dtype/vector/axis/step/dynamic-shape violations");
    kxc::Call minimum(kxc::relay::Op::Get("slice"), {data},
                      kxc::relay::SliceAttrs::Create({std::numeric_limits<int64_t>::min()},
                                                      {std::numeric_limits<int64_t>::max()},
                                                      {1}, {1}));
    kxc::Function minimum_function({data}, minimum);
    kxc::relay::InferTypePass(minimum_function);
    TEST_CHECK(CheckTensor(minimum.checked_type(), {2, 3, 4}, "float32"),
               "slice must clamp INT64_MIN safely without overflow");
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

    kxc::runtime::NDArray condition_bytes = kxc::runtime::NDArray::Empty(
        {2, 1}, kxc::runtime::DataTypeFromString("bool"), kxc::Device::CPU());
    const std::vector<uint8_t> condition_values{1, 0};
    condition_bytes.CopyFromBytes(condition_values.data(), condition_values.size());
    kxc::Var constant_x("constant_x", kxc::TensorType({1, 3}, "bool"));
    kxc::Var constant_y("constant_y", kxc::TensorType({2, 1}, "bool"));
    kxc::Call constant_where(kxc::relay::Op::Get("where"),
                             {kxc::Constant(condition_bytes), constant_x, constant_y});
    kxc::Function constant_function({constant_x, constant_y}, constant_where);
    kxc::relay::InferTypePass(constant_function);
    TEST_CHECK(CheckTensor(constant_where.checked_type(), {2, 3}, "bool") &&
                   kxc::relay::LowerToTIR(constant_function)->prim_func.defined(),
               "byte-backed bool Constant and bool branches must lower as TIR bool");

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

bool TestLayerNormInferAndLoweringContract() {
    kxc::Var data("data", kxc::TensorType({2, 3, 4}, "float32"));
    kxc::Var scale("scale", kxc::TensorType({3, 4}, "float32"));
    kxc::Var bias("bias", kxc::TensorType({3, 4}, "float32"));
    kxc::Call layer_norm(kxc::relay::Op::Get("nn_layer_norm"), {data, scale, bias},
                         kxc::relay::LayerNormAttrs::Create(-2, 1e-5f, "float64"));
    kxc::Function function({data, scale, bias}, layer_norm);
    kxc::relay::InferTypePass(function);
    TEST_CHECK(CheckTensor(layer_norm.checked_type(), {2, 3, 4}, "float32"),
               "LayerNorm must preserve data shape and dtype");
    TEST_CHECK(kxc::relay::LowerToTIR(function)->prim_func.defined(),
               "valid static LayerNorm should lower to TIR");
    TEST_CHECK(LowerUnits(function).size() == 1,
               "one LayerNorm Relay Call must produce one lowering unit");

    kxc::Var scalar("scalar", kxc::TensorType({}, "float32"));
    kxc::Var scalar_affine("scalar_affine", kxc::TensorType({}, "float32"));
    kxc::Call rank_zero(kxc::relay::Op::Get("nn_layer_norm"),
                        {scalar, scalar_affine, scalar_affine},
                        kxc::relay::LayerNormAttrs::Create());
    TEST_CHECK(ExpectThrow([&] {
                   kxc::relay::InferTypePass(
                       kxc::Function({scalar, scalar_affine}, rank_zero));
               }),
               "LayerNorm rank-zero data must fail type inference");

    kxc::Var wrong_dtype("wrong_dtype", kxc::TensorType({3, 4}, "float64"));
    kxc::Call dtype(kxc::relay::Op::Get("nn_layer_norm"), {data, wrong_dtype, bias},
                    kxc::relay::LayerNormAttrs::Create());
    TEST_CHECK(ExpectThrow([&] {
                   kxc::relay::InferTypePass(kxc::Function({data, wrong_dtype, bias}, dtype));
               }),
               "LayerNorm requires float32 data, scale, and bias");

    kxc::Var bad_scale("bad_scale", kxc::TensorType({4}, "float32"));
    kxc::Call scale_suffix(kxc::relay::Op::Get("nn_layer_norm"), {data, bad_scale, bias},
                           kxc::relay::LayerNormAttrs::Create(1));
    TEST_CHECK(ExpectThrow([&] {
                   kxc::relay::InferTypePass(
                       kxc::Function({data, bad_scale, bias}, scale_suffix));
               }),
               "LayerNorm scale shape must exactly equal the normalized suffix");
    kxc::Var bad_bias("bad_bias", kxc::TensorType({3, 1}, "float32"));
    kxc::Call bias_suffix(kxc::relay::Op::Get("nn_layer_norm"), {data, scale, bad_bias},
                          kxc::relay::LayerNormAttrs::Create(1));
    TEST_CHECK(ExpectThrow([&] {
                   kxc::relay::InferTypePass(
                       kxc::Function({data, scale, bad_bias}, bias_suffix));
               }),
               "LayerNorm bias shape must exactly equal the normalized suffix");

    kxc::Call invalid_axis(kxc::relay::Op::Get("nn_layer_norm"), {data, scale, bias},
                           kxc::relay::LayerNormAttrs::Create(3));
    TEST_CHECK(ExpectThrow([&] {
                   kxc::relay::InferTypePass(kxc::Function({data, scale, bias}, invalid_axis));
               }),
               "LayerNorm axis outside data rank must fail type inference");
    for (const float epsilon : {0.0f, -1e-5f,
                                std::numeric_limits<float>::infinity(),
                                std::numeric_limits<float>::quiet_NaN()}) {
        kxc::Call invalid_epsilon(kxc::relay::Op::Get("nn_layer_norm"), {data, scale, bias},
                                  kxc::relay::LayerNormAttrs::Create(1, epsilon, "float64"));
        TEST_CHECK(ExpectThrow([&] {
                       kxc::relay::InferTypePass(
                           kxc::Function({data, scale, bias}, invalid_epsilon));
                   }),
                   "LayerNorm epsilon must be finite and strictly positive");
    }
    kxc::Call invalid_accumulation(kxc::relay::Op::Get("nn_layer_norm"), {data, scale, bias},
                                   kxc::relay::LayerNormAttrs::Create(1, 1e-5f, "float32"));
    TEST_CHECK(ExpectThrow([&] {
                   kxc::relay::InferTypePass(
                       kxc::Function({data, scale, bias}, invalid_accumulation));
               }),
               "LayerNorm accumulation dtype must be float64");

    kxc::Var zero_data("zero_data", kxc::TensorType({2, 0, 4}, "float32"));
    kxc::Var zero_scale("zero_scale", kxc::TensorType({0, 4}, "float32"));
    kxc::Var zero_bias("zero_bias", kxc::TensorType({0, 4}, "float32"));
    kxc::Call zero_suffix(kxc::relay::Op::Get("nn_layer_norm"),
                          {zero_data, zero_scale, zero_bias},
                          kxc::relay::LayerNormAttrs::Create(1));
    TEST_CHECK(ExpectThrow([&] {
                   kxc::relay::InferTypePass(
                       kxc::Function({zero_data, zero_scale, zero_bias}, zero_suffix));
               }),
               "LayerNorm normalized suffix dimensions must be positive");

    const int64_t maximum = std::numeric_limits<int64_t>::max();
    kxc::Var huge_data("huge_data", kxc::TensorType({maximum, 2}, "float32"));
    kxc::Var huge_scale("huge_scale", kxc::TensorType({maximum, 2}, "float32"));
    kxc::Var huge_bias("huge_bias", kxc::TensorType({maximum, 2}, "float32"));
    kxc::Call overflowing(kxc::relay::Op::Get("nn_layer_norm"),
                          {huge_data, huge_scale, huge_bias},
                          kxc::relay::LayerNormAttrs::Create(0));
    TEST_CHECK(ExpectThrow([&] {
                   kxc::relay::LowerToTIR(
                       kxc::Function({huge_data, huge_scale, huge_bias}, overflowing));
               }),
               "LayerNorm lowering must reject normalized element-count overflow");
    return true;
}

bool TestExactTransformerOperatorSliceComposition() {
    kxc::Var embedding_table("embedding_table", kxc::TensorType({4, 2}, "float32"));
    kxc::Var token_ids("token_ids", kxc::TensorType({2}, "int64"));
    kxc::Var condition("condition", kxc::TensorType({2, 1}, "bool"));
    kxc::Var fallback("fallback", kxc::TensorType({1, 2}, "float32"));
    kxc::Var scale("scale", kxc::TensorType({2}, "float32"));
    kxc::Var bias("bias", kxc::TensorType({2}, "float32"));

    kxc::Call embedded(kxc::relay::Op::Get("gather"), {embedding_table, token_ids},
                       kxc::relay::GatherAttrs::Create(0));
    kxc::Call normalized(kxc::relay::Op::Get("nn_layer_norm"),
                         {embedded, scale, bias},
                         kxc::relay::LayerNormAttrs::Create(-1, 1e-5f, "float64"));
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

    kxc::relay::InferTypePass(function);
    TEST_CHECK(CheckTensor(embedded.checked_type(), {2, 2}, "float32") &&
                   CheckTensor(normalized.checked_type(), {2, 2}, "float32") &&
                   CheckTensor(selected.checked_type(), {2, 2}, "float32") &&
                   CheckTensor(prefix.checked_type(), {1, 2}, "float32") &&
                   CheckTensor(sequence.checked_type(), {3, 2}, "float32") &&
                   CheckTensor(context.checked_type(), {3, 2}, "float32"),
               "exact Transformer operator slice shapes must compose without dynamic claims");
    TEST_CHECK(kxc::relay::LowerToTIR(function)->prim_func.defined(),
               "exact Transformer operator slice must lower to TIR");
    TEST_CHECK(LowerUnits(function).size() == 9,
               "exact Transformer operator slice must preserve nine per-op units");
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
    TEST_CHECK(ExpectThrow([&] { LowerUnits(func); }),
               "per-unit lowering must reject negative static extents before producing TIR");
    return true;
}

bool TestStaticLoweringSizeGates() {
    const auto make_add = [](const std::vector<int64_t>& shape) {
        kxc::Array<int64_t> relay_shape;
        for (int64_t extent : shape) relay_shape.push_back(extent);
        kxc::Var lhs("lhs", kxc::TensorType(relay_shape, "float32"));
        kxc::Var rhs("rhs", kxc::TensorType(relay_shape, "float32"));
        return kxc::Function(
            {lhs, rhs}, kxc::Call(kxc::relay::Op::Get("add"), {lhs, rhs}));
    };
    const auto rejected_by_both = [&](const std::vector<int64_t>& shape) {
        const kxc::Function function = make_add(shape);
        return ExpectThrow([&] { kxc::relay::LowerToTIR(function); }) &&
               ExpectThrow([&] { LowerUnits(function); });
    };

    constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();
    TEST_CHECK(rejected_by_both({kInt32Max + 1}),
               "all lowering entries must reject an iteration extent above INT32_MAX");
    TEST_CHECK(rejected_by_both({kInt32Max, kInt32Max, 3}),
               "all lowering entries must reject row-major product overflow");
    TEST_CHECK(rejected_by_both({kInt32Max, kInt32Max}),
               "all lowering entries must reject tensor byte counts above int64/size_t");

    const kxc::Function zero = make_add({0, kInt32Max, kInt32Max});
    TEST_CHECK(kxc::relay::LowerToTIR(zero)->prim_func.defined() &&
                   LowerUnits(zero).size() == 1,
               "zero-element tensors with individually legal extents must remain lowerable");

    kxc::Var flatten_data("flatten_data",
                          kxc::TensorType({65536, 65536}, "float32"));
    kxc::Call flatten(kxc::relay::Op::Get("nn_flatten"), {flatten_data},
                      kxc::relay::FlattenAttrs::Create(0));
    const kxc::Function flatten_function({flatten_data}, flatten);
    TEST_CHECK(ExpectThrow([&] { kxc::relay::LowerToTIR(flatten_function); }) &&
                   ExpectThrow([&] { LowerUnits(flatten_function); }),
               "flattened iteration extents above INT32_MAX must fail closed");

    kxc::Var concat_lhs("concat_lhs", kxc::TensorType({kInt32Max}, "float32"));
    kxc::Var concat_rhs("concat_rhs", kxc::TensorType({1}, "float32"));
    kxc::Call concatenate(
        kxc::relay::Op::Get("concatenate"), {concat_lhs, concat_rhs},
        kxc::relay::ConcatenateAttrs::Create(0));
    const kxc::Function concatenate_function({concat_lhs, concat_rhs}, concatenate);
    TEST_CHECK(ExpectThrow([&] { kxc::relay::LowerToTIR(concatenate_function); }) &&
                   ExpectThrow([&] { LowerUnits(concatenate_function); }),
               "Concatenate output extents above INT32_MAX must fail closed");

    kxc::Var slice_data("slice_data",
                        kxc::TensorType({kInt32Max + 1}, "float32"));
    kxc::Call slice(kxc::relay::Op::Get("slice"), {slice_data},
                    kxc::relay::SliceAttrs::Create({0}, {1}, {0}, {1}));
    const kxc::Function slice_function({slice_data}, slice);
    TEST_CHECK(ExpectThrow([&] { kxc::relay::LowerToTIR(slice_function); }) &&
                   ExpectThrow([&] { LowerUnits(slice_function); }),
               "Slice input extents above INT32_MAX must fail closed");
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
    const std::vector<kxc::relay::LoweredFunction> units = LowerUnits(func);
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
        {"flatten_and_reshape_product_arithmetic", TestFlattenAndReshapeProductArithmetic},
        {"gather_infer_and_lowering_contract", TestGatherInferAndLoweringContract},
        {"concatenate_infer_and_lowering_contract", TestConcatenateInferAndLoweringContract},
        {"slice_infer_and_lowering_contract", TestSliceInferAndLoweringContract},
        {"where_infer_and_lowering_contract", TestWhereInferAndLoweringContract},
        {"layer_norm_infer_and_lowering_contract", TestLayerNormInferAndLoweringContract},
        {"exact_transformer_operator_slice", TestExactTransformerOperatorSliceComposition},
        {"softmax_infer_type_contract", TestSoftmaxInferTypeContract},
        {"negative_extent_lowering_gates", TestNegativeExtentLoweringGates},
        {"static_lowering_size_gates", TestStaticLoweringSizeGates},
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
