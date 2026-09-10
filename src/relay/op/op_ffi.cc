/*! \file src/relay/op/op_ffi.cc
 * \brief 注册 Relay MVP 算子的 canonical Python `_make` 构造入口。
 */

#include "kxc/ffi/packed_func.h"
#include "kxc/ffi/registration.h"
#include "kxc/relay/op.h"
#include "kxc/relay/relay.h"

#include <string>

namespace kxc {
namespace relay {

namespace {

// 通过稳定名称取得全局驻留的 Relay 算子对象。
inline const Op& GetOp(const std::string& name) { return Op::Get(name); }

}  // namespace

// 以下 Make* 函数为 Python FFI 构造规范 Relay Call，并把属性封装进对象系统。
// 构造二元加法调用。
Call MakeAdd(Expr lhs, Expr rhs) {
    return Call(GetOp("add"), {lhs, rhs});
}

// 构造二元减法调用。
Call MakeSubtract(Expr lhs, Expr rhs) {
    return Call(GetOp("subtract"), {lhs, rhs});
}

// 构造二元乘法调用。
Call MakeMul(Expr lhs, Expr rhs) {
    return Call(GetOp("mul"), {lhs, rhs});
}

// 构造二元除法调用。
Call MakeDivide(Expr lhs, Expr rhs) {
    return Call(GetOp("divide"), {lhs, rhs});
}

// 构造二元逐元素相等比较调用。
Call MakeEqual(Expr lhs, Expr rhs) {
    return Call(GetOp("equal"), {lhs, rhs});
}

// 构造一元逐元素取负调用。
Call MakeNeg(Expr data) {
    return Call(GetOp("neg"), {data});
}

// 构造一元 logistic sigmoid 调用。
Call MakeSigmoid(Expr data) {
    return Call(GetOp("sigmoid"), {data});
}

Call MakeTanh(Expr data) {
    return Call(GetOp("tanh"), {data});
}

Call MakeErf(Expr data) {
    return Call(GetOp("erf"), {data});
}

// 构造二元逐元素幂调用。
Call MakePow(Expr lhs, Expr rhs) {
    return Call(GetOp("pow"), {lhs, rhs});
}

// 构造三元 Where 调用。
Call MakeWhere(Expr condition, Expr x, Expr y) {
    return Call(GetOp("where"), {condition, x, y});
}

// 构造平方根调用。
Call MakeSqrt(Expr data) {
    return Call(GetOp("sqrt"), {data});
}

// 构造矩阵乘调用。
Call MakeMatmul(Expr lhs, Expr rhs) {
    return Call(GetOp("matmul"), {lhs, rhs});
}

// 构造带目标 dtype 的转换调用。
Call MakeCast(Expr data, int dtype) {
    return Call(GetOp("cast"), {data}, CastAttrs::Create(dtype));
}

// 构造均值归约调用，并保留对象容器中的轴列表。
Call MakeReduceMean(Expr data, Array<int64_t> axes, int64_t keepdims) {
    return Call(GetOp("reduce_mean"), {data}, ReduceMeanAttrs::Create(std::move(axes), keepdims));
}

// 构造最大/最小归约调用。
Call MakeReduceMax(Expr data, Array<int64_t> axes, int64_t keepdims) {
    return Call(GetOp("reduce_max"), {data}, ReduceMaxAttrs::Create(std::move(axes), keepdims));
}

Call MakeReduceMin(Expr data, Array<int64_t> axes, int64_t keepdims) {
    return Call(GetOp("reduce_min"), {data}, ReduceMinAttrs::Create(std::move(axes), keepdims));
}

// 构造 reshape 调用，并保留有符号目标维度。
Call MakeReshape(Expr data, Array<int64_t> newshape, int allowzero) {
    return Call(GetOp("reshape"), {data}, ReshapeAttrs::Create(std::move(newshape), allowzero));
}

// 构造 shape_of 调用，把输入维度物化为 int64 形状值。
Call MakeShapeOf(Expr data) {
    return Call(GetOp("shape_of"), {data});
}

// 构造受限形状表达式调用（链式折叠后的单元形态）。
Call MakeShapeExpr(Expr data, Array<int64_t> expr_kinds,
                   Array<int64_t> expr_values, Array<int64_t> expr_axes) {
    return Call(GetOp("shape_expr"), {data},
                ShapeExprAttrs::Create(std::move(expr_kinds), std::move(expr_values),
                                       std::move(expr_axes)));
}

// 构造控制输入 reshape 调用，携带已解析的受限形状表达式。
Call MakeReshapeDynamic(Expr data, Expr shape, Array<int64_t> expr_kinds,
                        Array<int64_t> expr_values, Array<int64_t> expr_axes) {
    return Call(GetOp("reshape_dynamic"), {data, shape},
                ReshapeDynamicAttrs::Create(std::move(expr_kinds),
                                            std::move(expr_values),
                                            std::move(expr_axes)));
}

// 构造受限目标 expand_dynamic 调用。
Call MakeExpandDynamic(Expr data, Expr shape, Array<int64_t> expr_kinds,
                Array<int64_t> expr_values, Array<int64_t> expr_axes) {
    return Call(GetOp("expand_dynamic"), {data, shape},
                ExpandDynamicAttrs::Create(std::move(expr_kinds), std::move(expr_values),
                                    std::move(expr_axes)));
}

// 构造常量目标 constant_of_shape 调用，携带显式填充 dtype 与标量值。
Call MakeConstantOfShape(Expr shape, Array<int64_t> target, int dtype_code,
                         double value) {
    return Call(GetOp("constant_of_shape"), {shape},
                ConstantOfShapeAttrs::Create(std::move(target), dtype_code, value));
}

Call MakeTrilu(Expr data, int upper, int64_t k) {
    return Call(GetOp("trilu"), {data}, TriluAttrs::Create(upper, k));
}

// 构造 squeeze 调用，axes 静态已知。
Call MakeSqueeze(Expr data, Array<int64_t> axes) {
    return Call(GetOp("squeeze"), {data}, SqueezeAttrs::Create(std::move(axes)));
}

// 构造 unsqueeze 调用，axes 静态已知。
Call MakeUnsqueeze(Expr data, Array<int64_t> axes) {
    return Call(GetOp("unsqueeze"), {data}, UnsqueezeAttrs::Create(std::move(axes)));
}

// 构造 expand 调用，并保留已解析的静态目标形状。
Call MakeExpand(Expr data, Array<int64_t> target_shape) {
    return Call(GetOp("expand"), {data}, ExpandAttrs::Create(std::move(target_shape)));
}

// 构造 softmax 调用。
Call MakeSoftmax(Expr data, int axis) {
    return Call(GetOp("softmax"), {data}, SoftmaxAttrs::Create(axis));
}

Call MakeMaskedSoftmax(Expr data, Expr mask, int axis) {
    return Call(GetOp("masked_softmax"), {data, mask}, SoftmaxAttrs::Create(axis));
}

// 构造 transpose 调用。
Call MakeTranspose(Expr data, Array<int64_t> axes) {
    return Call(GetOp("transpose"), {data}, TransposeAttrs::Create(std::move(axes)));
}

// 构造 Gather 调用。
Call MakeGather(Expr data, Expr indices, int axis) {
    return Call(GetOp("gather"), {data, indices}, GatherAttrs::Create(axis));
}

// 构造 exact-static positive-step slice 调用。
Call MakeSlice(Expr data, Array<int64_t> starts, Array<int64_t> ends,
               Array<int64_t> axes, Array<int64_t> steps) {
    return Call(GetOp("slice"), {data}, SliceAttrs::Create(
        std::move(starts), std::move(ends), std::move(axes), std::move(steps)));
}

// 构造 exact-static binary concatenate 调用。
Call MakeConcatenate(Expr lhs, Expr rhs, int axis) {
    return Call(GetOp("concatenate"), {lhs, rhs}, ConcatenateAttrs::Create(axis));
}

// 构造静态多路 Split 调用，并把分段长度保留在 canonical attrs 中。
Call MakeSplit(Expr data, int axis, Array<int64_t> sections) {
    return Call(GetOp("split"), {data}, SplitAttrs::Create(axis, std::move(sections)));
}

// 构造二维卷积调用及完整布局属性。
Call MakeNNConv2D(Expr data, Expr weight, Array<int64_t> strides,
                  Array<int64_t> padding, Array<int64_t> dilation, int groups,
                  int channels, Array<int64_t> kernel_size, std::string data_layout,
                  std::string kernel_layout, std::string out_layout, std::string out_dtype) {
    return Call(GetOp("nn_conv2d"), {data, weight},
                Conv2DAttrs::Create(std::move(strides), std::move(padding), std::move(dilation),
                                    groups, channels, std::move(kernel_size),
                                    std::move(data_layout), std::move(kernel_layout),
                                    std::move(out_layout), std::move(out_dtype)));
}

// 构造全连接调用。
Call MakeNNDense(Expr data, Expr weight, int units, std::string out_dtype) {
    return Call(GetOp("nn_dense"), {data, weight}, DenseAttrs::Create(units, std::move(out_dtype)));
}

// 构造 exact-static affine LayerNorm 调用。
Call MakeNNLayerNorm(Expr data, Expr scale, Expr bias, int axis, double epsilon,
                     std::string accumulation_dtype) {
    return Call(GetOp("nn_layer_norm"), {data, scale, bias},
                LayerNormAttrs::Create(axis, static_cast<float>(epsilon),
                                       std::move(accumulation_dtype)));
}

// 构造 ReLU 调用。
Call MakeNNRelu(Expr data) {
    return Call(GetOp("nn_relu"), {data}, ReluAttrs::Create());
}

// 构造最大池化调用。
Call MakeNNMaxPool2D(Expr data, Array<int64_t> strides, Array<int64_t> padding,
                     Array<int64_t> dilation, Array<int64_t> pool_size,
                     std::string layout, bool ceil_mode) {
    return Call(GetOp("nn_max_pool2d"), {data},
                MaxPool2DAttrs::Create(std::move(strides), std::move(padding),
                                       std::move(dilation), std::move(pool_size),
                                       std::move(layout), ceil_mode));
}

// 构造平均池化调用。
Call MakeNNAvgPool2D(Expr data, Array<int64_t> strides, Array<int64_t> padding,
                     Array<int64_t> dilation, Array<int64_t> pool_size,
                     std::string layout, bool ceil_mode) {
    return Call(GetOp("nn_avg_pool2d"), {data},
                MaxPool2DAttrs::Create(std::move(strides), std::move(padding),
                                       std::move(dilation), std::move(pool_size),
                                       std::move(layout), ceil_mode));
}

// 构造全局平均池化调用。
Call MakeNNGlobalAvgPool2D(Expr data) {
    return Call(GetOp("nn_global_avg_pool2d"), {data}, GlobalAvgPool2DAttrs::Create());
}

// 构造 flatten 调用。
Call MakeNNFlatten(Expr data, int axis) {
    return Call(GetOp("nn_flatten"), {data}, FlattenAttrs::Create(axis));
}

// 构造带缩放与转置属性的 GEMM 调用。
Call MakeNNGemm(Expr a, Expr b, Expr c, double alpha, double beta, int trans_a, int trans_b) {
    return Call(GetOp("nn_gemm"), {a, b, c},
                GemmAttrs::Create(static_cast<float>(alpha), static_cast<float>(beta), trans_a,
                                  trans_b));
}

// 将规范构造函数批量注册为 Python `_make` PackedFunc 入口。
KXC_REGISTER_GLOBAL("kxc.relay.op._make.add").set_body(ToPackedFunc(MakeAdd));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.subtract").set_body(ToPackedFunc(MakeSubtract));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.mul").set_body(ToPackedFunc(MakeMul));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.divide").set_body(ToPackedFunc(MakeDivide));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.equal").set_body(ToPackedFunc(MakeEqual));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.neg").set_body(ToPackedFunc(MakeNeg));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.sigmoid").set_body(ToPackedFunc(MakeSigmoid));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.tanh").set_body(ToPackedFunc(MakeTanh));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.erf").set_body(ToPackedFunc(MakeErf));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.pow").set_body(ToPackedFunc(MakePow));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.expand").set_body(ToPackedFunc(MakeExpand));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.where").set_body(ToPackedFunc(MakeWhere));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.sqrt").set_body(ToPackedFunc(MakeSqrt));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.matmul").set_body(ToPackedFunc(MakeMatmul));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.cast").set_body(ToPackedFunc(MakeCast));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.reduce_mean").set_body(ToPackedFunc(MakeReduceMean));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.reduce_max").set_body(ToPackedFunc(MakeReduceMax));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.reduce_min").set_body(ToPackedFunc(MakeReduceMin));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.reshape").set_body(ToPackedFunc(MakeReshape));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.shape_of").set_body(ToPackedFunc(MakeShapeOf));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.shape_expr")
    .set_body(ToPackedFunc(MakeShapeExpr));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.reshape_dynamic")
    .set_body(ToPackedFunc(MakeReshapeDynamic));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.expand_dynamic").set_body(ToPackedFunc(MakeExpandDynamic));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.constant_of_shape")
    .set_body(ToPackedFunc(MakeConstantOfShape));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.trilu").set_body(ToPackedFunc(MakeTrilu));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.squeeze").set_body(ToPackedFunc(MakeSqueeze));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.unsqueeze")
    .set_body(ToPackedFunc(MakeUnsqueeze));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.softmax").set_body(ToPackedFunc(MakeSoftmax));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.masked_softmax").set_body(ToPackedFunc(MakeMaskedSoftmax));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.transpose").set_body(ToPackedFunc(MakeTranspose));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.gather").set_body(ToPackedFunc(MakeGather));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.slice").set_body(ToPackedFunc(MakeSlice));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.concatenate").set_body(ToPackedFunc(MakeConcatenate));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.split").set_body(ToPackedFunc(MakeSplit));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.nn_conv2d").set_body(ToPackedFunc(MakeNNConv2D));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.nn_dense").set_body(ToPackedFunc(MakeNNDense));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.nn_layer_norm")
    .set_body(ToPackedFunc(MakeNNLayerNorm));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.nn_relu").set_body(ToPackedFunc(MakeNNRelu));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.nn_max_pool2d").set_body(ToPackedFunc(MakeNNMaxPool2D));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.nn_avg_pool2d").set_body(ToPackedFunc(MakeNNAvgPool2D));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.nn_global_avg_pool2d")
    .set_body(ToPackedFunc(MakeNNGlobalAvgPool2D));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.nn_flatten").set_body(ToPackedFunc(MakeNNFlatten));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.nn_gemm").set_body(ToPackedFunc(MakeNNGemm));

}  // namespace relay
}  // namespace kxc

namespace kxc::builtin_anchor {
void RelayOpFfi() {}
}  // namespace kxc::builtin_anchor
