/*! \file src/relay/op/op_ffi.cc
 * \brief 注册 Relay MVP 算子的 canonical Python `_make` 构造入口。
 */

#include "base/packedfunc.h"
#include "base/registry.h"
#include "relay/op.h"
#include "relay/relay.h"

#include <string>
#include <vector>

namespace kxc {
namespace relay {

namespace {

inline const Op& GetOp(const std::string& name) { return Op::Get(name); }

}  // namespace

Call MakeAdd(Expr lhs, Expr rhs) {
    return Call(GetOp("add"), {lhs, rhs});
}

Call MakeSubtract(Expr lhs, Expr rhs) {
    return Call(GetOp("subtract"), {lhs, rhs});
}

Call MakeMul(Expr lhs, Expr rhs) {
    return Call(GetOp("mul"), {lhs, rhs});
}

Call MakeDivide(Expr lhs, Expr rhs) {
    return Call(GetOp("divide"), {lhs, rhs});
}

Call MakeSqrt(Expr data) {
    return Call(GetOp("sqrt"), {data});
}

Call MakeMatmul(Expr lhs, Expr rhs) {
    return Call(GetOp("matmul"), {lhs, rhs});
}

Call MakeCast(Expr data, int dtype) {
    return Call(GetOp("cast"), {data}, CastAttrs::Create(dtype));
}

Call MakeReduceMean(Expr data, std::vector<int64_t> axes, int64_t keepdims) {
    return Call(GetOp("reduce_mean"), {data}, ReduceMeanAttrs::Create(std::move(axes), keepdims));
}

Call MakeReshape(Expr data, std::vector<int64_t> newshape, int allowzero) {
    return Call(GetOp("reshape"), {data}, ReshapeAttrs::Create(std::move(newshape), allowzero));
}

Call MakeSoftmax(Expr data, int axis) {
    return Call(GetOp("softmax"), {data}, SoftmaxAttrs::Create(axis));
}

Call MakeTranspose(Expr data, std::vector<int64_t> axes) {
    return Call(GetOp("transpose"), {data}, TransposeAttrs::Create(std::move(axes)));
}

Call MakeNNConv2D(Expr data, Expr weight, std::vector<int64_t> strides,
                  std::vector<int64_t> padding, std::vector<int64_t> dilation, int groups,
                  int channels, std::vector<int64_t> kernel_size, std::string data_layout,
                  std::string kernel_layout, std::string out_layout, std::string out_dtype) {
    return Call(GetOp("nn_conv2d"), {data, weight},
                Conv2DAttrs::Create(std::move(strides), std::move(padding), std::move(dilation),
                                    groups, channels, std::move(kernel_size),
                                    std::move(data_layout), std::move(kernel_layout),
                                    std::move(out_layout), std::move(out_dtype)));
}

Call MakeNNDense(Expr data, Expr weight, int units, std::string out_dtype) {
    return Call(GetOp("nn_dense"), {data, weight}, DenseAttrs::Create(units, std::move(out_dtype)));
}

Call MakeNNRelu(Expr data) {
    return Call(GetOp("nn_relu"), {data}, ReluAttrs::Create());
}

Call MakeNNMaxPool2D(Expr data, std::vector<int64_t> strides, std::vector<int64_t> padding,
                     std::vector<int64_t> dilation, std::vector<int64_t> pool_size,
                     std::string layout, bool ceil_mode) {
    return Call(GetOp("nn_max_pool2d"), {data},
                MaxPool2DAttrs::Create(std::move(strides), std::move(padding),
                                       std::move(dilation), std::move(pool_size),
                                       std::move(layout), ceil_mode));
}

Call MakeNNAvgPool2D(Expr data, std::vector<int64_t> strides, std::vector<int64_t> padding,
                     std::vector<int64_t> dilation, std::vector<int64_t> pool_size,
                     std::string layout, bool ceil_mode) {
    return Call(GetOp("nn_avg_pool2d"), {data},
                MaxPool2DAttrs::Create(std::move(strides), std::move(padding),
                                       std::move(dilation), std::move(pool_size),
                                       std::move(layout), ceil_mode));
}

Call MakeNNGlobalAvgPool2D(Expr data) {
    return Call(GetOp("nn_global_avg_pool2d"), {data}, GlobalAvgPool2DAttrs::Create());
}

Call MakeNNFlatten(Expr data, int axis) {
    return Call(GetOp("nn_flatten"), {data}, FlattenAttrs::Create(axis));
}

Call MakeNNGemm(Expr a, Expr b, Expr c, double alpha, double beta, int trans_a, int trans_b) {
    return Call(GetOp("nn_gemm"), {a, b, c},
                GemmAttrs::Create(static_cast<float>(alpha), static_cast<float>(beta), trans_a,
                                  trans_b));
}

KXC_REGISTER_GLOBAL("kxc.relay.op._make.add").set_body(ToPackedFunc(MakeAdd));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.subtract").set_body(ToPackedFunc(MakeSubtract));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.mul").set_body(ToPackedFunc(MakeMul));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.divide").set_body(ToPackedFunc(MakeDivide));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.sqrt").set_body(ToPackedFunc(MakeSqrt));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.matmul").set_body(ToPackedFunc(MakeMatmul));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.cast").set_body(ToPackedFunc(MakeCast));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.reduce_mean").set_body(ToPackedFunc(MakeReduceMean));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.reshape").set_body(ToPackedFunc(MakeReshape));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.softmax").set_body(ToPackedFunc(MakeSoftmax));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.transpose").set_body(ToPackedFunc(MakeTranspose));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.nn_conv2d").set_body(ToPackedFunc(MakeNNConv2D));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.nn_dense").set_body(ToPackedFunc(MakeNNDense));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.nn_relu").set_body(ToPackedFunc(MakeNNRelu));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.nn_max_pool2d").set_body(ToPackedFunc(MakeNNMaxPool2D));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.nn_avg_pool2d").set_body(ToPackedFunc(MakeNNAvgPool2D));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.nn_global_avg_pool2d")
    .set_body(ToPackedFunc(MakeNNGlobalAvgPool2D));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.nn_flatten").set_body(ToPackedFunc(MakeNNFlatten));
KXC_REGISTER_GLOBAL("kxc.relay.op._make.nn_gemm").set_body(ToPackedFunc(MakeNNGemm));

}  // namespace relay
}  // namespace kxc
