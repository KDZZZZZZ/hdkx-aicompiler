/*! \file src/relay/op/nn/dense.cc
 * \brief 注册 Relay 算子及其 FRelayToTE compute。
 */

#include "relay/op_macros.h"
#include "relay/op.h"
#include "relay/op_attr_types.h"
#include "relay/type_infer.h"
#include "te/topi/nn.h"
#include "te/topi/broadcast.h"
#include <stdexcept>

namespace kxc {
namespace relay {

namespace {
te::Tensor ScaleTensor(const te::Tensor& t, double scale, const std::string& name) {
    if (scale == 1.0) {
        return t;
    }
    return te::compute(t->shape, [&](const Array<kxc::tir::Var>& axis) {
        return t(axis) * kxc::te::topi::make_const(t->dtype, scale);
    }, name);
}
}

te::Tensor DenseCompute(const Attrs& attrs, const Array<te::Tensor>& inputs, const kxc::Type& out_type) {
    if (inputs.size() != 2) {
        throw std::runtime_error("nn_dense expects exactly 2 inputs");
    }
    return te::topi::dense(inputs[0], inputs[1], te::Tensor(), "T_dense");
}

te::Tensor GemmCompute(const Attrs& attrs, const Array<te::Tensor>& inputs, const kxc::Type& out_type) {
    if (inputs.size() != 3) {
        throw std::runtime_error("nn_gemm expects exactly 3 inputs");
    }
    auto* p = attrs.As<GemmAttrsNode>();
    if (!p) {
        throw std::runtime_error("nn_gemm expects GemmAttrs");
    }
    if (p->transA != 0) {
        throw std::runtime_error("nn_gemm currently does not support transA=1");
    }

    te::Tensor out;
    if (p->transB == 1) {
        // B is [N, K], Dense expects [N, K].
        out = te::topi::dense(inputs[0], inputs[1], te::Tensor(), "T_gemm_dense");
    } else {
        // B is [K, N]
        out = te::topi::matmul(inputs[0], inputs[1], "T_gemm_matmul");
    }

    out = ScaleTensor(out, p->alpha, "T_gemm_alpha");
    te::Tensor bias = inputs[2];
    if (p->beta != 1.0) {
        bias = ScaleTensor(inputs[2], p->beta, "T_gemm_beta");
    }
    return te::topi::add(out, bias, "T_gemm_out");
}

KXC_REGISTER_OP(nn_dense)
    .describe("Dense (fully connected) layer")
    .set_num_inputs(2) // data, weight
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("weight", "Tensor", "The weight tensor.")
    .set_attr<std::string>("TAttrs", "DenseAttrs")
    .set_attr<FInferType>("FInferType", DenseInferType)
    .set_attr<FRelayToTE>("FRelayToTE", DenseCompute);

// Gemm
KXC_REGISTER_OP(nn_gemm)
    .describe(R"doc(General matrix multiplication.
)doc")
    .set_num_inputs(3)
    .add_argument("A", "Tensor", "The first input tensor.")
    .add_argument("B", "Tensor", "The second input tensor.")
    .add_argument("C", "Tensor", "The third input tensor (bias).")
    .set_attr<std::string>("TAttrs", "GemmAttrs")
    .set_attr<FInferType>("FInferType", GemmInferType)
    .set_attr<FRelayToTE>("FRelayToTE", GemmCompute);

} // namespace relay
} // namespace kxc
