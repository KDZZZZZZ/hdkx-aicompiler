/*! \file src/relay/op/nn/softmax.cc
 * \brief 注册 Relay softmax 算子及其 TE lowering hook。
 */

#include "kxc/relay/op_attr_types.h"
#include "kxc/relay/op_macros.h"
#include "kxc/relay/type_infer.h"
#include "kxc/te/topi/broadcast.h"
#include "kxc/te/topi/elemwise.h"
#include "kxc/te/topi/reduction.h"

#include <stdexcept>
#include <string>

namespace kxc {
namespace relay {

namespace {

void RequireInputCount(const char* op_name, const Array<te::Tensor>& inputs, size_t expected) {
    if (inputs.size() != expected) {
        throw std::runtime_error(std::string(op_name) + " expects exactly " +
                                 std::to_string(expected) + " input(s)");
    }
}

void RequireTensorOutput(const char* op_name, const kxc::Type& out_type) {
    if (!out_type.As<TensorTypeNode>()) {
        throw std::runtime_error(std::string(op_name) + " expects TensorType output");
    }
}

int NormalizeAxis(const std::string& op_name, int64_t axis, int rank) {
    if (axis < 0) {
        axis += rank;
    }
    if (axis < 0 || axis >= rank) {
        throw std::runtime_error(op_name + " axis out of range");
    }
    return static_cast<int>(axis);
}

}  // namespace

te::Tensor SoftmaxCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                          const kxc::Type& out_type) {
    RequireInputCount("softmax", inputs, 1);
    RequireTensorOutput("softmax", out_type);
    if (inputs[0]->dtype.code != 2) {
        throw std::runtime_error("softmax lowering requires floating-point input");
    }

    const auto* softmax_attrs = attrs.As<SoftmaxAttrsNode>();
    const int rank = static_cast<int>(inputs[0]->shape.size());
    const int axis = NormalizeAxis("softmax", softmax_attrs ? softmax_attrs->axis : -1, rank);
    Array<int> reduce_axis = {axis};

    te::Tensor max_out = te::topi::max(inputs[0], reduce_axis, true, "T_softmax_max");
    te::Tensor shifted =
        te::topi::subtract(inputs[0], max_out, "T_softmax_shifted");
    te::Tensor exp_out = te::topi::exp(shifted, "T_softmax_exp");
    te::Tensor sum_out = te::topi::sum(exp_out, reduce_axis, true, "T_softmax_sum");
    return te::compute(
        inputs[0]->shape,
        [exp_out, sum_out, axis](const Array<kxc::tir::Var>& indices) {
            Array<kxc::tir::PrimExpr> denom_indices;
            for (size_t i = 0; i < indices.size(); ++i) {
                denom_indices.push_back(i == static_cast<size_t>(axis)
                                            ? kxc::tir::IntImm(0, kxc::tir::DataType::Int(64))
                                            : indices[i]);
            }
            return exp_out(indices) / sum_out(denom_indices);
        },
        "T_softmax");
}

KXC_REGISTER_OP(softmax)
    .describe(R"doc(Computes softmax along one tensor axis.)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<std::string>("TAttrs", "SoftmaxAttrs")
    .set_attr<FInferType>("FInferType", SoftmaxInferType)
    .set_attr<FRelayToTE>("FRelayToTE", SoftmaxCompute);

}  // namespace relay
}  // namespace kxc

namespace kxc::builtin_anchor {
void RelaySoftmaxOps() {}
}  // namespace kxc::builtin_anchor
