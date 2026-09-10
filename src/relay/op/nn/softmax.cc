/*! \file src/relay/op/nn/softmax.cc
 * \brief 注册 Relay softmax 算子及其 TE lowering hook。
 */

#include "kxc/relay/op_attr_types.h"
#include "kxc/relay/op_macros.h"
#include "kxc/relay/type_infer.h"
#include "kxc/te/topi/broadcast.h"
#include "kxc/te/topi/elemwise.h"
#include "kxc/te/topi/reduction.h"

#include <limits>
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

te::Tensor MaskedSoftmaxCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                const kxc::Type& out_type) {
    RequireInputCount("masked_softmax", inputs, 2);
    RequireTensorOutput("masked_softmax", out_type);
    const auto dtype = inputs[0]->dtype;
    if (dtype.code != kDLFloat || (dtype.bits != 32 && dtype.bits != 64) || dtype.lanes != 1 ||
        inputs[1]->dtype.code != kDLUInt || inputs[1]->dtype.bits != 1 || inputs[1]->dtype.lanes != 1) {
        throw std::runtime_error("masked_softmax lowering requires floating data and bool mask");
    }
    const auto* axis_attrs = attrs.As<SoftmaxAttrsNode>();
    if (attrs.defined() && !axis_attrs) throw std::runtime_error("masked_softmax requires SoftmaxAttrs");
    const int axis = NormalizeAxis("masked_softmax", axis_attrs ? axis_attrs->axis : -1,
                                   static_cast<int>(inputs[0]->shape.size()));
    const auto type = tir::DataType::Float(dtype.bits);
    const tir::PrimExpr zero = tir::FloatImm(0.0, type), one = tir::FloatImm(1.0, type);
    const tir::PrimExpr lowest = tir::FloatImm(dtype.bits == 32
        ? -static_cast<double>(std::numeric_limits<float>::max())
        : -std::numeric_limits<double>::max(), type);
    const auto mask = te::topi::broadcast_to(inputs[1], inputs[0]->shape, "T_masked_softmax_mask");
    const auto masked = te::compute(inputs[0]->shape,
        [data = inputs[0], mask, lowest](const Array<tir::Var>& indices) {
            return tir::Select(mask(indices), data(indices), lowest);
        }, "T_masked_softmax_logits");
    const auto maximum = te::topi::max(masked, {axis}, true, "T_masked_softmax_max");
    const auto reduced_indices = [axis](const Array<tir::Var>& indices) {
        Array<tir::PrimExpr> result;
        for (size_t i = 0; i < indices.size(); ++i) {
            result.push_back(i == static_cast<size_t>(axis)
                ? tir::IntImm(0, tir::DataType::Int(64)) : indices[i]);
        }
        return result;
    };
    const auto shifted = te::compute(inputs[0]->shape,
        [masked, mask, maximum, reduced_indices](const Array<tir::Var>& indices) {
            const auto max_value = maximum(reduced_indices(indices));
            // Masked payloads (including NaN/Inf) cannot enter arithmetic.
            // A finite sentinel also gives an empty row a finite maximum.
            return tir::Select(mask(indices), masked(indices), max_value) - max_value;
        }, "T_masked_softmax_shifted");
    const auto exp = te::topi::exp(shifted, "T_masked_softmax_exp");
    const auto weights = te::compute(inputs[0]->shape,
        [mask, exp, zero](const Array<tir::Var>& indices) {
            return tir::Select(mask(indices), exp(indices), zero);
        }, "T_masked_softmax_weights");
    const auto sum = te::topi::sum(weights, {axis}, true, "T_masked_softmax_sum");
    return te::compute(inputs[0]->shape,
        [weights, sum, zero, one, reduced_indices](const Array<tir::Var>& indices) {
            const auto denominator = sum(reduced_indices(indices));
            return weights(indices) / tir::Select(denominator == zero, one, denominator);
        }, "T_masked_softmax");
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
