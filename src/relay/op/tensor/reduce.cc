/*! \file src/relay/op/tensor/reduce.cc
 * \brief 注册 Relay reduce 算子及其 TE lowering hook。
 */

#include "relay/op_attr_types.h"
#include "relay/op_macros.h"
#include "relay/type_infer.h"
#include "te/topi/reduction.h"
#include "te/topi/utils.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

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

std::vector<int> NormalizeAxes(const std::string& op_name, const std::vector<int64_t>& axes,
                               int rank) {
    std::vector<int> out;
    if (axes.empty()) {
        for (int i = 0; i < rank; ++i) {
            out.push_back(i);
        }
        return out;
    }
    for (int64_t axis : axes) {
        if (axis < 0) {
            axis += rank;
        }
        if (axis < 0 || axis >= rank) {
            throw std::runtime_error(op_name + " axis out of range");
        }
        int normalized = static_cast<int>(axis);
        if (std::find(out.begin(), out.end(), normalized) != out.end()) {
            throw std::runtime_error(op_name + " duplicate axis");
        }
        out.push_back(normalized);
    }
    return out;
}

Array<int> ToAxisArray(const std::vector<int>& axes) {
    Array<int> out;
    for (int axis : axes) {
        out.push_back(axis);
    }
    return out;
}

int64_t ReductionElementCount(const te::Tensor& input, const std::vector<int>& axes) {
    int64_t count = 1;
    for (int axis : axes) {
        int64_t dim = 0;
        if (!te::topi::GetConstInt(input->shape[static_cast<size_t>(axis)], &dim) || dim < 0) {
            throw std::runtime_error("reduce_mean lowering requires static reduction shape");
        }
        count *= dim;
    }
    return count;
}

}  // namespace

te::Tensor ReduceMeanCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                             const kxc::Type& out_type) {
    RequireInputCount("reduce_mean", inputs, 1);
    RequireTensorOutput("reduce_mean", out_type);
    const auto* reduce_attrs = attrs.As<ReduceMeanAttrsNode>();
    const int rank = static_cast<int>(inputs[0]->shape.size());
    const std::vector<int> axes = NormalizeAxes(
        "reduce_mean", reduce_attrs ? reduce_attrs->axes : std::vector<int64_t>{}, rank);
    const bool keepdims = !reduce_attrs || reduce_attrs->keepdims != 0;

    te::Tensor sum_out = te::topi::sum(inputs[0], ToAxisArray(axes), keepdims, "T_reduce_mean_sum");
    const int64_t denominator = ReductionElementCount(inputs[0], axes);
    return te::compute(
        sum_out->shape,
        [sum_out, denominator](const Array<kxc::tir::Var>& indices) {
            return sum_out(indices) / te::topi::make_const(sum_out->dtype, denominator);
        },
        "T_reduce_mean");
}

KXC_REGISTER_OP(reduce_mean)
    .describe(R"doc(Computes the mean of elements across given dimensions.)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<std::string>("TAttrs", "ReduceMeanAttrs")
    .set_attr<FInferType>("FInferType", ReduceMeanInferType)
    .set_attr<FRelayToTE>("FRelayToTE", ReduceMeanCompute);

}  // namespace relay
}  // namespace kxc
