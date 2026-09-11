/*! \file src/relay/op/tensor/reduce.cc
 * \brief 注册 Relay reduce 算子及其 TE lowering hook。
 */

#include "kxc/relay/op_attr_types.h"
#include "kxc/relay/op_macros.h"
#include "kxc/relay/type_infer.h"
#include "kxc/te/topi/elemwise.h"
#include "kxc/te/topi/reduction.h"
#include "kxc/te/topi/utils.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace kxc {
namespace relay {

namespace {

// 校验归约算子的 TE 输入数量。
void RequireInputCount(const char* op_name, const Array<te::Tensor>& inputs, size_t expected) {
    if (inputs.size() != expected) {
        throw std::runtime_error(std::string(op_name) + " expects exactly " +
                                 std::to_string(expected) + " input(s)");
    }
}

// 确认归约结果已完成 TensorType 推导。
void RequireTensorOutput(const char* op_name, const kxc::Type& out_type) {
    if (!out_type.As<TensorTypeNode>()) {
        throw std::runtime_error(std::string(op_name) + " expects TensorType output");
    }
}

// 将负轴规范化为非负轴，并拒绝越界或重复轴。
std::vector<int> NormalizeAxes(const std::string& op_name, const Array<int64_t>& axes,
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

// 将标准库轴列表转换为对象系统 Array，供 TOPI 接口持有。
Array<int> ToAxisArray(const std::vector<int>& axes) {
    Array<int> out;
    for (int axis : axes) {
        out.push_back(axis);
    }
    return out;
}

// 计算静态归约域元素数，作为 reduce_mean 的除数。
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

// 将 Relay reduce_mean 降为 TOPI 求和与逐元素除法组合。
te::Tensor ReduceMeanCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                             const kxc::Type& out_type) {
    RequireInputCount("reduce_mean", inputs, 1);
    RequireTensorOutput("reduce_mean", out_type);
    const auto* reduce_attrs = attrs.As<ReduceMeanAttrsNode>();
    const int rank = static_cast<int>(inputs[0]->shape.size());
    const std::vector<int> axes = NormalizeAxes(
        "reduce_mean", reduce_attrs ? reduce_attrs->axes : Array<int64_t>{}, rank);
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

// 注册 reduce_mean 的属性、类型推导和 Relay-to-TE 入口。
KXC_REGISTER_OP(reduce_mean)
    .describe(R"doc(Computes the mean of elements across given dimensions.)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<std::string>("TAttrs", "ReduceMeanAttrs")
    .set_attr<FInferType>("FInferType", ReduceMeanInferType)
    .set_attr<FRelayToTE>("FRelayToTE", ReduceMeanCompute);

// 将 Relay reduce_max / reduce_min 降为 TOPI 最大/最小归约。
te::Tensor ReduceMaxCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                            const kxc::Type& out_type) {
    RequireInputCount("reduce_max", inputs, 1);
    RequireTensorOutput("reduce_max", out_type);
    const auto* reduce_attrs = attrs.As<ReduceMaxAttrsNode>();
    const int rank = static_cast<int>(inputs[0]->shape.size());
    const std::vector<int> axes = NormalizeAxes(
        "reduce_max", reduce_attrs ? reduce_attrs->axes : Array<int64_t>{}, rank);
    const bool keepdims = !reduce_attrs || reduce_attrs->keepdims != 0;
    return te::topi::max(inputs[0], ToAxisArray(axes), keepdims, "T_reduce_max");
}

te::Tensor ReduceMinCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                            const kxc::Type& out_type) {
    RequireInputCount("reduce_min", inputs, 1);
    RequireTensorOutput("reduce_min", out_type);
    const auto* reduce_attrs = attrs.As<ReduceMinAttrsNode>();
    const int rank = static_cast<int>(inputs[0]->shape.size());
    const std::vector<int> axes = NormalizeAxes(
        "reduce_min", reduce_attrs ? reduce_attrs->axes : Array<int64_t>{}, rank);
    const bool keepdims = !reduce_attrs || reduce_attrs->keepdims != 0;
    return te::topi::min(inputs[0], ToAxisArray(axes), keepdims, "T_reduce_min");
}

KXC_REGISTER_OP(reduce_max)
    .describe(R"doc(Computes the maximum of elements across given dimensions.)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<std::string>("TAttrs", "ReduceMaxAttrs")
    .set_attr<FInferType>("FInferType", ReduceMaxInferType)
    .set_attr<FRelayToTE>("FRelayToTE", ReduceMaxCompute);

KXC_REGISTER_OP(reduce_min)
    .describe(R"doc(Computes the minimum of elements across given dimensions.)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<std::string>("TAttrs", "ReduceMinAttrs")
    .set_attr<FInferType>("FInferType", ReduceMinInferType)
    .set_attr<FRelayToTE>("FRelayToTE", ReduceMinCompute);

// 图内 argmax：先在归约轴上求最大值，再对"等于最大值处取轴下标、否则取哨兵"
// 的 int64 张量做 min（select_last_index=0，默认）或 max（=1）。同值时
// select_min 取最小下标即 ONNX 的 first-occurrence 语义，select_max 取最大
// 下标即 last-occurrence。输出 int64。
te::Tensor ArgMaxCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                         const kxc::Type& out_type) {
    RequireInputCount("argmax", inputs, 1);
    RequireTensorOutput("argmax", out_type);
    const auto* arg_attrs = attrs.As<ArgMaxAttrsNode>();
    const auto* input = inputs[0].operator->();
    const int rank = static_cast<int>(input->shape.size());
    if (rank < 1) {
        throw std::runtime_error("argmax requires rank at least 1");
    }
    int axis = static_cast<int>(arg_attrs ? arg_attrs->axis : -1);
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) {
        throw std::runtime_error("argmax axis out of range");
    }
    const bool keepdims = !arg_attrs || arg_attrs->keepdims != 0;
    const bool select_last = arg_attrs && arg_attrs->select_last_index != 0;
    int64_t axis_extent = 0;
    if (!te::topi::GetConstInt(input->shape[static_cast<size_t>(axis)],
                               &axis_extent) ||
        axis_extent <= 0) {
        throw std::runtime_error("argmax reduction axis must be a positive static extent");
    }

    // Max value on the axis, kept so it can be broadcast back by indexing axis 0.
    te::Tensor max_value = te::topi::max(
        inputs[0], Array<int>{axis}, /*keepdims=*/true, "T_argmax_max");

    // selected[i] = axis index when data[i] equals the axis maximum, else the
    // reduction identity so it never wins the index reduction.
    const tir::DataType index_dtype = tir::DataType::Int(64);
    const int64_t identity =
        select_last ? std::numeric_limits<int64_t>::min()
                    : std::numeric_limits<int64_t>::max();
    te::Tensor selected = te::compute(
        input->shape,
        [input_tensor = inputs[0], max_value, axis, index_dtype, identity](
            const Array<tir::Var>& indices) {
            Array<tir::PrimExpr> reduced_indices;
            for (size_t i = 0; i < indices.size(); ++i) {
                if (static_cast<int>(i) == axis) {
                    reduced_indices.push_back(
                        tir::IntImm(0, tir::DataType::Int(32)));
                } else {
                    reduced_indices.push_back(te::AsPrimExpr(indices[i]));
                }
            }
            const tir::PrimExpr is_max =
                input_tensor(indices) == max_value(reduced_indices);
            Array<tir::PrimExpr> cast_arg;
            cast_arg.push_back(te::AsPrimExpr(indices[static_cast<size_t>(axis)]));
            const tir::PrimExpr index =
                tir::Call(index_dtype, "cast", std::move(cast_arg));
            return tir::Select(is_max, index,
                               tir::IntImm(identity, index_dtype));
        },
        "T_argmax_selected");
    return select_last
               ? te::topi::max(selected, Array<int>{axis}, keepdims, "T_argmax")
               : te::topi::min(selected, Array<int>{axis}, keepdims, "T_argmax");
}

KXC_REGISTER_OP(argmax)
    .describe(R"doc(Returns the indices of the maximum values along an axis.)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<std::string>("TAttrs", "ArgMaxAttrs")
    .set_attr<FInferType>("FInferType", ArgMaxInferType)
    .set_attr<FRelayToTE>("FRelayToTE", ArgMaxCompute);

}  // namespace relay
}  // namespace kxc

namespace kxc::builtin_anchor {
void RelayTensorReduceOps() {}
}  // namespace kxc::builtin_anchor
