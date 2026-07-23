/*! \file src/relay/op/nn/layer_norm.cc
 * \brief Registers exact-static float32 LayerNorm with float64 intermediates.
 */

#include "kxc/relay/op_attr_types.h"
#include "kxc/relay/op_macros.h"
#include "kxc/relay/type_infer.h"
#include "kxc/te/topi/elemwise.h"
#include "kxc/te/topi/reduction.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace kxc {
namespace relay {
namespace {

void RequireInputCount(const Array<te::Tensor>& inputs) {
    if (inputs.size() != 3) {
        throw std::runtime_error("nn_layer_norm expects exactly 3 inputs");
    }
}

}  // namespace

te::Tensor LayerNormCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                            const kxc::Type& out_type) {
    RequireInputCount(inputs);
    const auto* output = out_type.As<TensorTypeNode>();
    const auto* layer_norm_attrs = attrs.As<LayerNormAttrsNode>();
    if (!output || !layer_norm_attrs || output->dtype != "float32") {
        throw std::runtime_error("nn_layer_norm lowering requires checked float32 attrs and output");
    }
    if (inputs[0]->dtype.code != 2 || inputs[0]->dtype.bits != 32 ||
        inputs[1]->dtype.code != 2 || inputs[1]->dtype.bits != 32 ||
        inputs[2]->dtype.code != 2 || inputs[2]->dtype.bits != 32) {
        throw std::runtime_error("nn_layer_norm lowering requires float32 inputs");
    }

    const int rank = static_cast<int>(inputs[0]->shape.size());
    int axis = layer_norm_attrs->axis;
    if (axis < 0) axis += rank;
    if (rank == 0 || axis < 0 || axis >= rank ||
        !std::isfinite(layer_norm_attrs->epsilon) || layer_norm_attrs->epsilon <= 0.0f ||
        layer_norm_attrs->accumulation_dtype != "float64") {
        throw std::runtime_error("nn_layer_norm lowering received invalid checked attrs");
    }

    const size_t suffix_rank = static_cast<size_t>(rank - axis);
    if (inputs[1]->shape.size() != suffix_rank || inputs[2]->shape.size() != suffix_rank) {
        throw std::runtime_error(
            "nn_layer_norm lowering requires scale and bias shapes to equal data.shape[axis:]");
    }
    Array<int> reduce_axes;
    int64_t normalized_size = 1;
    for (int index = axis; index < rank; ++index) {
        int64_t extent = 0;
        int64_t scale_extent = 0;
        int64_t bias_extent = 0;
        const size_t suffix_index = static_cast<size_t>(index - axis);
        if (!te::topi::GetConstInt(inputs[0]->shape[static_cast<size_t>(index)], &extent) ||
            !te::topi::GetConstInt(inputs[1]->shape[suffix_index], &scale_extent) ||
            !te::topi::GetConstInt(inputs[2]->shape[suffix_index], &bias_extent) ||
            extent <= 0 || scale_extent != extent || bias_extent != extent) {
            throw std::runtime_error(
                "nn_layer_norm lowering requires positive exact normalized suffix dimensions");
        }
        if (normalized_size > std::numeric_limits<int64_t>::max() / extent) {
            throw std::runtime_error("nn_layer_norm normalized element count overflows int64");
        }
        normalized_size *= extent;
        reduce_axes.push_back(index);
    }

    const tir::DataType f32 = tir::DataType::Float(32);
    const tir::DataType f64 = tir::DataType::Float(64);
    const tir::PrimExpr divisor = tir::FloatImm(static_cast<double>(normalized_size), f64);
    const tir::PrimExpr epsilon = tir::FloatImm(
        static_cast<double>(layer_norm_attrs->epsilon), f64);
    const te::Tensor data_f64 = te::topi::cast(
        inputs[0], f64, "T_layer_norm_data_f64");
    const te::Tensor scale_f64 = te::topi::cast(
        inputs[1], f64, "T_layer_norm_scale_f64");
    const te::Tensor bias_f64 = te::topi::cast(
        inputs[2], f64, "T_layer_norm_bias_f64");
    const te::Tensor sum = te::topi::sum(
        data_f64, reduce_axes, true, "T_layer_norm_sum_f64");
    const te::Tensor mean = te::compute(
        sum->shape,
        [sum, divisor](const Array<tir::Var>& indices) { return sum(indices) / divisor; },
        "T_layer_norm_mean_f64");
    const te::Tensor centered = te::compute(
        data_f64->shape,
        [data_f64, mean, axis](const Array<tir::Var>& indices) {
            Array<tir::PrimExpr> mean_indices;
            for (size_t index = 0; index < indices.size(); ++index) {
                mean_indices.push_back(index < static_cast<size_t>(axis)
                                           ? indices[index]
                                           : tir::IntImm(0, tir::DataType::Int(64)));
            }
            return data_f64(indices) - mean(mean_indices);
        },
        "T_layer_norm_centered_f64");
    const te::Tensor squared = te::compute(
        centered->shape,
        [centered](const Array<tir::Var>& indices) {
            const tir::PrimExpr value = centered(indices);
            return value * value;
        },
        "T_layer_norm_squared_f64");
    const te::Tensor variance_sum = te::topi::sum(
        squared, reduce_axes, true, "T_layer_norm_variance_sum_f64");
    const te::Tensor variance = te::compute(
        variance_sum->shape,
        [variance_sum, divisor](const Array<tir::Var>& indices) {
            return variance_sum(indices) / divisor;
        },
        "T_layer_norm_variance_f64");
    const te::Tensor inv_std = te::compute(
        variance->shape,
        [variance, epsilon, f64](const Array<tir::Var>& indices) {
            return tir::FloatImm(1.0, f64) /
                   te::topi::sqrt(variance(indices) + epsilon);
        },
        "T_layer_norm_inv_std_f64");
    const te::Tensor affine_f64 = te::compute(
        data_f64->shape,
        [centered, inv_std, scale_f64, bias_f64, axis](const Array<tir::Var>& indices) {
            Array<tir::PrimExpr> reduced_indices;
            Array<tir::PrimExpr> affine_indices;
            for (size_t index = 0; index < indices.size(); ++index) {
                reduced_indices.push_back(index < static_cast<size_t>(axis)
                                              ? indices[index]
                                              : tir::IntImm(0, tir::DataType::Int(64)));
                if (index >= static_cast<size_t>(axis)) affine_indices.push_back(indices[index]);
            }
            return centered(indices) * inv_std(reduced_indices) *
                       scale_f64(affine_indices) +
                   bias_f64(affine_indices);
        },
        "T_layer_norm_affine_f64");
    return te::topi::cast(affine_f64, f32, "T_layer_norm");
}

KXC_REGISTER_OP(nn_layer_norm)
    .describe(R"doc(Exact-static float32 affine LayerNorm with float64 mean/variance intermediates.)doc")
    .set_num_inputs(3)
    .add_argument("data", "Tensor", "float32 input tensor.")
    .add_argument("scale", "Tensor", "float32 affine scale matching normalized suffix.")
    .add_argument("bias", "Tensor", "float32 affine bias matching normalized suffix.")
    .set_attr<std::string>("TAttrs", "LayerNormAttrs")
    .set_attr<FInferType>("FInferType", LayerNormInferType)
    .set_attr<FRelayToTE>("FRelayToTE", LayerNormCompute);

}  // namespace relay
}  // namespace kxc

namespace kxc::builtin_anchor {
void RelayLayerNormOps() {}
}  // namespace kxc::builtin_anchor
