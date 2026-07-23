/*! \file src/relay/op/tensor/transform.cc
 * \brief 注册 Relay tensor transform 算子及其 FRelayToTE compute。
 */

#include "kxc/relay/op_macros.h"
#include "kxc/relay/op.h"
#include "kxc/relay/op_attr_types.h"
#include "kxc/relay/type_infer.h"
#include "kxc/te/topi/elemwise.h"
#include "kxc/te/topi/transform.h"

#include <limits>
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

const TensorTypeNode* RequireTensorOutput(const char* op_name, const kxc::Type& out_type) {
    const auto* tensor_type = out_type.As<TensorTypeNode>();
    if (!tensor_type) {
        throw std::runtime_error(std::string(op_name) + " expects TensorType output");
    }
    return tensor_type;
}

te::Tensor RequireDefined(const char* op_name, const te::Tensor& tensor) {
    if (!tensor.defined()) {
        throw std::runtime_error(std::string(op_name) + " lowering returned undefined tensor");
    }
    return tensor;
}

Array<kxc::tir::PrimExpr> ShapeFromTensorType(const TensorTypeNode* type,
                                              const std::string& op_name) {
    Array<kxc::tir::PrimExpr> shape;
    for (int64_t dim : type->shape) {
        if (dim < 0) {
            throw std::runtime_error(op_name + " lowering requires static output shape");
        }
        shape.push_back(kxc::tir::IntImm(dim, kxc::tir::DataType::Int(64)));
    }
    return shape;
}

kxc::tir::PrimExpr LinearIndex(const Array<kxc::tir::PrimExpr>& indices,
                               const Array<kxc::tir::PrimExpr>& dims) {
    if (dims.empty()) {
        return kxc::tir::IntImm(0, kxc::tir::DataType::Int(64));
    }
    if (indices.size() != dims.size()) {
        throw std::runtime_error("reshape index rank mismatch");
    }
    kxc::tir::PrimExpr linear = indices[0];
    for (size_t i = 1; i < indices.size(); ++i) {
        linear = linear * dims[i] + indices[i];
    }
    return linear;
}

Array<kxc::tir::PrimExpr> UnflattenIndex(kxc::tir::PrimExpr flat,
                                         const Array<kxc::tir::PrimExpr>& dims) {
    Array<kxc::tir::PrimExpr> out;
    for (size_t i = 0; i < dims.size(); ++i) {
        out.push_back(kxc::tir::IntImm(0, kxc::tir::DataType::Int(64)));
    }
    kxc::tir::PrimExpr cur = flat;
    for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
        out[static_cast<size_t>(i)] = cur % dims[static_cast<size_t>(i)];
        cur = cur / dims[static_cast<size_t>(i)];
    }
    return out;
}

kxc::tir::DataType DTypeFromCastCode(int code) {
    switch (code) {
    case 0:
        return kxc::tir::DataType::Float(32);
    case 1:
        return kxc::tir::DataType::Int(32);
    case 2:
        return kxc::tir::DataType::Int(64);
    case 3:
        return kxc::tir::DataType::Float(64);
    case 4:
        return kxc::tir::DataType::Bool();
    case 5:
        return kxc::tir::DataType::Int(8);
    case 6:
        return kxc::tir::DataType::UInt(8);
    default:
        throw std::runtime_error("cast has unsupported dtype code: " + std::to_string(code));
    }
}

bool IsConcatenateDType(const kxc::tir::DataType& dtype) {
    return (dtype.code == 2 && (dtype.bits == 32 || dtype.bits == 64) && dtype.lanes == 1) ||
           (dtype.code == 0 && (dtype.bits == 8 || dtype.bits == 32 || dtype.bits == 64) &&
            dtype.lanes == 1) ||
           (dtype.code == 1 && dtype.bits == 8 && dtype.lanes == 1) ||
           dtype == kxc::tir::DataType::Bool();
}

bool MatchesRelayDType(const kxc::tir::DataType& dtype, const std::string& relay_dtype) {
    return (relay_dtype == "float32" && dtype == kxc::tir::DataType::Float(32)) ||
           (relay_dtype == "float64" && dtype == kxc::tir::DataType::Float(64)) ||
           (relay_dtype == "int32" && dtype == kxc::tir::DataType::Int(32)) ||
           (relay_dtype == "int64" && dtype == kxc::tir::DataType::Int(64)) ||
           (relay_dtype == "int8" && dtype == kxc::tir::DataType::Int(8)) ||
           (relay_dtype == "uint8" && dtype == kxc::tir::DataType::UInt(8)) ||
           (relay_dtype == "bool" && dtype == kxc::tir::DataType::Bool());
}

int64_t StaticExtent(const kxc::tir::PrimExpr& extent, const char* op_name) {
    const auto* immediate = extent.As<kxc::tir::IntImmNode>();
    if (!immediate || immediate->value < 0) {
        throw std::runtime_error(std::string(op_name) +
                                 " lowering requires static non-negative input shape");
    }
    return immediate->value;
}

Array<int> NormalizeTransposeAxes(const te::Tensor& input, const Attrs& attrs) {
    const int rank = static_cast<int>(input->shape.size());
    Array<int> axes;
    if (const auto* transpose_attrs = attrs.As<TransposeAttrsNode>()) {
        for (int64_t raw_axis : transpose_attrs->perm) {
            int axis = static_cast<int>(raw_axis);
            if (axis < 0) {
                axis += rank;
            }
            if (axis < 0 || axis >= rank) {
                throw std::runtime_error("transpose axis out of range");
            }
            axes.push_back(axis);
        }
    }
    if (axes.empty()) {
        for (int axis = rank - 1; axis >= 0; --axis) {
            axes.push_back(axis);
        }
    }
    if (static_cast<int>(axes.size()) != rank) {
        throw std::runtime_error("transpose perm rank mismatch");
    }
    std::vector<bool> seen(static_cast<size_t>(rank), false);
    for (int axis : axes) {
        if (seen[static_cast<size_t>(axis)]) {
            throw std::runtime_error("transpose duplicate axis");
        }
        seen[static_cast<size_t>(axis)] = true;
    }
    return axes;
}

}  // namespace

te::Tensor FlattenCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                          const kxc::Type& out_type) {
    RequireInputCount("nn_flatten", inputs, 1);
    RequireTensorOutput("nn_flatten", out_type);
    const auto* flatten_attrs = attrs.As<FlattenAttrsNode>();
    int axis = flatten_attrs ? flatten_attrs->axis : 1;
    int ndim = static_cast<int>(inputs[0]->shape.size());
    if (ndim <= 0) {
        throw std::runtime_error("nn_flatten requires rank >= 1");
    }
    if (axis < 0) {
        axis += ndim;
    }
    if (axis < 0 || axis > ndim) {
        throw std::runtime_error("nn_flatten axis out of range");
    }

    kxc::tir::PrimExpr outer = 1;
    kxc::tir::PrimExpr inner = 1;
    Array<kxc::tir::PrimExpr> outer_dims;
    Array<kxc::tir::PrimExpr> inner_dims;
    for (int i = 0; i < axis; ++i) {
        outer = outer * inputs[0]->shape[static_cast<size_t>(i)];
        outer_dims.push_back(inputs[0]->shape[static_cast<size_t>(i)]);
    }
    for (int i = axis; i < ndim; ++i) {
        inner = inner * inputs[0]->shape[static_cast<size_t>(i)];
        inner_dims.push_back(inputs[0]->shape[static_cast<size_t>(i)]);
    }

    te::Tensor out = te::compute(
        {outer, inner},
        [input = inputs[0], outer_dims, inner_dims](const Array<kxc::tir::Var>& indices) {
            Array<kxc::tir::PrimExpr> in_indices;
            auto outer_idx = UnflattenIndex(indices[0], outer_dims);
            auto inner_idx = UnflattenIndex(indices[1], inner_dims);
            for (const auto& value : outer_idx) in_indices.push_back(value);
            for (const auto& value : inner_idx) in_indices.push_back(value);
            return input(in_indices);
        },
        "T_flatten");
    return RequireDefined("nn_flatten", out);
}

te::Tensor ReshapeCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                          const kxc::Type& out_type) {
    (void)attrs;
    RequireInputCount("reshape", inputs, 1);
    const auto* tensor_type = RequireTensorOutput("reshape", out_type);
    Array<kxc::tir::PrimExpr> out_shape = ShapeFromTensorType(tensor_type, "reshape");

    te::Tensor out = te::compute(
        out_shape,
        [input = inputs[0], out_shape](const Array<kxc::tir::Var>& indices) {
            Array<kxc::tir::PrimExpr> out_indices;
            for (const auto& index : indices) {
                out_indices.push_back(index);
            }
            kxc::tir::PrimExpr flat = LinearIndex(out_indices, out_shape);
            Array<kxc::tir::PrimExpr> input_indices = UnflattenIndex(flat, input->shape);
            return input(input_indices);
        },
        "T_reshape");
    return RequireDefined("reshape", out);
}

te::Tensor TransposeCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                            const kxc::Type& out_type) {
    RequireInputCount("transpose", inputs, 1);
    RequireTensorOutput("transpose", out_type);
    Array<int> axes = NormalizeTransposeAxes(inputs[0], attrs);
    return RequireDefined("transpose", te::topi::transpose(inputs[0], axes, "T_transpose"));
}

te::Tensor CastCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                       const kxc::Type& out_type) {
    RequireInputCount("cast", inputs, 1);
    RequireTensorOutput("cast", out_type);
    const auto* cast_attrs = attrs.As<CastAttrsNode>();
    if (!cast_attrs) {
        throw std::runtime_error("cast expects CastAttrs");
    }
    return RequireDefined(
        "cast", te::topi::cast(inputs[0], DTypeFromCastCode(cast_attrs->to), "T_cast"));
}

te::Tensor ConcatenateCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                               const kxc::Type& out_type) {
    RequireInputCount("concatenate", inputs, 2);
    const auto* output_type = RequireTensorOutput("concatenate", out_type);
    const auto* concatenate_attrs = attrs.As<ConcatenateAttrsNode>();
    if (!concatenate_attrs) {
        throw std::runtime_error("concatenate expects ConcatenateAttrs");
    }
    if (!inputs[0].defined() || !inputs[1].defined()) {
        throw std::runtime_error("concatenate lowering received undefined input tensor");
    }
    const size_t rank = inputs[0]->shape.size();
    if (rank == 0 || inputs[1]->shape.size() != rank) {
        throw std::runtime_error("concatenate lowering requires equal rank >= 1 inputs");
    }
    int axis = concatenate_attrs->axis;
    if (axis < 0) axis += static_cast<int>(rank);
    if (axis < 0 || axis >= static_cast<int>(rank)) {
        throw std::runtime_error("concatenate lowering axis out of range");
    }
    if (!IsConcatenateDType(inputs[0]->dtype) || inputs[0]->dtype != inputs[1]->dtype) {
        throw std::runtime_error("concatenate lowering requires equal supported input dtypes");
    }
    if (output_type->shape.size() != rank ||
        !MatchesRelayDType(inputs[0]->dtype, output_type->dtype)) {
        throw std::runtime_error("concatenate lowering output type mismatch");
    }
    int64_t axis_sum = 0;
    for (size_t index = 0; index < rank; ++index) {
        const int64_t lhs_extent = StaticExtent(inputs[0]->shape[index], "concatenate");
        const int64_t rhs_extent = StaticExtent(inputs[1]->shape[index], "concatenate");
        if (static_cast<int>(index) != axis && lhs_extent != rhs_extent) {
            throw std::runtime_error("concatenate lowering non-axis dimensions must exactly match");
        }
        if (static_cast<int>(index) == axis) {
            if (lhs_extent > std::numeric_limits<int64_t>::max() - rhs_extent) {
                throw std::runtime_error("concatenate lowering axis extent sum overflows int64");
            }
            axis_sum = lhs_extent + rhs_extent;
        }
        const int64_t expected = static_cast<int>(index) == axis ? axis_sum : lhs_extent;
        if (output_type->shape[index] != expected) {
            throw std::runtime_error("concatenate lowering output shape disagrees with inputs");
        }
    }
    return RequireDefined("concatenate", te::topi::concatenate(inputs, axis, "T_concatenate"));
}

kxc::tir::PrimExpr TypedZero(kxc::tir::DataType dtype) {
    if (dtype.code == 2) {
        return kxc::tir::FloatImm(0.0, dtype);
    }
    return kxc::tir::IntImm(0, dtype);
}

te::Tensor GatherCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                         const kxc::Type& out_type) {
    RequireInputCount("gather", inputs, 2);
    const auto* output_type = RequireTensorOutput("gather", out_type);
    const auto* gather_attrs = attrs.As<GatherAttrsNode>();
    if (!gather_attrs) {
        throw std::runtime_error("gather expects GatherAttrs");
    }
    const int data_rank = static_cast<int>(inputs[0]->shape.size());
    int axis = gather_attrs->axis;
    if (axis < 0) axis += data_rank;
    if (data_rank < 1 || axis < 0 || axis >= data_rank) {
        throw std::runtime_error("gather axis out of range");
    }
    const kxc::tir::DataType index_dtype = inputs[1]->dtype;
    if (index_dtype.code != 0 || (index_dtype.bits != 32 && index_dtype.bits != 64)) {
        throw std::runtime_error("gather indices dtype must be int32 or int64");
    }
    const auto* extent = inputs[0]->shape[static_cast<size_t>(axis)].As<kxc::tir::IntImmNode>();
    if (!extent || extent->value < 0) {
        throw std::runtime_error("gather lowering requires static non-negative axis extent");
    }
    const kxc::tir::PrimExpr axis_extent = kxc::tir::IntImm(extent->value, index_dtype);
    const kxc::tir::PrimExpr zero_index = kxc::tir::IntImm(0, index_dtype);
    const kxc::tir::PrimExpr negative_extent = kxc::tir::IntImm(-extent->value, index_dtype);
    const size_t indices_rank = inputs[1]->shape.size();
    return RequireDefined("gather", te::compute(
        ShapeFromTensorType(output_type, "gather"),
        [data = inputs[0], indices = inputs[1], axis, indices_rank, axis_extent,
         zero_index, negative_extent](const Array<kxc::tir::Var>& output_indices) {
            Array<kxc::tir::PrimExpr> index_coordinates;
            for (size_t i = 0; i < indices_rank; ++i) {
                index_coordinates.push_back(output_indices[static_cast<size_t>(axis) + i]);
            }
            const kxc::tir::PrimExpr index = indices(index_coordinates);
            const kxc::tir::PrimExpr is_negative = index < zero_index;
            const kxc::tir::PrimExpr valid_negative =
                is_negative && !(index < negative_extent);
            const kxc::tir::PrimExpr valid_nonnegative =
                !is_negative && index < axis_extent;
            const kxc::tir::PrimExpr valid = valid_negative || valid_nonnegative;
            Array<kxc::tir::PrimExpr> data_coordinates;
            for (int i = 0; i < axis; ++i) data_coordinates.push_back(output_indices[i]);
            // The addition is reached only by the valid-negative Select branch.
            data_coordinates.push_back(kxc::tir::Select(
                valid_negative, index + axis_extent, index));
            for (size_t i = static_cast<size_t>(axis) + indices_rank;
                 i < output_indices.size(); ++i) {
                data_coordinates.push_back(output_indices[i]);
            }
            // Select is lowered lazily; invalid indices never form a TIR Load.
            return kxc::tir::Select(valid, data(data_coordinates), TypedZero(data->dtype));
        },
        "T_gather"));
}

KXC_REGISTER_OP(nn_flatten)
    .describe(R"doc(Flatten input tensor into a 2D tensor.)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<std::string>("TAttrs", "FlattenAttrs")
    .set_attr<FInferType>("FInferType", FlattenInferType)
    .set_attr<FRelayToTE>("FRelayToTE", FlattenCompute);

KXC_REGISTER_OP(reshape)
    .describe(R"doc(Reshape input tensor using a static target shape.)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<std::string>("TAttrs", "ReshapeAttrs")
    .set_attr<FInferType>("FInferType", ReshapeInferType)
    .set_attr<FRelayToTE>("FRelayToTE", ReshapeCompute);

KXC_REGISTER_OP(transpose)
    .describe(R"doc(Permute tensor dimensions.)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<std::string>("TAttrs", "TransposeAttrs")
    .set_attr<FInferType>("FInferType", TransposeInferType)
    .set_attr<FRelayToTE>("FRelayToTE", TransposeCompute);

KXC_REGISTER_OP(cast)
    .describe(R"doc(Cast input tensor to a target dtype.)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<std::string>("TAttrs", "CastAttrs")
    .set_attr<FInferType>("FInferType", CastInferType)
    .set_attr<FRelayToTE>("FRelayToTE", CastCompute);

KXC_REGISTER_OP(concatenate)
    .describe(R"doc(Concatenate exactly two static tensors along an axis into a fresh output.)doc")
    .set_num_inputs(2)
    .add_argument("lhs", "Tensor", "The left input tensor.")
    .add_argument("rhs", "Tensor", "The right input tensor.")
    .set_attr<std::string>("TAttrs", "ConcatenateAttrs")
    .set_attr<FInferType>("FInferType", ConcatenateInferType)
    .set_attr<FRelayToTE>("FRelayToTE", ConcatenateCompute);

KXC_REGISTER_OP(gather)
    .describe(R"doc(Gather slices along an axis; invalid runtime indices produce typed zero.)doc")
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The source tensor.")
    .add_argument("indices", "Tensor", "int32 or int64 gather indices.")
    .set_attr<std::string>("TAttrs", "GatherAttrs")
    .set_attr<FInferType>("FInferType", GatherInferType)
    .set_attr<FRelayToTE>("FRelayToTE", GatherCompute);

}  // namespace relay
}  // namespace kxc

namespace kxc::builtin_anchor {
void RelayTensorTransformOps() {}
}  // namespace kxc::builtin_anchor
