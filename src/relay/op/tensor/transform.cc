/*! \file src/relay/op/tensor/transform.cc
 * \brief 注册 Relay tensor transform 算子及其 FRelayToTE compute。
 */

#include "kxc/relay/op_macros.h"
#include "kxc/relay/op.h"
#include "kxc/relay/op_attr_types.h"
#include "kxc/relay/type_infer.h"
#include "kxc/te/topi/broadcast.h"
#include "kxc/te/topi/elemwise.h"
#include "kxc/te/topi/transform.h"
#include "kxc/te/topi/utils.h"

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

int64_t ClampPositiveStepEndpoint(int64_t endpoint, int64_t dim) {
    if (endpoint < 0) {
        return endpoint < -dim ? 0 : endpoint + dim;
    }
    return endpoint > dim ? dim : endpoint;
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

int64_t CheckedStaticProduct(const Array<kxc::tir::PrimExpr>& shape,
                             size_t begin, size_t end,
                             const char* op_name) {
    bool has_zero = false;
    std::vector<int64_t> extents;
    extents.reserve(end - begin);
    for (size_t index = begin; index < end; ++index) {
        const int64_t extent = StaticExtent(shape[index], op_name);
        has_zero = has_zero || extent == 0;
        extents.push_back(extent);
    }
    if (has_zero) return 0;
    int64_t product = 1;
    for (int64_t extent : extents) {
        if (product > std::numeric_limits<int64_t>::max() / extent) {
            throw std::runtime_error(std::string(op_name) +
                                     " flattened extent product overflows int64");
        }
        product *= extent;
    }
    return product;
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

    Array<kxc::tir::PrimExpr> outer_dims;
    Array<kxc::tir::PrimExpr> inner_dims;
    for (int i = 0; i < axis; ++i) {
        outer_dims.push_back(inputs[0]->shape[static_cast<size_t>(i)]);
    }
    for (int i = axis; i < ndim; ++i) {
        inner_dims.push_back(inputs[0]->shape[static_cast<size_t>(i)]);
    }
    const kxc::tir::PrimExpr outer = kxc::tir::IntImm(
        CheckedStaticProduct(inputs[0]->shape, 0, static_cast<size_t>(axis),
                             "nn_flatten"),
        kxc::tir::DataType::Int(64));
    const kxc::tir::PrimExpr inner = kxc::tir::IntImm(
        CheckedStaticProduct(inputs[0]->shape, static_cast<size_t>(axis),
                             static_cast<size_t>(ndim), "nn_flatten"),
        kxc::tir::DataType::Int(64));

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

te::Tensor ExpandCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                         const kxc::Type& out_type) {
    const auto* expand_attrs = attrs.As<ExpandAttrsNode>();
    if (!expand_attrs) {
        throw std::runtime_error("expand lowering requires ExpandAttrs");
    }
    RequireInputCount("expand", inputs, 1);
    RequireTensorOutput("expand", out_type);
    // 目标 shape 与 data 的兼容性在 lowering 前复校验：dim 1 或相等，且
    // data rank 不超过目标 rank，防止 broadcast_to 的越界索引。
    Array<kxc::tir::PrimExpr> target_shape;
    for (int64_t dim : expand_attrs->target_shape) {
        if (dim < 0) {
            throw std::runtime_error("expand target_shape must be non-negative");
        }
        target_shape.push_back(kxc::tir::IntImm(dim, kxc::tir::DataType::Int(64)));
    }
    if (inputs[0]->shape.size() > target_shape.size()) {
        throw std::runtime_error("expand data rank must not exceed the target rank");
    }
    const size_t offset = target_shape.size() - inputs[0]->shape.size();
    for (size_t i = 0; i < inputs[0]->shape.size(); ++i) {
        int64_t dim = 0;
        if (!te::topi::GetConstInt(inputs[0]->shape[i], &dim)) {
            continue;
        }
        int64_t target_dim = 0;
        if (!te::topi::GetConstInt(target_shape[offset + i], &target_dim)) {
            throw std::runtime_error("expand target dimensions must be static");
        }
        if (dim != target_dim && dim != 1) {
            throw std::runtime_error("expand data dimension " + std::to_string(dim) +
                                     " at axis " + std::to_string(i) +
                                     " must be 1 or equal to the target dimension " +
                                     std::to_string(target_dim));
        }
    }
    return RequireDefined("expand",
                          te::topi::broadcast_to(inputs[0], target_shape, "T_expand"));
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

te::Tensor SliceCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                        const kxc::Type& out_type) {
    RequireInputCount("slice", inputs, 1);
    const auto* output_type = RequireTensorOutput("slice", out_type);
    const auto* slice_attrs = attrs.As<SliceAttrsNode>();
    if (!slice_attrs) {
        throw std::runtime_error("slice expects SliceAttrs");
    }
    if (!inputs[0].defined() || !IsConcatenateDType(inputs[0]->dtype)) {
        throw std::runtime_error("slice lowering requires a defined supported input tensor");
    }
    const int rank = static_cast<int>(inputs[0]->shape.size());
    const size_t count = slice_attrs->starts.size();
    if (rank < 1 || count == 0 || slice_attrs->ends.empty() || slice_attrs->axes.empty() ||
        slice_attrs->steps.empty() || slice_attrs->ends.size() != count ||
        slice_attrs->axes.size() != count || slice_attrs->steps.size() != count ||
        output_type->shape.size() != static_cast<size_t>(rank) ||
        !MatchesRelayDType(inputs[0]->dtype, output_type->dtype)) {
        throw std::runtime_error("slice lowering input, attrs, or output type mismatch");
    }
    std::vector<int64_t> starts(static_cast<size_t>(rank), 0);
    std::vector<int64_t> expected_shape;
    expected_shape.reserve(static_cast<size_t>(rank));
    for (int axis = 0; axis < rank; ++axis) {
        expected_shape.push_back(StaticExtent(inputs[0]->shape[static_cast<size_t>(axis)], "slice"));
    }
    std::vector<bool> seen(static_cast<size_t>(rank), false);
    for (size_t index = 0; index < count; ++index) {
        if (slice_attrs->steps[index] != 1) {
            throw std::runtime_error("slice lowering requires every step to equal exactly +1");
        }
        int64_t raw_axis = slice_attrs->axes[index];
        if (raw_axis < 0) raw_axis += rank;
        if (raw_axis < 0 || raw_axis >= rank || seen[static_cast<size_t>(raw_axis)]) {
            throw std::runtime_error("slice lowering axes must be unique and in range");
        }
        const size_t axis = static_cast<size_t>(raw_axis);
        seen[axis] = true;
        const int64_t start = ClampPositiveStepEndpoint(slice_attrs->starts[index],
                                                        expected_shape[axis]);
        const int64_t end = ClampPositiveStepEndpoint(slice_attrs->ends[index],
                                                      expected_shape[axis]);
        starts[axis] = start;
        expected_shape[axis] = std::max(end - start, int64_t{0});
    }
    for (int axis = 0; axis < rank; ++axis) {
        if (output_type->shape[static_cast<size_t>(axis)] != expected_shape[static_cast<size_t>(axis)]) {
            throw std::runtime_error("slice lowering output shape disagrees with attrs and input");
        }
    }
    return RequireDefined("slice", te::compute(
        ShapeFromTensorType(output_type, "slice"),
        [input = inputs[0], starts](const Array<kxc::tir::Var>& indices) {
            Array<kxc::tir::PrimExpr> input_indices;
            for (size_t axis = 0; axis < indices.size(); ++axis) {
                input_indices.push_back(indices[axis] +
                    kxc::tir::IntImm(starts[axis], kxc::tir::DataType::Int(64)));
            }
            return input(input_indices);
        },
        "T_slice"));
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

KXC_REGISTER_OP(slice)
    .describe(R"doc(Exact-static ONNX/Python positive-step slice into a fresh output.)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<std::string>("TAttrs", "SliceAttrs")
    .set_attr<FInferType>("FInferType", SliceInferType)
    .set_attr<FRelayToTE>("FRelayToTE", SliceCompute);

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

KXC_REGISTER_OP(expand)
    .describe(R"doc(Expand data to a resolved static target shape with broadcast_to rules.)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor to expand.")
    .set_attr<std::string>("TAttrs", "ExpandAttrs")
    .set_attr<FInferType>("FInferType", ExpandInferType)
    .set_attr<FRelayToTE>("FRelayToTE", ExpandCompute);

}  // namespace relay
}  // namespace kxc

namespace kxc::builtin_anchor {
void RelayTensorTransformOps() {}
}  // namespace kxc::builtin_anchor
