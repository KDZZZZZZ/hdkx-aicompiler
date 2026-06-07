/*! \file src/relay/op/tensor/transform.cc
 * \brief 注册 Relay tensor transform 算子及其 FRelayToTE compute。
 */

#include "relay/op_macros.h"
#include "relay/op.h"
#include "relay/op_attr_types.h"
#include "relay/type_infer.h"
#include "te/topi/elemwise.h"
#include "te/topi/transform.h"

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

}  // namespace relay
}  // namespace kxc
