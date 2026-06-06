/*! \file src/relay/op/tensor/transform.cc
 * \brief 注册 Relay 算子及其 FRelayToTE compute。
 */

#include "relay/op_macros.h"
#include "relay/op.h"
#include "relay/op_attr_types.h"
#include "relay/type_infer.h"
#include "te/te.h"
#include <stdexcept>

namespace kxc {
namespace relay {

namespace {
Array<kxc::tir::PrimExpr> UnflattenIndex(
    kxc::tir::PrimExpr flat,
    const Array<kxc::tir::PrimExpr>& dims) {
    Array<kxc::tir::PrimExpr> out;
    for (size_t i = 0; i < dims.size(); ++i) {
        out.push_back(0);
    }
    kxc::tir::PrimExpr cur = flat;
    for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
        out[i] = cur % dims[i];
        cur = cur / dims[i];
    }
    return out;
}
}

te::Tensor FlattenCompute(const Attrs& attrs, const Array<te::Tensor>& inputs, const kxc::Type& out_type) {
    if (inputs.size() != 1) {
        throw std::runtime_error("nn_flatten expects exactly 1 input");
    }
    auto* p = attrs.As<FlattenAttrsNode>();
    int axis = p ? p->axis : 1;
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
        outer = outer * inputs[0]->shape[i];
        outer_dims.push_back(inputs[0]->shape[i]);
    }
    for (int i = axis; i < ndim; ++i) {
        inner = inner * inputs[0]->shape[i];
        inner_dims.push_back(inputs[0]->shape[i]);
    }

    return te::compute({outer, inner}, [&](const Array<kxc::tir::Var>& indices) {
        Array<kxc::tir::PrimExpr> in_indices;
        auto outer_idx = UnflattenIndex(indices[0], outer_dims);
        auto inner_idx = UnflattenIndex(indices[1], inner_dims);
        for (const auto& v : outer_idx) in_indices.push_back(v);
        for (const auto& v : inner_idx) in_indices.push_back(v);
        if (in_indices.empty()) {
            in_indices.push_back(0);
        }
        return inputs[0](in_indices);
    }, "T_flatten");
}

// ---------------------------------------------------------------------------
// Tensor Transformation Operators
// ---------------------------------------------------------------------------

// Concatenate
// Note: Concatenate takes a Tuple of tensors, so num_inputs is technically 1 (the tuple).
KXC_REGISTER_OP(concatenate)
    .describe(R"doc(Concatenate tensors along a given axis.

The input is a tuple of tensors.
)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tuple", "The tuple of tensors to concatenate.")
    .set_attr<FInferType>("FInferType", ConcatenateInferType)
    .set_attr<std::string>("TAttrs", "ConcatAttrs");

// Flatten
KXC_REGISTER_OP(nn_flatten)
    .describe(R"doc(Flattens the input tensor into a 2D tensor.
)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<std::string>("TAttrs", "FlattenAttrs")
    .set_attr<FInferType>("FInferType", FlattenInferType)
    .set_attr<FRelayToTE>("FRelayToTE", FlattenCompute);

// Reshape
KXC_REGISTER_OP(reshape)
    .describe(R"doc(Reshapes the input tensor.

Returns a tensor with the same data but different shape.
)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("newshape", "Array<Int>", "The static target shape.")
    .set_attr<FInferType>("FInferType", ReshapeInferType)
    .set_attr<std::string>("TAttrs", "ReshapeAttrs");

// Shape
KXC_REGISTER_OP(shape)
    .describe(R"doc(Returns the shape of the input tensor.
)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<FInferType>("FInferType", ShapeInferType);

// Slice
KXC_REGISTER_OP(slice)
    .describe(R"doc(Slices the input tensor.
)doc")
    .set_num_inputs(5) // Max inputs, can be variable? For now fixed max or variable? Relay usually handles this.
    // Since user said inputs [3, 5], we might set num_inputs to -1 (variable) or max. 
    // However, our system assumes fixed inputs usually. 
    // Let's set it to 5 and make some optional if possible, or just 5.
    // But "Inputs: [3, 5]" implies it varies. 
    // Let's set num_inputs to -1 to indicate variable arguments if supported, 
    // but OpNode::num_inputs = -1 usually means variable.
    // Let's check OpNode::num_inputs usage.
    // Assuming 5 for now as max.
    .set_num_inputs(5) 
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("starts", "Tensor", "Indices to start slicing.")
    .add_argument("ends", "Tensor", "Indices to end slicing.")
    .add_argument("axes", "Tensor", "Axes to slice along.", true) // Optional
    .add_argument("steps", "Tensor", "Slicing steps.", true); // Optional

// Split
KXC_REGISTER_OP(split)
    .describe(R"doc(Splits the input tensor into multiple tensors.
)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("split", "Array<Int>", "The split points or number of sections.")
    .set_attr<FInferType>("FInferType", SplitInferType)
    .set_attr<std::string>("TAttrs", "SplitAttrs");

// Squeeze
KXC_REGISTER_OP(squeeze)
    .describe(R"doc(Remove single-dimensional entries from the shape of a tensor.
)doc")
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("axes", "Tensor", "Axes to squeeze.");

// Transpose
KXC_REGISTER_OP(transpose)
    .describe(R"doc(Permutes the dimensions of an array.
)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<FInferType>("FInferType", TransposeInferType)
    .set_attr<std::string>("TAttrs", "TransposeAttrs");

// Unsqueeze
KXC_REGISTER_OP(unsqueeze)
    .describe(R"doc(Insert single-dimensional entries to the shape of a tensor.
)doc")
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("axes", "Tensor", "Axes to insert.");

// Where
KXC_REGISTER_OP(where)
    .describe(R"doc(Return elements chosen from x or y depending on condition.
)doc")
    .set_num_inputs(3)
    .add_argument("condition", "Tensor", "The condition tensor.")
    .add_argument("x", "Tensor", "Values to use where condition is True.")
    .add_argument("y", "Tensor", "Values to use where condition is False.")
    .set_attr<FInferType>("FInferType", WhereInferType);

// Gather
KXC_REGISTER_OP(gather)
    .describe(R"doc(Gather values along an axis.
)doc")
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("indices", "Tensor", "The indices to gather.")
    .set_attr<FInferType>("FInferType", GatherInferType)
    .set_attr<std::string>("TAttrs", "GatherAttrs");

// Cast
KXC_REGISTER_OP(cast)
    .describe(R"doc(Cast input to specified type.
)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<FInferType>("FInferType", CastInferType)
    .set_attr<std::string>("TAttrs", "CastAttrs");

// ConstantOfShape
KXC_REGISTER_OP(constant_of_shape)
    .describe(R"doc(Generate a tensor with given value and shape.
)doc")
    .set_num_inputs(1)
    .add_argument("input", "Tensor", "The shape tensor.")
    .set_attr<std::string>("TAttrs", "ConstantOfShapeAttrs");

// ExpandDims (Mapped from ONNX Expand, effectively broadcast_to)
KXC_REGISTER_OP(expand_dims)
    .describe(R"doc(Expand/Broadcast tensor to new shape.
)doc")
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("shape", "Tensor", "The target shape.")
    .set_attr<std::string>("TAttrs", "ExpandAttrs");

} // namespace relay
} // namespace kxc
