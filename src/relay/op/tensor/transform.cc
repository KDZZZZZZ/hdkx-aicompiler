#include "relay/op_macros.h"
#include "relay/relay.h"

namespace kxc {
namespace relay {

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
    .set_attr<std::string>("TAttrs", "ConcatAttrs");

// Flatten
KXC_REGISTER_OP(nn_flatten)
    .describe(R"doc(Flattens the input tensor into a 2D tensor.
)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<std::string>("TAttrs", "FlattenAttrs");

// Reshape
KXC_REGISTER_OP(reshape)
    .describe(R"doc(Reshapes the input tensor.

Returns a tensor with the same data but different shape.
)doc")
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("shape", "Tensor", "The target shape.")
    .set_attr<std::string>("TAttrs", "ReshapeAttrs");

// Shape
KXC_REGISTER_OP(shape)
    .describe(R"doc(Returns the shape of the input tensor.
)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.");

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
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("split", "Tensor", "The split points or sizes.")
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
    .add_argument("y", "Tensor", "Values to use where condition is False.");

// Gather
KXC_REGISTER_OP(gather)
    .describe(R"doc(Gather values along an axis.
)doc")
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("indices", "Tensor", "The indices to gather.")
    .set_attr<std::string>("TAttrs", "GatherAttrs");

// Cast
KXC_REGISTER_OP(cast)
    .describe(R"doc(Cast input to specified type.
)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
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
