#include "../include/relay/op_macros.h"
#include "../include/relay/op.h" // Need this for Attrs definitions if we want to verify or use them here
#include <iostream>

namespace kxc {
namespace relay {

// --- NN Ops ---
KXC_REGISTER_OP(nn_conv2d)
    .describe("2D convolution layer")
    .set_num_inputs(2) // data, weight
    .set_attr<std::string>("TAttrs", "Conv2DAttrs"); // Register the Attribute Type Key

KXC_REGISTER_OP(nn_relu)
    .describe("Rectified Linear Unit activation")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<std::string>("TAttrs", "ReluAttrs");

KXC_REGISTER_OP(nn_dense)
    .describe("Dense (fully connected) layer")
    .set_num_inputs(2) // data, weight
    .set_attr<std::string>("TAttrs", "DenseAttrs");

// Gemm
KXC_REGISTER_OP(nn_gemm)
    .describe(R"doc(General matrix multiplication.
)doc")
    .set_num_inputs(3)
    .add_argument("A", "Tensor", "The first input tensor.")
    .add_argument("B", "Tensor", "The second input tensor.")
    .add_argument("C", "Tensor", "The third input tensor (bias).")
    .set_attr<std::string>("TAttrs", "GemmAttrs");

KXC_REGISTER_OP(nn_max_pool2d)
    .describe("2D max pooling")
    .set_num_inputs(1)
    .set_attr<std::string>("TAttrs", "MaxPool2DAttrs");

KXC_REGISTER_OP(nn_softmax)
    .describe("Softmax activation")
    .set_num_inputs(1)
    .set_attr<std::string>("TAttrs", "SoftmaxAttrs");

// --- Tensor Ops ---
KXC_REGISTER_OP(add)
    .describe("Element-wise addition")
    .set_num_inputs(2);

KXC_REGISTER_OP(subtract)
    .describe("Element-wise subtraction")
    .set_num_inputs(2);

KXC_REGISTER_OP(multiply)
    .describe("Element-wise multiplication")
    .set_num_inputs(2);

// --- Control/Logic Ops ---
KXC_REGISTER_OP(greater)
    .describe("Element-wise greater than comparison")
    .set_num_inputs(2);

} // namespace relay
} // namespace kxc
