#include "../include/relay/op_macros.h"
#include "../include/base/op.h" // Need this for Attrs definitions if we want to verify or use them here
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
    .set_num_inputs(1);

KXC_REGISTER_OP(nn_dense)
    .describe("Dense (fully connected) layer")
    .set_num_inputs(2) // data, weight
    .set_attr<std::string>("TAttrs", "DenseAttrs");

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
