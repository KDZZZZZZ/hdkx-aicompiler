#include "relay/op_macros.h"
#include "relay/relay.h"
#include <vector>
#include <string>

namespace kxc {
namespace relay {

// ---------------------------------------------------------------------------
// 1. Operator Registration for Pooling
// ---------------------------------------------------------------------------

KXC_REGISTER_OP(nn_max_pool2d)
    .describe(R"doc(2D max pooling operation.

This operator performs max pooling on the input tensor.
)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<std::string>("TAttrs", "MaxPool2DAttrs");

KXC_REGISTER_OP(nn_avg_pool2d)
    .describe(R"doc(2D average pooling operation.

This operator performs average pooling on the input tensor.
)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    // Reusing MaxPool2DAttrs for AvgPool2D as they share structure (pool_size, strides, padding)
    // In real TVM, they might share a generic Pool2DAttrs.
    .set_attr<std::string>("TAttrs", "MaxPool2DAttrs"); 

KXC_REGISTER_OP(nn_global_avg_pool2d)
    .describe(R"doc(Global average pooling operation.

Reduces the spatial dimensions (H, W) to 1x1 by averaging.
)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<std::string>("TAttrs", "GlobalAvgPool2DAttrs");

} // namespace relay
} // namespace kxc
