#include "relay/op_macros.h"
#include "relay/relay.h"
#include <vector>
#include <string>

namespace kxc {
namespace relay {

// ---------------------------------------------------------------------------
// 1. Define Attribute Structure
// ---------------------------------------------------------------------------
// Conv2D attributes (strides, padding, dilation, etc.)
// Note: Ideally, this definition should be in a header file (e.g., include/relay/attrs/nn.h)
// so that other parts of the system (like Pass) can access it.
// For this example, we assume it's defined in include/relay/op.h or we re-use the one there.
// But for completeness of a standalone op file, we often define local helpers or refer to shared ones.

// Since Conv2DAttrs is already defined in include/relay/op.h, we just use it.
// In a real large project, you would include "include/relay/attrs/nn.h".

// ---------------------------------------------------------------------------
// 2. Operator Registration
// ---------------------------------------------------------------------------

KXC_REGISTER_OP(nn_conv2d)
    .describe(R"doc(2D convolution layer (e.g. spatial convolution over images).

This operator computes a 2D convolution of input `data` with `weight`.
The `data` input should have shape `(batch_size, in_channels, height, width)`
if layout is `NCHW`.
)doc")
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("weight", "Tensor", "The weight tensor.")
    .set_attr<std::string>("TAttrs", "Conv2DAttrs"); // Bind to C++ Attribute Struct

} // namespace relay
} // namespace kxc
