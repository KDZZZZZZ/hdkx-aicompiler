#include "relay/op_macros.h"
#include "relay/relay.h"

namespace kxc {
namespace relay {

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

} // namespace relay
} // namespace kxc
