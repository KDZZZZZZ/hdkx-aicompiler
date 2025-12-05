#include "relay/op_macros.h"
#include "base/relay.h"

namespace kxc {
namespace relay {

// ---------------------------------------------------------------------------
// Softmax Operator
// ---------------------------------------------------------------------------

// Softmax
KXC_REGISTER_OP(softmax)
    .describe(R"doc(Computes softmax.
)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<std::string>("TAttrs", "SoftmaxAttrs");

} // namespace relay
} // namespace kxc
