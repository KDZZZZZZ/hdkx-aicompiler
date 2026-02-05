#include "relay/op_macros.h"
#include "relay/relay.h"

namespace kxc {
namespace relay {

KXC_REGISTER_OP(nn_relu)
    .describe("Rectified Linear Unit activation")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<std::string>("TAttrs", "ReluAttrs");

} // namespace relay
} // namespace kxc
