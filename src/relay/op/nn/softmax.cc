/*! \file src/relay/op/nn/softmax.cc
 * \brief 注册 Relay 算子及其 FRelayToTE compute。
 */

#include "relay/op_macros.h"
#include "relay/relay.h"

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
