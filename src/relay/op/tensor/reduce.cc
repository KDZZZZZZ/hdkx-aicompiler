#include "relay/op_macros.h"
#include "base/relay.h"

namespace kxc {
namespace relay {

// ---------------------------------------------------------------------------
// Reduction Operators
// ---------------------------------------------------------------------------

// ReduceMean
KXC_REGISTER_OP(reduce_mean)
    .describe(R"doc(Computes the mean of elements across given dimensions.
)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<std::string>("TAttrs", "ReduceMeanAttrs");

} // namespace relay
} // namespace kxc
