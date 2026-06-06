/*! \file src/relay/op/tensor/reduce.cc
 * \brief 注册 Relay 算子及其 FRelayToTE compute。
 */

#include "relay/op_macros.h"
#include "relay/relay.h"
#include "relay/op_attr_types.h"
#include "relay/type_infer.h"

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
    .set_attr<FInferType>("FInferType", ReduceMeanInferType)
    .set_attr<std::string>("TAttrs", "ReduceMeanAttrs");

} // namespace relay
} // namespace kxc
