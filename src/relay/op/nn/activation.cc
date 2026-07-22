/*! \file src/relay/op/nn/activation.cc
 * \brief 注册 Relay 算子及其 FRelayToTE compute。
 */

#include "kxc/relay/op_macros.h"
#include "kxc/relay/op_attr_types.h"
#include "kxc/relay/type_infer.h"
#include "kxc/te/topi/nn.h"
#include <stdexcept>

namespace kxc {
namespace relay {

te::Tensor ReluCompute(const Attrs& attrs, const Array<te::Tensor>& inputs, const kxc::Type& out_type) {
    if (inputs.size() != 1) {
        throw std::runtime_error("nn_relu expects exactly 1 input");
    }
    return te::topi::relu(inputs[0], "T_relu");
}

KXC_REGISTER_OP(nn_relu)
    .describe("Rectified Linear Unit activation")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<std::string>("TAttrs", "ReluAttrs")
    .set_attr<FInferType>("FInferType", UnarySameInferType)
    .set_attr<FRelayToTE>("FRelayToTE", ReluCompute);

} // namespace relay
} // namespace kxc

namespace kxc::builtin_anchor {
void RelayActivationOps() {}
}  // namespace kxc::builtin_anchor
