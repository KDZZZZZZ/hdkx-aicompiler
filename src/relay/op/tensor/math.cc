/*! \file src/relay/op/tensor/math.cc
 * \brief Registers Relay tensor math operators and compiler hooks.
 */

#include "relay/op_macros.h"
#include "relay/op_attr_types.h"
#include "relay/type_infer.h"
#include "te/topi/broadcast.h"
#include <stdexcept>

namespace kxc {
namespace relay {

te::Tensor AddCompute(const Attrs& attrs, const Array<te::Tensor>& inputs, const kxc::Type& out_type) {
    (void)attrs;
    (void)out_type;
    if (inputs.size() != 2) {
        throw std::runtime_error("add expects exactly 2 inputs");
    }
    return te::topi::add(inputs[0], inputs[1], "T_add");
}

// ---------------------------------------------------------------------------
// Math/Arithmetic Operators
// ---------------------------------------------------------------------------

// Add
KXC_REGISTER_OP(add)
    .describe(R"doc(Element-wise addition.
)doc")
    .set_num_inputs(2)
    .add_argument("lhs", "Tensor", "The left hand side input tensor.")
    .add_argument("rhs", "Tensor", "The right hand side input tensor.")
    .set_attr<FInferType>("FInferType", AddInferType)
    .set_attr<FRelayToTE>("FRelayToTE", AddCompute);

KXC_REGISTER_OP(matmul)
    .describe(R"doc(Matrix multiplication.
)doc")
    .set_num_inputs(2)
    .add_argument("a", "Tensor", "The first input tensor.")
    .add_argument("b", "Tensor", "The second input tensor.")
    .set_attr<FInferType>("FInferType", MatMulInferType);

// Mul (Element-wise multiplication)
KXC_REGISTER_OP(mul)
    .describe(R"doc(Element-wise multiplication.
)doc")
    .set_num_inputs(2)
    .add_argument("lhs", "Tensor", "The left hand side input tensor.")
    .add_argument("rhs", "Tensor", "The right hand side input tensor.")
    .set_attr<FInferType>("FInferType", MultiplyInferType);

// Pow
KXC_REGISTER_OP(pow)
    .describe(R"doc(Power of elements.
)doc")
    .set_num_inputs(2)
    .add_argument("x", "Tensor", "The input base tensor.")
    .add_argument("y", "Tensor", "The exponent tensor.")
    .set_attr<FInferType>("FInferType", PowInferType);

// Sqrt
KXC_REGISTER_OP(sqrt)
    .describe(R"doc(Square root of elements.
)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<FInferType>("FInferType", UnarySameInferType);

// Subtract
KXC_REGISTER_OP(subtract)
    .describe(R"doc(Element-wise subtraction.
)doc")
    .set_num_inputs(2)
    .add_argument("lhs", "Tensor", "The left hand side input tensor.")
    .add_argument("rhs", "Tensor", "The right hand side input tensor.")
    .set_attr<FInferType>("FInferType", SubtractInferType);

// Divide
KXC_REGISTER_OP(divide)
    .describe(R"doc(Element-wise division.
)doc")
    .set_num_inputs(2)
    .add_argument("lhs", "Tensor", "The left hand side input tensor.")
    .add_argument("rhs", "Tensor", "The right hand side input tensor.")
    .set_attr<FInferType>("FInferType", DivideInferType)
    .set_attr<std::string>("TAttrs", "DivAttrs");

// Equal
KXC_REGISTER_OP(equal)
    .describe(R"doc(Element-wise equal comparison.
)doc")
    .set_num_inputs(2)
    .add_argument("lhs", "Tensor", "The left hand side input tensor.")
    .add_argument("rhs", "Tensor", "The right hand side input tensor.")
    .set_attr<FInferType>("FInferType", EqualInferType)
    .set_attr<std::string>("TAttrs", "EqualAttrs");

// Erf
KXC_REGISTER_OP(erf)
    .describe(R"doc(Error function.
)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<FInferType>("FInferType", UnarySameInferType)
    .set_attr<std::string>("TAttrs", "ErfAttrs");

} // namespace relay
} // namespace kxc
