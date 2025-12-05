#include "relay/op_macros.h"
#include "base/relay.h"

namespace kxc {
namespace relay {

// ---------------------------------------------------------------------------
// Math/Arithmetic Operators
// ---------------------------------------------------------------------------

// MatMul
KXC_REGISTER_OP(matmul)
    .describe(R"doc(Matrix multiplication.
)doc")
    .set_num_inputs(2)
    .add_argument("a", "Tensor", "The first input tensor.")
    .add_argument("b", "Tensor", "The second input tensor.");
    // .set_attr<std::string>("TAttrs", "MatMulAttrs"); // No attributes mentioned in OP_TODO

// Mul (Element-wise multiplication)
KXC_REGISTER_OP(mul)
    .describe(R"doc(Element-wise multiplication.
)doc")
    .set_num_inputs(2)
    .add_argument("lhs", "Tensor", "The left hand side input tensor.")
    .add_argument("rhs", "Tensor", "The right hand side input tensor.");

// Pow
KXC_REGISTER_OP(pow)
    .describe(R"doc(Power of elements.
)doc")
    .set_num_inputs(2)
    .add_argument("x", "Tensor", "The input base tensor.")
    .add_argument("y", "Tensor", "The exponent tensor.");

// Sqrt
KXC_REGISTER_OP(sqrt)
    .describe(R"doc(Square root of elements.
)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.");

// Sub
KXC_REGISTER_OP(sub)
    .describe(R"doc(Element-wise subtraction.
)doc")
    .set_num_inputs(2)
    .add_argument("lhs", "Tensor", "The left hand side input tensor.")
    .add_argument("rhs", "Tensor", "The right hand side input tensor.");

// Divide
KXC_REGISTER_OP(divide)
    .describe(R"doc(Element-wise division.
)doc")
    .set_num_inputs(2)
    .add_argument("lhs", "Tensor", "The dividend.")
    .add_argument("rhs", "Tensor", "The divisor.")
    .set_attr<std::string>("TAttrs", "DivAttrs");

// Equal
KXC_REGISTER_OP(equal)
    .describe(R"doc(Element-wise equality comparison.
)doc")
    .set_num_inputs(2)
    .add_argument("lhs", "Tensor", "The left hand side input tensor.")
    .add_argument("rhs", "Tensor", "The right hand side input tensor.")
    .set_attr<std::string>("TAttrs", "EqualAttrs");

// Erf
KXC_REGISTER_OP(erf)
    .describe(R"doc(Computes the error function.
)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<std::string>("TAttrs", "ErfAttrs");

} // namespace relay
} // namespace kxc
