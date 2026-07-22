/*! \file src/relay/op/tensor/math.cc
 * \brief 注册 Relay tensor math 算子及其编译 hook。
 */

#include "kxc/relay/op_macros.h"
#include "kxc/relay/op_attr_types.h"
#include "kxc/relay/type_infer.h"
#include "kxc/te/topi/broadcast.h"
#include "kxc/te/topi/elemwise.h"
#include "kxc/te/topi/nn.h"

#include <stdexcept>

namespace kxc {
namespace relay {

namespace {

void RequireInputCount(const char* op_name, const Array<te::Tensor>& inputs, size_t expected) {
    if (inputs.size() != expected) {
        throw std::runtime_error(std::string(op_name) + " expects exactly " +
                                 std::to_string(expected) + " input(s)");
    }
}

void RequireTensorOutput(const char* op_name, const kxc::Type& out_type) {
    if (!out_type.As<TensorTypeNode>()) {
        throw std::runtime_error(std::string(op_name) + " expects TensorType output");
    }
}

te::Tensor RequireDefined(const char* op_name, const te::Tensor& tensor) {
    if (!tensor.defined()) {
        throw std::runtime_error(std::string(op_name) + " lowering returned undefined tensor");
    }
    return tensor;
}

}  // namespace

te::Tensor AddCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                      const kxc::Type& out_type) {
    (void)attrs;
    RequireInputCount("add", inputs, 2);
    RequireTensorOutput("add", out_type);
    return RequireDefined("add", te::topi::add(inputs[0], inputs[1], "T_add"));
}

te::Tensor SubtractCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                           const kxc::Type& out_type) {
    (void)attrs;
    RequireInputCount("subtract", inputs, 2);
    RequireTensorOutput("subtract", out_type);
    return RequireDefined("subtract", te::topi::subtract(inputs[0], inputs[1], "T_subtract"));
}

te::Tensor MultiplyCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                           const kxc::Type& out_type) {
    (void)attrs;
    RequireInputCount("mul", inputs, 2);
    RequireTensorOutput("mul", out_type);
    return RequireDefined("mul", te::topi::multiply(inputs[0], inputs[1], "T_mul"));
}

te::Tensor DivideCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                         const kxc::Type& out_type) {
    (void)attrs;
    RequireInputCount("divide", inputs, 2);
    RequireTensorOutput("divide", out_type);
    return RequireDefined("divide", te::topi::divide(inputs[0], inputs[1], "T_divide"));
}

te::Tensor SqrtCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                       const kxc::Type& out_type) {
    (void)attrs;
    RequireInputCount("sqrt", inputs, 1);
    RequireTensorOutput("sqrt", out_type);
    return RequireDefined("sqrt", te::topi::sqrt(inputs[0], "T_sqrt"));
}

te::Tensor MatMulCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                         const kxc::Type& out_type) {
    (void)attrs;
    RequireInputCount("matmul", inputs, 2);
    RequireTensorOutput("matmul", out_type);
    return RequireDefined("matmul", te::topi::matmul(inputs[0], inputs[1], "T_matmul"));
}

KXC_REGISTER_OP(add)
    .describe(R"doc(Element-wise addition.)doc")
    .set_num_inputs(2)
    .add_argument("lhs", "Tensor", "The left hand side input tensor.")
    .add_argument("rhs", "Tensor", "The right hand side input tensor.")
    .set_attr<FInferType>("FInferType", AddInferType)
    .set_attr<FRelayToTE>("FRelayToTE", AddCompute);

KXC_REGISTER_OP(subtract)
    .describe(R"doc(Element-wise subtraction.)doc")
    .set_num_inputs(2)
    .add_argument("lhs", "Tensor", "The left hand side input tensor.")
    .add_argument("rhs", "Tensor", "The right hand side input tensor.")
    .set_attr<FInferType>("FInferType", SubtractInferType)
    .set_attr<FRelayToTE>("FRelayToTE", SubtractCompute);

KXC_REGISTER_OP(mul)
    .describe(R"doc(Element-wise multiplication.)doc")
    .set_num_inputs(2)
    .add_argument("lhs", "Tensor", "The left hand side input tensor.")
    .add_argument("rhs", "Tensor", "The right hand side input tensor.")
    .set_attr<FInferType>("FInferType", MultiplyInferType)
    .set_attr<FRelayToTE>("FRelayToTE", MultiplyCompute);

KXC_REGISTER_OP(divide)
    .describe(R"doc(Element-wise division.)doc")
    .set_num_inputs(2)
    .add_argument("lhs", "Tensor", "The left hand side input tensor.")
    .add_argument("rhs", "Tensor", "The right hand side input tensor.")
    .set_attr<FInferType>("FInferType", DivideInferType)
    .set_attr<FRelayToTE>("FRelayToTE", DivideCompute);

KXC_REGISTER_OP(sqrt)
    .describe(R"doc(Square root of elements.)doc")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<FInferType>("FInferType", UnarySameInferType)
    .set_attr<FRelayToTE>("FRelayToTE", SqrtCompute);

KXC_REGISTER_OP(matmul)
    .describe(R"doc(Matrix multiplication.)doc")
    .set_num_inputs(2)
    .add_argument("a", "Tensor", "The first input tensor.")
    .add_argument("b", "Tensor", "The second input tensor.")
    .set_attr<FInferType>("FInferType", MatMulInferType)
    .set_attr<FRelayToTE>("FRelayToTE", MatMulCompute);

}  // namespace relay
}  // namespace kxc

namespace kxc::builtin_anchor {
void RelayTensorMathOps() {}
}  // namespace kxc::builtin_anchor
