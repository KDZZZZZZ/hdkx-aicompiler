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

const TensorTypeNode* RequireTensorOutput(const char* op_name, const kxc::Type& out_type) {
    const auto* output = out_type.As<TensorTypeNode>();
    if (!output) {
        throw std::runtime_error(std::string(op_name) + " expects TensorType output");
    }
    return output;
}

bool MatchesRelayDType(kxc::tir::DataType dtype, const std::string& relay_dtype) {
    return (relay_dtype == "float32" && dtype == kxc::tir::DataType::Float(32)) ||
           (relay_dtype == "float64" && dtype == kxc::tir::DataType::Float(64)) ||
           (relay_dtype == "int32" && dtype == kxc::tir::DataType::Int(32)) ||
           (relay_dtype == "int64" && dtype == kxc::tir::DataType::Int(64)) ||
           (relay_dtype == "int8" && dtype == kxc::tir::DataType::Int(8)) ||
           (relay_dtype == "uint8" && dtype == kxc::tir::DataType::UInt(8)) ||
           (relay_dtype == "bool" && dtype == kxc::tir::DataType::Bool());
}

// equal 内核层已验证的同类型输入 dtype 集合，与 IsEqualInputDType 的字符串集合一致。
bool MatchesEqualInputDType(kxc::tir::DataType dtype) {
    return dtype == kxc::tir::DataType::Int(32) || dtype == kxc::tir::DataType::Int(64) ||
           dtype == kxc::tir::DataType::Float(32);
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

te::Tensor EqualCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                        const kxc::Type& out_type) {
    (void)attrs;
    RequireInputCount("equal", inputs, 2);
    const auto* output = RequireTensorOutput("equal", out_type);
    if (inputs[0]->dtype != inputs[1]->dtype) {
        throw std::runtime_error("equal input dtypes must match");
    }
    if (!MatchesEqualInputDType(inputs[0]->dtype)) {
        throw std::runtime_error(
            "equal supports same-dtype int32, int64, or float32 inputs before lowering");
    }
    if (output->dtype != "bool") {
        throw std::runtime_error("equal output dtype must be bool");
    }
    return RequireDefined("equal", te::topi::equal(inputs[0], inputs[1], "T_equal"));
}

te::Tensor NegCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                      const kxc::Type& out_type) {
    (void)attrs;
    RequireInputCount("neg", inputs, 1);
    const auto* output = RequireTensorOutput("neg", out_type);
    if (inputs[0]->dtype != kxc::tir::DataType::Float(32) ||
        output->dtype != "float32") {
        throw std::runtime_error("neg supports only float32 input and output before lowering");
    }
    return RequireDefined("neg", te::topi::negative(inputs[0], "T_neg"));
}

te::Tensor SigmoidCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                          const kxc::Type& out_type) {
    (void)attrs;
    RequireInputCount("sigmoid", inputs, 1);
    const auto* output = RequireTensorOutput("sigmoid", out_type);
    if (inputs[0]->dtype != kxc::tir::DataType::Float(32) ||
        output->dtype != "float32") {
        throw std::runtime_error(
            "sigmoid supports only float32 input and output before lowering");
    }
    return RequireDefined("sigmoid", te::topi::sigmoid(inputs[0], "T_sigmoid"));
}

te::Tensor WhereCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                        const kxc::Type& out_type) {
    if (attrs.defined()) {
        throw std::runtime_error("where does not accept attrs");
    }
    RequireInputCount("where", inputs, 3);
    const auto* output = RequireTensorOutput("where", out_type);
    if (inputs[0]->dtype != kxc::tir::DataType::Bool()) {
        throw std::runtime_error("where condition dtype must be bool");
    }
    if (inputs[1]->dtype != inputs[2]->dtype ||
        !MatchesRelayDType(inputs[1]->dtype, output->dtype)) {
        throw std::runtime_error("where output and branch dtypes must match");
    }
    return RequireDefined("where", te::topi::where(inputs[0], inputs[1], inputs[2], "T_where"));
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

KXC_REGISTER_OP(where)
    .describe(R"doc(Select x or y according to a boolean condition with exact trailing-axis broadcast.)doc")
    .set_num_inputs(3)
    .add_argument("condition", "Tensor", "Boolean selection condition.")
    .add_argument("x", "Tensor", "Value selected when condition is true.")
    .add_argument("y", "Tensor", "Value selected when condition is false.")
    .set_attr<FInferType>("FInferType", WhereInferType)
    .set_attr<FRelayToTE>("FRelayToTE", WhereCompute);

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
