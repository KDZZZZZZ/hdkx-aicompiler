#include "../include/relay/op_macros.h"
#include "../include/base/relay.h"
#include <iostream>
#include <vector>
#include <cassert>

using namespace kxc;

// 1. Define Attribute Struct with Defaults (C++ Side)
// This answers "how to set optional parameters defaults" for Attributes.
class BiasAddAttrsNode : public BaseAttrsNode {
public:
    int axis = 1; // Default value
    
    const TypeIndex GetTypeId() const override { return kKXC_OBJECT_TYPE + 30; }
};

class BiasAddAttrs : public Attrs {
public:
    using Attrs::Attrs;
    static BiasAddAttrs Create(int axis = 1) {
        BiasAddAttrsNode* node = new BiasAddAttrsNode();
        node->axis = axis;
        BiasAddAttrs attrs;
        attrs.object_ = node;
        if (attrs.object_) attrs.object_->IncRef();
        return attrs;
    }
    const BiasAddAttrsNode* operator->() const { return static_cast<const BiasAddAttrsNode*>(object_); }
};

// 2. Register Operator with Arguments and Attribute Binding
// This answers "how to register ... set required and optional parameters"
KXC_REGISTER_OP(nn_bias_add)
    .describe("Add bias to input")
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input data")
    .add_argument("bias", "Tensor", "The bias data")
    .set_attr<std::string>("TAttrs", "BiasAddAttrs"); // Link to Attrs

// Example of Op with Optional Input
KXC_REGISTER_OP(nn_dropout)
    .describe("Dropout layer")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "Input data")
    // In TVM/Relay, optional tensors are often handled by variable arg counts or explicit None
    // Here we document it.
    .add_argument("mask", "Tensor", "Dropout mask (optional)", true, "None");


void test_op_params() {
    std::cout << "Testing Operator Parameter Registration..." << std::endl;

    // 1. Inspect BiasAdd
    const Op& op = Op::Get("nn_bias_add");
    assert(op->num_inputs == 2);
    assert(op->arguments.size() == 2);
    assert(op->arguments[0].name == "data");
    assert(op->arguments[1].name == "bias");
    
    // Check Attribute Binding
    auto it = op->attrs.find("TAttrs");
    assert(it != op->attrs.end());
    assert(std::any_cast<std::string>(it->second) == "BiasAddAttrs");

    // 2. Create BiasAdd with Default Attributes
    // User code normally parses arguments or uses a builder.
    // Here we show that the underlying C++ struct holds the defaults.
    BiasAddAttrs attrs = BiasAddAttrs::Create(); // Uses default axis=1
    assert(attrs->axis == 1);
    std::cout << "BiasAddAttrs default axis: " << attrs->axis << std::endl;

    // 3. Inspect Dropout (Optional Input)
    const Op& dropout = Op::Get("nn_dropout");
    assert(dropout->arguments.size() == 2);
    assert(dropout->arguments[1].name == "mask");
    assert(dropout->arguments[1].is_optional == true);
    std::cout << "Dropout optional arg: " << dropout->arguments[1].name << std::endl;

    std::cout << "PASS: Operator Parameter Registration" << std::endl;
}

int main() {
    test_op_params();
    return 0;
}
