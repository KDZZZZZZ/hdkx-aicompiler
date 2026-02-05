#include "../include/relay/op.h"
#include "../include/relay/relay.h"
#include <iostream>
#include <vector>
#include <cassert>

using namespace kxc;

void test_op_and_attrs() {
    std::cout << "Testing Op, Call, and Attributes..." << std::endl;

    // 1. Get an Operator
    Op conv2d = Op::Get("nn.conv2d");
    assert(conv2d->name == "nn.conv2d");
    std::cout << "Op fetched: " << conv2d->name << std::endl;

    // 2. Create Attributes for Conv2D
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> padding = {1, 1};
    Conv2DAttrs attrs = Conv2DAttrs::Create(strides, padding);
    assert(attrs->strides[0] == 1);
    assert(attrs->data_layout == "NCHW");
    std::cout << "Conv2DAttrs created with padding: " << attrs->padding[0] << std::endl;

    // 3. Create Inputs (Data and Weight)
    Var data("data");
    Var weight("weight");
    std::vector<Expr> args = {data, weight};

    // 4. Create Call Node (The 'Output' Expr)
    // conv2d(data, weight, attrs=...)
    Call call(conv2d, args, attrs);
    
    // 5. Verification
    // Check Op
    const OpNode* op_node = call->op.As<OpNode>();
    assert(op_node != nullptr);
    assert(op_node->name == "nn.conv2d");

    // Check Args
    assert(call->args.size() == 2);
    const VarNode* arg0 = call->args[0].As<VarNode>();
    assert(arg0->vid->name_hint == "data");

    // Check Attrs
    const Conv2DAttrsNode* attr_node = call->attrs.As<Conv2DAttrsNode>();
    assert(attr_node != nullptr);
    assert(attr_node->strides[0] == 1);

    std::cout << "PASS: Op and Attrs Structure" << std::endl;
}

int main() {
    test_op_and_attrs();
    return 0;
}
