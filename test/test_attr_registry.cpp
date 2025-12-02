#include "../include/relay/op_macros.h"
#include "../include/base/relay.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <any>

using namespace kxc;

void test_attr_registry() {
    std::cout << "Testing Attribute Registry..." << std::endl;

    // 1. Verify Op Attribute Metadata
    const Op& conv2d = Op::Get("nn_conv2d");
    auto it = conv2d->attrs.find("TAttrs");
    assert(it != conv2d->attrs.end());
    assert(std::any_cast<std::string>(it->second) == "Conv2DAttrs");
    std::cout << "nn_conv2d TAttrs: " << std::any_cast<std::string>(it->second) << std::endl;

    // 2. Verify Concrete Attributes Creation
    // Conv2D
    auto conv_attrs = Conv2DAttrs::Create({1, 1}, {1, 1}, "NCHW");
    assert(conv_attrs->data_layout == "NCHW");
    std::cout << "Conv2DAttrs created." << std::endl;

    // Dense
    auto dense_attrs = DenseAttrs::Create(128);
    assert(dense_attrs->units == 128);
    std::cout << "DenseAttrs created." << std::endl;

    // Softmax
    auto softmax_attrs = SoftmaxAttrs::Create(1);
    assert(softmax_attrs->axis == 1);
    std::cout << "SoftmaxAttrs created." << std::endl;

    std::cout << "PASS: Attribute Registry and Types" << std::endl;
}

int main() {
    test_attr_registry();
    return 0;
}
