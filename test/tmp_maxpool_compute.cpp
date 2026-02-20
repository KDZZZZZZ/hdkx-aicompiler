#include "relay/op.h"
#include "te/te.h"
#include <iostream>

using namespace kxc;
using namespace kxc::relay;

namespace kxc {
namespace relay {
te::Tensor MaxPool2DCompute(const Attrs& attrs, const Array<te::Tensor>& inputs, const kxc::Type& out_type);
}
}

int main() {
    try {
        te::Tensor x = te::placeholder({1, 64, 112, 112}, tir::DataType::Float(32), "x");
        MaxPool2DAttrs attrs = MaxPool2DAttrs::Create({2, 2}, {1, 1, 1, 1}, {1, 1}, {3, 3}, "NCHW", false);
        te::Tensor y = MaxPool2DCompute(attrs, {x}, Type());
        std::cout << "name=" << y->name << " rank=" << y->shape.size() << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "err: " << e.what() << std::endl;
        return 1;
    }
}

