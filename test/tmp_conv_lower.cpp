/*! \file test/tmp_conv_lower.cpp
 * \brief 定义编译器核心路径、pass、codegen 和 profiling 的 C++ 测试入口。
 */

#include "relay/relay.h"
#include "relay/op.h"
#include "relay/transforms/lower.h"
#include <iostream>

using namespace kxc;
using namespace kxc::relay;

int main() {
    try {
        Var input("input", TensorType({1, 3, 224, 224}, "float32"));
        runtime::NDArray w_data({64, 3, 7, 7}, "float32");
        runtime::NDArray b_data({64}, "float32");
        Constant weight(w_data);
        Constant bias(b_data);

        std::vector<int64_t> strides = {2, 2};
        std::vector<int64_t> pads = {3, 3, 3, 3};
        std::vector<int64_t> dil = {1, 1};
        std::vector<int64_t> ksize = {7, 7};
        Conv2DAttrs attrs = Conv2DAttrs::Create(strides, pads, dil, 1, 64, ksize, "NCHW", "OIHW", "", "");

        std::vector<Expr> args = {input, weight, bias};
        Call conv(Op::Get("nn_conv2d"), args, attrs);
        Function fn({input}, conv);

        tir::PrimFunc pf = LowerToTIR(fn);
        std::cout << "ok: params=" << pf->params.size() << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "err: " << e.what() << std::endl;
        return 1;
    }
}

