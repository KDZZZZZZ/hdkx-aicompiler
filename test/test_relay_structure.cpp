#include "../include/base/relay.h"
#include <iostream>
#include <vector>
#include <cassert>

using namespace kxc;

void test_relay_structure() {
    std::cout << "Testing Relay Structure..." << std::endl;

    // 1. Test Var
    Var x("x");
    assert(x->vid->name_hint == "x");
    std::cout << "Var created: " << x->vid->name_hint << std::endl;

    // 2. Test Constant
    std::vector<int64_t> shape = {2, 3};
    Tensor t(shape, "float32");
    Constant c(t);
    assert(c->data->shape[0] == 2);
    std::cout << "Constant created with shape: " << c->data->shape[0] << "," << c->data->shape[1] << std::endl;

    // 3. Test Call
    // Create a dummy Op (using Var as Op for now, though usually it's an OpNode)
    Var add_op("add"); 
    std::vector<Expr> args = {x, c};
    Call call(add_op, args);
    assert(call->args.size() == 2);
    assert(call->op.defined());
    std::cout << "Call created with " << call->args.size() << " args." << std::endl;

    // 4. Test Function
    std::vector<Var> params = {x};
    Function func(params, call);
    assert(func->params.size() == 1);
    assert(func->body.defined());
    std::cout << "Function created." << std::endl;

    std::cout << "PASS: Relay Structure" << std::endl;
}

int main() {
    test_relay_structure();
    return 0;
}
