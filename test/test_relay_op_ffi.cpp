#include "base/registry.h"
#include "base/packedfunc.h"
#include "relay/relay.h"
#include "relay/op.h"
// #include "relay/attrs.h"
#include "base/arena.h"
#include <iostream>
#include <vector>
#include <cassert>

using namespace kxc;
// using namespace kxc::relay;

// Helper to check and exit
void CheckFunc(const PackedFunc& f, const std::string& name) {
    if (!f.defined()) {
        std::cerr << "ERROR: Function " << name << " is not defined (not found in Registry)." << std::endl;
        exit(1);
    }
}

void TestMakeAdd() {
    std::cout << "Testing MakeAdd..." << std::endl;
    auto f = Registry::Global().Get("kxc.relay.op._make.add");
    CheckFunc(f, "kxc.relay.op._make.add");

    Var x("x");
    Var y("y");
    
    // Call PackedFunc
    Expr res = f(x, y); 
    
    const CallNode* call = res.As<CallNode>();
    assert(call != nullptr);
    const OpNode* op = call->op.As<OpNode>();
    assert(op != nullptr);
    assert(op->name == "add");
    assert(call->args.size() == 2);
    
    std::cout << "MakeAdd passed." << std::endl;
}

void TestMakeConv2D() {
    std::cout << "Testing MakeConv2D..." << std::endl;
    auto f = Registry::Global().Get("kxc.relay.op._make.conv2d");
    CheckFunc(f, "kxc.relay.op._make.conv2d");

    Var data("data");

    Var weight("weight");
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> padding = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    int groups = 1;
    int channels = 32;
    std::vector<int64_t> kernel_size = {3, 3};
    std::string data_layout = "NCHW";
    std::string kernel_layout = "OIHW";
    std::string out_layout = "";
    std::string out_dtype = "";

    Expr res = f(data, weight, strides, padding, dilation, groups, channels, kernel_size, data_layout, kernel_layout, out_layout, out_dtype);
    
    const CallNode* call = res.As<CallNode>();
    assert(call != nullptr);
    const OpNode* op = call->op.As<OpNode>();
    assert(op != nullptr);
    assert(op->name == "nn_conv2d");
    
    const Conv2DAttrsNode* attrs = call->attrs.As<Conv2DAttrsNode>();
    assert(attrs != nullptr);
    assert(attrs->channels == 32);
    assert(attrs->strides[0] == 1);
    
    std::cout << "MakeConv2D passed." << std::endl;
}

void TestMakeReduceMean() {
    std::cout << "Testing MakeReduceMean..." << std::endl;
    auto f = Registry::Global().Get("kxc.relay.op._make.reduce_mean");
    CheckFunc(f, "kxc.relay.op._make.reduce_mean");

    Var data("data");
    std::vector<int64_t> axes = {1};
    int64_t keepdims = 1; 
    
    Expr res = f(data, axes, keepdims);
    
    const CallNode* call = res.As<CallNode>();
    assert(call != nullptr);
    const OpNode* op = call->op.As<OpNode>();
    assert(op != nullptr);
    assert(op->name == "reduce_mean");
    
    const ReduceMeanAttrsNode* attrs = call->attrs.As<ReduceMeanAttrsNode>();
    assert(attrs != nullptr);
    assert(attrs->keepdims == 1);
    assert(attrs->axes[0] == 1);
    
    std::cout << "MakeReduceMean passed." << std::endl;
}

void TestMakeMatMul() {
    std::cout << "Testing MakeMatMul..." << std::endl;
    auto f = Registry::Global().Get("kxc.relay.op._make.matmul");
    CheckFunc(f, "kxc.relay.op._make.matmul");

    Var a("a"), b("b");
    Expr res = f(a, b);
    const CallNode* call = res.As<CallNode>();
    assert(call != nullptr);
    const OpNode* op = call->op.As<OpNode>();
    assert(op->name == "matmul");
    std::cout << "MakeMatMul passed." << std::endl;
}

void TestMakeDense() {
    std::cout << "Testing MakeDense..." << std::endl;
    auto f = Registry::Global().Get("kxc.relay.op._make.dense");
    CheckFunc(f, "kxc.relay.op._make.dense");

    Var data("data"), weight("weight");
    int units = 64;
    std::string out_dtype = "float32";
    Expr res = f(data, weight, units, out_dtype);
    const CallNode* call = res.As<CallNode>();
    const OpNode* op = call->op.As<OpNode>();
    assert(op->name == "nn_dense");
    const DenseAttrsNode* attrs = call->attrs.As<DenseAttrsNode>();
    assert(attrs != nullptr);
    assert(attrs->units == 64);
    std::cout << "MakeDense passed." << std::endl;
}

void TestMakeRelu() {
    std::cout << "Testing MakeRelu..." << std::endl;
    auto f = Registry::Global().Get("kxc.relay.op._make.relu");
    CheckFunc(f, "kxc.relay.op._make.relu");

    Var data("data");
    Expr res = f(data);
    const CallNode* call = res.As<CallNode>();
    const OpNode* op = call->op.As<OpNode>();
    assert(op->name == "nn_relu");
    std::cout << "MakeRelu passed." << std::endl;
}

void TestMakeSoftmax() {
    std::cout << "Testing MakeSoftmax..." << std::endl;
    auto f = Registry::Global().Get("kxc.relay.op._make.softmax");
    CheckFunc(f, "kxc.relay.op._make.softmax");

    Var data("data");
    int axis = 1;
    Expr res = f(data, axis);
    const CallNode* call = res.As<CallNode>();
    const OpNode* op = call->op.As<OpNode>();
    assert(op->name == "nn_softmax");
    const SoftmaxAttrsNode* attrs = call->attrs.As<SoftmaxAttrsNode>();
    assert(attrs != nullptr);
    assert(attrs->axis == 1);
    std::cout << "MakeSoftmax passed." << std::endl;
}

int main() {
    try {
        // Initialize Arena for memory management test
        Arena arena;
        current_arena = &arena;
        
        TestMakeAdd();
        TestMakeConv2D();
        TestMakeReduceMean();
        TestMakeMatMul();
        TestMakeDense();
        TestMakeRelu();
        TestMakeSoftmax();
        
        std::cout << "All tests passed!" << std::endl;
        
        current_arena = nullptr;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        current_arena = nullptr;
        return 1;
    }
    return 0;
}
