#include "relay/relay.h"
#include "relay/op.h"
#include "relay/transforms/lower.h"
#include "tir/stmt.h"
#include "tir/expr.h"
#include <iostream>
#include <cassert>

using namespace kxc;
using namespace kxc::relay;

// Helper to verify Stmt structure
void VerifyTIR(tir::Stmt stmt) {
    if (!stmt.defined()) {
        std::cerr << "Error: Lowering returned undefined statement." << std::endl;
        exit(1);
    }

    // Expect For -> For -> Store (since shape is [1, 1] default in our mock)
    // Actually shape is 2D in mock.
    
    const tir::ForNode* outer_loop = stmt.As<tir::ForNode>();
    if (!outer_loop) {
        std::cerr << "Error: Expected outer For loop." << std::endl;
        exit(1);
    }
    std::cout << "Outer Loop: " << outer_loop->loop_var->name_hint << std::endl;
    
    const tir::ForNode* inner_loop = outer_loop->body.As<tir::ForNode>();
    if (!inner_loop) {
        std::cerr << "Error: Expected inner For loop." << std::endl;
        exit(1);
    }
    std::cout << "Inner Loop: " << inner_loop->loop_var->name_hint << std::endl;
    
    const tir::StoreNode* store = inner_loop->body.As<tir::StoreNode>();
    if (!store) {
        std::cerr << "Error: Expected Store statement." << std::endl;
        exit(1);
    }
    std::cout << "Store to: " << store->buffer_var->name_hint << std::endl;
    
    // Check computation: inputs[0] + inputs[1]
    const tir::AddNode* add = store->value.As<tir::AddNode>();
    if (!add) {
        std::cerr << "Error: Expected Add operation in store value." << std::endl;
        exit(1);
    }
    std::cout << "Operation verified: Add" << std::endl;
}

void TestLowerAdd() {
    std::cout << "Testing Lower Add..." << std::endl;
    
    // 1. Create Relay Graph
    Var x("x");
    Var y("y");
    
    // Fetch Add Op
    const kxc::relay::Op& add_op = kxc::relay::Op::Get("add");
    Call add_call(add_op, {x, y});
    
    Function func({x, y}, add_call);
    
    // 2. Lower
    tir::Stmt stmt = LowerToTIR(func);
    
    // 3. Verify
    VerifyTIR(stmt);
    
    std::cout << "PASS: Lower Add" << std::endl;
}

int main() {
    try {
        TestLowerAdd();
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
