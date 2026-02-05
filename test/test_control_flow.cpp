#include "../include/relay/op_macros.h"
#include "../include/relay/relay.h"
#include <iostream>
#include <vector>
#include <cassert>

using namespace kxc;

// Register a new operator using the macro
// Note: The macro takes a token, not a string literal
KXC_REGISTER_OP(test_add)
    .describe("Element-wise addition")
    .set_num_inputs(2);

KXC_REGISTER_OP(test_relu)
    .describe("Rectified Linear Unit")
    .set_num_inputs(1);

void test_control_flow_and_registry() {
    std::cout << "Testing Control Flow and Registry..." << std::endl;

    // 1. Verify Macro Registration
    const Op& add_op = Op::Get("test_add");
    assert(add_op->name == "test_add");
    assert(add_op->description == "Element-wise addition");
    std::cout << "Registered Op fetched: " << add_op->name << std::endl;

    // 2. Test If
    Var x("x");
    Var y("y");
    Expr cond = x; // Dummy condition
    Expr true_branch = x;
    Expr false_branch = y;
    If if_expr(cond, true_branch, false_branch);
    
    assert(if_expr->cond.defined());
    const VarNode* t_branch = if_expr->true_branch.As<VarNode>();
    assert(t_branch->vid->name_hint == "x");
    std::cout << "If expression created." << std::endl;

    // 3. Test Let
    // let v = x in v
    Var v("v");
    Let let_expr(v, x, v);
    assert(let_expr->var->vid->name_hint == "v");
    std::cout << "Let expression created." << std::endl;

    // 4. Test Tuple
    std::vector<Expr> fields = {x, y};
    Tuple tuple(fields);
    assert(tuple->fields.size() == 2);
    std::cout << "Tuple created." << std::endl;

    // 5. Test TupleGetItem
    TupleGetItem tgi(tuple, 0);
    assert(tgi->index == 0);
    std::cout << "TupleGetItem created." << std::endl;

    std::cout << "PASS: Control Flow and Registry" << std::endl;
}

int main() {
    test_control_flow_and_registry();
    return 0;
}
