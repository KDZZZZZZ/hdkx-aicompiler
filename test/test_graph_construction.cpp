#include "../include/base/relay.h"
#include "../include/base/op.h"
#include <iostream>
#include <vector>
#include <cassert>

using namespace kxc;

// Helper to make a Call
Expr CallOp(const std::string& op_name, std::vector<Expr> args) {
    Op op = Op::Get(op_name);
    return Call(op, std::move(args));
}

void test_complex_graph() {
    std::cout << "Building complex graph with control flow..." << std::endl;

    /*
     * We will build a graph that represents:
     * 
     * fn (x, w1, w2) {
     *   let t1 = nn.conv2d(x, w1)
     *   let t2 = nn.relu(t1)
     *   let mean = mean(t2)  <-- simplifying, assume t2 is scalar for condition
     *   
     *   if (mean > 0.5) {
     *      return nn.dense(t2, w2)
     *   } else {
     *      return t2
     *   }
     * }
     */

    // 1. Define Parameters
    Var x("x");
    Var w1("w1");
    Var w2("w2");

    // 2. Build Computation Body
    
    // t1 = conv2d(x, w1)
    Expr t1 = CallOp("nn_conv2d", {x, w1});
    
    // t2 = relu(t1)
    // We want to bind this to a variable 'v_t2' so we can reference it in multiple places
    // (in the condition and in the branches)
    // But in pure functional form (Relay), we can just use the Expr t2 repeatedly.
    // However, using Let binding makes the graph structure explicit about reuse.
    Expr t2_expr = CallOp("nn_relu", {t1});
    Var v_t2("v_t2"); // The variable that holds the result of relu

    // Condition: v_t2 > constant(0.5)
    // Note: In real Relay, we might need a ReduceMean op first if v_t2 is a tensor.
    // Let's assume we have a 'mean' op or just compare directly for this structural test.
    std::vector<int64_t> scalar_shape = {1};
    Tensor val_0_5(scalar_shape, "float32"); // Value 0.5
    Constant c_0_5(val_0_5);
    
    Expr condition = CallOp("greater", {v_t2, c_0_5});

    // True Branch: dense(v_t2, w2)
    Expr true_branch = CallOp("nn_dense", {v_t2, w2});

    // False Branch: v_t2
    Expr false_branch = v_t2;

    // If Expression
    Expr if_body = If(condition, true_branch, false_branch);

    // Let Binding: let v_t2 = relu(...) in if(...)
    Expr body = Let(v_t2, t2_expr, if_body);

    // 3. Create Function
    std::vector<Var> params = {x, w1, w2};
    Function func(params, body);

    // 4. Verification
    std::cout << "Graph built successfully." << std::endl;
    
    // Verify Function Params
    assert(func->params.size() == 3);
    assert(func->params[0]->vid->name_hint == "x");

    // Verify Body is Let
    const LetNode* let_node = func->body.As<LetNode>();
    assert(let_node != nullptr);
    assert(let_node->var->vid->name_hint == "v_t2");
    
    // Verify Let Value is Call(relu)
    const CallNode* relu_call = let_node->value.As<CallNode>();
    assert(relu_call != nullptr);
    const OpNode* relu_op = relu_call->op.As<OpNode>();
    assert(relu_op->name == "nn_relu");

    // Verify Let Body is If
    const IfNode* if_node = let_node->body.As<IfNode>();
    assert(if_node != nullptr);
    
    // Verify Condition is Call(greater)
    const CallNode* cond_call = if_node->cond.As<CallNode>();
    assert(cond_call != nullptr);
    const OpNode* greater_op = cond_call->op.As<OpNode>();
    assert(greater_op->name == "greater");

    std::cout << "PASS: Complex Control Flow Graph" << std::endl;
}

int main() {
    test_complex_graph();
    return 0;
}
