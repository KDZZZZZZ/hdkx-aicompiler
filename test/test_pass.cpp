#include "../include/base/pass.h"
#include "../include/base/relay.h"
#include "../include/relay/op_macros.h"
#include <iostream>
#include <cassert>

using namespace kxc;

// A simple pass that replaces a specific variable with another expression
class VarSubstitutor : public RelayPass {
public:
    VarSubstitutor(Var target, Expr replacement) 
        : target_(target), replacement_(replacement) {}

    // Override using Expr
    Expr VisitVar(const VarNode* op, const Expr& ref) override {
        if (op->vid->name_hint == target_->vid->name_hint) {
            return replacement_;
        }
        return ref;
    }

private:
    Var target_;
    Expr replacement_;
};

void test_pass_infrastructure() {
    std::cout << "Testing Pass Infrastructure..." << std::endl;

    // 1. Setup Graph: f(x) = x
    Var x("x");
    Var y("y");
    
    // Graph: Call(op, {x})
    // We'll use a dummy Op
    Var op_var("dummy_op"); 
    std::vector<Expr> args = {x};
    Call call(op_var, args);

    // 2. Run Pass: Replace x with y
    VarSubstitutor substitutor(x, y);
    Expr new_graph = substitutor.Mutate(call);

    // 3. Verify
    // The new graph should be Call(op, {y})
    const CallNode* new_call_node = new_graph.As<CallNode>();
    assert(new_call_node != nullptr);
    assert(new_call_node->args.size() == 1);
    
    const VarNode* arg0 = new_call_node->args[0].As<VarNode>();
    assert(arg0 != nullptr);
    assert(arg0->vid->name_hint == "y"); // Should be y now
    
    std::cout << "Variable substitution successful: x -> y" << std::endl;

    // 4. Test Copy-On-Write (No change)
    // Run pass again replacing 'z' (which doesn't exist) -> should return original ref
    Var z("z");
    VarSubstitutor no_op_pass(z, y);
    Expr same_graph = no_op_pass.Mutate(call);
    
    assert(same_graph.get() == call.get()); // Pointers should be identical
    std::cout << "Copy-On-Write verification successful." << std::endl;

    std::cout << "PASS: Pass Infrastructure" << std::endl;
}

int main() {
    test_pass_infrastructure();
    return 0;
}
