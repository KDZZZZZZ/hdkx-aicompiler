#include "te/te.h"
#include <iostream>
#include <vector>
#include <cassert>

using namespace kxc;
using namespace kxc::te;
using namespace kxc::tir;

namespace kxc {
    thread_local Arena* current_arena = nullptr;
}

void test_placeholder() {
    std::cout << "Testing Placeholder..." << std::endl;
    Var n("n"), m("m");
    Tensor A = placeholder({n, m}, DataType::Float(32), "A");
    
    assert(A->shape.size() == 2);
    assert(A->op.defined());
    assert(A->op.As<PlaceholderOpNode>());
    assert(A->name == "A");
    std::cout << "Placeholder created successfully." << std::endl;
}

void test_compute() {
    std::cout << "Testing Compute..." << std::endl;
    Var n("n"), m("m");
    Tensor A = placeholder({n, m}, DataType::Float(32), "A");
    
    Tensor B = compute({n, m}, [&](const Array<Var>& axis) {
        Var i = axis[0];
        Var j = axis[1];
        // B[i, j] = A[i, j] + 1.0
        return A(i, j) + 1.0f;
    }, "B");
    
    assert(B->shape.size() == 2);
    assert(B->op.defined());
    assert(B->op.As<ComputeOpNode>());
    
    const ComputeOpNode* compute_op = B->op.As<ComputeOpNode>();
    assert(compute_op->axis.size() == 2);
    assert(compute_op->body.size() == 1);
    
    std::cout << "Compute created successfully." << std::endl;
}

void test_schedule() {
    std::cout << "Testing Schedule..." << std::endl;
    Var n("n"), m("m");
    Tensor A = placeholder({n, m}, DataType::Float(32), "A");
    Tensor B = compute({n, m}, [&](const Array<Var>& axis) {
        return A(axis) + 1.0f;
    }, "B");
    
    Schedule s = create_schedule({B->op});
    assert(s->outputs.size() == 1);
    assert(s->stages.size() >= 1);
    
    std::cout << "Schedule created successfully." << std::endl;
}

int main() {
    test_placeholder();
    test_compute();
    test_schedule();
    std::cout << "All TE tests passed!" << std::endl;
    return 0;
}
