#include "base/packedfunc.h"
#include "base/tensor.h"
#include <iostream>
#include <cassert>

namespace kxc {
    thread_local Arena* current_arena = nullptr;
}

using namespace kxc;

void test_tensor_arg() {
    std::cout << "Testing Tensor argument..." << std::endl;
    
    // Create a Tensor
    Tensor t({2, 3}, "float32");
    
    // Create a PackedFunc that takes a Tensor
    PackedFunc f = ToPackedFunc([](Tensor t_arg) {
        std::cout << "Inside PackedFunc: Tensor received." << std::endl;
        assert(t_arg.defined());
        assert(t_arg->ndim == 2);
        assert(t_arg->shape[0] == 2);
        assert(t_arg->shape[1] == 3);
        
        // Check data type
        assert(t_arg->dtype.code == 2); // kDLFloat
        assert(t_arg->dtype.bits == 32);
        
        std::cout << "Tensor verification passed." << std::endl;
    });
    
    // Call it
    f(t);
}

void test_tensor_return() {
    std::cout << "Testing Tensor return..." << std::endl;
    // Not implemented yet - RetValue needs support for Tensor
    // Currently RetValue supports ObjectRef, so we can return ObjectRef and cast.
    // But direct conversion to Tensor from RetValue needs modification in RetValue::operator=
    // However, since Tensor is not ObjectRef, RetValue doesn't have operator=(Tensor).
    // The user only asked for input support.
    // "support putting this tensor as packedfunc input"
    // So I will skip return test for now unless requested.
}

int main() {
    try {
        test_tensor_arg();
        std::cout << "All tests passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
