#include "te/topi/broadcast.h"
#include "te/topi/elemwise.h"
#include "te/topi/reduction.h"
#include "te/topi/nn.h"
#include "te/topi/transform.h"
#include <iostream>
#include "base/arena.h"

namespace kxc {
    thread_local Arena* current_arena = nullptr;
}

using namespace kxc;
using namespace kxc::te;
using namespace kxc::te::topi;

int main() {
    // 1. Placeholder
    Tensor A = placeholder({32, 32}, DataType::Float(32), "A");
    Tensor B = placeholder({32, 32}, DataType::Float(32), "B");
    
    // 2. Broadcast Add
    Tensor C = add(A, B);
    
    // 3. Elemwise Relu
    Tensor D = relu(C);
    
    // 4. Reduction Sum
    Tensor E = sum(D, {0});
    
    // 5. Matmul
    Tensor F = dense(A, B);
    
    std::cout << "Test Compile Success" << std::endl;
    return 0;
}
