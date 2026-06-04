/*! \file include/relay/common_ops.h
 * \brief 定义 Relay IR 节点、算子注册、attrs 和 Relay 到 TE lowering 属性。
 */

#pragma once
#include "op_macros.h"

// Common Operators Registration
// In a real project, these might be distributed across different files
// e.g., src/relay/op/nn/convolution.cc, src/relay/op/tensor/transform.cc

namespace kxc {
namespace relay {

// We use a dummy function to ensure static initialization happens if included
// In a real shared library, the static initializers run automatically on load.
inline void RegisterCommonOps() {
    // This function body is empty, but calling it ensures the translation unit is linked
    // However, for header-only or direct inclusion, we define macros here.
    // But macros define static variables. Defining them in a header included by multiple
    // files will cause "multiple definition" errors for the static variables 
    // if they are not inline or distinct.
    
    // BETTER APPROACH for this test/demo:
    // Define a .cc file that registers them.
}

}
}
