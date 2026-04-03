#pragma once

#ifdef KXC_USE_LLVM

#include <memory>
#include <string>

#include "codegen/compiled_kernel.h"

namespace llvm {
class LLVMContext;
class Module;
namespace orc {
class LLJIT;
}
}  // namespace llvm

namespace kxc {
namespace codegen {

class LLVMJITEngine {
public:
    LLVMJITEngine();
    ~LLVMJITEngine();

    // 编译LLVM Module → 可执行函数指针
    // 注意: module的LLVMContext必须保持存活直到JIT完成
    CompiledKernel Compile(std::unique_ptr<llvm::Module> module,
                           std::unique_ptr<llvm::LLVMContext> context,
                           const std::string& func_name,
                           int opt_level = 2);

    // 优化LLVM Module
    void Optimize(llvm::Module* module, int opt_level);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace codegen
}  // namespace kxc

#endif  // KXC_USE_LLVM
