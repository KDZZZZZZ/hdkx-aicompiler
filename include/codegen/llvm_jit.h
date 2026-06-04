/*! \file include/codegen/llvm_jit.h
 * \brief 定义 codegen 后端、C/LLVM codegen、JIT 和 compiled kernel 抽象。
 */

#pragma once

#if KXC_USE_LLVM

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

/*! \brief LLVM ORC JIT 封装，负责优化 module 并导出可调用 kernel。 */
class LLVMJITEngine {
public:
    LLVMJITEngine();
    ~LLVMJITEngine();

    /*! \brief 编译 LLVM Module 并返回可执行 CompiledKernel。 */
    CompiledKernel Compile(std::unique_ptr<llvm::Module> module,
                           std::unique_ptr<llvm::LLVMContext> context,
                           const std::string& func_name,
                           int opt_level = 2);

    /*! \brief 对 LLVM Module 执行 opt_level 指定级别的优化。 */
    void Optimize(llvm::Module* module, int opt_level);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace codegen
}  // namespace kxc

#endif  // KXC_USE_LLVM
