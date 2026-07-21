/*! \file include/codegen/llvm_jit.h
 * \brief 定义每模块独占 ORC LLJIT 的 LLVM 编译入口。
 */

#pragma once

#if KXC_USE_LLVM

#include <memory>

#include "codegen/compiled_kernel.h"

namespace llvm {
class LLVMContext;
class Module;
}  // namespace llvm

namespace kxc::codegen {

/*!
 * \brief 把单个 LLVM Module 优化、验证并封装为强类型 CompiledKernel。
 *
 * 每次 Compile 都创建独立 LLJIT。LLVMJITEngine 本身不持有可执行资源，返回的
 * launcher 是 module、context、函数地址和 JIT 生命周期的唯一所有者。
 */
class LLVMJITEngine {
public:
    LLVMJITEngine() = default;
    ~LLVMJITEngine() = default;

    /*! \brief 编译 module，并原样保留签名与启动元数据 ObjectRef。 */
    CompiledKernel Compile(std::unique_ptr<llvm::Module> module,
                           std::unique_ptr<llvm::LLVMContext> context,
                           const KernelSignature& signature,
                           const KernelLaunchMetadata& launch_metadata,
                           int opt_level = 2) const;

    /*! \brief 按 0 到 3 的优化等级运行 LLVM 默认 module pipeline。 */
    void Optimize(llvm::Module* module, int opt_level) const;
};

}  // namespace kxc::codegen

#endif  // KXC_USE_LLVM
