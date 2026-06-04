/*! \file src/codegen/compiled_kernel.cc
 * \brief 实现 C/LLVM codegen、LLVM JIT 和 compiled kernel 调用封装。
 */

#include "codegen/compiled_kernel.h"

#include <memory>
#include <stdexcept>

#ifdef __linux__
#include <dlfcn.h>
#endif

#if KXC_USE_LLVM
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#endif

namespace kxc {
namespace codegen {

CompiledKernelNode::~CompiledKernelNode() {
#ifdef __linux__
    if (lib_handle) {
        dlclose(lib_handle);
        lib_handle = nullptr;
    }
#endif

#if KXC_USE_LLVM
    // Clean up JIT resource if it's a shared_ptr to LLJIT
    if (backend == CodeGenBackend::kLLVM && jit_resource) {
        auto* jit_ptr = static_cast<std::shared_ptr<std::unique_ptr<llvm::orc::LLJIT>>*>(jit_resource);
        delete jit_ptr;
        jit_resource = nullptr;
    }
#endif

    func_ptr = nullptr;
}

void CompiledKernel::operator()(const std::vector<void*>& args) const {
    auto* node = operator->();
    if (!node || !node->func_ptr) {
        throw std::runtime_error("CompiledKernel: not ready or func_ptr is null");
    }
    // 标准kernel ABI: int32_t kernel(void* arg0, void* arg1, ...)
    // 通过一个args数组指针调用
    using KernelFunc = int32_t (*)(void**);
    auto fn = reinterpret_cast<KernelFunc>(node->func_ptr);
    int32_t ret = fn(const_cast<void**>(args.data()));
    if (ret != 0) {
        throw std::runtime_error("Kernel " + node->kernel_name +
                                 " returned error code " + std::to_string(ret));
    }
}

bool CompiledKernel::IsReady() const {
    auto* node = operator->();
    return node && node->func_ptr != nullptr;
}

}  // namespace codegen
}  // namespace kxc
