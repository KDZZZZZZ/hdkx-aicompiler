/*! \file include/codegen/backend.h
 * \brief 定义不依赖具体代码生成器实现的后端标识。
 */

#pragma once

namespace kxc::codegen {

/*! \brief 代码生成路径标识；是否可执行由具体产物契约决定。 */
enum class CodeGenBackend : int {
    /*! \brief 通过 LLVM IR 和 ORC JIT 生成本机 CPU 机器码。 */
    kLLVM = 0,
    /*! \brief 仅生成诊断 C 源码，不表示存在可执行 C runtime backend。 */
    kC = 1,
    /*! \brief 通过 CUDA C、NVRTC 和 Driver API 生成并加载 GPU 内核。 */
    kCUDA = 2,
};

}  // namespace kxc::codegen
