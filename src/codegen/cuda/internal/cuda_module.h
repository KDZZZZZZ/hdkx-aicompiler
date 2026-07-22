/*! \file src/codegen/cuda/internal/cuda_module.h
 * \brief 定义 CUDA 源码经 NVRTC 编译、Driver API 加载并封装为 CompiledKernel 的边界。
 */

#pragma once

#include <string>
#include <vector>

#include "../../internal/compiled_kernel.h"

namespace kxc::codegen {

/*!
 * \brief 控制单次 NVRTC 编译的输入选项。
 *
 * architecture 使用 NVRTC 接受的虚拟架构名称，例如 compute_75；additional_options
 * 用于传入诊断、快速数学等后端私有开关，调用方不需要接触 nvrtcProgram 句柄。
 */
struct CUDACompileOptions {
    std::string architecture;
    std::vector<std::string> additional_options;
    std::string source_name{"kxc_kernel.cu"};
};

/*!
 * \brief 编译和加载 CUDA 内核的无状态入口。
 *
 * 每次 Compile 都创建独立的 CUmodule 所有者。返回的 CompiledKernel 通过 launcher
 * 强持有 module；异步启动又通过 AsyncOperation 强持有 CompiledKernel，因此 module
 * 不会在仍有 GPU 工作访问其代码时提前卸载。
 */
class CUDAModule {
public:
    /*! \brief 使用 NVRTC 把 CUDA C 源码编译为 PTX；失败异常包含完整编译日志。 */
    static std::string CompileToPTX(const std::string& source,
                                    const CUDACompileOptions& options);

    /*!
     * \brief 编译源码、加载 PTX、解析 signature.symbol，并组装可异步启动的内核。
     * \param source 已生成的 CUDA C 源码。
     * \param signature 有序 NDArray 参数 ABI；符号名也是 Driver lookup 的事实来源。
     * \param metadata CUDA 设备、grid、block 与动态 shared memory 契约。
     * \param options NVRTC 架构和附加编译选项。
     */
    static CompiledKernel Compile(const std::string& source,
                                  KernelSignature signature,
                                  KernelLaunchMetadata metadata,
                                  const CUDACompileOptions& options);

    /*! \brief Loads one PTX module and returns symbol launchers sharing its owner. */
    static std::vector<CompiledKernel> CompileMany(
        const std::string& source,
        const std::vector<KernelSignature>& signatures,
        const std::vector<KernelLaunchMetadata>& metadata,
        const CUDACompileOptions& options);
};

}  // namespace kxc::codegen
