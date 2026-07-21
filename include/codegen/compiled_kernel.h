/*! \file include/codegen/compiled_kernel.h
 * \brief 定义后端私有 launcher 与已编译内核的强类型所有权边界。
 */

#pragma once

#include <memory>
#include <utility>

#include "base/device_stream.h"
#include "base/ndarray.h"
#include "codegen/kernel_signature.h"

namespace kxc::codegen {

/*!
 * \brief 后端 launcher 的内部多态接口。
 *
 * CompiledModule 在进入该接口前已经完成全部 NDArray 契约校验。具体后端只负责
 * 参数 ABI 打包、提交执行以及通过 AsyncOperation 保活异步资源。
 */
class KernelLauncher {
public:
    virtual ~KernelLauncher() = default;

    /*! \brief 返回底层可执行资源是否已经具备启动条件。 */
    virtual bool IsReady() const noexcept = 0;

    /*!
     * \brief 启动已经完成公共契约校验的参数。
     * \param arguments 按 KernelSignature 顺序排列的 NDArray。
     * \param stream 与启动元数据设备一致的显式执行流。
     * \param executable_owner 异步后端需要保活的 CompiledKernel ObjectRef。
     */
    virtual AsyncOperation Launch(
        const Array<runtime::NDArray>& arguments,
        const DeviceStream& stream,
        const ObjectRef& executable_owner) const = 0;
};

/*! \brief 强持有签名、启动元数据和后端 launcher 的内部可执行节点。 */
class CompiledKernelNode final : public Object {
public:
    /*! \brief 一次性接管完整后端契约，节点不存在半初始化状态。 */
    CompiledKernelNode(KernelSignature signature,
                       KernelLaunchMetadata launch_metadata,
                       std::shared_ptr<const KernelLauncher> launcher)
        : signature(std::move(signature)),
          launch_metadata(std::move(launch_metadata)),
          launcher(std::move(launcher)) {}

    /*! \brief 后端入口的参数顺序与张量契约。 */
    KernelSignature signature;
    /*! \brief 后端种类、设备和启动尺寸。 */
    KernelLaunchMetadata launch_metadata;
    /*! \brief LLVM、CUDA 等后端资源的强类型所有者。 */
    std::shared_ptr<const KernelLauncher> launcher;

    KXC_OBJECT_DECLARE
};

KXC_OBJECT_DEFINE_WITH_KEY(CompiledKernelNode, "kxc.codegen.CompiledKernelNode")

/*! \brief codegen 内部可执行句柄，不向调用方暴露函数指针或后端原生资源。 */
class CompiledKernel : public ObjectRef {
public:
    /*! \brief 组装并校验 launcher、签名与启动元数据。 */
    CompiledKernel(KernelSignature signature,
                   KernelLaunchMetadata launch_metadata,
                   std::shared_ptr<const KernelLauncher> launcher);
    /*! \brief 从通用对象引用恢复 CompiledKernel，并检查运行时节点类型。 */
    explicit CompiledKernel(const ObjectRef& ref);

    /*! \brief 把已验证参数交给后端，并把当前可执行对象交给异步操作保活。 */
    AsyncOperation Launch(const Array<runtime::NDArray>& arguments,
                          const DeviceStream& stream) const;
    /*! \brief 返回 launcher 是否存在且已经可用。 */
    bool IsReady() const noexcept;
    /*! \brief 返回参数签名句柄。 */
    KernelSignature signature() const;
    /*! \brief 返回启动元数据句柄。 */
    KernelLaunchMetadata launch_metadata() const;
    /*! \brief 返回经过动态类型检查的只读节点。 */
    const CompiledKernelNode* operator->() const;
};

}  // namespace kxc::codegen
