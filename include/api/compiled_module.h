/*! \file include/api/compiled_module.h
 * \brief 定义后端无关、以 NDArray 启动的编译模块对象。
 */

#pragma once

#include <memory>
#include <utility>

#include "base/profiling.h"
#include "base/target.h"
#include "codegen/compiled_kernel.h"
#include "tir/stmt.h"

namespace kxc::api {

/*! \brief 完整持有编译产物、常量 payload 和后端可执行资源的对象节点。 */
class CompiledModuleNode final : public Object {
public:
    /*! \brief 一次性接管完整编译产物，避免节点暴露半初始化状态。 */
    CompiledModuleNode(Target target,
                       tir::PrimFunc prim_func,
                       codegen::KernelSignature signature,
                       codegen::KernelLaunchMetadata launch_metadata,
                       Map<String, runtime::NDArray> constants,
                       codegen::CompiledKernel executable,
                       std::shared_ptr<profiling::ProfileContext> profile_context)
        : target_(std::move(target)),
          prim_func_(std::move(prim_func)),
          signature_(std::move(signature)),
          launch_metadata_(std::move(launch_metadata)),
          constants_(std::move(constants)),
          executable_(std::move(executable)),
          profile_context_(std::move(profile_context)) {}

    KXC_OBJECT_DECLARE

private:
    friend class CompiledModule;

    /*! \brief 编译与执行设备的唯一目标快照。 */
    Target target_;
    /*! \brief 用于诊断和后续 artifact 导出的 lowered TIR。 */
    tir::PrimFunc prim_func_;
    /*! \brief 公共 Launch 校验所依据的唯一参数契约。 */
    codegen::KernelSignature signature_;
    /*! \brief 后端设备和启动尺寸。 */
    codegen::KernelLaunchMetadata launch_metadata_;
    /*! \brief 按稳定 key 强持有编译常量。 */
    Map<String, runtime::NDArray> constants_;
    /*! \brief 隐藏具体后端资源的内部可执行对象。 */
    codegen::CompiledKernel executable_;
    /*! \brief 可选 profiling 上下文，与模块共享执行期生命周期。 */
    std::shared_ptr<profiling::ProfileContext> profile_context_;
};

KXC_OBJECT_DEFINE_WITH_KEY(CompiledModuleNode, "kxc.api.CompiledModuleNode")

/*! \brief 只接受强类型 NDArray 和显式 DeviceStream 的编译模块句柄。 */
class CompiledModule : public ObjectRef {
public:
    /*!
     * \brief 组装 Compiler 各阶段产物，并立即校验跨对象不变量。
     *
     * 该构造函数供 Compiler 和后端集成使用，不提供旧 Run ABI 或裸资源入口。
     */
    CompiledModule(Target target,
                   tir::PrimFunc prim_func,
                   codegen::KernelSignature signature,
                   codegen::KernelLaunchMetadata launch_metadata,
                   Map<String, runtime::NDArray> constants,
                   codegen::CompiledKernel executable,
                   std::shared_ptr<profiling::ProfileContext> profile_context = nullptr);
    /*! \brief 从 ObjectRef 恢复模块，并执行运行时节点类型检查。 */
    explicit CompiledModule(const ObjectRef& ref);

    /*! \brief 校验全部参数后，通过内部 executable 启动内核。 */
    AsyncOperation Launch(const Array<runtime::NDArray>& ordered_arguments,
                          const DeviceStream& stream) const;

    /*! \brief 返回签名句柄；参数数组由 KernelSignature 自身提供防别名副本。 */
    codegen::KernelSignature signature() const;
    /*! \brief 返回启动元数据句柄。 */
    codegen::KernelLaunchMetadata launch_metadata() const;
    /*! \brief 返回独立 Map，防止调用方改写模块内部常量表。 */
    Map<String, runtime::NDArray> constants() const;
    /*! \brief 返回目标快照。 */
    Target target() const;
    /*! \brief 返回 lowered TIR 诊断产物。 */
    tir::PrimFunc prim_func() const;
    /*! \brief 返回内部 executable 是否已经可启动。 */
    bool IsReady() const noexcept;
    /*! \brief 返回稳定、简短的模块状态文本。 */
    String GetStatus() const;
    /*! \brief 返回 profiling bundle 路径；未启用时为空字符串。 */
    String GetProfileBundlePath() const;
    /*! \brief 返回经过动态类型检查的只读节点。 */
    const CompiledModuleNode* operator->() const;
};

}  // namespace kxc::api
