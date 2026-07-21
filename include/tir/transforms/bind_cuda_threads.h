/*! \file include/tir/transforms/bind_cuda_threads.h
 * \brief 声明一维 CUDA 线程绑定调度及其强类型结果。
 */

#pragma once

#include "base/target.h"
#include "codegen/kernel_signature.h"
#include "tir/stmt.h"

namespace kxc::tir {

/*! \brief 调度后 PrimFunc 保存同源启动元数据使用的稳定 attr key。 */
constexpr const char* kCudaLaunchMetadataAttr = "kxc.cuda.launch_metadata";
/*! \brief 调度后 PrimFunc 保存原始一维工作量使用的稳定 attr key。 */
constexpr const char* kCudaWorkSizeAttr = "kxc.cuda.work_size";

/*! \brief 原子持有 CUDA 调度后 TIR 与其唯一启动元数据。 */
class CudaScheduleResultNode final : public Object {
public:
    /*! \brief 一次性构造完整结果，节点不存在半初始化状态。 */
    CudaScheduleResultNode(PrimFunc prim_func,
                           codegen::KernelLaunchMetadata launch_metadata);

    KXC_OBJECT_DECLARE

private:
    friend class CudaScheduleResult;
    PrimFunc prim_func_;
    codegen::KernelLaunchMetadata launch_metadata_;
};

KXC_OBJECT_DEFINE_WITH_KEY(CudaScheduleResultNode, "kxc.tir.CudaScheduleResultNode")

/*! \brief CUDA thread-binding pass 的类型安全返回句柄。 */
class CudaScheduleResult : public ObjectRef {
public:
    /*! \brief 组合并校验调度后函数与同源启动元数据。 */
    CudaScheduleResult(PrimFunc prim_func,
                       codegen::KernelLaunchMetadata launch_metadata);
    /*! \brief 从对象系统恢复结果并重新校验节点内容。 */
    explicit CudaScheduleResult(const ObjectRef& ref);

    /*! \brief 返回带 ThreadBinding 和 metadata attr 的 PrimFunc。 */
    PrimFunc prim_func() const;
    /*! \brief 返回与 PrimFunc attr 共享对象身份的启动元数据。 */
    codegen::KernelLaunchMetadata launch_metadata() const;
    /*! \brief 校验结果内部对象身份和 CUDA backend/device 语义。 */
    void Validate() const;
    /*! \brief 返回经过动态类型检查的只读节点。 */
    const CudaScheduleResultNode* operator->() const;
};

/*! \brief 把可证明独立的一维外层循环映射到 blockIdx.x/threadIdx.x。 */
CudaScheduleResult BindCudaThreads(const PrimFunc& function, const Target& target);

/*! \brief 从调度后 PrimFunc 恢复强类型启动元数据。 */
codegen::KernelLaunchMetadata GetCudaLaunchMetadata(const PrimFunc& function);

}  // namespace kxc::tir
