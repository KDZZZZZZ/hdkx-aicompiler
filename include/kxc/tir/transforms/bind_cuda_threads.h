/*! \file include/kxc/tir/transforms/bind_cuda_threads.h
 * \brief 声明独立输出到 CUDA 线程的调度及其强类型结果。
 */

#pragma once

#include <cstdint>

#include "kxc/target/target.h"
#include "kxc/tir/stmt.h"

namespace kxc::tir {

/*! \brief 调度后 PrimFunc 保存同源启动元数据使用的稳定 attr key。 */
constexpr const char* kCudaLaunchMetadataAttr = "kxc.cuda.launch_metadata";
/*! \brief 调度后 PrimFunc 保存展平输出工作量使用的稳定 attr key。 */
constexpr const char* kCudaWorkSizeAttr = "kxc.cuda.work_size";
/*! \brief 编译器按 runtime-extent 参数顺序提供的有限上界；CUDA 证明消费。 */
constexpr const char* kCudaRuntimeExtentBoundsAttr = "kxc.cuda.runtime_extent_upper_bounds";

struct CudaLaunchConfig {
    uint32_t grid_x{1};
    uint32_t grid_y{1};
    uint32_t grid_z{1};
    uint32_t block_x{1};
    uint32_t block_y{1};
    uint32_t block_z{1};
    uint64_t dynamic_shared_memory_bytes{0};
};

/*! \brief 原子持有 CUDA 调度后 TIR 与其唯一启动元数据。 */
class CudaScheduleResultNode final : public Object {
public:
    /*! \brief 一次性构造完整结果，节点不存在半初始化状态。 */
    CudaScheduleResultNode(PrimFunc prim_func, CudaLaunchConfig launch_config);

    KXC_OBJECT_DECLARE

private:
    friend class CudaScheduleResult;
    PrimFunc prim_func_;
    CudaLaunchConfig launch_config_;
};


/*! \brief CUDA thread-binding pass 的类型安全返回句柄。 */
class CudaScheduleResult : public ObjectRef {
public:
    /*! \brief 组合并校验调度后函数与同源启动元数据。 */
    CudaScheduleResult(PrimFunc prim_func, CudaLaunchConfig launch_config);
    /*! \brief 从对象系统恢复结果并重新校验节点内容。 */
    explicit CudaScheduleResult(const ObjectRef& ref);

    /*! \brief 返回带 ThreadBinding 和 metadata attr 的 PrimFunc。 */
    PrimFunc prim_func() const;
    /*! \brief 返回与 PrimFunc attr 共享对象身份的启动元数据。 */
    CudaLaunchConfig launch_config() const;
    /*! \brief 校验结果内部对象身份和 CUDA backend/device 语义。 */
    void Validate() const;
    /*! \brief 返回经过动态类型检查的只读节点。 */
    const CudaScheduleResultNode* operator->() const;
};

/*!
 * \brief 将可证明的行主序输出域展平映射到 blockIdx.x/threadIdx.x。
 *
 * 支持标量输出及完美嵌套数据循环；线程内串行归约须先初始化，
 * 且只能读写该线程独占的输出元素。多阶段张量按共有独立坐标分配线程，
 * 消除只读输入复制/转换后压缩临时存储（每线程最多 64 KiB）。
 * 独立输出还支持带完整范围保护的只读间接 Load；索引整数与地址须逐式证明安全。
 * 有限上界的 runtime-extent ABI 支持紧凑输出、串行归约和多阶段临时存储：
 * 上界决定 launch，实际形状决定 guard/坐标；符号地址须在整个域内成立。
 * 有界多阶段按共有前缀划分行，证明行内读取后按上界压缩私有 Allocate；
 * 每线程仍限 64 KiB，非前缀坐标保持串行。静态只读表可使用经过完整分支
 * 保护的有界整数索引；索引张量本身的地址须按实际形状证明。
 * 形状重排支持已证明正除数的商/余数、单例广播和分支内的静态坐标范围；
 * 符号前缀与常量尾段的拼接可按 j<P 将 [0,P+C) 分为两个安全读取域。
 * 分支范围不会传播到兄弟分支或后续访问。未证明的间接地址仍拒绝。
 * 空外层迭代域生成工作量为零的显式 guarded launch。
 * 无法证明完整初始化、阶段顺序、线程所有权或索引范围时均拒绝。
 */
CudaScheduleResult BindCudaThreads(const PrimFunc& function, const Target& target);

/*! \brief 从调度后 PrimFunc 恢复强类型启动元数据。 */
CudaLaunchConfig GetCudaLaunchConfig(const PrimFunc& function);

}  // namespace kxc::tir
