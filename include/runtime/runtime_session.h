/*! \file include/runtime/runtime_session.h
 * \brief 定义消费 CompiledModule 的强类型同步与异步执行会话。
 */

#pragma once

#include <utility>

#include "api/compiled_module.h"

namespace kxc::runtime {

/*!
 * \brief 同时返回自动分配的输出和后端完成句柄。
 *
 * outputs 持有张量元数据与 Storage；completion 额外保活本次 launch 的全部
 * 参数 Storage。尚未完成的异步后端还会由 completion 保活 executable，使
 * 调用方可以分别管理结果与完成状态。
 */
struct RunAsyncResult final {
    /*! \brief 按 KernelSignature 输出参数顺序排列的自动分配张量。 */
    Array<NDArray> outputs;
    /*! \brief 可等待或轮询的后端完成状态。 */
    AsyncOperation completion;
};

/*! \brief 不可变持有一个 ready CompiledModule 的 RuntimeSession 节点。 */
class RuntimeSessionNode final : public Object {
public:
    /*! \brief 一次性接管 ready module，节点不暴露半初始化状态。 */
    explicit RuntimeSessionNode(api::CompiledModule compiled_module)
        : module(std::move(compiled_module)) {}

    KXC_OBJECT_DECLARE

private:
    friend class RuntimeSession;
    /*! \brief 会话唯一消费的已编译模块，不包含编译或 cache 状态。 */
    api::CompiledModule module;
};

KXC_OBJECT_DEFINE_WITH_KEY(RuntimeSessionNode, "kxc.runtime.RuntimeSessionNode")

/*!
 * \brief 为 CompiledModule 自动完成输入校验、常量绑定和静态输出分配。
 *
 * 会话不编译 Relay、不缓存 module，也不进行 shape specialization。动态输入
 * 只按签名校验；动态输出在没有 shape function 的当前模型中明确拒绝。
 */
class RuntimeSession : public ObjectRef {
public:
    /*! \brief 从 ready CompiledModule 创建不可变执行会话。 */
    explicit RuntimeSession(api::CompiledModule module);
    /*! \brief 从对象系统引用恢复会话，并重新校验节点和模块状态。 */
    explicit RuntimeSession(const ObjectRef& ref);

    /*! \brief 使用模块设备的默认 stream 同步执行并返回自动分配的输出。 */
    Array<NDArray> Run(const Array<NDArray>& inputs) const;
    /*! \brief 在显式 stream 上异步执行并同时返回输出与 completion。 */
    RunAsyncResult RunAsync(const Array<NDArray>& inputs,
                            const DeviceStream& stream) const;

    /*! \brief 返回经过动态类型检查的只读节点。 */
    const RuntimeSessionNode* operator->() const;
};

}  // namespace kxc::runtime
