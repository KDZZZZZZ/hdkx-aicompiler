/*! \file include/kxc/runtime/session.h
 * \brief 定义消费 CompiledModule 的同步与异步执行会话。
 */

#pragma once

#include "kxc/runtime/compiled_module.h"
#include "kxc/runtime/executable_plan.h"
#include "kxc/runtime/task_plan.h"

namespace kxc::runtime {

enum class RuntimeExecutionMode : int32_t {
    kPerCall = 0,
    kTaskDAG = 1,
};

struct RunAsyncResult final {
    Array<NDArray> outputs;
    AsyncOperation completion;
};

class RuntimeSessionNode;

class RuntimeSession : public ObjectRef {
public:
    RuntimeSession(api::CompiledModule module, ExecutablePlan plan);
    RuntimeSession(api::CompiledModule module, ExecutablePlan plan,
                   RuntimeExecutionMode mode);
    RuntimeSession(api::CompiledModule module, FrozenTaskPlan plan);
    explicit RuntimeSession(const ObjectRef& ref);

    Array<NDArray> Run(const Array<NDArray>& inputs) const;
    RunAsyncResult RunAsync(const Array<NDArray>& inputs,
                            const DeviceStream& stream) const;
    bool UsesTaskDAG() const;

private:
    const RuntimeSessionNode* operator->() const;
};

}  // namespace kxc::runtime
