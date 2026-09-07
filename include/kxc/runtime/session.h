/*! \file include/kxc/runtime/session.h
 * \brief 定义消费 CompiledModule 的同步与异步执行会话。
 */

#pragma once

#include "kxc/runtime/compiled_module.h"
#include "kxc/runtime/executable_plan.h"

namespace kxc::runtime {

struct RunAsyncResult final {
    Array<NDArray> outputs;
    AsyncOperation completion;
};

class RuntimeSessionNode;

class RuntimeSession : public ObjectRef {
public:
    RuntimeSession(api::CompiledModule module, ExecutablePlan plan);
    explicit RuntimeSession(const ObjectRef& ref);

    Array<NDArray> Run(const Array<NDArray>& inputs) const;
    RunAsyncResult RunAsync(const Array<NDArray>& inputs,
                            const DeviceStream& stream) const;

    /*! \brief Committed valid length (tokens) of one session-owned state
     *  value in the dynamic stateful mode; rejected for unknown ids and for
     *  modes without state extent metadata. The value changes only after a
     *  successful run completion commits it. */
    int64_t StateExtent(int64_t state_value_id) const;
    /*! \brief Diagnostic handle to one session-owned state buffer; treat as
     *  read-only. The returned NDArray shares the session's storage. */
    NDArray StateValue(int64_t state_value_id) const;

private:
    const RuntimeSessionNode* operator->() const;
};

}  // namespace kxc::runtime
