/*! \file include/kxc/runtime/session.h
 * \brief 定义消费 CompiledModule 的同步与异步执行会话。
 */

#pragma once

#include "kxc/runtime/compiled_module.h"
#include "kxc/runtime/executable_plan.h"
#include "kxc/runtime/task_executor.h"
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

struct TaskDAGSelectionResult final {
    RuntimeExecutionMode requested_mode{RuntimeExecutionMode::kPerCall};
    RuntimeExecutionMode selected_mode{RuntimeExecutionMode::kPerCall};
    FallbackReason fallback_reason{FallbackReason::kNone};
    String diagnostic;
};

class RuntimeSessionNode;

class RuntimeSession : public ObjectRef {
public:
    RuntimeSession(api::CompiledModule module, ExecutablePlan plan);
    RuntimeSession(api::CompiledModule module, ExecutablePlan plan,
                   RuntimeExecutionMode mode);
    RuntimeSession(api::CompiledModule module, ExecutablePlan plan,
                   RuntimeExecutionMode mode, RuntimeObserver observer);
    RuntimeSession(api::CompiledModule module, PlanVariant variant);
    RuntimeSession(api::CompiledModule module, PlanVariant variant,
                   RuntimeExecutionMode mode,
                   RuntimeObserver observer = {});
    RuntimeSession(api::CompiledModule module, FrozenTaskPlan plan);
    RuntimeSession(api::CompiledModule module, FrozenTaskPlan plan,
                   RuntimeObserver observer);
    explicit RuntimeSession(const ObjectRef& ref);

    Array<NDArray> Run(const Array<NDArray>& inputs) const;
    RunAsyncResult RunAsync(const Array<NDArray>& inputs,
                            const DeviceStream& stream) const;
    bool UsesTaskDAG() const;
    TaskDAGSelectionResult TaskDAGSelection() const;
    /*! \brief Returns the accepted declaration; it is not provenance proof. */
    SelectedArtifactManifest artifact_manifest() const;

private:
    const RuntimeSessionNode* operator->() const;
};

}  // namespace kxc::runtime
