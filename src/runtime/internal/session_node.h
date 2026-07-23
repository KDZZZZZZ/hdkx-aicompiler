#pragma once

#include <unordered_map>
#include <utility>

#include "kxc/runtime/session.h"

namespace kxc::runtime {

class RuntimeSessionNode final : public Object {
public:
    RuntimeSessionNode(
        api::CompiledModule compiled_module, ExecutablePlan executable_plan,
        SelectedArtifactManifest selected_artifacts, Device execution_device,
        std::unordered_map<int64_t, String> constant_keys,
        TaskDAGSelectionResult task_dag_selection,
        RuntimeObserver runtime_observer)
        : module(std::move(compiled_module)),
          plan(std::move(executable_plan)),
          manifest(std::move(selected_artifacts)),
          device(std::move(execution_device)),
          constant_keys_by_value(std::move(constant_keys)),
          selection(std::move(task_dag_selection)),
          observer(std::move(runtime_observer)) {}

    RuntimeSessionNode(
        api::CompiledModule compiled_module, FrozenTaskPlan frozen_task_plan,
        Device execution_device,
        std::unordered_map<int64_t, String> constant_keys,
        TaskDAGSelectionResult task_dag_selection,
        RuntimeObserver runtime_observer)
        : module(std::move(compiled_module)),
          task_plan(std::move(frozen_task_plan)),
          manifest(task_plan.manifest()),
          device(std::move(execution_device)),
          constant_keys_by_value(std::move(constant_keys)),
          selection(std::move(task_dag_selection)),
          observer(std::move(runtime_observer)) {}

    api::CompiledModule module;
    ExecutablePlan plan;
    FrozenTaskPlan task_plan;
    SelectedArtifactManifest manifest;
    Device device;
    std::unordered_map<int64_t, String> constant_keys_by_value;
    TaskDAGSelectionResult selection;
    RuntimeObserver observer;
    KXC_OBJECT_DECLARE
};

}  // namespace kxc::runtime
