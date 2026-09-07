#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>

#include "kxc/runtime/execution_observer.h"
#include "kxc/runtime/session.h"

namespace kxc::runtime {

class RuntimeSessionNode final : public Object {
public:
    RuntimeSessionNode(
        api::CompiledModule compiled_module, ExecutablePlan executable_plan,
        Device execution_device, std::unordered_map<int64_t, String> constant_keys,
        std::unordered_map<int64_t, size_t> storage_alignments,
        std::unordered_map<int64_t, NDArray> initial_states,
        std::shared_ptr<ExecutionObserver> execution_observer = nullptr)
        : module(std::move(compiled_module)),
          plan(std::move(executable_plan)),
          device(std::move(execution_device)),
          constant_keys_by_value(std::move(constant_keys)),
          required_alignment_by_storage(std::move(storage_alignments)),
          states_by_value(std::move(initial_states)),
          observer(std::move(execution_observer)) {}

    api::CompiledModule module;
    ExecutablePlan plan;
    Device device;
    std::unordered_map<int64_t, String> constant_keys_by_value;
    std::unordered_map<int64_t, size_t> required_alignment_by_storage;
    std::unordered_map<int64_t, NDArray> states_by_value;
    /*! \brief 从模块继承的执行观测器；模块未启用观测时为空，
     *  Run/RunAsync 行为与未装配时完全一致。 */
    std::shared_ptr<ExecutionObserver> observer;
    mutable std::mutex state_mutex;
    mutable AsyncOperation state_completion;
    KXC_OBJECT_DECLARE
};

}  // namespace kxc::runtime
