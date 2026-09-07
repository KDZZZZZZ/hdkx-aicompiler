#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>

#include "kxc/runtime/execution_observer.h"
#include "kxc/runtime/session.h"

namespace kxc::runtime {

/*! \brief Committed dynamic stateful length book, shared between the session
 *  node and completion handles so a pending commit survives session
 *  destruction while the completion stays alive. All access is serialized by
 *  its own mutex; run serialization is the session's state_mutex. */
struct StatefulLengthBook final {
    std::mutex mutex;
    /*! \brief Committed valid length per state value id; starts at zero. */
    std::unordered_map<int64_t, int64_t> lengths;
    /*! \brief Set when a run failed after its first kernel submit; the
     *  session must be explicitly reconstructed. */
    bool poisoned{false};
};

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
          observer(std::move(execution_observer)),
          length_book(std::make_shared<StatefulLengthBook>()) {}

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
    /*! \brief 动态有状态合同的已提交长度与失败标记；completion 句柄经由
     *  shared_ptr 保活，会话销毁后待提交仍可安全落地。 */
    std::shared_ptr<StatefulLengthBook> length_book;
    KXC_OBJECT_DECLARE
};

}  // namespace kxc::runtime
