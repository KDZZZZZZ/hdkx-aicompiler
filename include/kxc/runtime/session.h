/*! \file include/kxc/runtime/session.h
 * \brief 定义消费 CompiledModule 的同步与异步执行会话。
 */

#pragma once

#include "kxc/runtime/compiled_module.h"
#include "kxc/runtime/executable_plan.h"
#include "kxc/runtime/execution_observer.h"

namespace kxc::runtime {

struct RunAsyncResult final {
    Array<NDArray> outputs;
    AsyncOperation completion;
};

struct RequestResult final {
    uint64_t request_id{0};
    Array<NDArray> outputs;
};

class RuntimeSessionNode;

class RuntimeSession : public ObjectRef {
public:
    RuntimeSession(api::CompiledModule module, ExecutablePlan plan);
    explicit RuntimeSession(const ObjectRef& ref);

    /*! \brief Validate the executable contract without allocating session state. */
    static void Validate(const api::CompiledModule& module, const ExecutablePlan& plan);

    /*! \brief Execute caller inputs in plan order. In bounded stateful mode,
     *  omit inputs named by StateOutputBinding::input_value_id: the session
     *  supplies their committed valid prefixes. */
    Array<NDArray> Run(const Array<NDArray>& inputs) const;
    /*! \brief 带调用方关联字段的同步运行入口。
     *  metadata 只用于运行观测，不参与输入校验、kernel ABI 或产物 identity。 */
    Array<NDArray> Run(const Array<NDArray>& inputs,
                       const ExecutionMetadata& metadata) const;
    /*! \brief Submit through stream. Static or bounded external state on
     *  CPU/LLVM or CUDA completes its kernels and state copies before returning; the
     *  committed extent is already visible and the completion is ready. */
    RunAsyncResult RunAsync(const Array<NDArray>& inputs,
                            const DeviceStream& stream) const;
    /*! \brief 带调用方关联字段的异步运行入口。 */
    RunAsyncResult RunAsync(const Array<NDArray>& inputs,
                            const DeviceStream& stream,
                            const ExecutionMetadata& metadata) const;

    /*! \brief Execute an explicitly supplied module with this session's plan
     *  and state. The signature, launch, invocation and storage contracts must
     *  equal the original module's contracts, checked before allocation/launch.
     *  The caller establishes semantic equivalence; this API neither selects
     *  nor compiles a replacement. The original module remains the default.
     *  Observation and completion retention use the supplied module. */
    RunAsyncResult RunAsyncWithModule(const api::CompiledModule& module,
        const Array<NDArray>& inputs, const DeviceStream& stream,
        const ExecutionMetadata& metadata = {}) const;

    /*! \brief Committed valid length (tokens) of one session-owned state
     *  value in the dynamic stateful mode; rejected for unknown ids and for
     *  modes without state extent metadata. The value changes only after a
     *  successful run completion commits it. */
    int64_t StateExtent(int64_t state_value_id) const;
    /*! \brief Diagnostic handle to one session-owned state buffer; treat as
     *  read-only. The returned NDArray shares the session's storage. */
    NDArray StateValue(int64_t state_value_id) const;

    /*! \brief Seed a session-owned capacity state without changing its
     *  invalid-region fill.  The source may contain only the valid prefix.
     *  Only [0, valid_extent) is copied;
     *  subsequent decode runs append into the next slot. */
    void InitializeState(int64_t state_value_id, const NDArray& contents,
                         int64_t valid_extent) const;

    /*! \brief Admit one request into a free session-owned slot. Initial states
     *  follow plan.state_value_ids() order, carry batch size 1 and a shared
     *  valid extent; empty states are allowed only at extent zero. Returned
     *  nonzero ids are session-local and never reused. Initial device tensors
     *  must be ready for synchronous copying before this call. */
    uint64_t AdmitRequest(const Array<NDArray>& initial_states = {},
                          int64_t valid_extent = 0) const;
    /*! \brief Snapshot one step's caller inputs (each batch size 1). At most
     *  one step may be queued per request; reject invalid inputs before queue
     *  mutation. Input device tensors must be ready for synchronous copying.
     *  All batching methods reject concurrent/reentrant operations. */
    void EnqueueRequest(uint64_t request_id, const Array<NDArray>& inputs) const;
    /*! \brief Execute the oldest queued request with compatible queued peers,
     *  up to max_batch_size. Compatibility means equal committed extent and
     *  equal non-batch input shapes. Empty queue returns an empty vector.
     *  CPU/LLVM and CUDA are synchronous, including state commit. No compilation or
     *  padding occurs. Returned output rows own their storage independently of
     *  request departure and slot reuse. */
    std::vector<RequestResult> RunNextBatch(const ExecutionMetadata& metadata = {}) const;
    /*! \brief Execute on an explicit same-device stream. Validate the stream
     *  before packing inputs, even when the queue is empty. The metadata-only
     *  overload uses the device's default stream. */
    std::vector<RequestResult> RunNextBatch(const DeviceStream& stream,
                                           const ExecutionMetadata& metadata) const;
    /*! \brief Execute one batch with an explicitly supplied compatible module.
     *  Check the same replacement contract as RunAsyncWithModule before packing
     *  inputs. Queue, slots and committed state remain in this session; the
     *  supplied module owns this batch's execution observation. */
    std::vector<RequestResult> RunNextBatchWithModule(const api::CompiledModule& module,
        const DeviceStream& stream, const ExecutionMetadata& metadata = {}) const;
    /*! \brief Cancel any queued step and release the slot. Old ids fail closed. */
    void ReleaseRequest(uint64_t request_id) const;
    int64_t RequestExtent(uint64_t request_id) const;
    /*! \brief Copy the committed prefix, so diagnostics cannot mutate a slot
     *  or retain access after another request acquires it. */
    NDArray CopyRequestState(uint64_t request_id, int64_t state_value_id) const;

private:
    std::vector<RequestResult> RunNextBatchImpl(const api::CompiledModule& module,
        const DeviceStream& stream, const ExecutionMetadata& metadata) const;
    RunAsyncResult RunAsyncImpl(const api::CompiledModule& module,
        const Array<NDArray>& inputs,
        const DeviceStream& stream, const ExecutionMetadata& metadata,
        const std::vector<size_t>* request_slots) const;
    const RuntimeSessionNode* operator->() const;
};

}  // namespace kxc::runtime
