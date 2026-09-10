/*! \file include/kxc/runtime/executable_plan.h
 * \brief Defines the runtime-neutral ordered kernel execution contract.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <dlpack/dlpack.h>

#include "kxc/runtime/device.h"
#include "kxc/support/container.h"

namespace kxc::runtime {

/*! \brief Defines whether a produced value owns storage or writes its alias source. */
enum class ValueWriteMode : uint8_t {
    kAllocate = 0,
    kInPlace = 1,
};

/*! \brief Selects the immutable graph execution and allocation contract. */
enum class ExecutablePlanMode : uint8_t {
    kStatic = 0,
    kDynamicFreshOutputV1 = 1,
    /*! \brief Session-owned persistent state with a dynamic valid extent.
     *
     *  Independently versioned dynamic stateful contract (M2 v1): the session
     *  owns the cache storage and the committed valid length, appends are
     *  declared in-place through alias outputs, and kernels read the
     *  state-sourced runtime extents injected by the session. This mode never
     *  relaxes the fresh-output rejection contract. */
    kDynamicStatefulV1 = 2,
    /*! \brief Session-owned capacity state whose append is copied from a
     *  statically-shaped producer into the next valid slot.
     *
     *  This mode is the bridge for capacity-padded Transformer decode graphs:
     *  the compiled graph keeps a fixed past/present shape, while the session
     *  owns the past buffers and valid cursor.  The cursor is runtime data and
     *  is never part of the kernel ABI. CPU/LLVM and CUDA finish the state
     *  copy before RunAsync returns; valid extent commits only after it. */
    kStaticStatefulExternalV1 = 3,
    /*! \brief Bounded kernels read compact prefixes of session-owned capacity
     *  state. CPU/LLVM and CUDA pack valid prefixes into reusable session
     *  storage before launch, then finish kernels and copy each produced
     *  append before RunAsync returns and the extent is committed. */
    kBoundedStatefulExternalV1 = 4,
};

/*! \brief One produced tensor segment appended to session-owned state.
 *  Static mode reads source_slot == capacity from a capacity-padded source.
 *  Bounded mode reads the committed cursor from a valid-prefix source whose
 *  length is cursor + append_count. Both commit extent only after copying. */
struct StateOutputBinding final {
    int64_t state_value_id{-1};
    int64_t source_value_id{-1};
    int64_t source_extent_axis{-1};
    int64_t source_slot{-1};
    int64_t append_count{1};
    /*! \brief Bounded mode's logical graph input, supplied by the session.
     *  -1 in the static mode. Bounded source_slot is -1 and means the
     *  committed extent; input_value_id is assigned by BindBoundedStateOutputs. */
    int64_t input_value_id{-1};
};

/*! \brief Explicit serving ABI for independent requests along leading axis 0.
 *  All graph inputs/outputs carry that axis; non-batch shapes and committed
 *  state extents must agree within a batch. Physical state axis 0 is the
 *  finite request-slot capacity. The model producer asserts row independence;
 *  shape guards alone do not prove it. CPU/LLVM or CUDA bounded state v1. */
struct RequestBatchingContract final {
    int64_t max_batch_size{1};
};

/*! \brief One graph-input axis referenced by a shared-shape guard. */
struct GraphInputAxisReference final {
    size_t input_index{0};
    size_t axis{0};
};

/*! \brief Finite preflight contract for one wildcard graph-input axis. */
struct GraphInputAxisGuard final {
    size_t input_index{0};
    size_t axis{0};
    int64_t lower{0};
    int64_t upper{0};
    int64_t divisible_by{1};
    std::optional<GraphInputAxisReference> equal_to;
};

/*! \brief Complete runtime metadata for one stable graph value. */
class ValueSpecNode final : public Object {
public:
    int64_t value_id{-1};
    int64_t storage_id{-1};
    DLDataType dtype{};
    Device device;
    bool is_input{false};
    bool is_constant{false};
    bool is_output{false};
    bool is_alias{false};
    bool is_async_live{false};
    bool is_state{false};
    int64_t alias_source_value_id{-1};
    ValueWriteMode write_mode{ValueWriteMode::kAllocate};
    int64_t valid_bytes{-1};
    /*! \brief Declared physical capacity along the extent axis; -1 when the
     *  value carries no dynamic state-extent contract. Must equal
     *  shape[state_extent_axis] when declared. */
    int64_t state_capacity{-1};
    /*! \brief Axis whose runtime extent is the valid length; -1 when unset. */
    int64_t state_extent_axis{-1};
    /*! \brief Fill value applied to the invalid capacity region at session
     *  state construction; must be finite. Non-state values keep 0.0. */
    double state_fill{0.0};

    KXC_OBJECT_DECLARE

private:
    friend class ValueSpec;
    Array<int64_t> shape_;
    bool shape_defined_{false};
};

/*! \brief Type-safe immutable handle for a graph value contract. */
class ValueSpec : public ObjectRef {
public:
    ValueSpec(int64_t value_id, int64_t storage_id, Array<int64_t> shape,
              DLDataType dtype, Device device, bool is_input = false,
              bool is_constant = false, bool is_output = false,
              bool is_alias = false, bool is_async_live = false,
              bool is_state = false, int64_t alias_source_value_id = -1,
              ValueWriteMode write_mode = ValueWriteMode::kAllocate,
              int64_t valid_bytes = -1, int64_t state_capacity = -1,
              int64_t state_extent_axis = -1, double state_fill = 0.0);
    explicit ValueSpec(const ObjectRef& ref);

    Array<int64_t> shape() const;
    void Validate() const;
    const ValueSpecNode* operator->() const;
};

/*! \brief One ordered kernel invocation over stable graph value ids. */
class KernelCallNode final : public Object {
public:
    String symbol;

    KXC_OBJECT_DECLARE

private:
    friend class KernelCall;
    Array<int64_t> input_value_ids_;
    Array<int64_t> output_value_ids_;
};

/*! \brief Type-safe immutable handle for an ordered kernel invocation. */
class KernelCall : public ObjectRef {
public:
    KernelCall(String symbol, Array<int64_t> input_value_ids,
               Array<int64_t> output_value_ids);
    explicit KernelCall(const ObjectRef& ref);

    Array<int64_t> input_value_ids() const;
    Array<int64_t> output_value_ids() const;
    void Validate() const;
    const KernelCallNode* operator->() const;
};

/*! \brief Ordered runtime-neutral execution plan for a compiled graph. */
class ExecutablePlanNode final : public Object {
public:
    KXC_OBJECT_DECLARE

private:
    friend class ExecutablePlan;
    Array<ValueSpec> values_;
    Array<KernelCall> calls_;
    Array<int64_t> input_value_ids_;
    Array<int64_t> constant_value_ids_;
    Array<int64_t> output_value_ids_;
    Array<int64_t> state_value_ids_;
    ExecutablePlanMode mode_{ExecutablePlanMode::kStatic};
    std::vector<GraphInputAxisGuard> graph_input_guards_;
    /*! \brief Per call (in call order), the state value ids whose committed
     *  lengths feed that call's runtime-extent arguments in ABI order. */
    std::vector<std::vector<int64_t>> state_extent_bindings_;
    /*! \brief Graph input value id carrying the uint64[1] append count;
     *  required (>= 0) in the dynamic stateful mode. */
    int64_t state_count_input_value_id_{-1};
    std::vector<StateOutputBinding> state_output_bindings_;
    std::optional<RequestBatchingContract> request_batching_;
};

/*! \brief Validated immutable graph ABI and kernel call order. */
class ExecutablePlan : public ObjectRef {
public:
    ExecutablePlan() = default;
    ExecutablePlan(Array<ValueSpec> values, Array<KernelCall> calls,
                   Array<int64_t> input_value_ids,
                   Array<int64_t> constant_value_ids,
                   Array<int64_t> output_value_ids,
                   Array<int64_t> state_value_ids = {},
                   ExecutablePlanMode mode = ExecutablePlanMode::kStatic,
                   std::vector<GraphInputAxisGuard> graph_input_guards = {},
                   std::vector<std::vector<int64_t>> state_extent_bindings = {},
                   int64_t state_count_input_value_id = -1,
                   std::vector<StateOutputBinding> state_output_bindings = {},
                   std::optional<RequestBatchingContract> request_batching = std::nullopt);
    explicit ExecutablePlan(const ObjectRef& ref);

    Array<ValueSpec> values() const;
    Array<KernelCall> calls() const;
    Array<int64_t> input_value_ids() const;
    Array<int64_t> constant_value_ids() const;
    Array<int64_t> output_value_ids() const;
    Array<int64_t> state_value_ids() const;
    ExecutablePlanMode mode() const;
    std::vector<GraphInputAxisGuard> graph_input_guards() const;
    std::vector<std::vector<int64_t>> state_extent_bindings() const;
    int64_t state_count_input_value_id() const;
    std::vector<StateOutputBinding> state_output_bindings() const;
    std::optional<RequestBatchingContract> request_batching() const;
    /*! \brief Declare the leading-axis independent-request ABI on an already
     *  bound bounded-state plan. Physical slot capacity may exceed max_batch_size
     *  but must fit the compiled batch bounds. No compilation occurs here. */
    ExecutablePlan BindRequestBatching(int64_t max_batch_size) const;
    /*! \brief Attach session-owned capacity state to a compiled static plan.
     *  Selected graph inputs become state sources; selected graph outputs
     *  become private append sources retained through the state commit. */
    ExecutablePlan BindStateOutputs(std::vector<StateOutputBinding> bindings,
                                    double state_fill = 0.0) const;
    /*! \brief Attach fixed-capacity state to a bounded fresh-output plan.
     *  Each binding names an original graph input as state_value_id and its
     *  present output as source_value_id; source_slot and input_value_id must
     *  be -1. Physical shapes follow binding order and fix all dimensions,
     *  including batch. The returned plan retains logical graph input order;
     *  RuntimeSession supplies the bound prefixes and accepts only the other
     *  inputs. Kernel contracts and graph guards remain unchanged. */
    ExecutablePlan BindBoundedStateOutputs(
        std::vector<StateOutputBinding> bindings,
        std::vector<Array<int64_t>> physical_shapes,
        double state_fill = 0.0) const;
    void Validate() const;
    const ExecutablePlanNode* operator->() const;
};

}  // namespace kxc::runtime
