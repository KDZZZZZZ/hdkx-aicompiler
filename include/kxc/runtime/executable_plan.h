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
                   int64_t state_count_input_value_id = -1);
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
    void Validate() const;
    const ExecutablePlanNode* operator->() const;
};

}  // namespace kxc::runtime
