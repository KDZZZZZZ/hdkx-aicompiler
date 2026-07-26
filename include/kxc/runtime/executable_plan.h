/*! \file include/kxc/runtime/executable_plan.h
 * \brief Defines the runtime-neutral ordered kernel execution contract.
 */

#pragma once

#include <cstdint>

#include <dlpack/dlpack.h>

#include "kxc/runtime/device.h"
#include "kxc/support/container.h"

namespace kxc::runtime {

/*! \brief Defines whether a produced value owns storage or writes its alias source. */
enum class ValueWriteMode : uint8_t {
    kAllocate = 0,
    kInPlace = 1,
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
              int64_t valid_bytes = -1);
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
};

/*! \brief Validated immutable graph ABI and kernel call order. */
class ExecutablePlan : public ObjectRef {
public:
    ExecutablePlan() = default;
    ExecutablePlan(Array<ValueSpec> values, Array<KernelCall> calls,
                   Array<int64_t> input_value_ids,
                   Array<int64_t> constant_value_ids,
                   Array<int64_t> output_value_ids,
                   Array<int64_t> state_value_ids = {});
    explicit ExecutablePlan(const ObjectRef& ref);

    Array<ValueSpec> values() const;
    Array<KernelCall> calls() const;
    Array<int64_t> input_value_ids() const;
    Array<int64_t> constant_value_ids() const;
    Array<int64_t> output_value_ids() const;
    Array<int64_t> state_value_ids() const;
    void Validate() const;
    const ExecutablePlanNode* operator->() const;
};

}  // namespace kxc::runtime
