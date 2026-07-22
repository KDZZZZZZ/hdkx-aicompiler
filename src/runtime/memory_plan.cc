/*! \file src/runtime/memory_plan.cc
 * \brief Last-use storage-slot planning for sequential single-stream execution.
 */

#include "internal/memory_plan.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace kxc::runtime::internal {
namespace {

bool SameShape(const Array<int64_t>& lhs, const Array<int64_t>& rhs) {
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i] != rhs[i]) return false;
    }
    return true;
}

bool SameContract(const ValueSpec& lhs, const ValueSpec& rhs) {
    return lhs->dtype.code == rhs->dtype.code &&
           lhs->dtype.bits == rhs->dtype.bits &&
           lhs->dtype.lanes == rhs->dtype.lanes &&
           lhs->device == rhs->device && SameShape(lhs.shape(), rhs.shape());
}

bool Reusable(const ValueSpec& value) {
    return !value->is_input && !value->is_constant && !value->is_output &&
           !value->is_alias && !value->is_async_live;
}

struct Slot final {
    int64_t storage_id{-1};
    int64_t release_after{-1};
    ValueSpec contract;
};

}  // namespace

ExecutablePlan PlanMemory(const ExecutablePlan& plan) {
    plan.Validate();
    const Array<ValueSpec> values = plan.values();
    const Array<KernelCall> calls = plan.calls();
    std::unordered_map<int64_t, ValueSpec> by_id;
    std::unordered_map<int64_t, int64_t> producer;
    std::unordered_map<int64_t, int64_t> last_use;
    for (const auto& value : values) by_id.emplace(value->value_id, value);
    for (size_t call_index = 0; call_index < calls.size(); ++call_index) {
        for (int64_t input : calls[call_index].input_value_ids()) {
            last_use[input] = static_cast<int64_t>(call_index);
        }
        for (int64_t output : calls[call_index].output_value_ids()) {
            producer[output] = static_cast<int64_t>(call_index);
            last_use.emplace(output, static_cast<int64_t>(call_index));
        }
    }

    std::unordered_map<int64_t, int64_t> storage_by_value;
    for (const auto& value : values) {
        if (!Reusable(value)) storage_by_value[value->value_id] = value->value_id;
    }
    std::vector<Slot> slots;
    for (size_t call_index = 0; call_index < calls.size(); ++call_index) {
        for (int64_t value_id : calls[call_index].output_value_ids()) {
            const ValueSpec& value = by_id.at(value_id);
            if (!Reusable(value)) continue;
            Slot* selected = nullptr;
            for (auto& slot : slots) {
                if (slot.release_after < static_cast<int64_t>(call_index) &&
                    SameContract(slot.contract, value) &&
                    (!selected || slot.storage_id < selected->storage_id)) {
                    selected = &slot;
                }
            }
            if (selected) {
                storage_by_value[value_id] = selected->storage_id;
                selected->release_after = last_use.at(value_id);
                selected->contract = value;
            } else {
                storage_by_value[value_id] = value_id;
                slots.push_back(
                    Slot{value_id, last_use.at(value_id), value});
            }
        }
    }

    Array<ValueSpec> planned_values;
    for (const auto& value : values) {
        const auto storage = storage_by_value.find(value->value_id);
        if (storage == storage_by_value.end()) {
            throw std::logic_error(
                "Memory planner did not assign every graph value");
        }
        planned_values.push_back(ValueSpec(
            value->value_id, storage->second, value.shape(), value->dtype,
            value->device, value->is_input, value->is_constant,
            value->is_output, value->is_alias, value->is_async_live));
    }
    return ExecutablePlan(
        std::move(planned_values), calls, plan.input_value_ids(),
        plan.constant_value_ids(), plan.output_value_ids());
}

}  // namespace kxc::runtime::internal
