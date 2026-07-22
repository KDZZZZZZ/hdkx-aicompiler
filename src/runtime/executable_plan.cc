/*! \file src/runtime/executable_plan.cc
 * \brief Implements the runtime-neutral ordered kernel execution contract.
 */

#include "kxc/runtime/executable_plan.h"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "internal/executable_plan_validation.h"
#include "kxc/support/object_registration.h"

namespace kxc::runtime {

KXC_OBJECT_DEFINE_WITH_KEY(ValueSpecNode, "kxc.runtime.ValueSpecNode")
KXC_OBJECT_DEFINE_WITH_KEY(KernelCallNode, "kxc.runtime.KernelCallNode")
KXC_OBJECT_DEFINE_WITH_KEY(ExecutablePlanNode, "kxc.runtime.ExecutablePlanNode")

namespace {

bool IsSupportedDType(DLDataType dtype) {
    switch (dtype.code) {
        case kDLInt:
        case kDLUInt:
        case kDLFloat:
        case kDLBfloat:
        case kDLComplex:
        case kDLBool:
            return true;
        default:
            return false;
    }
}

bool IsSourceValue(const ValueSpec& value) {
    return value->is_input || value->is_constant;
}

bool SameStorageContract(const ValueSpec& lhs, const ValueSpec& rhs) {
    if (lhs->dtype.code != rhs->dtype.code ||
        lhs->dtype.bits != rhs->dtype.bits ||
        lhs->dtype.lanes != rhs->dtype.lanes || lhs->device != rhs->device) {
        return false;
    }
    const Array<int64_t> lhs_shape = lhs.shape();
    const Array<int64_t> rhs_shape = rhs.shape();
    if (lhs_shape.size() != rhs_shape.size()) return false;
    for (size_t i = 0; i < lhs_shape.size(); ++i) {
        if (lhs_shape[i] != rhs_shape[i]) return false;
    }
    return true;
}

bool CanShareStorage(const ValueSpec& value) {
    return !IsSourceValue(value) && !value->is_output && !value->is_alias &&
           !value->is_async_live;
}

template <typename T>
Array<T> CopyArray(const Array<T>& source) {
    Array<T> result;
    for (const auto& value : source) result.push_back(value);
    return result;
}

void ValidateOrderedRoleList(
    const Array<int64_t>& ids, bool ValueSpecNode::*expected_role,
    const char* list_name,
    const std::unordered_map<int64_t, ValueSpec>& values_by_id) {
    std::unordered_set<int64_t> seen;
    for (int64_t id : ids) {
        const auto it = values_by_id.find(id);
        if (it == values_by_id.end()) {
            throw std::invalid_argument(std::string(list_name) +
                                        " references an unknown value id");
        }
        if (!(it->second.operator->()->*expected_role)) {
            throw std::invalid_argument(std::string(list_name) +
                                        " contains a value with the wrong kind");
        }
        if (!seen.insert(id).second) {
            throw std::invalid_argument(std::string(list_name) +
                                        " contains a duplicate value id");
        }
    }

    for (const auto& entry : values_by_id) {
        if ((entry.second.operator->()->*expected_role) &&
            seen.count(entry.first) == 0) {
            throw std::invalid_argument(std::string(list_name) +
                                        " omits a value with the required kind");
        }
    }
}

}  // namespace

ValueSpec::ValueSpec(int64_t value_id, int64_t storage_id, Array<int64_t> shape,
                     DLDataType dtype, Device device, bool is_input,
                     bool is_constant, bool is_output, bool is_alias,
                     bool is_async_live) {
    auto* node = new ValueSpecNode();
    node->value_id = value_id;
    node->storage_id = storage_id;
    node->dtype = dtype;
    node->shape_ = CopyArray(shape);
    node->shape_defined_ = true;
    node->device = std::move(device);
    node->is_input = is_input;
    node->is_constant = is_constant;
    node->is_output = is_output;
    node->is_alias = is_alias;
    node->is_async_live = is_async_live;
    SetData(node);
    Validate();
}

ValueSpec::ValueSpec(const ObjectRef& ref) : ObjectRef(ref) {
    if (defined() && !As<ValueSpecNode>()) {
        SetData(nullptr);
        throw std::invalid_argument("ObjectRef does not contain ValueSpecNode");
    }
    if (defined()) Validate();
}

Array<int64_t> ValueSpec::shape() const {
    return CopyArray(operator->()->shape_);
}

void ValueSpec::Validate() const {
    const auto* node = operator->();
    if (node->value_id < 0) {
        throw std::invalid_argument("ValueSpec value_id must be non-negative");
    }
    if (node->storage_id < 0) {
        throw std::invalid_argument("ValueSpec storage_id must be non-negative");
    }
    if (node->is_input && node->is_constant) {
        throw std::invalid_argument(
            "ValueSpec cannot be both an input and a constant");
    }
    if (!node->shape_defined_) {
        throw std::invalid_argument("ValueSpec shape metadata must be defined");
    }
    if (!node->device.defined()) {
        throw std::invalid_argument("ValueSpec device metadata must be defined");
    }
    if (!IsSupportedDType(node->dtype) || node->dtype.bits == 0 ||
        node->dtype.bits % 8 != 0 || node->dtype.lanes == 0) {
        throw std::invalid_argument("ValueSpec dtype metadata is invalid");
    }
    for (int64_t dimension : node->shape_) {
        constexpr int64_t kDynamicDimension = -1;
        if (dimension < kDynamicDimension) {
            throw std::invalid_argument("ValueSpec shape contains an invalid dimension");
        }
        if (dimension == kDynamicDimension && !node->is_input) {
            throw std::invalid_argument(
                "Only input ValueSpecs may contain dynamic dimensions");
        }
    }

}

const ValueSpecNode* ValueSpec::operator->() const {
    const auto* node = As<ValueSpecNode>();
    if (!node) throw std::runtime_error("undefined or invalid ValueSpec");
    return node;
}

KernelCall::KernelCall(String symbol, Array<int64_t> input_value_ids,
                       Array<int64_t> output_value_ids) {
    auto* node = new KernelCallNode();
    node->symbol = std::move(symbol);
    node->input_value_ids_ = CopyArray(input_value_ids);
    node->output_value_ids_ = CopyArray(output_value_ids);
    SetData(node);
    Validate();
}

KernelCall::KernelCall(const ObjectRef& ref) : ObjectRef(ref) {
    if (defined() && !As<KernelCallNode>()) {
        SetData(nullptr);
        throw std::invalid_argument("ObjectRef does not contain KernelCallNode");
    }
    if (defined()) Validate();
}

Array<int64_t> KernelCall::input_value_ids() const {
    return CopyArray(operator->()->input_value_ids_);
}

Array<int64_t> KernelCall::output_value_ids() const {
    return CopyArray(operator->()->output_value_ids_);
}

void KernelCall::Validate() const {
    const auto* node = operator->();
    if (std::string(node->symbol).empty()) {
        throw std::invalid_argument("KernelCall symbol must not be empty");
    }
    if (node->output_value_ids_.empty()) {
        throw std::invalid_argument("KernelCall requires at least one output");
    }
    for (int64_t id : node->input_value_ids_) {
        if (id < 0) {
            throw std::invalid_argument("KernelCall input value ids must be non-negative");
        }
    }
    std::unordered_set<int64_t> output_ids;
    for (int64_t id : node->output_value_ids_) {
        if (id < 0 || !output_ids.insert(id).second) {
            throw std::invalid_argument(
                "KernelCall output value ids must be non-negative and unique");
        }
    }
}

const KernelCallNode* KernelCall::operator->() const {
    const auto* node = As<KernelCallNode>();
    if (!node) throw std::runtime_error("undefined or invalid KernelCall");
    return node;
}

ExecutablePlan::ExecutablePlan(Array<ValueSpec> values, Array<KernelCall> calls,
                               Array<int64_t> input_value_ids,
                               Array<int64_t> constant_value_ids,
                               Array<int64_t> output_value_ids) {
    auto* node = new ExecutablePlanNode();
    node->values_ = CopyArray(values);
    node->calls_ = CopyArray(calls);
    node->input_value_ids_ = CopyArray(input_value_ids);
    node->constant_value_ids_ = CopyArray(constant_value_ids);
    node->output_value_ids_ = CopyArray(output_value_ids);
    SetData(node);
    Validate();
}

ExecutablePlan::ExecutablePlan(const ObjectRef& ref) : ObjectRef(ref) {
    if (defined() && !As<ExecutablePlanNode>()) {
        SetData(nullptr);
        throw std::invalid_argument("ObjectRef does not contain ExecutablePlanNode");
    }
    if (defined()) Validate();
}

Array<ValueSpec> ExecutablePlan::values() const {
    return CopyArray(operator->()->values_);
}

Array<KernelCall> ExecutablePlan::calls() const {
    return CopyArray(operator->()->calls_);
}

Array<int64_t> ExecutablePlan::input_value_ids() const {
    return CopyArray(operator->()->input_value_ids_);
}

Array<int64_t> ExecutablePlan::constant_value_ids() const {
    return CopyArray(operator->()->constant_value_ids_);
}

Array<int64_t> ExecutablePlan::output_value_ids() const {
    return CopyArray(operator->()->output_value_ids_);
}

void ExecutablePlan::Validate() const {
    internal::ValidateExecutablePlan(*this);
}

const ExecutablePlanNode* ExecutablePlan::operator->() const {
    const auto* node = As<ExecutablePlanNode>();
    if (!node) throw std::runtime_error("undefined or invalid ExecutablePlan");
    return node;
}

namespace internal {

void ValidateExecutablePlan(const ExecutablePlan& plan) {
    const Array<ValueSpec> values = plan.values();
    const Array<KernelCall> calls = plan.calls();
    if (values.empty()) {
        throw std::invalid_argument("ExecutablePlan requires value metadata");
    }
    if (plan.output_value_ids().empty()) {
        throw std::invalid_argument("ExecutablePlan requires at least one graph output");
    }

    std::unordered_map<int64_t, ValueSpec> values_by_id;
    for (const auto& value : values) {
        value.Validate();
        if (!values_by_id.emplace(value->value_id, value).second) {
            throw std::invalid_argument("ExecutablePlan value ids must be unique");
        }
    }

    const Array<int64_t> input_ids = plan.input_value_ids();
    const Array<int64_t> constant_ids = plan.constant_value_ids();
    const Array<int64_t> output_ids = plan.output_value_ids();
    ValidateOrderedRoleList(input_ids, &ValueSpecNode::is_input, "input_value_ids",
                            values_by_id);
    ValidateOrderedRoleList(constant_ids, &ValueSpecNode::is_constant,
                            "constant_value_ids", values_by_id);
    ValidateOrderedRoleList(output_ids, &ValueSpecNode::is_output,
                            "output_value_ids",
                            values_by_id);

    std::unordered_set<int64_t> available;
    for (int64_t id : input_ids) available.insert(id);
    for (int64_t id : constant_ids) available.insert(id);

    std::unordered_map<int64_t, int> producer_counts;
    std::unordered_map<int64_t, int64_t> producer_index;
    std::unordered_map<int64_t, int64_t> last_use;
    std::unordered_set<std::string> symbols;
    for (size_t call_index = 0; call_index < calls.size(); ++call_index) {
        const auto& call = calls[call_index];
        call.Validate();
        if (!symbols.insert(std::string(call->symbol)).second) {
            throw std::invalid_argument("ExecutablePlan KernelCall symbols must be unique");
        }
        for (int64_t input_id : call.input_value_ids()) {
            last_use[input_id] = static_cast<int64_t>(call_index);
            if (values_by_id.count(input_id) == 0) {
                throw std::invalid_argument(
                    "KernelCall references an undefined input value id");
            }
            if (available.count(input_id) == 0) {
                throw std::invalid_argument(
                    "KernelCall input is not available before the call");
            }
        }
        for (int64_t output_id : call.output_value_ids()) {
            const auto value_it = values_by_id.find(output_id);
            if (value_it == values_by_id.end()) {
                throw std::invalid_argument(
                    "KernelCall references an undefined output value id");
            }
            if (IsSourceValue(value_it->second)) {
                throw std::invalid_argument("KernelCall cannot produce a source value");
            }
            if (++producer_counts[output_id] != 1) {
                throw std::invalid_argument(
                    "ExecutablePlan value has more than one producer");
            }
            producer_index[output_id] = static_cast<int64_t>(call_index);
            last_use.emplace(output_id, static_cast<int64_t>(call_index));
        }
        for (int64_t output_id : call.output_value_ids()) available.insert(output_id);
    }

    for (const auto& entry : values_by_id) {
        if (!IsSourceValue(entry.second) && producer_counts[entry.first] != 1) {
            throw std::invalid_argument(
                "Every non-source ExecutablePlan value requires exactly one producer");
        }
    }
    for (int64_t output_id : output_ids) {
        if (available.count(output_id) == 0) {
            throw std::invalid_argument("ExecutablePlan graph output is unavailable");
        }
    }

    std::unordered_map<int64_t, std::vector<ValueSpec>> values_by_storage;
    for (const auto& value : values) {
        values_by_storage[value->storage_id].push_back(value);
    }
    for (const auto& storage : values_by_storage) {
        const auto& shared = storage.second;
        for (size_t i = 0; i < shared.size(); ++i) {
            for (size_t j = i + 1; j < shared.size(); ++j) {
                if (!CanShareStorage(shared[i]) ||
                    !CanShareStorage(shared[j]) ||
                    !SameStorageContract(shared[i], shared[j])) {
                    throw std::invalid_argument(
                        "ExecutablePlan storage sharing violates a value contract");
                }
                const int64_t first_begin =
                    producer_index.at(shared[i]->value_id);
                const int64_t first_end = last_use.at(shared[i]->value_id);
                const int64_t second_begin =
                    producer_index.at(shared[j]->value_id);
                const int64_t second_end = last_use.at(shared[j]->value_id);
                if (!(first_end < second_begin || second_end < first_begin)) {
                    throw std::invalid_argument(
                        "ExecutablePlan storage lifetimes overlap");
                }
            }
        }
    }
}

}  // namespace internal
}  // namespace kxc::runtime
