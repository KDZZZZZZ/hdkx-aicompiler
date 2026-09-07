/*! \file src/runtime/executable_plan.cc
 * \brief Implements the runtime-neutral ordered kernel execution contract.
 */

#include "kxc/runtime/executable_plan.h"

#include <cmath>
#include <limits>
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
    return value->is_input || value->is_constant || value->is_state;
}

size_t StaticNBytes(const ValueSpec& value) {
    size_t elements = 1;
    bool has_zero_dimension = false;
    for (int64_t dimension : value.shape()) {
        if (dimension < 0) {
            throw std::invalid_argument("ValueSpec valid bytes require a static shape");
        }
        has_zero_dimension = has_zero_dimension || dimension == 0;
        if (dimension != 0 &&
            elements > std::numeric_limits<size_t>::max() /
                           static_cast<size_t>(dimension)) {
            throw std::overflow_error("ValueSpec element count overflow");
        }
        if (dimension != 0) elements *= static_cast<size_t>(dimension);
    }
    if (has_zero_dimension) return 0;
    const size_t element_bytes =
        static_cast<size_t>(value->dtype.bits / 8) * value->dtype.lanes;
    if (elements > std::numeric_limits<size_t>::max() / element_bytes) {
        throw std::overflow_error("ValueSpec byte size overflow");
    }
    return elements * element_bytes;
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

namespace internal {

bool SameValueStorageContract(const ValueSpec& lhs, const ValueSpec& rhs) {
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

bool IsValueStorageReusable(const ValueSpec& value) {
    return !value->is_input && !value->is_constant && !value->is_output &&
           !value->is_alias && !value->is_async_live && !value->is_state;
}

}  // namespace internal

ValueSpec::ValueSpec(int64_t value_id, int64_t storage_id, Array<int64_t> shape,
                     DLDataType dtype, Device device, bool is_input,
                     bool is_constant, bool is_output, bool is_alias,
                     bool is_async_live, bool is_state,
                     int64_t alias_source_value_id, ValueWriteMode write_mode,
                     int64_t valid_bytes, int64_t state_capacity,
                     int64_t state_extent_axis, double state_fill) {
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
    node->is_state = is_state;
    node->alias_source_value_id = alias_source_value_id;
    node->write_mode = write_mode;
    node->valid_bytes = valid_bytes;
    node->state_capacity = state_capacity;
    node->state_extent_axis = state_extent_axis;
    node->state_fill = state_fill;
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
    if (node->is_state &&
        (node->is_input || node->is_constant || node->is_output ||
         node->is_alias)) {
        throw std::invalid_argument(
            "ValueSpec state cannot also be an input, constant, graph output, or alias");
    }
    if (node->alias_source_value_id < -1) {
        throw std::invalid_argument("ValueSpec alias source id is invalid");
    }
    if (node->alias_source_value_id == -1 &&
        node->write_mode != ValueWriteMode::kAllocate) {
        throw std::invalid_argument(
            "ValueSpec write mode requires an alias source");
    }
    if (node->alias_source_value_id != -1 &&
        (!node->is_alias || node->write_mode != ValueWriteMode::kInPlace ||
         node->is_input || node->is_constant || node->is_state)) {
        throw std::invalid_argument(
            "ValueSpec alias source requires a produced in-place alias");
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
    }
    if (node->valid_bytes < -1) {
        throw std::invalid_argument("ValueSpec valid bytes is invalid");
    }
    if (node->valid_bytes != -1 &&
        static_cast<uint64_t>(node->valid_bytes) > StaticNBytes(*this)) {
        throw std::invalid_argument("ValueSpec valid bytes exceeds tensor capacity");
    }
    if ((node->state_capacity == -1) != (node->state_extent_axis == -1)) {
        throw std::invalid_argument(
            "ValueSpec state capacity and extent axis must be declared together");
    }
    if (node->state_extent_axis != -1) {
        if (!node->is_state) {
            throw std::invalid_argument(
                "ValueSpec state extent metadata requires a state value");
        }
        if (node->state_capacity < 0 ||
            node->state_extent_axis < 0 ||
            node->state_extent_axis >= static_cast<int64_t>(node->shape_.size())) {
            throw std::invalid_argument(
                "ValueSpec state extent metadata is out of range");
        }
        if (node->shape_[node->state_extent_axis] != node->state_capacity) {
            throw std::invalid_argument(
                "ValueSpec state capacity must equal the declared extent axis size");
        }
    }
    if (!node->is_state && node->state_fill != 0.0) {
        throw std::invalid_argument(
            "ValueSpec fill metadata requires a state value");
    }
    if (node->is_state && !std::isfinite(node->state_fill)) {
        throw std::invalid_argument(
            "ValueSpec state fill must be a finite value");
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

ExecutablePlan::ExecutablePlan(
    Array<ValueSpec> values, Array<KernelCall> calls,
    Array<int64_t> input_value_ids, Array<int64_t> constant_value_ids,
    Array<int64_t> output_value_ids, Array<int64_t> state_value_ids,
    ExecutablePlanMode mode,
    std::vector<GraphInputAxisGuard> graph_input_guards,
    std::vector<std::vector<int64_t>> state_extent_bindings,
    int64_t state_count_input_value_id) {
    auto* node = new ExecutablePlanNode();
    node->values_ = CopyArray(values);
    node->calls_ = CopyArray(calls);
    node->input_value_ids_ = CopyArray(input_value_ids);
    node->constant_value_ids_ = CopyArray(constant_value_ids);
    node->output_value_ids_ = CopyArray(output_value_ids);
    node->state_value_ids_ = CopyArray(state_value_ids);
    node->mode_ = mode;
    node->graph_input_guards_ = std::move(graph_input_guards);
    node->state_extent_bindings_ = std::move(state_extent_bindings);
    node->state_count_input_value_id_ = state_count_input_value_id;
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

Array<int64_t> ExecutablePlan::state_value_ids() const {
    return CopyArray(operator->()->state_value_ids_);
}

ExecutablePlanMode ExecutablePlan::mode() const {
    return operator->()->mode_;
}

std::vector<GraphInputAxisGuard> ExecutablePlan::graph_input_guards() const {
    return operator->()->graph_input_guards_;
}

std::vector<std::vector<int64_t>> ExecutablePlan::state_extent_bindings() const {
    return operator->()->state_extent_bindings_;
}

int64_t ExecutablePlan::state_count_input_value_id() const {
    return operator->()->state_count_input_value_id_;
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
    const Array<int64_t> state_ids = plan.state_value_ids();
    ValidateOrderedRoleList(input_ids, &ValueSpecNode::is_input, "input_value_ids",
                            values_by_id);
    ValidateOrderedRoleList(constant_ids, &ValueSpecNode::is_constant,
                            "constant_value_ids", values_by_id);
    ValidateOrderedRoleList(output_ids, &ValueSpecNode::is_output,
                            "output_value_ids",
                            values_by_id);
    ValidateOrderedRoleList(state_ids, &ValueSpecNode::is_state,
                            "state_value_ids", values_by_id);

    const ExecutablePlanMode mode = plan.mode();
    const std::vector<GraphInputAxisGuard> graph_guards =
        plan.graph_input_guards();
    if (mode == ExecutablePlanMode::kStatic) {
        if (!graph_guards.empty()) {
            throw std::invalid_argument(
                "Static ExecutablePlan cannot declare graph input guards");
        }
        for (const auto& value : values) {
            for (int64_t dimension : value.shape()) {
                if (dimension == -1 && !value->is_input) {
                    throw std::invalid_argument(
                        "Static ExecutablePlan only allows wildcard input values");
                }
            }
        }
    } else if (mode == ExecutablePlanMode::kDynamicFreshOutputV1) {
        std::unordered_set<int64_t> storage_ids;
        for (const auto& value : values) {
            if (value->is_state || value->is_alias ||
                value->alias_source_value_id != -1 ||
                value->write_mode != ValueWriteMode::kAllocate) {
                throw std::invalid_argument(
                    "Dynamic fresh-output ExecutablePlan rejects state, alias, and donation");
            }
            if (value->valid_bytes != -1) {
                throw std::invalid_argument(
                    "Dynamic fresh-output ExecutablePlan requires logical valid extents");
            }
            if (!storage_ids.insert(value->storage_id).second) {
                throw std::invalid_argument(
                    "Dynamic fresh-output ExecutablePlan forbids graph storage reuse");
            }
            if (value->is_constant) {
                for (int64_t dimension : value.shape()) {
                    if (dimension == -1) {
                        throw std::invalid_argument(
                            "Dynamic fresh-output constants require static shapes");
                    }
                }
            }
        }
        if (!state_ids.empty()) {
            throw std::invalid_argument(
                "Dynamic fresh-output ExecutablePlan rejects persistent state");
        }

        std::vector<std::pair<size_t, size_t>> wildcard_axes;
        for (size_t input_index = 0; input_index < input_ids.size();
             ++input_index) {
            const Array<int64_t> shape =
                values_by_id.at(input_ids[input_index]).shape();
            for (size_t axis = 0; axis < shape.size(); ++axis) {
                if (shape[axis] == -1) {
                    wildcard_axes.emplace_back(input_index, axis);
                }
            }
        }
        if (graph_guards.size() != wildcard_axes.size()) {
            throw std::invalid_argument(
                "Dynamic fresh-output graph guards must cover every wildcard input axis");
        }
        for (size_t index = 0; index < graph_guards.size(); ++index) {
            const GraphInputAxisGuard& guard = graph_guards[index];
            if (guard.input_index != wildcard_axes[index].first ||
                guard.axis != wildcard_axes[index].second) {
                throw std::invalid_argument(
                    "Dynamic fresh-output graph guards must follow input-axis order");
            }
            if (guard.lower < 0 || guard.upper < guard.lower ||
                guard.divisible_by <= 0) {
                throw std::invalid_argument(
                    "Dynamic fresh-output graph guard bounds are invalid");
            }
            const int64_t remainder = guard.lower % guard.divisible_by;
            if (remainder != 0 &&
                guard.divisible_by - remainder > guard.upper - guard.lower) {
                throw std::invalid_argument(
                    "Dynamic fresh-output graph guard admits no divisible extent");
            }
            if (!guard.equal_to) continue;
            const GraphInputAxisReference& reference = *guard.equal_to;
            if (reference.input_index >= input_ids.size()) {
                throw std::invalid_argument(
                    "Dynamic fresh-output equality guard references an unknown input");
            }
            const Array<int64_t> reference_shape =
                values_by_id.at(input_ids[reference.input_index]).shape();
            if (reference.axis >= reference_shape.size() ||
                reference.input_index > guard.input_index ||
                (reference.input_index == guard.input_index &&
                 reference.axis >= guard.axis)) {
                throw std::invalid_argument(
                    "Dynamic fresh-output equality guards require a prior input axis");
            }
        }
    } else if (mode == ExecutablePlanMode::kDynamicStatefulV1) {
        if (!graph_guards.empty()) {
            throw std::invalid_argument(
                "Dynamic stateful ExecutablePlan cannot declare graph input guards");
        }
        if (state_ids.empty()) {
            throw std::invalid_argument(
                "Dynamic stateful ExecutablePlan requires persistent state");
        }
        std::unordered_set<int64_t> state_id_set;
        for (int64_t state_id : state_ids) state_id_set.insert(state_id);
        for (const auto& value : values) {
            for (int64_t dimension : value.shape()) {
                if (dimension < 0) {
                    throw std::invalid_argument(
                        "Dynamic stateful ExecutablePlan requires fully static shapes");
                }
            }
            if (value->valid_bytes != -1) {
                throw std::invalid_argument(
                    "Dynamic stateful ExecutablePlan derives valid extents from "
                    "session state metadata, not static valid bytes");
            }
        }
        // Append-count input: an explicit uint64[1] graph input describes how
        // many tokens this run appends to every declared state.
        if (plan.state_count_input_value_id() < 0) {
            throw std::invalid_argument(
                "Dynamic stateful ExecutablePlan requires an append-count input");
        }
        {
            const auto count_it =
                values_by_id.find(plan.state_count_input_value_id());
            if (count_it == values_by_id.end() || !count_it->second->is_input ||
                count_it->second->dtype.code != kDLUInt ||
                count_it->second->dtype.bits != 64 ||
                count_it->second->dtype.lanes != 1 ||
                count_it->second.shape().size() != 1 ||
                count_it->second.shape()[0] != 1) {
                throw std::invalid_argument(
                    "Dynamic stateful append-count input must be a uint64[1] graph input");
            }
        }
        // Every declared state carries capacity/extent metadata, is appended
        // exactly once through an in-place alias producer, and each call's
        // state extent bindings reference declared states in ABI order.
        std::unordered_map<int64_t, int> append_producers;
        for (const auto& value : values) {
            if (value->state_extent_axis == -1 && value->is_state) {
                throw std::invalid_argument(
                    "Dynamic stateful ExecutablePlan state requires capacity metadata");
            }
            if (value->alias_source_value_id != -1 &&
                state_id_set.count(value->alias_source_value_id) != 0) {
                ++append_producers[value->alias_source_value_id];
            }
        }
        for (int64_t state_id : state_ids) {
            if (append_producers[state_id] != 1) {
                throw std::invalid_argument(
                    "Dynamic stateful ExecutablePlan state must be appended by "
                    "exactly one in-place alias producer");
            }
        }
        const std::vector<std::vector<int64_t>> bindings =
            plan.state_extent_bindings();
        if (bindings.size() != calls.size()) {
            throw std::invalid_argument(
                "Dynamic stateful ExecutablePlan requires one extent binding list per call");
        }
        for (const auto& bound : bindings) {
            for (int64_t state_id : bound) {
                if (state_id_set.count(state_id) == 0) {
                    throw std::invalid_argument(
                        "Dynamic stateful extent binding references a non-state value");
                }
            }
        }
    } else {
        throw std::invalid_argument("ExecutablePlan mode is unsupported");
    }

    std::unordered_set<int64_t> available;
    for (int64_t id : input_ids) available.insert(id);
    for (int64_t id : constant_ids) available.insert(id);
    for (int64_t id : state_ids) available.insert(id);

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
    for (int64_t state_id : state_ids) {
        if (last_use.count(state_id) == 0) {
            throw std::invalid_argument(
                "ExecutablePlan state value is never consumed");
        }
    }
    for (int64_t output_id : output_ids) {
        if (available.count(output_id) == 0) {
            throw std::invalid_argument("ExecutablePlan graph output is unavailable");
        }
    }
    std::unordered_set<int64_t> donated_sources;
    for (const auto& value : values) {
        if (value->alias_source_value_id == -1) continue;
        const auto source_it = values_by_id.find(value->alias_source_value_id);
        if (source_it == values_by_id.end() ||
            source_it->first == value->value_id || source_it->second->is_constant ||
            source_it->second->is_output ||
            source_it->second->storage_id != value->storage_id ||
            !SameValueStorageContract(source_it->second, value)) {
            throw std::invalid_argument("ExecutablePlan alias source violates a value contract");
        }
        const int64_t write_at = producer_index.at(value->value_id);
        bool producer_consumes_source = false;
        for (int64_t input_id : calls[static_cast<size_t>(write_at)].input_value_ids()) {
            producer_consumes_source =
                producer_consumes_source || input_id == source_it->first;
        }
        if (!producer_consumes_source) {
            throw std::invalid_argument(
                "ExecutablePlan alias producer does not consume its source");
        }
        if (!donated_sources.insert(source_it->first).second) {
            throw std::invalid_argument(
                "ExecutablePlan alias source is donated more than once");
        }
        if (last_use.at(source_it->first) > write_at) {
            throw std::invalid_argument(
                "ExecutablePlan alias source remains live after an in-place write");
        }
    }

    const auto alias_root = [&](int64_t value_id) {
        for (size_t depth = 0; depth <= values.size(); ++depth) {
            const auto value = values_by_id.find(value_id);
            if (value == values_by_id.end()) {
                throw std::logic_error(
                    "ExecutablePlan alias root references an unknown value");
            }
            if (value->second->alias_source_value_id == -1) return value_id;
            value_id = value->second->alias_source_value_id;
        }
        throw std::logic_error("ExecutablePlan alias chain contains a cycle");
    };

    std::unordered_map<int64_t, std::vector<ValueSpec>> values_by_storage;
    for (const auto& value : values) {
        values_by_storage[value->storage_id].push_back(value);
    }
    for (const auto& storage : values_by_storage) {
        const auto& shared = storage.second;
        for (size_t i = 0; i < shared.size(); ++i) {
            for (size_t j = i + 1; j < shared.size(); ++j) {
                const bool declared_alias =
                    alias_root(shared[i]->value_id) ==
                    alias_root(shared[j]->value_id);
                if (!declared_alias &&
                    (!IsValueStorageReusable(shared[i]) ||
                     !IsValueStorageReusable(shared[j]) ||
                     !SameValueStorageContract(shared[i], shared[j]))) {
                    throw std::invalid_argument(
                        "ExecutablePlan storage sharing violates a value contract");
                }
                if (declared_alias) continue;
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
