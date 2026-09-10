/*! \file src/runtime/executable_plan.cc
 * \brief Implements the runtime-neutral ordered kernel execution contract.
 */

#include "kxc/runtime/executable_plan.h"

#include <algorithm>
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
    int64_t state_count_input_value_id,
    std::vector<StateOutputBinding> state_output_bindings,
    std::optional<RequestBatchingContract> request_batching,
    std::optional<StructuredSchedule> structured_schedule) {
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
    node->state_output_bindings_ = std::move(state_output_bindings);
    node->request_batching_ = request_batching;
    node->structured_schedule_ = std::move(structured_schedule);
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

std::vector<StateOutputBinding> ExecutablePlan::state_output_bindings() const {
    return operator->()->state_output_bindings_;
}

std::optional<RequestBatchingContract> ExecutablePlan::request_batching() const {
    return operator->()->request_batching_;
}

std::optional<StructuredSchedule> ExecutablePlan::structured_schedule() const {
    return operator->()->structured_schedule_;
}

ExecutablePlan ExecutablePlan::BindRequestBatching(int64_t max_batch_size) const {
    return ExecutablePlan(values(), calls(), input_value_ids(), constant_value_ids(),
        output_value_ids(), state_value_ids(), mode(), graph_input_guards(),
        state_extent_bindings(), state_count_input_value_id(), state_output_bindings(),
        RequestBatchingContract{max_batch_size});
}

ExecutablePlan ExecutablePlan::BindStateOutputs(
    std::vector<StateOutputBinding> bindings, double state_fill) const {
    Validate();
    if (mode() != ExecutablePlanMode::kStatic || !state_value_ids().empty() ||
        bindings.empty()) {
        throw std::invalid_argument(
            "BindStateOutputs requires a stateless static plan and nonempty bindings");
    }
    if (!std::isfinite(state_fill) ||
        !std::isfinite(static_cast<float>(state_fill))) {
        throw std::invalid_argument("BindStateOutputs fill must be finite float32");
    }
    std::unordered_map<int64_t, ValueSpec> by_id;
    for (const auto& value : values()) by_id.emplace(value->value_id, value);
    std::unordered_map<int64_t, StateOutputBinding> state_bindings;
    std::unordered_set<int64_t> append_sources;
    for (const auto& binding : bindings) {
        const auto state = by_id.find(binding.state_value_id);
        const auto source = by_id.find(binding.source_value_id);
        if (state == by_id.end() || source == by_id.end() ||
            !state->second->is_input || state->second->is_output ||
            !source->second->is_output || source->second->is_input ||
            binding.input_value_id != -1 || binding.source_extent_axis < 0 ||
            binding.source_extent_axis >=
                static_cast<int64_t>(state->second.shape().size()) ||
            !state_bindings.emplace(binding.state_value_id, binding).second ||
            !append_sources.insert(binding.source_value_id).second) {
            throw std::invalid_argument(
                "BindStateOutputs requires one-to-one graph input/output bindings");
        }
    }
    Array<ValueSpec> bound_values;
    for (const auto& value : values()) {
        const auto state = state_bindings.find(value->value_id);
        const bool is_state = state != state_bindings.end();
        const bool is_append_source = append_sources.count(value->value_id) != 0;
        const int64_t axis = is_state ? state->second.source_extent_axis : -1;
        bound_values.push_back(ValueSpec(
            value->value_id, value->storage_id, value.shape(), value->dtype,
            value->device, value->is_input && !is_state, value->is_constant,
            value->is_output && !is_append_source, value->is_alias,
            value->is_async_live || is_append_source, is_state,
            value->alias_source_value_id, value->write_mode, value->valid_bytes,
            is_state ? value.shape()[axis] : -1, axis,
            is_state ? state_fill : 0.0));
    }
    Array<int64_t> bound_inputs;
    Array<int64_t> bound_states;
    for (int64_t id : input_value_ids()) {
        if (state_bindings.count(id) != 0) {
            bound_states.push_back(id);
        } else {
            bound_inputs.push_back(id);
        }
    }
    Array<int64_t> bound_outputs;
    for (int64_t id : output_value_ids()) {
        if (append_sources.count(id) == 0) bound_outputs.push_back(id);
    }
    return ExecutablePlan(
        std::move(bound_values), calls(), std::move(bound_inputs),
        constant_value_ids(), std::move(bound_outputs), std::move(bound_states),
        ExecutablePlanMode::kStaticStatefulExternalV1, {}, {}, -1,
        std::move(bindings));
}

ExecutablePlan ExecutablePlan::BindBoundedStateOutputs(
    std::vector<StateOutputBinding> bindings,
    std::vector<Array<int64_t>> physical_shapes, double state_fill) const {
    Validate();
    if (mode() != ExecutablePlanMode::kDynamicFreshOutputV1 ||
        bindings.empty() || bindings.size() != physical_shapes.size() ||
        !std::isfinite(state_fill) || !std::isfinite(static_cast<float>(state_fill))) {
        throw std::invalid_argument(
            "BindBoundedStateOutputs requires a fresh bounded plan, physical shapes and finite fill");
    }
    std::unordered_map<int64_t, ValueSpec> by_id;
    int64_t next_value = 0, next_storage = 0;
    for (const auto& value : values()) {
        by_id.emplace(value->value_id, value);
        next_value = std::max(next_value, value->value_id);
        next_storage = std::max(next_storage, value->storage_id);
    }
    std::unordered_map<int64_t, size_t> state_indices;
    std::unordered_set<int64_t> append_sources;
    Array<ValueSpec> bound_values;
    Array<int64_t> bound_states;
    for (size_t index = 0; index < bindings.size(); ++index) {
        auto& binding = bindings[index];
        const auto input = by_id.find(binding.state_value_id);
        const auto source = by_id.find(binding.source_value_id);
        const int64_t axis = binding.source_extent_axis;
        if (input == by_id.end() || source == by_id.end() ||
            !input->second->is_input || input->second->is_output ||
            !source->second->is_output || source->second->is_input ||
            binding.input_value_id != -1 || binding.source_slot != -1 ||
            axis < 0 || axis >= static_cast<int64_t>(physical_shapes[index].size()) ||
            physical_shapes[index].size() != input->second.shape().size() ||
            !state_indices.emplace(binding.state_value_id, index).second ||
            !append_sources.insert(binding.source_value_id).second) {
            throw std::invalid_argument(
                "BindBoundedStateOutputs requires one-to-one input/output bindings and physical shapes");
        }
        if (next_value == std::numeric_limits<int64_t>::max() ||
            next_storage == std::numeric_limits<int64_t>::max()) {
            throw std::overflow_error("BindBoundedStateOutputs value/storage id overflow");
        }
        binding.input_value_id = ++next_value;
        const auto& value = input->second;
        bound_values.push_back(ValueSpec(binding.input_value_id, ++next_storage,
            value.shape(), value->dtype, value->device, true));
        bound_states.push_back(binding.state_value_id);
    }
    for (const auto& value : values()) {
        const auto found = state_indices.find(value->value_id);
        if (found != state_indices.end()) {
            const auto& binding = bindings[found->second];
            const auto& shape = physical_shapes[found->second];
            bound_values.push_back(ValueSpec(value->value_id, value->storage_id,
                shape, value->dtype, value->device, false, false, false, false,
                false, true, -1, ValueWriteMode::kAllocate, -1,
                shape[binding.source_extent_axis], binding.source_extent_axis, state_fill));
        } else if (append_sources.count(value->value_id)) {
            bound_values.push_back(ValueSpec(value->value_id, value->storage_id,
                value.shape(), value->dtype, value->device, false, false, false,
                false, true));
        } else {
            bound_values.push_back(value);
        }
    }
    const auto read_id = [&](int64_t value_id) {
        const auto found = state_indices.find(value_id);
        return found == state_indices.end() ? value_id
            : bindings[found->second].input_value_id;
    };
    Array<int64_t> bound_inputs, bound_outputs;
    for (int64_t id : input_value_ids()) bound_inputs.push_back(read_id(id));
    for (int64_t id : output_value_ids()) {
        if (!append_sources.count(id)) bound_outputs.push_back(id);
    }
    Array<KernelCall> bound_calls;
    for (const auto& call : calls()) {
        Array<int64_t> inputs;
        for (int64_t id : call.input_value_ids()) inputs.push_back(read_id(id));
        bound_calls.push_back(KernelCall(call->symbol, std::move(inputs), call.output_value_ids()));
    }
    return ExecutablePlan(std::move(bound_values), std::move(bound_calls),
        std::move(bound_inputs), constant_value_ids(), std::move(bound_outputs),
        std::move(bound_states), ExecutablePlanMode::kBoundedStatefulExternalV1,
        graph_input_guards(), {}, -1, std::move(bindings));
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

/*! \brief Validate the optional structured topology layered on one plan.
 *
 *  Kernel tasks reference calls by index; each call point is executed by
 *  exactly one kernel task. Branch/loop tasks reference regions by id. Phi and
 *  loop-carried results are produced by the topology, so they are collected
 *  into `structured_produced`. Region reachability is checked from the entry
 *  region so no region can be silently unreachable. */
void ValidateStructuredSchedule(
    const StructuredSchedule& schedule, size_t call_count,
    const Array<int64_t>& input_ids, const Array<int64_t>& constant_ids,
    const Array<int64_t>& state_ids, const Array<int64_t>& output_ids,
    const std::unordered_map<int64_t, ValueSpec>& values_by_id,
    std::unordered_set<int64_t>* structured_produced) {
    if (schedule.schema_version != StructuredSchedule::kSchemaVersion) {
        throw std::invalid_argument(
            "ExecutablePlan structured schedule version is unsupported");
    }
    if (schedule.regions.empty() || schedule.region_order.empty()) {
        throw std::invalid_argument(
            "ExecutablePlan structured schedule requires regions");
    }
    std::unordered_map<int64_t, const StructuredRegion*> region_by_id;
    for (const auto& region : schedule.regions) {
        if (!region_by_id.emplace(region.id, &region).second) {
            throw std::invalid_argument(
                "ExecutablePlan structured region ids must be unique");
        }
    }
    if (region_by_id.count(schedule.entry_region) == 0) {
        throw std::invalid_argument(
            "ExecutablePlan structured entry region is undefined");
    }
    if (schedule.region_order.size() != schedule.regions.size()) {
        throw std::invalid_argument(
            "ExecutablePlan structured region order must list every region once");
    }
    std::unordered_set<int64_t> ordered;
    for (int64_t region_id : schedule.region_order) {
        if (region_by_id.count(region_id) == 0 ||
            !ordered.insert(region_id).second) {
            throw std::invalid_argument(
                "ExecutablePlan structured region order is not a permutation");
        }
    }
    const auto require_value = [&](int64_t value_id, const char* context) {
        if (values_by_id.count(value_id) == 0) {
            throw std::invalid_argument(
                std::string("ExecutablePlan structured ") + context +
                " references an undefined value");
        }
    };
    const auto require_region = [&](int64_t region_id, const char* context) {
        if (region_by_id.count(region_id) == 0) {
            throw std::invalid_argument(
                std::string("ExecutablePlan structured ") + context +
                " references an undefined region");
        }
    };

    std::unordered_set<int64_t> referenced_calls;
    std::unordered_set<int64_t> reachable;
    reachable.insert(schedule.entry_region);
    for (const auto& region : schedule.regions) {
        for (int64_t value_id : region.live_ins) require_value(value_id, "live-in");
        for (int64_t value_id : region.live_outs) require_value(value_id, "live-out");
        std::unordered_set<int64_t> task_ids;
        for (const auto& task : region.tasks) {
            if (!task_ids.insert(task.id).second) {
                throw std::invalid_argument(
                    "ExecutablePlan structured task ids must be unique per region");
            }
            for (int64_t value_id : task.inputs) require_value(value_id, "task input");
            for (int64_t value_id : task.outputs) require_value(value_id, "task output");
            switch (task.kind) {
                case StructuredTaskKind::kKernel:
                    if (task.call_index < 0 ||
                        static_cast<size_t>(task.call_index) >= call_count) {
                        throw std::invalid_argument(
                            "ExecutablePlan structured kernel task references an unknown call");
                    }
                    if (!referenced_calls.insert(task.call_index).second) {
                        throw std::invalid_argument(
                            "ExecutablePlan structured call point is executed by more than one task");
                    }
                    break;
                case StructuredTaskKind::kBranch:
                    require_value(task.branch.predicate, "branch predicate");
                    require_region(task.branch.then_region, "branch then region");
                    require_region(task.branch.else_region, "branch else region");
                    reachable.insert(task.branch.then_region);
                    reachable.insert(task.branch.else_region);
                    for (const auto& phi : task.branch.phis) {
                        require_value(phi.result, "phi result");
                        require_value(phi.then_value, "phi then value");
                        require_value(phi.else_value, "phi else value");
                        structured_produced->insert(phi.result);
                    }
                    break;
                case StructuredTaskKind::kLoop:
                    require_region(task.loop.condition_region, "loop condition region");
                    require_region(task.loop.body_region, "loop body region");
                    require_value(task.loop.condition_value, "loop condition value");
                    if (task.loop.max_trip_count <= 0) {
                        throw std::invalid_argument(
                            "ExecutablePlan structured loop requires a positive trip bound");
                    }
                    reachable.insert(task.loop.condition_region);
                    reachable.insert(task.loop.body_region);
                    for (const auto& carried : task.loop.carried) {
                        require_value(carried.result, "loop result");
                        require_value(carried.initial, "loop initial");
                        require_value(carried.body_argument, "loop body argument");
                        require_value(carried.backedge, "loop backedge");
                        // Both the loop result and the per-iteration body
                        // argument are produced by the topology, not by a call.
                        structured_produced->insert(carried.result);
                        structured_produced->insert(carried.body_argument);
                    }
                    break;
            }
        }
    }
    if (referenced_calls.size() != call_count) {
        throw std::invalid_argument(
            "ExecutablePlan structured schedule must execute every call point exactly once");
    }
    if (reachable.size() != schedule.regions.size()) {
        throw std::invalid_argument(
            "ExecutablePlan structured schedule has an unreachable region");
    }
    (void)input_ids;
    (void)constant_ids;
    (void)state_ids;
    (void)output_ids;
}

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
    const bool bounded_stateful = mode == ExecutablePlanMode::kBoundedStatefulExternalV1;
    if (plan.request_batching() && !bounded_stateful) {
        throw std::invalid_argument("Request batching requires a bounded stateful plan");
    }
    if (mode != ExecutablePlanMode::kStaticStatefulExternalV1 &&
        !bounded_stateful &&
        !plan.state_output_bindings().empty()) {
        throw std::invalid_argument(
            "ExecutablePlan output-to-state bindings require static external stateful mode");
    }
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
    } else if (mode == ExecutablePlanMode::kDynamicFreshOutputV1 || bounded_stateful) {
        std::unordered_set<int64_t> storage_ids;
        for (const auto& value : values) {
            if ((!bounded_stateful && value->is_state) || value->is_alias ||
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
            if (value->is_constant || value->is_state) {
                for (int64_t dimension : value.shape()) {
                    if (dimension == -1) {
                        throw std::invalid_argument(
                            "Dynamic fresh-output constants require static shapes");
                    }
                }
            }
        }
        if (!bounded_stateful && !state_ids.empty()) {
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
    } else if (mode == ExecutablePlanMode::kStaticStatefulExternalV1) {
        if (!graph_guards.empty()) {
            throw std::invalid_argument(
                "Static external stateful ExecutablePlan cannot declare graph input guards");
        }
        if (state_ids.empty()) {
            throw std::invalid_argument(
                "Static external stateful ExecutablePlan requires persistent state");
        }
        if (!plan.state_extent_bindings().empty() ||
            plan.state_count_input_value_id() != -1) {
            throw std::invalid_argument(
                "Static external stateful ExecutablePlan does not use runtime extent ABI");
        }
        std::unordered_set<int64_t> state_id_set;
        for (int64_t state_id : state_ids) state_id_set.insert(state_id);
        std::unordered_set<int64_t> bound_states;
        std::unordered_set<int64_t> bound_sources;
        const std::vector<StateOutputBinding> bindings =
            plan.state_output_bindings();
        if (bindings.size() != state_ids.size()) {
            throw std::invalid_argument(
                "Static external stateful ExecutablePlan requires one output binding per state");
        }
        for (const auto& binding : bindings) {
            if (state_id_set.count(binding.state_value_id) == 0) {
                throw std::invalid_argument(
                    "Static external stateful output binding references a non-state value");
            }
            if (!bound_states.insert(binding.state_value_id).second ||
                binding.source_value_id < 0 ||
                !bound_sources.insert(binding.source_value_id).second) {
                throw std::invalid_argument(
                    "Static external stateful output bindings must be one-to-one");
            }
            const ValueSpec& state =
                values_by_id.at(binding.state_value_id);
            const auto source_it = values_by_id.find(binding.source_value_id);
            if (source_it == values_by_id.end()) {
                throw std::invalid_argument(
                    "Static external stateful output binding references an unknown source");
            }
            const ValueSpec& source = source_it->second;
            if (source->is_input || source->is_constant || source->is_state ||
                source->is_alias || source->is_output ||
                !source->is_async_live ||
                source->alias_source_value_id != -1 ||
                source->write_mode != ValueWriteMode::kAllocate) {
                throw std::invalid_argument(
                    "Static external stateful source must be a private allocated output");
            }
            if (state->dtype.code != kDLFloat || state->dtype.bits != 32 ||
                state->dtype.lanes != 1 || source->dtype.code != kDLFloat ||
                source->dtype.bits != 32 || source->dtype.lanes != 1 ||
                !std::isfinite(static_cast<float>(state->state_fill)) ||
                (state->device != Device::CPU() && state->device.device_type() != kCUDA) ||
                source->device != state->device) {
                throw std::invalid_argument(
                    "Static external stateful v1 requires float32 state and sources on one CPU or CUDA device");
            }
            if (state->state_extent_axis < 0 ||
                binding.source_extent_axis < 0 ||
                binding.source_extent_axis >=
                    static_cast<int64_t>(source.shape().size()) ||
                state->state_extent_axis >=
                    static_cast<int64_t>(state.shape().size()) ||
                binding.source_extent_axis != state->state_extent_axis ||
                source.shape().size() != state.shape().size()) {
                throw std::invalid_argument(
                    "Static external stateful source and state extent axes do not match");
            }
            const Array<int64_t> state_shape = state.shape();
            const Array<int64_t> source_shape = source.shape();
            const int64_t axis = state->state_extent_axis;
            if (binding.append_count <= 0 ||
                binding.append_count > state->state_capacity ||
                source_shape[axis] < 0 || binding.source_slot < 0 || binding.input_value_id != -1 ||
                binding.source_slot != state->state_capacity ||
                binding.source_slot > source_shape[axis] ||
                source_shape[axis] - binding.source_slot != binding.append_count) {
                throw std::invalid_argument(
                    "Static external stateful source slot does not describe one capacity append");
            }
            for (size_t dim = 0; dim < state_shape.size(); ++dim) {
                if (dim == static_cast<size_t>(axis)) continue;
                if (state_shape[dim] != source_shape[dim]) {
                    throw std::invalid_argument(
                        "Static external stateful source and state shapes differ");
                }
            }
        }
        if (bound_states.size() != state_ids.size()) {
            throw std::invalid_argument(
                "Static external stateful output bindings omit a state");
        }
        for (const auto& value : values) {
            for (int64_t dimension : value.shape()) {
                if (dimension < 0) {
                    throw std::invalid_argument(
                        "Static external stateful ExecutablePlan requires fully static shapes");
                }
            }
            if (value->is_alias || value->alias_source_value_id != -1 ||
                value->write_mode != ValueWriteMode::kAllocate) {
                throw std::invalid_argument(
                    "Static external stateful ExecutablePlan rejects kernel alias and donation");
            }
            if (value->valid_bytes != -1) {
                throw std::invalid_argument(
                    "Static external stateful ExecutablePlan derives valid extents from session state");
            }
        }
    } else {
        throw std::invalid_argument("ExecutablePlan mode is unsupported");
    }

    if (bounded_stateful) {
        if (state_ids.empty() || !plan.state_extent_bindings().empty() ||
            plan.state_count_input_value_id() != -1 ||
            plan.state_output_bindings().size() != state_ids.size()) {
            throw std::invalid_argument("Bounded stateful plan requires one prefix/output binding per state");
        }
        std::unordered_set<int64_t> bound_states, bound_inputs, bound_sources;
        for (const auto& binding : plan.state_output_bindings()) {
            const auto state_it = values_by_id.find(binding.state_value_id);
            const auto input_it = values_by_id.find(binding.input_value_id);
            const auto source_it = values_by_id.find(binding.source_value_id);
            if (state_it == values_by_id.end() || input_it == values_by_id.end() ||
                source_it == values_by_id.end() ||
                !bound_states.insert(binding.state_value_id).second ||
                !bound_inputs.insert(binding.input_value_id).second ||
                !bound_sources.insert(binding.source_value_id).second) {
                throw std::invalid_argument("Bounded stateful bindings must be one-to-one declared values");
            }
            const ValueSpec& state = state_it->second;
            const ValueSpec& input = input_it->second;
            const ValueSpec& source = source_it->second;
            if (!state->is_state || !std::isfinite(static_cast<float>(state->state_fill)) ||
                !input->is_input || input->is_output ||
                source->is_input || source->is_constant || source->is_state ||
                source->is_output || !source->is_async_live) {
                throw std::invalid_argument("Bounded stateful binding requires state, graph input and private output");
            }
            for (const auto& value : {state, input, source}) {
                if (value->dtype.code != kDLFloat || value->dtype.bits != 32 ||
                    value->dtype.lanes != 1 || value->device != state->device ||
                    (value->device != Device::CPU() && value->device.device_type() != kCUDA)) {
                    throw std::invalid_argument("Bounded stateful bindings require float32 state, prefix and source on one CPU or CUDA device");
                }
            }
            const int64_t axis = state->state_extent_axis;
            const auto state_shape = state.shape();
            const auto input_shape = input.shape();
            const auto source_shape = source.shape();
            if (axis < 0 || binding.source_extent_axis != axis ||
                input_shape.size() != state_shape.size() || source_shape.size() != state_shape.size() ||
                input_shape[axis] != -1 || source_shape[axis] != -1 ||
                binding.source_slot != -1 || binding.append_count <= 0 ||
                binding.append_count > state->state_capacity) {
                throw std::invalid_argument("Bounded stateful extent axes or append contract are invalid");
            }
            for (size_t dim = 0; dim < state_shape.size(); ++dim) {
                if (state_shape[dim] <= 0 ||
                    (static_cast<int64_t>(dim) != axis &&
                     (input_shape[dim] != source_shape[dim] ||
                      (input_shape[dim] != -1 && input_shape[dim] != state_shape[dim])))) {
                    throw std::invalid_argument("Bounded stateful physical and logical layouts differ");
                }
            }
            (void)StaticNBytes(state);
            for (const auto& guard : graph_guards) {
                if (input_ids[guard.input_index] != binding.input_value_id ||
                    guard.axis == static_cast<size_t>(axis)) continue;
                const int64_t dimension = state_shape[guard.axis];
                if (dimension < guard.lower || dimension > guard.upper ||
                    dimension % guard.divisible_by != 0) {
                    throw std::invalid_argument("Bounded stateful physical shape violates graph input bounds");
                }
            }
        }
    }

    if (const auto batching = plan.request_batching()) {
        const auto first_state = values_by_id.at(state_ids[0]);
        for (const auto& value : values) {
            if (value->device != first_state->device ||
                (value->device != Device::CPU() && value->device.device_type() != kCUDA)) {
                throw std::invalid_argument("Request batching requires one CPU:0 or CUDA device");
            }
        }
        if (input_ids.size() <= state_ids.size()) {
            throw std::invalid_argument("Request batching requires a caller input per step");
        }
        const int64_t slots = first_state.shape()[0];
        if (batching->max_batch_size <= 0 || batching->max_batch_size > slots) {
            throw std::invalid_argument("Request batching max batch must fit physical slots");
        }
        const int64_t append_count = plan.state_output_bindings()[0].append_count;
        for (const auto& guard : graph_guards) {
            if (guard.axis != 0 && guard.equal_to && guard.equal_to->axis == 0) {
                throw std::invalid_argument("Request batching cannot equate batch and non-batch axes");
            }
        }
        for (const auto& binding : plan.state_output_bindings()) {
            const auto state = values_by_id.at(binding.state_value_id);
            if (state->state_extent_axis == 0 || state.shape()[0] != slots ||
                binding.append_count != append_count) {
                throw std::invalid_argument("Request batching requires uniform slots and appends with a non-batch state axis");
            }
        }
        for (size_t index = 0; index < input_ids.size(); ++index) {
            const auto shape = values_by_id.at(input_ids[index]).shape();
            if (shape.empty() || shape[0] != -1) {
                throw std::invalid_argument("Request batching requires a bounded leading axis on every input");
            }
            for (const auto& guard : graph_guards) {
                if (guard.input_index != index || guard.axis != 0) continue;
                if (guard.lower > 1 || guard.upper < slots || guard.divisible_by != 1 ||
                    (index == 0 && guard.equal_to) ||
                    (index != 0 && (!guard.equal_to || guard.equal_to->input_index != 0 ||
                                    guard.equal_to->axis != 0))) {
                    throw std::invalid_argument("Request batching requires shared leading-axis guards admitting all batch sizes");
                }
            }
        }
        for (int64_t id : output_ids) {
            const auto shape = values_by_id.at(id).shape();
            if (shape.empty() || shape[0] != -1) {
                throw std::invalid_argument("Request batching requires a bounded leading axis on every output");
            }
        }
    }

    std::unordered_set<int64_t> available;
    for (int64_t id : input_ids) available.insert(id);
    for (int64_t id : constant_ids) available.insert(id);
    for (int64_t id : state_ids) available.insert(id);

    const std::optional<StructuredSchedule> structured =
        plan.structured_schedule();
    std::unordered_set<int64_t> structured_produced;
    if (structured) {
        ValidateStructuredSchedule(*structured, calls.size(), input_ids,
                                   constant_ids, state_ids, output_ids,
                                   values_by_id, &structured_produced);
    }

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
            if (!structured && available.count(input_id) == 0) {
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
        if (!IsSourceValue(entry.second) &&
            producer_counts[entry.first] != 1 &&
            structured_produced.count(entry.first) == 0) {
            throw std::invalid_argument(
                "Every non-source ExecutablePlan value requires exactly one producer");
        }
    }
    for (const auto& binding : plan.state_output_bindings()) {
        const auto& call = calls[static_cast<size_t>(producer_index.at(binding.source_value_id))];
        bool consumes_state = false;
        for (int64_t input_id : call.input_value_ids()) {
            consumes_state = consumes_state || input_id ==
                (bounded_stateful ? binding.input_value_id : binding.state_value_id);
        }
        if (!consumes_state) {
            throw std::invalid_argument(
                "Static external stateful source producer must consume its state");
        }
    }
    for (int64_t state_id : state_ids) {
        if (bounded_stateful && last_use.count(state_id) != 0) {
            throw std::invalid_argument("Bounded stateful kernels must consume compact prefixes, not capacity state");
        }
        if (!bounded_stateful && last_use.count(state_id) == 0) {
            throw std::invalid_argument(
                "ExecutablePlan state value is never consumed");
        }
    }
    for (int64_t output_id : output_ids) {
        if (!structured && available.count(output_id) == 0) {
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
