/*! \file src/runtime/task_plan.cc
 * \brief Implements frozen Region/task-DAG validation for the exact single-stream path.
 */

#include "kxc/runtime/task_plan.h"

#include <algorithm>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "kxc/support/object_registration.h"

namespace kxc::runtime {

KXC_OBJECT_DEFINE_WITH_KEY(RegionSpecNode, "kxc.runtime.RegionSpecNode")
KXC_OBJECT_DEFINE_WITH_KEY(TaskSpecNode, "kxc.runtime.TaskSpecNode")
KXC_OBJECT_DEFINE_WITH_KEY(FrozenTaskPlanNode,
                           "kxc.runtime.FrozenTaskPlanNode")

namespace {

template <typename T>
Array<T> CopyArray(const Array<T>& source) {
    Array<T> result;
    for (const auto& value : source) result.push_back(value);
    return result;
}

void ValidateIds(const Array<int64_t>& ids, const std::string& context,
                 bool allow_duplicates = false) {
    std::unordered_set<int64_t> seen;
    for (int64_t id : ids) {
        if (id < 0 || (!allow_duplicates && !seen.insert(id).second)) {
            throw std::invalid_argument(
                context + " must contain non-negative" +
                (allow_duplicates ? "" : " unique") + " ids");
        }
    }
}

bool IsDataProducer(TaskKind kind) {
    return kind == TaskKind::kKernel || kind == TaskKind::kCopy ||
           kind == TaskKind::kShapeEval;
}

bool IsSourceValue(const ValueSpec& value) {
    return value->is_input || value->is_constant;
}

bool SameShape(const Array<int64_t>& lhs, const Array<int64_t>& rhs) {
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i] != rhs[i]) return false;
    }
    return true;
}

bool SameStorageContract(const ValueSpec& lhs, const ValueSpec& rhs) {
    return lhs->dtype.code == rhs->dtype.code &&
           lhs->dtype.bits == rhs->dtype.bits &&
           lhs->dtype.lanes == rhs->dtype.lanes &&
           lhs->device == rhs->device && SameShape(lhs.shape(), rhs.shape());
}

bool CanShareStorage(const ValueSpec& value) {
    return !IsSourceValue(value) && !value->is_output && !value->is_alias &&
           !value->is_async_live;
}

std::set<int64_t> ToSet(const Array<int64_t>& values) {
    return std::set<int64_t>(values.begin(), values.end());
}

void ValidateRegionKind(RegionKind kind) {
    switch (kind) {
        case RegionKind::kPerCall:
        case RegionKind::kFusion:
        case RegionKind::kLibrary:
        case RegionKind::kControlFlow:
            return;
    }
    throw std::invalid_argument("RegionSpec kind is invalid");
}

void ValidateRegionEffect(RegionEffect effect) {
    switch (effect) {
        case RegionEffect::kPure:
        case RegionEffect::kOrdered:
            return;
    }
    throw std::invalid_argument("RegionSpec effect is invalid");
}

void ValidateRegionAlias(RegionAlias alias) {
    switch (alias) {
        case RegionAlias::kNoAlias:
        case RegionAlias::kConservative:
            return;
    }
    throw std::invalid_argument("RegionSpec alias contract is invalid");
}

void ValidateTaskKind(TaskKind kind) {
    switch (kind) {
        case TaskKind::kKernel:
        case TaskKind::kCopy:
        case TaskKind::kEvent:
        case TaskKind::kShapeEval:
        case TaskKind::kAllocate:
        case TaskKind::kSync:
            return;
    }
    throw std::invalid_argument("TaskSpec kind is invalid");
}

template <typename T>
void ValidateRoleList(
    const Array<int64_t>& ids, bool ValueSpecNode::*role,
    const std::string& name,
    const std::unordered_map<int64_t, T>& values_by_id) {
    std::unordered_set<int64_t> seen;
    for (int64_t id : ids) {
        const auto value = values_by_id.find(id);
        if (value == values_by_id.end() ||
            !(value->second.operator->()->*role) || !seen.insert(id).second) {
            throw std::invalid_argument(name + " does not match value roles");
        }
    }
    for (const auto& value : values_by_id) {
        if ((value.second.operator->()->*role) && !seen.count(value.first)) {
            throw std::invalid_argument(name + " omits a required value");
        }
    }
}

struct GraphIndex final {
    std::unordered_map<int64_t, size_t> task_index;
    std::vector<size_t> topological_order;
    std::vector<std::vector<bool>> reaches;
};

GraphIndex BuildGraphIndex(const Array<TaskSpec>& tasks) {
    GraphIndex result;
    for (size_t i = 0; i < tasks.size(); ++i) {
        if (!result.task_index.emplace(tasks[i]->task_id, i).second) {
            throw std::invalid_argument("FrozenTaskPlan task ids must be unique");
        }
    }

    std::vector<std::vector<size_t>> successors(tasks.size());
    std::vector<size_t> indegree(tasks.size(), 0);
    for (size_t i = 0; i < tasks.size(); ++i) {
        for (int64_t dependency : tasks[i].dependency_task_ids()) {
            const auto predecessor = result.task_index.find(dependency);
            if (predecessor == result.task_index.end()) {
                throw std::invalid_argument(
                    "TaskSpec dependency references an unknown task");
            }
            successors[predecessor->second].push_back(i);
            ++indegree[i];
        }
    }

    using Ready = std::pair<int64_t, size_t>;
    std::priority_queue<Ready, std::vector<Ready>, std::greater<Ready>> ready;
    for (size_t i = 0; i < tasks.size(); ++i) {
        if (indegree[i] == 0) ready.emplace(tasks[i]->task_id, i);
    }
    while (!ready.empty()) {
        const size_t current = ready.top().second;
        ready.pop();
        result.topological_order.push_back(current);
        for (size_t successor : successors[current]) {
            if (--indegree[successor] == 0) {
                ready.emplace(tasks[successor]->task_id, successor);
            }
        }
    }
    if (result.topological_order.size() != tasks.size()) {
        throw std::invalid_argument("FrozenTaskPlan task dependencies contain a cycle");
    }

    result.reaches.assign(tasks.size(), std::vector<bool>(tasks.size(), false));
    for (size_t from = 0; from < successors.size(); ++from) {
        for (size_t to : successors[from]) result.reaches[from][to] = true;
    }
    for (size_t via = 0; via < tasks.size(); ++via) {
        for (size_t from = 0; from < tasks.size(); ++from) {
            if (!result.reaches[from][via]) continue;
            for (size_t to = 0; to < tasks.size(); ++to) {
                result.reaches[from][to] =
                    result.reaches[from][to] || result.reaches[via][to];
            }
        }
    }
    return result;
}

bool HappensBefore(const GraphIndex& graph, int64_t from, int64_t to) {
    return graph.reaches.at(graph.task_index.at(from))
                         .at(graph.task_index.at(to));
}

}  // namespace

RegionSpec::RegionSpec(int64_t region_id, RegionKind kind, String semantic_key,
                       Array<int64_t> task_ids,
                       Array<int64_t> live_in_value_ids,
                       Array<int64_t> live_out_value_ids,
                       Array<int64_t> constant_value_ids, RegionEffect effect,
                       RegionAlias alias) {
    auto* node = new RegionSpecNode();
    node->region_id = region_id;
    node->kind = kind;
    node->semantic_key = std::move(semantic_key);
    node->task_ids_ = CopyArray(task_ids);
    node->live_in_value_ids_ = CopyArray(live_in_value_ids);
    node->live_out_value_ids_ = CopyArray(live_out_value_ids);
    node->constant_value_ids_ = CopyArray(constant_value_ids);
    node->effect = effect;
    node->alias = alias;
    SetData(node);
    Validate();
}

RegionSpec::RegionSpec(const ObjectRef& ref) : ObjectRef(ref) {
    if (defined() && !As<RegionSpecNode>()) {
        SetData(nullptr);
        throw std::invalid_argument("ObjectRef does not contain RegionSpecNode");
    }
    if (defined()) Validate();
}

Array<int64_t> RegionSpec::task_ids() const {
    return CopyArray(operator->()->task_ids_);
}
Array<int64_t> RegionSpec::live_in_value_ids() const {
    return CopyArray(operator->()->live_in_value_ids_);
}
Array<int64_t> RegionSpec::live_out_value_ids() const {
    return CopyArray(operator->()->live_out_value_ids_);
}
Array<int64_t> RegionSpec::constant_value_ids() const {
    return CopyArray(operator->()->constant_value_ids_);
}

void RegionSpec::Validate() const {
    const auto* node = operator->();
    if (node->region_id < 0) {
        throw std::invalid_argument("RegionSpec region_id must be non-negative");
    }
    ValidateRegionKind(node->kind);
    ValidateRegionEffect(node->effect);
    ValidateRegionAlias(node->alias);
    if (node->task_ids_.empty()) {
        throw std::invalid_argument("RegionSpec requires at least one task");
    }
    ValidateIds(node->task_ids_, "RegionSpec task_ids");
    ValidateIds(node->live_in_value_ids_, "RegionSpec live_in_value_ids");
    ValidateIds(node->live_out_value_ids_, "RegionSpec live_out_value_ids");
    ValidateIds(node->constant_value_ids_, "RegionSpec constant_value_ids");
    const auto live_ins = ToSet(node->live_in_value_ids_);
    for (int64_t constant : node->constant_value_ids_) {
        if (!live_ins.count(constant)) {
            throw std::invalid_argument(
                "RegionSpec constants must be region live-ins");
        }
    }
}

const RegionSpecNode* RegionSpec::operator->() const {
    const auto* node = As<RegionSpecNode>();
    if (!node) throw std::runtime_error("undefined or invalid RegionSpec");
    return node;
}

TaskSpec::TaskSpec(int64_t task_id, TaskKind kind, Device device,
                   Array<int64_t> input_value_ids,
                   Array<int64_t> output_value_ids,
                   Array<int64_t> dependency_task_ids, String symbol,
                   int64_t artifact_generation, int64_t stream_id,
                   uint64_t alignment) {
    auto* node = new TaskSpecNode();
    node->task_id = task_id;
    node->kind = kind;
    node->device = std::move(device);
    node->input_value_ids_ = CopyArray(input_value_ids);
    node->output_value_ids_ = CopyArray(output_value_ids);
    node->dependency_task_ids_ = CopyArray(dependency_task_ids);
    node->symbol = std::move(symbol);
    node->artifact_generation = artifact_generation;
    node->stream_id = stream_id;
    node->alignment = alignment;
    SetData(node);
    Validate();
}

TaskSpec::TaskSpec(const ObjectRef& ref) : ObjectRef(ref) {
    if (defined() && !As<TaskSpecNode>()) {
        SetData(nullptr);
        throw std::invalid_argument("ObjectRef does not contain TaskSpecNode");
    }
    if (defined()) Validate();
}

Array<int64_t> TaskSpec::input_value_ids() const {
    return CopyArray(operator->()->input_value_ids_);
}
Array<int64_t> TaskSpec::output_value_ids() const {
    return CopyArray(operator->()->output_value_ids_);
}
Array<int64_t> TaskSpec::dependency_task_ids() const {
    return CopyArray(operator->()->dependency_task_ids_);
}

void TaskSpec::Validate() const {
    const auto* node = operator->();
    if (node->task_id < 0 || node->stream_id < 0 ||
        node->artifact_generation < 0 || !node->device.defined()) {
        throw std::invalid_argument("TaskSpec identity or device is invalid");
    }
    ValidateTaskKind(node->kind);
    ValidateIds(node->input_value_ids_, "TaskSpec input_value_ids", true);
    ValidateIds(node->output_value_ids_, "TaskSpec output_value_ids");
    ValidateIds(node->dependency_task_ids_, "TaskSpec dependency_task_ids");
    for (int64_t dependency : node->dependency_task_ids_) {
        if (dependency == node->task_id) {
            throw std::invalid_argument("TaskSpec cannot depend on itself");
        }
    }

    const bool has_symbol = !std::string(node->symbol).empty();
    switch (node->kind) {
        case TaskKind::kKernel:
            if (!has_symbol || node->output_value_ids_.empty() ||
                node->alignment != 0) {
                throw std::invalid_argument(
                    "Kernel task requires a symbol and outputs only");
            }
            break;
        case TaskKind::kCopy:
            if (has_symbol || node->input_value_ids_.size() != 1 ||
                node->output_value_ids_.size() != 1 || node->alignment != 0 ||
                node->artifact_generation != 0) {
                throw std::invalid_argument(
                    "Copy task requires one input and one output");
            }
            break;
        case TaskKind::kShapeEval:
            if (!has_symbol || node->output_value_ids_.empty() ||
                node->alignment != 0 || node->artifact_generation != 0) {
                throw std::invalid_argument(
                    "ShapeEval task requires a program key and outputs");
            }
            break;
        case TaskKind::kAllocate:
            if (has_symbol || !node->input_value_ids_.empty() ||
                node->output_value_ids_.size() != 1 || node->alignment == 0 ||
                (node->alignment & (node->alignment - 1)) != 0 ||
                node->artifact_generation != 0) {
                throw std::invalid_argument(
                    "Allocate task requires one output and power-of-two alignment");
            }
            break;
        case TaskKind::kEvent:
        case TaskKind::kSync:
            if (has_symbol || !node->input_value_ids_.empty() ||
                !node->output_value_ids_.empty() || node->alignment != 0 ||
                node->artifact_generation != 0) {
                throw std::invalid_argument(
                    "Event and Sync tasks carry only dependencies");
            }
            break;
    }
}

const TaskSpecNode* TaskSpec::operator->() const {
    const auto* node = As<TaskSpecNode>();
    if (!node) throw std::runtime_error("undefined or invalid TaskSpec");
    return node;
}

FrozenTaskPlan::FrozenTaskPlan(int32_t version, Array<ValueSpec> values,
                               Array<TaskSpec> tasks,
                               Array<RegionSpec> regions,
                               Array<int64_t> input_value_ids,
                               Array<int64_t> constant_value_ids,
                               Array<int64_t> output_value_ids) {
    auto* node = new FrozenTaskPlanNode();
    node->version = version;
    node->values_ = CopyArray(values);
    node->tasks_ = CopyArray(tasks);
    node->regions_ = CopyArray(regions);
    node->input_value_ids_ = CopyArray(input_value_ids);
    node->constant_value_ids_ = CopyArray(constant_value_ids);
    node->output_value_ids_ = CopyArray(output_value_ids);
    SetData(node);
    Validate();
}

FrozenTaskPlan::FrozenTaskPlan(const ObjectRef& ref) : ObjectRef(ref) {
    if (defined() && !As<FrozenTaskPlanNode>()) {
        SetData(nullptr);
        throw std::invalid_argument(
            "ObjectRef does not contain FrozenTaskPlanNode");
    }
    if (defined()) Validate();
}

Array<ValueSpec> FrozenTaskPlan::values() const {
    return CopyArray(operator->()->values_);
}
Array<TaskSpec> FrozenTaskPlan::tasks() const {
    return CopyArray(operator->()->tasks_);
}
Array<RegionSpec> FrozenTaskPlan::regions() const {
    return CopyArray(operator->()->regions_);
}
Array<int64_t> FrozenTaskPlan::input_value_ids() const {
    return CopyArray(operator->()->input_value_ids_);
}
Array<int64_t> FrozenTaskPlan::constant_value_ids() const {
    return CopyArray(operator->()->constant_value_ids_);
}
Array<int64_t> FrozenTaskPlan::output_value_ids() const {
    return CopyArray(operator->()->output_value_ids_);
}

Array<int64_t> FrozenTaskPlan::topological_task_ids() const {
    const Array<TaskSpec> plan_tasks = tasks();
    const GraphIndex graph = BuildGraphIndex(plan_tasks);
    Array<int64_t> result;
    for (size_t index : graph.topological_order) {
        result.push_back(plan_tasks[index]->task_id);
    }
    return result;
}

void FrozenTaskPlan::Validate() const {
    const auto* node = operator->();
    if (node->version != kFrozenTaskPlanVersion) {
        throw std::invalid_argument("FrozenTaskPlan version is unsupported");
    }
    if (node->values_.empty() || node->output_value_ids_.empty()) {
        throw std::invalid_argument(
            "FrozenTaskPlan requires values and graph outputs");
    }

    std::unordered_map<int64_t, ValueSpec> values_by_id;
    for (const auto& value : node->values_) {
        value.Validate();
        if (!values_by_id.emplace(value->value_id, value).second) {
            throw std::invalid_argument(
                "FrozenTaskPlan value ids must be unique");
        }
        if (value->is_alias) {
            throw std::invalid_argument(
                "FrozenTaskPlan v1 does not execute aliased values");
        }
    }
    ValidateRoleList(node->input_value_ids_, &ValueSpecNode::is_input,
                     "FrozenTaskPlan input_value_ids", values_by_id);
    ValidateRoleList(node->constant_value_ids_, &ValueSpecNode::is_constant,
                     "FrozenTaskPlan constant_value_ids", values_by_id);
    ValidateRoleList(node->output_value_ids_, &ValueSpecNode::is_output,
                     "FrozenTaskPlan output_value_ids", values_by_id);

    const Device plan_device = node->values_[0]->device;
    for (const auto& value : node->values_) {
        if (value->device != plan_device) {
            throw std::invalid_argument(
                "FrozenTaskPlan v1 requires one physical device");
        }
    }
    for (const auto& task : node->tasks_) {
        task.Validate();
        if (task->device != plan_device || task->stream_id != 0) {
            throw std::invalid_argument(
                "FrozenTaskPlan v1 requires one device and stream 0");
        }
    }

    const GraphIndex graph = BuildGraphIndex(node->tasks_);
    std::unordered_map<int64_t, TaskSpec> tasks_by_id;
    for (const auto& task : node->tasks_) {
        tasks_by_id.emplace(task->task_id, task);
    }

    std::unordered_map<int64_t, int64_t> producer_by_value;
    std::unordered_map<int64_t, int64_t> allocation_by_value;
    std::unordered_map<int64_t, std::unordered_set<int64_t>> consumers_by_value;
    for (const auto& task : node->tasks_) {
        for (int64_t input : task.input_value_ids()) {
            if (!values_by_id.count(input)) {
                throw std::invalid_argument(
                    "TaskSpec input references an unknown value");
            }
            consumers_by_value[input].insert(task->task_id);
        }
        for (int64_t output : task.output_value_ids()) {
            const auto value = values_by_id.find(output);
            if (value == values_by_id.end()) {
                throw std::invalid_argument(
                    "TaskSpec output references an unknown value");
            }
            if (IsSourceValue(value->second)) {
                throw std::invalid_argument(
                    "A task cannot produce or allocate a source value");
            }
            if (task->kind == TaskKind::kAllocate) {
                if (!allocation_by_value.emplace(output, task->task_id).second) {
                    throw std::invalid_argument(
                        "A value has more than one Allocate task");
                }
            } else if (IsDataProducer(task->kind) &&
                       !producer_by_value.emplace(output, task->task_id).second) {
                throw std::invalid_argument(
                    "A value has more than one data producer");
            }
        }
    }

    for (const auto& value : values_by_id) {
        if (IsSourceValue(value.second)) continue;
        if (!producer_by_value.count(value.first) ||
            !allocation_by_value.count(value.first)) {
            throw std::invalid_argument(
                "Every produced value requires one producer and Allocate task");
        }
        const TaskSpec& producer = tasks_by_id.at(producer_by_value.at(value.first));
        const int64_t allocation = allocation_by_value.at(value.first);
        if (ToSet(producer.dependency_task_ids()).count(allocation) == 0) {
            throw std::invalid_argument(
                "A data producer must directly depend on its Allocate task");
        }
    }

    for (const auto& task : node->tasks_) {
        for (int64_t input : task.input_value_ids()) {
            const auto producer = producer_by_value.find(input);
            if (producer != producer_by_value.end() &&
                !HappensBefore(graph, producer->second, task->task_id)) {
                throw std::invalid_argument(
                    "A task input producer must precede its consumer");
            }
        }
        if (task->kind == TaskKind::kCopy &&
            !SameStorageContract(
                values_by_id.at(task.input_value_ids()[0]),
                values_by_id.at(task.output_value_ids()[0]))) {
            throw std::invalid_argument(
                "Copy task values require matching exact tensor contracts");
        }
    }
    for (int64_t output : node->output_value_ids_) {
        if (!IsSourceValue(values_by_id.at(output)) &&
            !producer_by_value.count(output)) {
            throw std::invalid_argument(
                "FrozenTaskPlan graph output is unavailable");
        }
    }

    std::unordered_set<int64_t> region_ids;
    std::unordered_map<int64_t, int64_t> region_by_task;
    std::unordered_set<int64_t> conservative_alias_values;
    std::vector<RegionSpec> ordered_regions;
    for (const auto& region : node->regions_) {
        region.Validate();
        if (region->effect == RegionEffect::kOrdered) {
            ordered_regions.push_back(region);
        }
        if (!region_ids.insert(region->region_id).second) {
            throw std::invalid_argument(
                "FrozenTaskPlan region ids must be unique");
        }
        for (int64_t task_id : region.task_ids()) {
            if (!tasks_by_id.count(task_id) ||
                !region_by_task.emplace(task_id, region->region_id).second) {
                throw std::invalid_argument(
                    "Every task must belong to exactly one known region");
            }
            if (region->alias == RegionAlias::kConservative) {
                const TaskSpec& task = tasks_by_id.at(task_id);
                for (int64_t input : task.input_value_ids()) {
                    conservative_alias_values.insert(input);
                }
                for (int64_t output : task.output_value_ids()) {
                    conservative_alias_values.insert(output);
                }
            }
        }

        const std::set<int64_t> region_tasks = ToSet(region.task_ids());
        std::set<int64_t> expected_live_ins;
        std::set<int64_t> expected_live_outs;
        std::set<int64_t> expected_constants;
        for (int64_t task_id : region_tasks) {
            const TaskSpec& task = tasks_by_id.at(task_id);
            for (int64_t input : task.input_value_ids()) {
                const auto producer = producer_by_value.find(input);
                if (producer == producer_by_value.end() ||
                    !region_tasks.count(producer->second)) {
                    expected_live_ins.insert(input);
                    if (values_by_id.at(input)->is_constant) {
                        expected_constants.insert(input);
                    }
                }
            }
            if (!IsDataProducer(task->kind)) continue;
            for (int64_t output : task.output_value_ids()) {
                bool crosses_boundary = values_by_id.at(output)->is_output;
                for (int64_t consumer : consumers_by_value[output]) {
                    crosses_boundary = crosses_boundary ||
                                       !region_tasks.count(consumer);
                }
                if (crosses_boundary) expected_live_outs.insert(output);
            }
        }
        if (expected_live_ins != ToSet(region.live_in_value_ids()) ||
            expected_live_outs != ToSet(region.live_out_value_ids()) ||
            expected_constants != ToSet(region.constant_value_ids())) {
            throw std::invalid_argument(
                "RegionSpec live-in/live-out/constant boundary is incomplete");
        }
    }
    if (region_by_task.size() != node->tasks_.size()) {
        throw std::invalid_argument(
            "Every FrozenTaskPlan task requires one region");
    }
    const auto region_before = [&](const RegionSpec& lhs,
                                   const RegionSpec& rhs) {
        for (int64_t lhs_task : lhs.task_ids()) {
            for (int64_t rhs_task : rhs.task_ids()) {
                if (!HappensBefore(graph, lhs_task, rhs_task)) return false;
            }
        }
        return true;
    };
    for (size_t i = 0; i < ordered_regions.size(); ++i) {
        for (size_t j = i + 1; j < ordered_regions.size(); ++j) {
            if (!region_before(ordered_regions[i], ordered_regions[j]) &&
                !region_before(ordered_regions[j], ordered_regions[i])) {
                throw std::invalid_argument(
                    "Ordered regions require an explicit total dependency order");
            }
        }
    }

    std::unordered_map<int64_t, std::vector<ValueSpec>> values_by_storage;
    for (const auto& value : node->values_) {
        values_by_storage[value->storage_id].push_back(value);
    }
    const auto lifetime_before = [&](int64_t lhs, int64_t rhs) {
        const int64_t rhs_allocation = allocation_by_value.at(rhs);
        const auto consumers = consumers_by_value.find(lhs);
        if (consumers == consumers_by_value.end() || consumers->second.empty()) {
            return HappensBefore(graph, producer_by_value.at(lhs),
                                 rhs_allocation);
        }
        for (int64_t consumer : consumers->second) {
            if (!HappensBefore(graph, consumer, rhs_allocation)) return false;
        }
        return true;
    };
    for (const auto& storage : values_by_storage) {
        const auto& shared = storage.second;
        for (size_t i = 0; i < shared.size(); ++i) {
            for (size_t j = i + 1; j < shared.size(); ++j) {
                if (!CanShareStorage(shared[i]) ||
                    !CanShareStorage(shared[j]) ||
                    conservative_alias_values.count(shared[i]->value_id) ||
                    conservative_alias_values.count(shared[j]->value_id) ||
                    !SameStorageContract(shared[i], shared[j]) ||
                    !(lifetime_before(shared[i]->value_id,
                                      shared[j]->value_id) ||
                      lifetime_before(shared[j]->value_id,
                                      shared[i]->value_id))) {
                    throw std::invalid_argument(
                        "FrozenTaskPlan storage sharing violates dependency-aware liveness");
                }
            }
        }
    }
}

const FrozenTaskPlanNode* FrozenTaskPlan::operator->() const {
    const auto* node = As<FrozenTaskPlanNode>();
    if (!node) throw std::runtime_error("undefined or invalid FrozenTaskPlan");
    return node;
}

}  // namespace kxc::runtime
