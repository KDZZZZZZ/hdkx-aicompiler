/*! \file src/runtime/task_plan.cc
 * \brief Implements frozen Region/task-DAG validation for the exact single-stream path.
 */

#include "kxc/runtime/task_plan.h"

#include <algorithm>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "kxc/support/object_registration.h"

namespace kxc::runtime {

KXC_OBJECT_DEFINE_WITH_KEY(SelectedArtifactBindingNode,
                           "kxc.runtime.SelectedArtifactBindingNode")
KXC_OBJECT_DEFINE_WITH_KEY(SelectedArtifactManifestNode,
                           "kxc.runtime.SelectedArtifactManifestNode")
KXC_OBJECT_DEFINE_WITH_KEY(PlanVariantNode, "kxc.runtime.PlanVariantNode")
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

std::string Text(const String& value) { return std::string(value); }

void AppendField(std::string* out, const std::string& name,
                 const std::string& value) {
    *out += std::to_string(name.size()) + ":" + name + "=" +
            std::to_string(value.size()) + ":" + value + ";";
}

std::string Hex(const std::string& value) {
    static constexpr char kDigits[] = "0123456789abcdef";
    std::string result;
    result.reserve(value.size() * 2);
    for (unsigned char byte : value) {
        result.push_back(kDigits[byte >> 4]);
        result.push_back(kDigits[byte & 0x0f]);
    }
    return result;
}

String Fingerprint(const char* kind, const std::string& canonical) {
    return String(std::string(kind) + ":" + Hex(canonical));
}

void AppendIds(std::string* out, const std::string& name,
               const Array<int64_t>& ids, bool ordered = true) {
    std::vector<int64_t> values(ids.begin(), ids.end());
    if (!ordered) std::sort(values.begin(), values.end());
    for (int64_t value : values) {
        AppendField(out, name, std::to_string(value));
    }
}

std::string ValueCanonical(const ValueSpec& value) {
    std::string result;
    AppendField(&result, "dtype_code", std::to_string(value->dtype.code));
    AppendField(&result, "dtype_bits", std::to_string(value->dtype.bits));
    AppendField(&result, "dtype_lanes", std::to_string(value->dtype.lanes));
    AppendField(&result, "device", value->device.ToString());
    for (int64_t dimension : value.shape()) {
        AppendField(&result, "dimension", std::to_string(dimension));
    }
    AppendField(&result, "is_input", value->is_input ? "1" : "0");
    AppendField(&result, "is_constant", value->is_constant ? "1" : "0");
    AppendField(&result, "is_output", value->is_output ? "1" : "0");
    AppendField(&result, "is_alias", value->is_alias ? "1" : "0");
    AppendField(&result, "is_async_live", value->is_async_live ? "1" : "0");
    return result;
}

std::unordered_map<int64_t, ValueSpec> ValueIndex(
    const Array<ValueSpec>& values) {
    std::unordered_map<int64_t, ValueSpec> result;
    for (const auto& value : values) result.emplace(value->value_id, value);
    return result;
}

std::string ArrayContract(const NDArray& value) {
    std::string result;
    AppendField(&result, "dtype_code", std::to_string(value.dtype().code));
    AppendField(&result, "dtype_bits", std::to_string(value.dtype().bits));
    AppendField(&result, "dtype_lanes", std::to_string(value.dtype().lanes));
    AppendField(&result, "device", value.device().ToString());
    for (int64_t dimension : value.shape()) {
        AppendField(&result, "dimension", std::to_string(dimension));
    }
    AppendField(&result, "bytes", std::to_string(value.NBytes()));
    AppendField(&result, "byte_offset",
                std::to_string(value->byte_offset));
    AppendField(&result, "storage_alignment",
                std::to_string(value.storage()->alignment));
    AppendField(&result, "contiguous", value.IsContiguous() ? "1" : "0");
    return result;
}

std::string ModuleEntryCanonical(
    const api::CompiledModule& module, const String& symbol,
    const Array<int64_t>& inputs, const Array<int64_t>& outputs,
    const std::unordered_map<int64_t, ValueSpec>& values,
    const std::unordered_map<int64_t, uint64_t>& output_alignments) {
    if (!module.defined() || !module.IsReady() || !module.HasFunction(symbol)) {
        throw std::invalid_argument(
            "exact ABI fingerprint requires a ready matching module entry");
    }
    const codegen::KernelSignature signature = module.signature(symbol);
    const codegen::KernelLaunchMetadata launch =
        module.launch_metadata(symbol);
    signature.Validate();
    launch.Validate();

    std::string result;
    AppendField(&result, "entry_symbol", Text(symbol));
    AppendField(&result, "signature", signature.ToString());
    AppendField(&result, "module_invocation_abi",
                module.invocation_contract(symbol).CanonicalBytes());
    AppendField(&result, "launch", launch.ToString());
    for (int64_t value_id : inputs) {
        const auto found = values.find(value_id);
        if (found == values.end()) {
            throw std::invalid_argument(
                "exact ABI fingerprint input references an unknown value");
        }
        AppendField(&result, "input", ValueCanonical(found->second));
    }
    for (int64_t value_id : outputs) {
        const auto found = values.find(value_id);
        if (found == values.end()) {
            throw std::invalid_argument(
                "exact ABI fingerprint output references an unknown value");
        }
        AppendField(&result, "output", ValueCanonical(found->second));
        const auto alignment = output_alignments.find(value_id);
        if (alignment != output_alignments.end()) {
            AppendField(&result, "allocation_alignment",
                        std::to_string(alignment->second));
        }
    }
    const Map<String, NDArray> constants = module.constants();
    for (const auto& argument : signature.arguments()) {
        if (argument->role != codegen::KernelArgRole::kConstant) continue;
        if (!constants.count(argument->constant_key)) {
            throw std::invalid_argument(
                "exact ABI fingerprint constant has no module binding");
        }
        AppendField(&result, "constant_key", Text(argument->constant_key));
        AppendField(&result, "constant_contract",
                    ArrayContract(constants.at(argument->constant_key)));
    }
    return result;
}

std::string BindingCanonical(const SelectedArtifactBinding& binding) {
    std::string result;
    AppendField(&result, "binding_kind",
                std::to_string(static_cast<int32_t>(binding->kind)));
    AppendField(&result, "invocation_id",
                std::to_string(binding->invocation_id));
    AppendField(&result, "artifact_identity", Text(binding->artifact_identity));
    AppendField(&result, "generation", std::to_string(binding->generation));
    AppendField(&result, "exact_abi", Text(binding->exact_abi_fingerprint));
    AppendField(&result, "entry_symbol", Text(binding->entry_symbol));
    AppendField(&result, "entry_binding", Text(binding->entry_binding));
    return result;
}

void AppendBindings(std::string* out,
                    const Array<SelectedArtifactBinding>& bindings) {
    std::vector<SelectedArtifactBinding> ordered(bindings.begin(),
                                                  bindings.end());
    std::sort(ordered.begin(), ordered.end(),
              [](const SelectedArtifactBinding& lhs,
                 const SelectedArtifactBinding& rhs) {
                  if (lhs->kind != rhs->kind) {
                      return static_cast<int32_t>(lhs->kind) <
                             static_cast<int32_t>(rhs->kind);
                  }
                  return lhs->invocation_id < rhs->invocation_id;
              });
    for (const auto& binding : ordered) {
        AppendField(out, "selected_artifact", BindingCanonical(binding));
    }
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

bool IsDataProducer(TaskKind kind) { return kind == TaskKind::kKernel; }

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

bool IsSourceValue(const ValueSpec& value) {
    return value->is_input || value->is_constant;
}

bool CanShareStorage(const ValueSpec& value) {
    return !IsSourceValue(value) && !value->is_output && !value->is_alias &&
           !value->is_async_live;
}

std::set<int64_t> ToSet(const Array<int64_t>& values) {
    return std::set<int64_t>(values.begin(), values.end());
}

void ValidateRegionKind(RegionKind kind) {
    if (kind != RegionKind::kPerCall) {
        throw std::invalid_argument("RegionSpec kind is invalid");
    }
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
    if (kind != TaskKind::kKernel && kind != TaskKind::kAllocate) {
        throw std::invalid_argument("TaskSpec kind is invalid");
    }
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

SelectedArtifactBinding::SelectedArtifactBinding(
    ArtifactBindingKind kind, int64_t invocation_id, String artifact_identity,
    uint64_t generation, String exact_abi_fingerprint, String entry_symbol,
    String entry_binding) {
    auto* node = new SelectedArtifactBindingNode();
    node->kind = kind;
    node->invocation_id = invocation_id;
    node->artifact_identity = std::move(artifact_identity);
    node->generation = generation;
    node->exact_abi_fingerprint = std::move(exact_abi_fingerprint);
    node->entry_symbol = std::move(entry_symbol);
    node->entry_binding = std::move(entry_binding);
    SetData(node);
    Validate();
}

SelectedArtifactBinding::SelectedArtifactBinding(const ObjectRef& ref)
    : ObjectRef(ref) {
    if (defined() && !As<SelectedArtifactBindingNode>()) {
        SetData(nullptr);
        throw std::invalid_argument(
            "ObjectRef does not contain SelectedArtifactBindingNode");
    }
    if (defined()) Validate();
}

void SelectedArtifactBinding::Validate() const {
    const auto* node = operator->();
    if (node->kind != ArtifactBindingKind::kCall &&
        node->kind != ArtifactBindingKind::kTask) {
        throw std::invalid_argument(
            "artifact observability declaration binding kind is invalid");
    }
    if (node->invocation_id < 0 || Text(node->artifact_identity).empty() ||
        Text(node->exact_abi_fingerprint).empty() ||
        Text(node->entry_symbol).empty() || Text(node->entry_binding).empty()) {
        throw std::invalid_argument(
            "artifact observability declaration requires complete caller "
            "identity, ABI, and entry fields");
    }
}

const SelectedArtifactBindingNode* SelectedArtifactBinding::operator->() const {
    const auto* node = As<SelectedArtifactBindingNode>();
    if (!node) {
        throw std::runtime_error(
            "undefined or invalid SelectedArtifactBinding");
    }
    return node;
}

SelectedArtifactManifest::SelectedArtifactManifest(
    String plan_fingerprint, Array<SelectedArtifactBinding> bindings,
    std::shared_ptr<const void> retention_lease) {
    auto* node = new SelectedArtifactManifestNode();
    node->plan_fingerprint = std::move(plan_fingerprint);
    node->bindings_ = CopyArray(bindings);
    node->retention_lease_ = std::move(retention_lease);
    SetData(node);
    Validate();
}

SelectedArtifactManifest::SelectedArtifactManifest(const ObjectRef& ref)
    : ObjectRef(ref) {
    if (defined() && !As<SelectedArtifactManifestNode>()) {
        SetData(nullptr);
        throw std::invalid_argument(
            "ObjectRef does not contain SelectedArtifactManifestNode");
    }
    if (defined()) Validate();
}

Array<SelectedArtifactBinding> SelectedArtifactManifest::bindings() const {
    return CopyArray(operator->()->bindings_);
}

std::shared_ptr<const void> SelectedArtifactManifest::retention_lease() const {
    return operator->()->retention_lease_;
}

void SelectedArtifactManifest::Validate() const {
    const auto* node = operator->();
    if (node->version != kSelectedArtifactManifestVersion ||
        Text(node->plan_fingerprint).empty() || node->bindings_.empty()) {
        throw std::invalid_argument(
            "artifact observability declaration version, plan fingerprint, "
            "or bindings are invalid");
    }
    std::set<std::pair<int32_t, int64_t>> locators;
    for (const auto& binding : node->bindings_) {
        binding.Validate();
        if (!locators
                 .emplace(static_cast<int32_t>(binding->kind),
                          binding->invocation_id)
                 .second) {
            throw std::invalid_argument(
                "artifact observability declaration locators must be unique");
        }
    }
}

const SelectedArtifactManifestNode* SelectedArtifactManifest::operator->() const {
    const auto* node = As<SelectedArtifactManifestNode>();
    if (!node) {
        throw std::runtime_error(
            "undefined or invalid SelectedArtifactManifest");
    }
    return node;
}

PlanVariant::PlanVariant(ExecutablePlan plan,
                         SelectedArtifactManifest manifest) {
    auto* node = new PlanVariantNode();
    node->plan_ = std::move(plan);
    node->manifest_ = std::move(manifest);
    SetData(node);
    Validate();
}

PlanVariant::PlanVariant(const ObjectRef& ref) : ObjectRef(ref) {
    if (defined() && !As<PlanVariantNode>()) {
        SetData(nullptr);
        throw std::invalid_argument("ObjectRef does not contain PlanVariantNode");
    }
    if (defined()) Validate();
}

ExecutablePlan PlanVariant::plan() const { return operator->()->plan_; }

SelectedArtifactManifest PlanVariant::manifest() const {
    return operator->()->manifest_;
}

void PlanVariant::Validate() const {
    const auto* node = operator->();
    if (!node->plan_.defined() || !node->manifest_.defined()) {
        throw std::invalid_argument(
            "PlanVariant requires an executable plan and artifact observability declaration");
    }
    node->plan_.Validate();
    node->manifest_.Validate();
    const Array<KernelCall> calls = node->plan_.calls();
    const Array<SelectedArtifactBinding> bindings =
        node->manifest_.bindings();
    if (bindings.size() != calls.size()) {
        throw std::invalid_argument(
            "PlanVariant requires one declared artifact per ordered call");
    }
    std::unordered_map<int64_t, SelectedArtifactBinding> by_call;
    for (const auto& binding : bindings) {
        if (binding->kind != ArtifactBindingKind::kCall ||
            !by_call.emplace(binding->invocation_id, binding).second) {
            throw std::invalid_argument(
                "PlanVariant artifact declaration must contain unique call bindings");
        }
    }
    for (size_t index = 0; index < calls.size(); ++index) {
        const auto found = by_call.find(static_cast<int64_t>(index));
        if (found == by_call.end() ||
            Text(found->second->entry_symbol) != Text(calls[index]->symbol)) {
            throw std::invalid_argument(
                "PlanVariant call and manifest entry bindings do not match");
        }
    }
    if (Text(node->manifest_->plan_fingerprint) !=
        Text(ComputePlanVariantFingerprint(node->plan_, bindings))) {
        throw std::invalid_argument("PlanVariant plan fingerprint does not match");
    }
}

const PlanVariantNode* PlanVariant::operator->() const {
    const auto* node = As<PlanVariantNode>();
    if (!node) throw std::runtime_error("undefined or invalid PlanVariant");
    return node;
}

String ComputeEntryBindingFingerprint(
    ArtifactBindingKind kind, int64_t invocation_id,
    const String& artifact_identity, uint64_t generation,
    const String& exact_abi_fingerprint, const String& entry_symbol) {
    if ((kind != ArtifactBindingKind::kCall &&
         kind != ArtifactBindingKind::kTask) ||
        invocation_id < 0 || Text(artifact_identity).empty() ||
        Text(exact_abi_fingerprint).empty() || Text(entry_symbol).empty()) {
        throw std::invalid_argument(
            "entry binding fingerprint requires complete binding fields");
    }
    std::string canonical;
    AppendField(&canonical, "kind", "runtime-entry-binding-v1");
    AppendField(&canonical, "binding_kind",
                std::to_string(static_cast<int32_t>(kind)));
    AppendField(&canonical, "invocation_id", std::to_string(invocation_id));
    AppendField(&canonical, "artifact_identity", Text(artifact_identity));
    AppendField(&canonical, "generation", std::to_string(generation));
    AppendField(&canonical, "exact_abi", Text(exact_abi_fingerprint));
    AppendField(&canonical, "entry_symbol", Text(entry_symbol));
    return Fingerprint("runtime-entry-binding-v1", canonical);
}

String ComputeCallExactAbiFingerprint(const api::CompiledModule& module,
                                      const ExecutablePlan& plan,
                                      int64_t call_index) {
    if (!plan.defined()) {
        throw std::invalid_argument(
            "call ABI fingerprint requires an executable plan");
    }
    plan.Validate();
    const Array<KernelCall> calls = plan.calls();
    if (call_index < 0 || static_cast<size_t>(call_index) >= calls.size()) {
        throw std::invalid_argument("call ABI fingerprint index is invalid");
    }
    const KernelCall& call = calls[static_cast<size_t>(call_index)];
    std::string canonical;
    AppendField(&canonical, "kind", "runtime-call-exact-abi-v1");
    AppendField(&canonical, "contract",
                ModuleEntryCanonical(module, call->symbol,
                                     call.input_value_ids(),
                                     call.output_value_ids(),
                                     ValueIndex(plan.values()), {}));
    return Fingerprint("runtime-call-exact-abi-v1", canonical);
}

String ComputeTaskExactAbiFingerprint(const api::CompiledModule& module,
                                      const FrozenTaskPlan& plan,
                                      int64_t task_id) {
    if (!plan.defined()) {
        throw std::invalid_argument(
            "task ABI fingerprint requires a frozen task plan");
    }
    plan.Validate();
    TaskSpec selected{ObjectRef()};
    std::unordered_map<int64_t, uint64_t> alignments;
    for (const auto& task : plan.tasks()) {
        if (task->kind == TaskKind::kAllocate) {
            alignments.emplace(task.output_value_ids()[0], task->alignment);
        }
        if (task->task_id == task_id) selected = task;
    }
    if (!selected.defined() || selected->kind != TaskKind::kKernel) {
        throw std::invalid_argument(
            "task ABI fingerprint requires a known kernel task");
    }
    std::string canonical;
    AppendField(&canonical, "kind", "runtime-task-exact-abi-v1");
    AppendField(&canonical, "contract",
                ModuleEntryCanonical(module, selected->symbol,
                                     selected.input_value_ids(),
                                     selected.output_value_ids(),
                                     ValueIndex(plan.values()), alignments));
    return Fingerprint("runtime-task-exact-abi-v1", canonical);
}

String ComputePlanVariantFingerprint(
    const ExecutablePlan& plan,
    const Array<SelectedArtifactBinding>& bindings) {
    if (!plan.defined()) {
        throw std::invalid_argument(
            "plan fingerprint requires an executable plan");
    }
    std::string canonical;
    AppendField(&canonical, "kind", "runtime-plan-variant-v1");
    const Array<ValueSpec> plan_values = plan.values();
    std::vector<ValueSpec> values(plan_values.begin(), plan_values.end());
    std::sort(values.begin(), values.end(),
              [](const ValueSpec& lhs, const ValueSpec& rhs) {
                  return lhs->value_id < rhs->value_id;
              });
    for (const auto& value : values) {
        AppendField(&canonical, "value", ValueCanonical(value));
    }
    const Array<KernelCall> calls = plan.calls();
    for (size_t index = 0; index < calls.size(); ++index) {
        std::string call;
        AppendField(&call, "call_index", std::to_string(index));
        AppendField(&call, "entry_symbol", Text(calls[index]->symbol));
        AppendIds(&call, "input_value_id", calls[index].input_value_ids());
        AppendIds(&call, "output_value_id", calls[index].output_value_ids());
        AppendField(&canonical, "call", call);
    }
    AppendIds(&canonical, "graph_input", plan.input_value_ids());
    AppendIds(&canonical, "graph_constant", plan.constant_value_ids());
    AppendIds(&canonical, "graph_output", plan.output_value_ids());
    AppendBindings(&canonical, bindings);
    return Fingerprint("runtime-plan-variant-v1", canonical);
}

String ComputeFrozenTaskPlanFingerprint(
    const FrozenTaskPlan& plan,
    const Array<SelectedArtifactBinding>& bindings) {
    if (!plan.defined()) {
        throw std::invalid_argument(
            "task plan fingerprint requires a frozen task plan");
    }
    std::string canonical;
    AppendField(&canonical, "kind", "runtime-frozen-task-plan-v1");
    AppendField(&canonical, "version", std::to_string(plan->version));
    const Array<ValueSpec> plan_values = plan.values();
    std::vector<ValueSpec> values(plan_values.begin(), plan_values.end());
    std::sort(values.begin(), values.end(),
              [](const ValueSpec& lhs, const ValueSpec& rhs) {
                  return lhs->value_id < rhs->value_id;
              });
    for (const auto& value : values) {
        AppendField(&canonical, "value", ValueCanonical(value));
    }
    const Array<TaskSpec> plan_tasks = plan.tasks();
    std::vector<TaskSpec> tasks(plan_tasks.begin(), plan_tasks.end());
    std::sort(tasks.begin(), tasks.end(),
              [](const TaskSpec& lhs, const TaskSpec& rhs) {
                  return lhs->task_id < rhs->task_id;
              });
    for (const auto& task : tasks) {
        std::string item;
        AppendField(&item, "task_id", std::to_string(task->task_id));
        AppendField(&item, "kind",
                    std::to_string(static_cast<int32_t>(task->kind)));
        AppendField(&item, "device", task->device.ToString());
        AppendField(&item, "stream_id", std::to_string(task->stream_id));
        AppendField(&item, "symbol", Text(task->symbol));
        AppendField(&item, "generation",
                    std::to_string(task->artifact_generation));
        AppendField(&item, "alignment", std::to_string(task->alignment));
        AppendIds(&item, "input_value_id", task.input_value_ids());
        AppendIds(&item, "output_value_id", task.output_value_ids());
        AppendIds(&item, "dependency_task_id",
                  task.dependency_task_ids(), false);
        AppendField(&canonical, "task", item);
    }
    const Array<RegionSpec> plan_regions = plan.regions();
    std::vector<RegionSpec> regions(plan_regions.begin(), plan_regions.end());
    std::sort(regions.begin(), regions.end(),
              [](const RegionSpec& lhs, const RegionSpec& rhs) {
                  return lhs->region_id < rhs->region_id;
              });
    for (const auto& region : regions) {
        std::string item;
        AppendField(&item, "region_id", std::to_string(region->region_id));
        AppendField(&item, "kind",
                    std::to_string(static_cast<int32_t>(region->kind)));
        AppendField(&item, "semantic_key", Text(region->semantic_key));
        AppendField(&item, "effect",
                    std::to_string(static_cast<int32_t>(region->effect)));
        AppendField(&item, "alias",
                    std::to_string(static_cast<int32_t>(region->alias)));
        AppendIds(&item, "task_id", region.task_ids(), false);
        AppendIds(&item, "live_in", region.live_in_value_ids(), false);
        AppendIds(&item, "live_out", region.live_out_value_ids(), false);
        AppendIds(&item, "constant", region.constant_value_ids(), false);
        AppendField(&canonical, "region", item);
    }
    AppendIds(&canonical, "graph_input", plan.input_value_ids());
    AppendIds(&canonical, "graph_constant", plan.constant_value_ids());
    AppendIds(&canonical, "graph_output", plan.output_value_ids());
    AppendBindings(&canonical, bindings);
    return Fingerprint("runtime-frozen-task-plan-v1", canonical);
}

PlanVariant MakePlanVariant(
    const api::CompiledModule& module, const ExecutablePlan& plan,
    const Array<ArtifactSelection>& selections,
    std::shared_ptr<const void> retention_lease) {
    if (!plan.defined()) {
        throw std::invalid_argument("MakePlanVariant requires a plan");
    }
    plan.Validate();
    const Array<KernelCall> calls = plan.calls();
    if (selections.size() != calls.size()) {
        throw std::invalid_argument(
            "MakePlanVariant requires one caller declaration per call");
    }
    std::unordered_map<int64_t, ArtifactSelection> by_call;
    for (const auto& selection : selections) {
        if (selection.invocation_id < 0 ||
            Text(selection.artifact_identity).empty() ||
            !by_call.emplace(selection.invocation_id, selection).second) {
            throw std::invalid_argument(
                "MakePlanVariant caller declarations require unique complete call identities");
        }
    }
    Array<SelectedArtifactBinding> bindings;
    for (size_t index = 0; index < calls.size(); ++index) {
        const int64_t call_id = static_cast<int64_t>(index);
        const auto selected = by_call.find(call_id);
        if (selected == by_call.end()) {
            throw std::invalid_argument(
                "MakePlanVariant caller declaration omits an ordered call");
        }
        const String abi =
            ComputeCallExactAbiFingerprint(module, plan, call_id);
        const String entry = calls[index]->symbol;
        bindings.push_back(SelectedArtifactBinding(
            ArtifactBindingKind::kCall, call_id,
            selected->second.artifact_identity, selected->second.generation,
            abi, entry,
            ComputeEntryBindingFingerprint(
                ArtifactBindingKind::kCall, call_id,
                selected->second.artifact_identity,
                selected->second.generation, abi, entry)));
    }
    const String fingerprint =
        ComputePlanVariantFingerprint(plan, bindings);
    return PlanVariant(
        plan, SelectedArtifactManifest(fingerprint, std::move(bindings),
                                       std::move(retention_lease)));
}

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
        case TaskKind::kAllocate:
            if (has_symbol || !node->input_value_ids_.empty() ||
                node->output_value_ids_.size() != 1 || node->alignment == 0 ||
                (node->alignment & (node->alignment - 1)) != 0 ||
                node->artifact_generation != 0) {
                throw std::invalid_argument(
                    "Allocate task requires one output and power-of-two alignment");
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
                               Array<int64_t> output_value_ids,
                               SelectedArtifactManifest manifest) {
    auto* node = new FrozenTaskPlanNode();
    node->version = version;
    node->values_ = CopyArray(values);
    node->tasks_ = CopyArray(tasks);
    node->regions_ = CopyArray(regions);
    node->input_value_ids_ = CopyArray(input_value_ids);
    node->constant_value_ids_ = CopyArray(constant_value_ids);
    node->output_value_ids_ = CopyArray(output_value_ids);
    node->manifest_ = std::move(manifest);
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
SelectedArtifactManifest FrozenTaskPlan::manifest() const {
    return operator->()->manifest_;
}
FrozenTaskPlan FrozenTaskPlan::WithManifest(
    SelectedArtifactManifest manifest) const {
    if (!manifest.defined()) {
        throw std::invalid_argument(
            "FrozenTaskPlan manifest must be defined");
    }
    return FrozenTaskPlan(operator->()->version, values(), tasks(), regions(),
                          input_value_ids(), constant_value_ids(),
                          output_value_ids(), std::move(manifest));
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
        for (int64_t dimension : value.shape()) {
            if (dimension < 0) {
                throw std::invalid_argument(
                    "FrozenTaskPlan v1 requires exact static shapes");
            }
        }
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
        if (region->effect == RegionEffect::kOrdered) {
            std::vector<int64_t> ordered_actions;
            for (int64_t task_id : region_tasks) {
                if (tasks_by_id.at(task_id)->kind != TaskKind::kAllocate) {
                    ordered_actions.push_back(task_id);
                }
            }
            for (size_t i = 0; i < ordered_actions.size(); ++i) {
                for (size_t j = i + 1; j < ordered_actions.size(); ++j) {
                    if (!HappensBefore(graph, ordered_actions[i],
                                       ordered_actions[j]) &&
                        !HappensBefore(graph, ordered_actions[j],
                                       ordered_actions[i])) {
                        throw std::invalid_argument(
                            "Ordered region actions require explicit dependencies");
                    }
                }
            }
        }
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

    if (node->manifest_.defined()) {
        node->manifest_.Validate();
        const Array<SelectedArtifactBinding> bindings =
            node->manifest_.bindings();
        size_t kernel_count = 0;
        std::unordered_map<int64_t, SelectedArtifactBinding> by_task;
        for (const auto& binding : bindings) {
            if (binding->kind != ArtifactBindingKind::kTask ||
                !by_task.emplace(binding->invocation_id, binding).second) {
                throw std::invalid_argument(
                    "FrozenTaskPlan artifact declaration must contain unique task bindings");
            }
        }
        for (const auto& task : node->tasks_) {
            if (task->kind != TaskKind::kKernel) continue;
            ++kernel_count;
            const auto binding = by_task.find(task->task_id);
            if (binding == by_task.end() ||
                binding->second->generation !=
                    static_cast<uint64_t>(task->artifact_generation) ||
                Text(binding->second->entry_symbol) != Text(task->symbol)) {
                throw std::invalid_argument(
                    "FrozenTaskPlan kernel and artifact binding do not match");
            }
        }
        if (kernel_count != bindings.size()) {
            throw std::invalid_argument(
                "FrozenTaskPlan requires one declared artifact per kernel task");
        }
        if (Text(node->manifest_->plan_fingerprint) !=
            Text(ComputeFrozenTaskPlanFingerprint(*this, bindings))) {
            throw std::invalid_argument(
                "FrozenTaskPlan plan fingerprint does not match");
        }
    }
}

const FrozenTaskPlanNode* FrozenTaskPlan::operator->() const {
    const auto* node = As<FrozenTaskPlanNode>();
    if (!node) throw std::runtime_error("undefined or invalid FrozenTaskPlan");
    return node;
}

FrozenTaskPlan AttachSelectedArtifacts(
    const api::CompiledModule& module, const FrozenTaskPlan& plan,
    const Array<ArtifactSelection>& selections,
    std::shared_ptr<const void> retention_lease) {
    if (!plan.defined()) {
        throw std::invalid_argument(
            "AttachSelectedArtifacts requires a frozen task plan");
    }
    plan.Validate();
    std::unordered_map<int64_t, TaskSpec> kernels;
    for (const auto& task : plan.tasks()) {
        if (task->kind == TaskKind::kKernel) {
            kernels.emplace(task->task_id, task);
        }
    }
    if (selections.size() != kernels.size()) {
        throw std::invalid_argument(
            "AttachSelectedArtifacts requires one caller declaration per kernel task");
    }
    std::unordered_map<int64_t, ArtifactSelection> by_task;
    for (const auto& selection : selections) {
        if (selection.invocation_id < 0 ||
            Text(selection.artifact_identity).empty() ||
            !by_task.emplace(selection.invocation_id, selection).second) {
            throw std::invalid_argument(
                "task artifact declarations require unique complete caller identities");
        }
    }
    Array<SelectedArtifactBinding> bindings;
    std::vector<int64_t> task_ids;
    task_ids.reserve(kernels.size());
    for (const auto& item : kernels) task_ids.push_back(item.first);
    std::sort(task_ids.begin(), task_ids.end());
    for (int64_t task_id : task_ids) {
        const auto selection = by_task.find(task_id);
        if (selection == by_task.end()) {
            throw std::invalid_argument(
                "task artifact declaration omits a kernel task");
        }
        const TaskSpec& task = kernels.at(task_id);
        if (selection->second.generation !=
            static_cast<uint64_t>(task->artifact_generation)) {
            throw std::invalid_argument(
                "task artifact declaration generation does not match the task");
        }
        const String abi =
            ComputeTaskExactAbiFingerprint(module, plan, task_id);
        bindings.push_back(SelectedArtifactBinding(
            ArtifactBindingKind::kTask, task_id,
            selection->second.artifact_identity,
            selection->second.generation, abi, task->symbol,
            ComputeEntryBindingFingerprint(
                ArtifactBindingKind::kTask, task_id,
                selection->second.artifact_identity,
                selection->second.generation, abi, task->symbol)));
    }
    const String fingerprint =
        ComputeFrozenTaskPlanFingerprint(plan, bindings);
    return plan.WithManifest(SelectedArtifactManifest(
        fingerprint, std::move(bindings), std::move(retention_lease)));
}

}  // namespace kxc::runtime
