/*! \file src/runtime/session.cc
 * \brief Executes a validated multi-kernel graph plan on one device stream.
 */

#include "kxc/runtime/session.h"
#include "kxc/runtime/task_executor.h"
#include "kxc/support/object_registration.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "internal/compiled_module_node.h"
#include "internal/kernel_argument_validation.h"
#include "internal/session_node.h"
#include "internal/value_table.h"

namespace kxc::runtime {

KXC_OBJECT_DEFINE_WITH_KEY(RuntimeSessionNode, "kxc.runtime.RuntimeSessionNode")

namespace {

struct ValidatedPlanContract final {
    Device device;
    std::unordered_map<int64_t, String> constant_keys_by_value;
};

struct RuntimeExecutionState final {
    api::CompiledModule module;
    ExecutablePlan plan;
    FrozenTaskPlan task_plan;
    SelectedArtifactManifest manifest;
    std::shared_ptr<internal::ValueTable> values;
    Array<AsyncOperation> prior_operations;
};

template <typename T>
Array<T> CopyArray(const Array<T>& source) {
    Array<T> result;
    for (const auto& value : source) result.push_back(value);
    return result;
}

std::unordered_map<int64_t, ValueSpec> IndexValues(
    const ExecutablePlan& plan) {
    std::unordered_map<int64_t, ValueSpec> result;
    for (const auto& value : plan.values()) {
        result.emplace(value->value_id, value);
    }
    return result;
}

const ValueSpec& FindValue(
    const std::unordered_map<int64_t, ValueSpec>& values, int64_t value_id,
    const std::string& context) {
    const auto it = values.find(value_id);
    if (it == values.end()) {
        throw std::invalid_argument(context + " references unknown value " +
                                    std::to_string(value_id));
    }
    return it->second;
}

void ValidateShape(const Array<int64_t>& expected,
                   const Array<int64_t>& actual,
                   const std::string& context, bool allow_dynamic) {
    if (expected.size() != actual.size()) {
        throw std::invalid_argument(context + " rank does not match");
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (allow_dynamic && expected[i] == codegen::kDynamicDimension) continue;
        if (expected[i] != actual[i]) {
            throw std::invalid_argument(
                context + " shape does not match at axis " + std::to_string(i) +
                ": kernel expects " + std::to_string(expected[i]) +
                ", plan declares " + std::to_string(actual[i]));
        }
    }
}

void ValidateValueContract(const ValueSpec& value,
                           const codegen::KernelArgSpec& argument,
                           const std::string& context) {
    if (!api::SameDType(value->dtype, argument->dtype) ||
        value->device != argument->device) {
        throw std::invalid_argument(context + " dtype or device does not match");
    }
    ValidateShape(argument.shape(), value.shape(), context, false);
}

void ValidateRuntimeValue(const ValueSpec& value, const NDArray& array,
                          const std::string& context) {
    if (!array.defined()) {
        throw std::invalid_argument(context + " requires a defined NDArray");
    }
    if (!api::SameDType(value->dtype, array.dtype())) {
        throw std::invalid_argument(context + " dtype does not match");
    }
    if (value->device != array.device()) {
        throw std::invalid_argument(
            context + " device expected " + value->device.ToString() +
            ", actual " + array.device().ToString());
    }
    ValidateShape(value.shape(), array.shape(), context, value->is_input);
}

ValidatedPlanContract ValidateModuleAndPlan(
    const api::CompiledModule& module, const ExecutablePlan& plan) {
    if (!module.defined() || !module.IsReady()) {
        throw std::invalid_argument(
            "RuntimeSession requires a defined, ready CompiledModule");
    }
    if (!plan.defined()) {
        throw std::invalid_argument(
            "RuntimeSession requires a defined ExecutablePlan");
    }
    plan.Validate();
    const Map<String, NDArray>& module_constants =
        api::internal::BorrowCompiledModuleConstants(module);

    const Array<String> module_symbols = module.symbols();
    const Array<KernelCall> calls = plan.calls();
    if (module_symbols.empty() || module_symbols.size() != calls.size()) {
        throw std::invalid_argument(
            "RuntimeSession requires one plan call per module entry");
    }

    std::unordered_set<std::string> expected_symbols;
    for (const auto& symbol : module_symbols) {
        expected_symbols.insert(std::string(symbol));
    }

    const auto values = IndexValues(plan);
    const Device device = module.launch_metadata(module_symbols[0])->device;
    for (const auto& item : values) {
        if (item.second->device != device) {
            throw std::invalid_argument(
                "RuntimeSession v1 requires every plan value on one device");
        }
    }

    ValidatedPlanContract result;
    result.device = device;
    std::unordered_map<std::string, int64_t> value_by_constant_key;
    std::unordered_set<std::string> call_symbols;
    for (size_t call_index = 0; call_index < calls.size(); ++call_index) {
        const KernelCall& call = calls[call_index];
        const std::string symbol = std::string(call->symbol);
        const std::string context = "RuntimeSession call[" +
                                    std::to_string(call_index) + "] '" + symbol +
                                    "'";
        if (!module.HasFunction(call->symbol) ||
            expected_symbols.count(symbol) == 0 ||
            !call_symbols.insert(symbol).second) {
            throw std::invalid_argument(context +
                                        " does not match a unique module entry");
        }
        const codegen::KernelSignature signature = module.signature(call->symbol);
        const codegen::KernelLaunchMetadata metadata =
            module.launch_metadata(call->symbol);
        signature.Validate();
        metadata.Validate();
        if (metadata->device != device) {
            throw std::invalid_argument(
                context + " targets a different execution device");
        }

        Array<int64_t> regular_inputs;
        Array<int64_t> constant_inputs;
        for (int64_t value_id : call.input_value_ids()) {
            const ValueSpec& value = FindValue(values, value_id, context);
            (value->is_constant ? constant_inputs : regular_inputs)
                .push_back(value_id);
        }
        const Array<int64_t> outputs = call.output_value_ids();
        size_t regular_index = 0;
        size_t constant_index = 0;
        size_t output_index = 0;
        const Array<codegen::KernelArgSpec> arguments = signature.arguments();
        for (size_t argument_index = 0; argument_index < arguments.size();
             ++argument_index) {
            const auto& argument = arguments[argument_index];
            int64_t value_id = -1;
            switch (argument->role) {
                case codegen::KernelArgRole::kInput:
                    if (regular_index >= regular_inputs.size()) {
                        throw std::invalid_argument(
                            context + " has fewer plan inputs than its signature");
                    }
                    value_id = regular_inputs[regular_index++];
                    break;
                case codegen::KernelArgRole::kConstant:
                    if (constant_index >= constant_inputs.size()) {
                        throw std::invalid_argument(
                            context + " has fewer constants than its signature");
                    }
                    value_id = constant_inputs[constant_index++];
                    break;
                case codegen::KernelArgRole::kOutput:
                    if (output_index >= outputs.size()) {
                        throw std::invalid_argument(
                            context + " has fewer outputs than its signature");
                    }
                    value_id = outputs[output_index++];
                    break;
            }
            const ValueSpec& value = FindValue(values, value_id, context);
            ValidateValueContract(value, argument,
                                  context + " value " +
                                      std::to_string(value_id));
            if (argument->role == codegen::KernelArgRole::kConstant) {
                const std::string key = std::string(argument->constant_key);
                if (!module_constants.count(argument->constant_key)) {
                    throw std::invalid_argument(
                        context + " has no module constant binding");
                }
                api::ValidateKernelArgument(
                    signature, argument_index, argument,
                    module_constants.at(argument->constant_key),
                    module_constants);
                const auto existing = result.constant_keys_by_value.find(value_id);
                if (existing != result.constant_keys_by_value.end() &&
                    std::string(existing->second) != key) {
                    throw std::invalid_argument(
                        context + " maps one constant value to multiple keys");
                }
                const auto reverse = value_by_constant_key.emplace(key, value_id);
                if (!reverse.second && reverse.first->second != value_id) {
                    throw std::invalid_argument(
                        context + " maps one constant key to multiple values");
                }
                result.constant_keys_by_value[value_id] = argument->constant_key;
            }
        }
        if (regular_index != regular_inputs.size() ||
            constant_index != constant_inputs.size() ||
            output_index != outputs.size()) {
            throw std::invalid_argument(
                context + " plan arity does not match its signature");
        }
    }
    if (call_symbols != expected_symbols) {
        throw std::invalid_argument(
            "RuntimeSession plan and module symbols are not identical");
    }

    const Array<int64_t> constant_ids = plan.constant_value_ids();
    if (result.constant_keys_by_value.size() != constant_ids.size() ||
        module_constants.size() != constant_ids.size()) {
        throw std::invalid_argument(
            "RuntimeSession plan constants do not match the module constant pool");
    }
    for (int64_t value_id : constant_ids) {
        const auto key_it = result.constant_keys_by_value.find(value_id);
        if (key_it == result.constant_keys_by_value.end() ||
            !module_constants.count(key_it->second)) {
            throw std::invalid_argument(
                "RuntimeSession constant value has no module binding");
        }
        ValidateRuntimeValue(
            FindValue(values, value_id, "RuntimeSession constant"),
            module_constants.at(key_it->second),
            "RuntimeSession constant value " + std::to_string(value_id));
    }
    return result;
}

void ValidatePlanManifest(const api::CompiledModule& module,
                          const PlanVariant& variant) {
    variant.Validate();
    const ExecutablePlan plan = variant.plan();
    const Array<KernelCall> calls = plan.calls();
    for (const auto& binding : variant.manifest().bindings()) {
        const std::string context =
            "RuntimeSession call artifact declaration[" +
            std::to_string(binding->invocation_id) + "]";
        if (binding->generation != 0) {
            throw std::invalid_argument(
                context + " requires unsupported non-zero generation");
        }
        const size_t call_index =
            static_cast<size_t>(binding->invocation_id);
        if (call_index >= calls.size() ||
            std::string(binding->entry_symbol) !=
                std::string(calls[call_index]->symbol)) {
            throw std::invalid_argument(
                context + " entry binding does not match the ordered call");
        }
        const String abi = ComputeCallExactAbiFingerprint(
            module, plan, binding->invocation_id);
        if (std::string(binding->exact_abi_fingerprint) != std::string(abi)) {
            throw std::invalid_argument(
                context + " exact ABI fingerprint does not match the module");
        }
        const String entry_binding = ComputeEntryBindingFingerprint(
            ArtifactBindingKind::kCall, binding->invocation_id,
            binding->artifact_identity, binding->generation, abi,
            binding->entry_symbol);
        if (std::string(binding->entry_binding) !=
            std::string(entry_binding)) {
            throw std::invalid_argument(
                context + " canonical entry binding does not match");
        }
    }
}

void ValidateTaskManifest(const api::CompiledModule& module,
                          const FrozenTaskPlan& plan) {
    if (!plan.manifest().defined()) {
        throw std::invalid_argument(
            "RuntimeSession task DAG requires a trusted upper-plane artifact declaration");
    }
    for (const auto& binding : plan.manifest().bindings()) {
        const std::string context =
            "RuntimeSession task artifact declaration[" +
            std::to_string(binding->invocation_id) + "]";
        if (binding->generation != 0) {
            throw std::invalid_argument(
                context + " requires unsupported non-zero generation");
        }
        const String abi = ComputeTaskExactAbiFingerprint(
            module, plan, binding->invocation_id);
        if (std::string(binding->exact_abi_fingerprint) != std::string(abi)) {
            throw std::invalid_argument(
                context + " exact ABI fingerprint does not match the module");
        }
        const String entry_binding = ComputeEntryBindingFingerprint(
            ArtifactBindingKind::kTask, binding->invocation_id,
            binding->artifact_identity, binding->generation, abi,
            binding->entry_symbol);
        if (std::string(binding->entry_binding) !=
            std::string(entry_binding)) {
            throw std::invalid_argument(
                context + " canonical entry binding does not match");
        }
    }
}

ValidatedPlanContract ValidateModuleAndTaskPlan(
    const api::CompiledModule& module, const FrozenTaskPlan& plan) {
    if (!module.defined() || !module.IsReady()) {
        throw std::invalid_argument(
            "RuntimeSession requires a defined, ready CompiledModule");
    }
    if (!plan.defined()) {
        throw std::invalid_argument(
            "RuntimeSession requires a defined FrozenTaskPlan");
    }
    plan.Validate();
    const Map<String, NDArray>& module_constants =
        api::internal::BorrowCompiledModuleConstants(module);
    for (const auto& region : plan.regions()) {
        switch (region->kind) {
            case RegionKind::kPerCall:
                break;
            case RegionKind::kFusion:
                throw std::invalid_argument(
                    "RuntimeSession task DAG does not execute fusion regions "
                    "without a fusion execution contract");
            case RegionKind::kLibrary:
                throw std::invalid_argument(
                    "RuntimeSession task DAG requires a library descriptor, "
                    "workspace, stream, and error ABI");
            case RegionKind::kControlFlow:
                throw std::invalid_argument(
                    "RuntimeSession task DAG does not execute control-flow regions");
        }
    }

    const Array<String> module_symbols = module.symbols();
    if (module_symbols.empty()) {
        throw std::invalid_argument(
            "RuntimeSession task DAG requires module entries");
    }
    std::unordered_set<std::string> expected_symbols;
    for (const auto& symbol : module_symbols) {
        expected_symbols.insert(std::string(symbol));
    }

    const auto values = [&] {
        std::unordered_map<int64_t, ValueSpec> result;
        for (const auto& value : plan.values()) {
            result.emplace(value->value_id, value);
        }
        return result;
    }();
    const Device device = module.launch_metadata(module_symbols[0])->device;
    for (const auto& item : values) {
        if (item.second->device != device) {
            throw std::invalid_argument(
                "RuntimeSession task DAG values must match the module device");
        }
    }

    ValidatedPlanContract result;
    result.device = device;
    std::unordered_map<std::string, int64_t> value_by_constant_key;
    std::unordered_map<int64_t, uint64_t> allocation_alignment;
    for (const auto& task : plan.tasks()) {
        if (task->kind == TaskKind::kAllocate) {
            allocation_alignment.emplace(task.output_value_ids()[0],
                                         task->alignment);
        }
    }
    size_t kernel_count = 0;
    for (const auto& task : plan.tasks()) {
        if (task->kind == TaskKind::kShapeEval) {
            throw std::invalid_argument(
                "RuntimeSession task DAG requires shape-eval integration");
        }
        if (task->kind != TaskKind::kKernel) continue;
        ++kernel_count;
        const std::string symbol = std::string(task->symbol);
        const std::string context =
            "RuntimeSession task[" + std::to_string(task->task_id) +
            "] '" + symbol + "'";
        if (task->artifact_generation != 0) {
            throw std::invalid_argument(
                context + " requires an unsupported artifact generation");
        }
        if (!module.HasFunction(task->symbol) ||
            expected_symbols.count(symbol) == 0) {
            throw std::invalid_argument(
                context + " does not match a module entry");
        }
        const codegen::KernelSignature signature = module.signature(task->symbol);
        const codegen::KernelLaunchMetadata metadata =
            module.launch_metadata(task->symbol);
        signature.Validate();
        metadata.Validate();
        if (metadata->device != device || task->device != device) {
            throw std::invalid_argument(
                context + " targets a different execution device");
        }

        Array<int64_t> regular_inputs;
        Array<int64_t> constant_inputs;
        for (int64_t value_id : task.input_value_ids()) {
            const ValueSpec& value = FindValue(values, value_id, context);
            (value->is_constant ? constant_inputs : regular_inputs)
                .push_back(value_id);
        }
        const Array<int64_t> outputs = task.output_value_ids();
        size_t regular_index = 0;
        size_t constant_index = 0;
        size_t output_index = 0;
        const Array<codegen::KernelArgSpec> arguments = signature.arguments();
        for (size_t argument_index = 0; argument_index < arguments.size();
             ++argument_index) {
            const auto& argument = arguments[argument_index];
            int64_t value_id = -1;
            switch (argument->role) {
                case codegen::KernelArgRole::kInput:
                    if (regular_index >= regular_inputs.size()) {
                        throw std::invalid_argument(
                            context + " has fewer inputs than its signature");
                    }
                    value_id = regular_inputs[regular_index++];
                    break;
                case codegen::KernelArgRole::kConstant:
                    if (constant_index >= constant_inputs.size()) {
                        throw std::invalid_argument(
                            context + " has fewer constants than its signature");
                    }
                    value_id = constant_inputs[constant_index++];
                    break;
                case codegen::KernelArgRole::kOutput:
                    if (output_index >= outputs.size()) {
                        throw std::invalid_argument(
                            context + " has fewer outputs than its signature");
                    }
                    value_id = outputs[output_index++];
                    break;
            }
            const ValueSpec& value = FindValue(values, value_id, context);
            ValidateValueContract(
                value, argument,
                context + " value " + std::to_string(value_id));
            if (argument->role == codegen::KernelArgRole::kOutput &&
                allocation_alignment.at(value_id) < argument->alignment) {
                throw std::invalid_argument(
                    context + " output allocation alignment is insufficient");
            }
            if (argument->role == codegen::KernelArgRole::kConstant) {
                const std::string key = std::string(argument->constant_key);
                if (!module_constants.count(argument->constant_key)) {
                    throw std::invalid_argument(
                        context + " has no module constant binding");
                }
                api::ValidateKernelArgument(
                    signature, argument_index, argument,
                    module_constants.at(argument->constant_key),
                    module_constants);
                const auto existing =
                    result.constant_keys_by_value.find(value_id);
                if (existing != result.constant_keys_by_value.end() &&
                    std::string(existing->second) != key) {
                    throw std::invalid_argument(
                        context + " maps one constant value to multiple keys");
                }
                const auto reverse = value_by_constant_key.emplace(key, value_id);
                if (!reverse.second && reverse.first->second != value_id) {
                    throw std::invalid_argument(
                        context + " maps one constant key to multiple values");
                }
                result.constant_keys_by_value[value_id] = argument->constant_key;
            }
        }
        if (regular_index != regular_inputs.size() ||
            constant_index != constant_inputs.size() ||
            output_index != outputs.size()) {
            throw std::invalid_argument(
                context + " arity does not match its signature");
        }
    }
    if (kernel_count == 0) {
        throw std::invalid_argument(
            "RuntimeSession task DAG requires at least one kernel task");
    }

    const Array<int64_t> constant_ids = plan.constant_value_ids();
    if (result.constant_keys_by_value.size() != constant_ids.size()) {
        throw std::invalid_argument(
            "RuntimeSession task constants do not match the module constant pool");
    }
    for (int64_t value_id : constant_ids) {
        const auto key = result.constant_keys_by_value.find(value_id);
        if (key == result.constant_keys_by_value.end() ||
            !module_constants.count(key->second)) {
            throw std::invalid_argument(
                "RuntimeSession task constant has no module binding");
        }
        ValidateRuntimeValue(
            FindValue(values, value_id, "RuntimeSession task constant"),
            module_constants.at(key->second),
            "RuntimeSession task constant value " + std::to_string(value_id));
    }
    ValidateTaskManifest(module, plan);
    return result;
}

void ValidateStoredSession(const RuntimeSessionNode& node) {
    ValidatedPlanContract checked;
    if (node.task_plan.defined()) {
        checked = ValidateModuleAndTaskPlan(node.module, node.task_plan);
    } else {
        checked = ValidateModuleAndPlan(node.module, node.plan);
        if (node.manifest.defined()) {
            ValidatePlanManifest(
                node.module, PlanVariant(node.plan, node.manifest));
        }
    }
    const bool selected_task_dag =
        node.selection.selected_mode == RuntimeExecutionMode::kTaskDAG;
    if (selected_task_dag != node.task_plan.defined() ||
        checked.device != node.device ||
        checked.constant_keys_by_value.size() !=
            node.constant_keys_by_value.size()) {
        throw std::invalid_argument(
            "RuntimeSession stored plan, selection, or metadata is inconsistent");
    }
    for (const auto& item : checked.constant_keys_by_value) {
        const auto stored = node.constant_keys_by_value.find(item.first);
        if (stored == node.constant_keys_by_value.end() ||
            std::string(stored->second) != std::string(item.second)) {
            throw std::invalid_argument(
                "RuntimeSession stored constant mapping is inconsistent");
        }
    }
}

Array<NDArray> PrepareCallArguments(
    const api::CompiledModule& module, const KernelCall& call,
    const std::unordered_map<int64_t, ValueSpec>& values,
    const std::shared_ptr<internal::ValueTable>& table) {
    const codegen::KernelSignature signature = module.signature(call->symbol);
    Array<int64_t> regular_inputs;
    Array<int64_t> constant_inputs;
    for (int64_t value_id : call.input_value_ids()) {
        const ValueSpec& value =
            FindValue(values, value_id, "RuntimeSession execution");
        (value->is_constant ? constant_inputs : regular_inputs)
            .push_back(value_id);
    }

    const Array<int64_t> output_ids = call.output_value_ids();
    size_t regular_index = 0;
    size_t constant_index = 0;
    size_t output_index = 0;
    Array<NDArray> ordered;
    for (const auto& argument : signature.arguments()) {
        int64_t value_id = -1;
        switch (argument->role) {
            case codegen::KernelArgRole::kInput:
                value_id = regular_inputs[regular_index++];
                break;
            case codegen::KernelArgRole::kConstant:
                value_id = constant_inputs[constant_index++];
                break;
            case codegen::KernelArgRole::kOutput: {
                value_id = output_ids[output_index++];
                if (table->Contains(value_id)) {
                    throw std::logic_error(
                        "RuntimeSession attempted to allocate a produced value twice");
                }
                const ValueSpec& value =
                    FindValue(values, value_id, "RuntimeSession output");
                table->Allocate(value, argument->alignment);
                break;
            }
        }
        ordered.push_back(table->Get(value_id));
    }
    return ordered;
}

void ValidateBoundSourceArguments(
    const api::CompiledModule& module, const ExecutablePlan& plan,
    const std::unordered_map<int64_t, ValueSpec>& values,
    const std::shared_ptr<internal::ValueTable>& table) {
    const Map<String, NDArray>& constants =
        api::internal::BorrowCompiledModuleConstants(module);
    for (const auto& call : plan.calls()) {
        const codegen::KernelSignature signature = module.signature(call->symbol);
        Array<int64_t> regular_inputs;
        Array<int64_t> constant_inputs;
        for (int64_t value_id : call.input_value_ids()) {
            const ValueSpec& value =
                FindValue(values, value_id, "RuntimeSession source preflight");
            (value->is_constant ? constant_inputs : regular_inputs)
                .push_back(value_id);
        }
        size_t regular_index = 0;
        size_t constant_index = 0;
        const Array<codegen::KernelArgSpec> arguments = signature.arguments();
        for (size_t argument_index = 0; argument_index < arguments.size();
             ++argument_index) {
            const auto& argument = arguments[argument_index];
            if (argument->role == codegen::KernelArgRole::kOutput) continue;
            const int64_t value_id =
                argument->role == codegen::KernelArgRole::kConstant
                    ? constant_inputs[constant_index++]
                    : regular_inputs[regular_index++];
            if (!table->Contains(value_id)) continue;
            api::ValidateKernelArgument(signature, argument_index, argument,
                                        table->Get(value_id), constants);
        }
    }
}

FrozenTaskPlan BuildPerCallTaskPlan(const api::CompiledModule& module,
                                    const ExecutablePlan& plan) {
    const auto values = IndexValues(plan);
    std::unordered_map<int64_t, size_t> consumers;
    for (const auto& call : plan.calls()) {
        std::unordered_set<int64_t> seen;
        for (int64_t input : call.input_value_ids()) {
            if (seen.insert(input).second) ++consumers[input];
        }
    }
    std::unordered_set<int64_t> graph_outputs;
    for (int64_t output : plan.output_value_ids()) graph_outputs.insert(output);

    Array<ValueSpec> task_values;
    for (const auto& value : plan.values()) {
        task_values.push_back(ValueSpec(
            value->value_id, value->value_id, value.shape(), value->dtype,
            value->device, value->is_input, value->is_constant,
            value->is_output, value->is_alias, value->is_async_live));
    }

    Array<TaskSpec> tasks;
    Array<RegionSpec> regions;
    int64_t next_task_id = 0;
    int64_t previous_kernel = -1;
    int64_t region_id = 0;
    for (const auto& call : plan.calls()) {
        Array<int64_t> region_tasks;
        Array<int64_t> allocation_tasks;
        Array<int64_t> allocation_dependencies;
        if (previous_kernel >= 0) {
            allocation_dependencies.push_back(previous_kernel);
        }

        Array<uint64_t> output_alignments;
        for (const auto& argument : module.signature(call->symbol).arguments()) {
            if (argument->role == codegen::KernelArgRole::kOutput) {
                output_alignments.push_back(argument->alignment);
            }
        }
        const Array<int64_t> outputs = call.output_value_ids();
        for (size_t i = 0; i < outputs.size(); ++i) {
            const int64_t task_id = next_task_id++;
            tasks.push_back(TaskSpec(
                task_id, TaskKind::kAllocate, values.at(outputs[i])->device, {},
                {outputs[i]}, allocation_dependencies, String(), 0, 0,
                output_alignments[i]));
            allocation_tasks.push_back(task_id);
            region_tasks.push_back(task_id);
        }

        Array<int64_t> kernel_dependencies = allocation_tasks;
        if (previous_kernel >= 0) kernel_dependencies.push_back(previous_kernel);
        const int64_t kernel_task_id = next_task_id++;
        tasks.push_back(TaskSpec(
            kernel_task_id, TaskKind::kKernel,
            module.launch_metadata(call->symbol)->device,
            call.input_value_ids(), outputs, std::move(kernel_dependencies),
            call->symbol));
        region_tasks.push_back(kernel_task_id);

        Array<int64_t> live_ins;
        Array<int64_t> constants;
        std::unordered_set<int64_t> seen_live_ins;
        for (int64_t input : call.input_value_ids()) {
            if (!seen_live_ins.insert(input).second) continue;
            live_ins.push_back(input);
            if (values.at(input)->is_constant) constants.push_back(input);
        }
        Array<int64_t> live_outs;
        for (int64_t output : outputs) {
            if (consumers[output] != 0 || graph_outputs.count(output)) {
                live_outs.push_back(output);
            }
        }
        regions.push_back(RegionSpec(
            region_id++, RegionKind::kPerCall, String(),
            std::move(region_tasks), std::move(live_ins),
            std::move(live_outs), std::move(constants),
            RegionEffect::kOrdered, RegionAlias::kNoAlias));
        previous_kernel = kernel_task_id;
    }

    return PlanTaskMemory(FrozenTaskPlan(
        kFrozenTaskPlanVersion, std::move(task_values), std::move(tasks),
        std::move(regions), plan.input_value_ids(), plan.constant_value_ids(),
        plan.output_value_ids()));
}

struct FallbackDecision final {
    FallbackReason reason{FallbackReason::kNone};
    std::string diagnostic;
};

FallbackDecision InspectOrderedPlanForTaskDAG(const ExecutablePlan& plan) {
    for (const auto& value : plan.values()) {
        for (int64_t dimension : value.shape()) {
            if (dimension < 0) {
                return {FallbackReason::kUnsupportedDynamicInput,
                        "task DAG adapter does not support dynamic input value " +
                            std::to_string(value->value_id)};
            }
        }
    }
    for (const auto& value : plan.values()) {
        if (value->is_alias) {
            return {FallbackReason::kUnsupportedAlias,
                    "task DAG adapter does not support alias value " +
                        std::to_string(value->value_id)};
        }
    }
    return {};
}

FallbackDecision InspectFrozenPlanForRuntime(const FrozenTaskPlan& plan) {
    for (const auto& region : plan.regions()) {
        if (region->alias != RegionAlias::kNoAlias) {
            return {FallbackReason::kUnsupportedAlias,
                    "task DAG runtime does not support conservative alias regions"};
        }
        switch (region->kind) {
            case RegionKind::kPerCall:
                break;
            case RegionKind::kFusion:
                return {
                    FallbackReason::kUnsupportedFusion,
                    "task DAG runtime does not execute fusion regions without "
                    "a fusion execution contract"};
            case RegionKind::kLibrary:
                return {FallbackReason::kUnsupportedLibrary,
                        "task DAG runtime requires a library execution contract"};
            case RegionKind::kControlFlow:
                return {FallbackReason::kUnsupportedControlFlow,
                        "task DAG runtime does not execute control-flow regions"};
        }
    }
    for (const auto& task : plan.tasks()) {
        if (task->kind == TaskKind::kShapeEval) {
            return {FallbackReason::kUnsupportedShapeEvaluation,
                    "task DAG runtime requires shape-evaluation integration"};
        }
    }
    return {};
}

void EmitFallback(const RuntimeObserver& observer, FallbackReason reason,
                  const std::string& diagnostic) {
    if (!observer) return;
    RuntimeEvent event;
    event.kind = RuntimeEventKind::kFallback;
    event.fallback_reason = reason;
    event.diagnostic = String(diagnostic);
    try {
        observer(event);
    } catch (...) {
        // Audit hooks are synchronous but never control plan selection.
    }
}

TaskDAGSelectionResult SelectionResult(
    RuntimeExecutionMode requested, RuntimeExecutionMode selected,
    FallbackReason reason = FallbackReason::kNone,
    std::string diagnostic = {}) {
    return TaskDAGSelectionResult{requested, selected, reason,
                                  String(std::move(diagnostic))};
}

FrozenTaskPlan AdaptPlanVariantToTaskDAG(
    const api::CompiledModule& module, const PlanVariant& variant) {
    FrozenTaskPlan task_plan = BuildPerCallTaskPlan(module, variant.plan());
    std::unordered_map<int64_t, SelectedArtifactBinding> calls;
    for (const auto& binding : variant.manifest().bindings()) {
        calls.emplace(binding->invocation_id, binding);
    }
    Array<ArtifactSelection> selections;
    size_t call_index = 0;
    for (const auto& task : task_plan.tasks()) {
        if (task->kind != TaskKind::kKernel) continue;
        const auto selected = calls.find(static_cast<int64_t>(call_index++));
        if (selected == calls.end()) {
            throw std::logic_error(
                "task DAG adapter lost an ordered call artifact declaration");
        }
        selections.push_back(ArtifactSelection{
            task->task_id, selected->second->artifact_identity,
            selected->second->generation});
    }
    if (call_index != calls.size()) {
        throw std::logic_error(
            "task DAG adapter did not preserve every ordered call artifact declaration");
    }
    return AttachSelectedArtifacts(
        module, task_plan, selections,
        variant.manifest().retention_lease());
}

Array<NDArray> PrepareTaskArguments(
    const api::CompiledModule& module, const TaskSpec& task,
    const std::unordered_map<int64_t, ValueSpec>& values,
    const std::shared_ptr<internal::ValueTable>& table) {
    const codegen::KernelSignature signature = module.signature(task->symbol);
    Array<int64_t> regular_inputs;
    Array<int64_t> constant_inputs;
    for (int64_t value_id : task.input_value_ids()) {
        const ValueSpec& value =
            FindValue(values, value_id, "RuntimeSession task execution");
        (value->is_constant ? constant_inputs : regular_inputs)
            .push_back(value_id);
    }
    const Array<int64_t> outputs = task.output_value_ids();
    size_t regular_index = 0;
    size_t constant_index = 0;
    size_t output_index = 0;
    Array<NDArray> ordered;
    for (const auto& argument : signature.arguments()) {
        int64_t value_id = -1;
        switch (argument->role) {
            case codegen::KernelArgRole::kInput:
                value_id = regular_inputs[regular_index++];
                break;
            case codegen::KernelArgRole::kConstant:
                value_id = constant_inputs[constant_index++];
                break;
            case codegen::KernelArgRole::kOutput:
                value_id = outputs[output_index++];
                if (!table->Contains(value_id)) {
                    throw std::logic_error(
                        "RuntimeSession kernel output was not allocated");
                }
                break;
        }
        ordered.push_back(table->Get(value_id));
    }
    return ordered;
}

void ValidateBoundTaskSources(
    const api::CompiledModule& module, const FrozenTaskPlan& plan,
    const std::unordered_map<int64_t, ValueSpec>& values,
    const std::shared_ptr<internal::ValueTable>& table) {
    const Map<String, NDArray>& constants =
        api::internal::BorrowCompiledModuleConstants(module);
    for (const auto& task : plan.tasks()) {
        if (task->kind != TaskKind::kKernel) continue;
        const codegen::KernelSignature signature = module.signature(task->symbol);
        Array<int64_t> regular_inputs;
        Array<int64_t> constant_inputs;
        for (int64_t value_id : task.input_value_ids()) {
            const ValueSpec& value =
                FindValue(values, value_id, "RuntimeSession task preflight");
            (value->is_constant ? constant_inputs : regular_inputs)
                .push_back(value_id);
        }
        size_t regular_index = 0;
        size_t constant_index = 0;
        const Array<codegen::KernelArgSpec> arguments = signature.arguments();
        for (size_t argument_index = 0; argument_index < arguments.size();
             ++argument_index) {
            const auto& argument = arguments[argument_index];
            if (argument->role == codegen::KernelArgRole::kOutput) continue;
            const int64_t value_id =
                argument->role == codegen::KernelArgRole::kConstant
                    ? constant_inputs[constant_index++]
                    : regular_inputs[regular_index++];
            if (!table->Contains(value_id)) continue;
            api::ValidateKernelArgument(signature, argument_index, argument,
                                        table->Get(value_id), constants);
        }
    }
}

}  // namespace

RuntimeSession::RuntimeSession(api::CompiledModule module, ExecutablePlan plan)
    : RuntimeSession(std::move(module), std::move(plan),
                     RuntimeExecutionMode::kPerCall, {}) {}

RuntimeSession::RuntimeSession(api::CompiledModule module, ExecutablePlan plan,
                               RuntimeExecutionMode mode)
    : RuntimeSession(std::move(module), std::move(plan), mode, {}) {}

RuntimeSession::RuntimeSession(api::CompiledModule module, ExecutablePlan plan,
                               RuntimeExecutionMode mode,
                               RuntimeObserver observer) {
    if (mode != RuntimeExecutionMode::kPerCall &&
        mode != RuntimeExecutionMode::kTaskDAG) {
        throw std::invalid_argument("RuntimeSession execution mode is invalid");
    }
    ValidatedPlanContract contract = ValidateModuleAndPlan(module, plan);
    TaskDAGSelectionResult selection =
        SelectionResult(mode, RuntimeExecutionMode::kPerCall);
    if (mode == RuntimeExecutionMode::kTaskDAG) {
#if KXC_ENABLE_REGION_TASK_DAG
        FallbackDecision fallback = InspectOrderedPlanForTaskDAG(plan);
        if (fallback.reason == FallbackReason::kNone) {
            fallback = {
                FallbackReason::kMissingArtifactManifest,
                "task DAG adapter requires a trusted upper-plane artifact "
                "declaration in PlanVariant"};
        }
#else
        const FallbackDecision fallback{
            FallbackReason::kFeatureDisabled,
            "task DAG support is disabled by KXC_ENABLE_REGION_TASK_DAG"};
#endif
        selection = SelectionResult(mode, RuntimeExecutionMode::kPerCall,
                                    fallback.reason, fallback.diagnostic);
        EmitFallback(observer, fallback.reason, fallback.diagnostic);
    }
    SetData(new RuntimeSessionNode(
        std::move(module), std::move(plan), SelectedArtifactManifest(),
        std::move(contract.device),
        std::move(contract.constant_keys_by_value), std::move(selection),
        std::move(observer)));
}

RuntimeSession::RuntimeSession(api::CompiledModule module, PlanVariant variant)
    : RuntimeSession(std::move(module), std::move(variant),
                     RuntimeExecutionMode::kPerCall, {}) {}

RuntimeSession::RuntimeSession(api::CompiledModule module, PlanVariant variant,
                               RuntimeExecutionMode mode,
                               RuntimeObserver observer) {
    if (mode != RuntimeExecutionMode::kPerCall &&
        mode != RuntimeExecutionMode::kTaskDAG) {
        throw std::invalid_argument("RuntimeSession execution mode is invalid");
    }
    variant.Validate();
    const ExecutablePlan plan = variant.plan();
    ValidatedPlanContract contract = ValidateModuleAndPlan(module, plan);
    ValidatePlanManifest(module, variant);
    TaskDAGSelectionResult selection =
        SelectionResult(mode, RuntimeExecutionMode::kPerCall);
    if (mode == RuntimeExecutionMode::kTaskDAG) {
#if KXC_ENABLE_REGION_TASK_DAG
        const FallbackDecision fallback = InspectOrderedPlanForTaskDAG(plan);
        if (fallback.reason != FallbackReason::kNone) {
            selection = SelectionResult(mode, RuntimeExecutionMode::kPerCall,
                                        fallback.reason,
                                        fallback.diagnostic);
            EmitFallback(observer, fallback.reason, fallback.diagnostic);
        } else {
            try {
                FrozenTaskPlan task_plan =
                    AdaptPlanVariantToTaskDAG(module, variant);
                ValidatedPlanContract task_contract =
                    ValidateModuleAndTaskPlan(module, task_plan);
                SetData(new RuntimeSessionNode(
                    std::move(module), std::move(task_plan),
                    std::move(task_contract.device),
                    std::move(task_contract.constant_keys_by_value),
                    SelectionResult(mode, RuntimeExecutionMode::kTaskDAG),
                    std::move(observer)));
                return;
            } catch (const std::bad_alloc&) {
                throw;
            } catch (const std::exception& error) {
                const std::string diagnostic =
                    "task DAG adapter invariant failed: " +
                    std::string(error.what());
                EmitFallback(observer, FallbackReason::kAdapterBug,
                             diagnostic);
                throw std::logic_error(diagnostic);
            }
        }
#else
        const FallbackDecision fallback{
            FallbackReason::kFeatureDisabled,
            "task DAG support is disabled by KXC_ENABLE_REGION_TASK_DAG"};
        selection = SelectionResult(mode, RuntimeExecutionMode::kPerCall,
                                    fallback.reason, fallback.diagnostic);
        EmitFallback(observer, fallback.reason, fallback.diagnostic);
#endif
    }
    SetData(new RuntimeSessionNode(
        std::move(module), plan, variant.manifest(),
        std::move(contract.device),
        std::move(contract.constant_keys_by_value), std::move(selection),
        std::move(observer)));
}

RuntimeSession::RuntimeSession(api::CompiledModule module,
                               FrozenTaskPlan plan)
    : RuntimeSession(std::move(module), std::move(plan), {}) {}

RuntimeSession::RuntimeSession(api::CompiledModule module,
                               FrozenTaskPlan plan,
                               RuntimeObserver observer) {
#if KXC_ENABLE_REGION_TASK_DAG
    plan.Validate();
    const FallbackDecision unsupported = InspectFrozenPlanForRuntime(plan);
    if (unsupported.reason != FallbackReason::kNone) {
        EmitFallback(observer, unsupported.reason, unsupported.diagnostic);
        throw std::invalid_argument(unsupported.diagnostic);
    }
    ValidatedPlanContract contract = ValidateModuleAndTaskPlan(module, plan);
    SetData(new RuntimeSessionNode(
        std::move(module), std::move(plan), std::move(contract.device),
        std::move(contract.constant_keys_by_value),
        SelectionResult(RuntimeExecutionMode::kTaskDAG,
                        RuntimeExecutionMode::kTaskDAG),
        std::move(observer)));
#else
    const std::string diagnostic =
        "task DAG support is disabled by KXC_ENABLE_REGION_TASK_DAG";
    EmitFallback(observer, FallbackReason::kFeatureDisabled, diagnostic);
    (void)module;
    (void)plan;
    throw std::invalid_argument(diagnostic);
#endif
}

RuntimeSession::RuntimeSession(const ObjectRef& ref) : ObjectRef(ref) {
    if (defined() && !As<RuntimeSessionNode>()) {
        SetData(nullptr);
        throw std::invalid_argument(
            "ObjectRef does not contain RuntimeSessionNode");
    }
    if (defined()) ValidateStoredSession(*operator->());
}

Array<NDArray> RuntimeSession::Run(const Array<NDArray>& inputs) const {
    const DeviceStream stream = DeviceStream::Default(operator->()->device);
    RunAsyncResult result = RunAsync(inputs, stream);
    result.completion.Wait();
    return result.outputs;
}

RunAsyncResult RuntimeSession::RunAsync(const Array<NDArray>& inputs,
                                        const DeviceStream& stream) const {
    const auto* node = operator->();
    if (!stream.defined() || !stream.As<DeviceStreamNode>()) {
        throw std::invalid_argument(
            "RuntimeSession RunAsync requires a defined DeviceStream");
    }
    if (stream.device() != node->device) {
        throw std::invalid_argument(
            "RuntimeSession stream device expected " + node->device.ToString() +
            ", actual " + stream.device().ToString());
    }

    if (node->task_plan.defined()) {
        const Array<int64_t> input_ids = node->task_plan.input_value_ids();
        if (inputs.size() != input_ids.size()) {
            throw std::invalid_argument(
                "RuntimeSession input count expected " +
                std::to_string(input_ids.size()) + ", actual " +
                std::to_string(inputs.size()));
        }
        std::unordered_map<int64_t, ValueSpec> values;
        for (const auto& value : node->task_plan.values()) {
            values.emplace(value->value_id, value);
        }
        auto table = std::make_shared<internal::ValueTable>();
        for (size_t i = 0; i < inputs.size(); ++i) {
            const ValueSpec& spec =
                FindValue(values, input_ids[i], "RuntimeSession graph input");
            ValidateRuntimeValue(
                spec, inputs[i],
                "RuntimeSession input[" + std::to_string(i) + "]");
            table->Bind(spec, inputs[i]);
        }

        const Map<String, NDArray>& constants =
            api::internal::BorrowCompiledModuleConstants(node->module);
        for (int64_t value_id : node->task_plan.constant_value_ids()) {
            const auto key = node->constant_keys_by_value.find(value_id);
            if (key == node->constant_keys_by_value.end() ||
                !constants.count(key->second)) {
                throw std::logic_error(
                    "RuntimeSession validated task constant binding disappeared");
            }
            const ValueSpec& spec =
                FindValue(values, value_id, "RuntimeSession task constant");
            table->Bind(spec, constants.at(key->second));
        }
        ValidateBoundTaskSources(node->module, node->task_plan, values, table);

        Array<AsyncOperation> operations;
        (void)ExecuteTasksDeterministically(
            node->task_plan,
            [&](const TaskSpec& task) {
                switch (task->kind) {
                    case TaskKind::kAllocate: {
                        const int64_t value_id = task.output_value_ids()[0];
                        table->Allocate(
                            FindValue(values, value_id,
                                      "RuntimeSession task allocation"),
                            task->alignment);
                        break;
                    }
                    case TaskKind::kKernel: {
                        const Array<NDArray> arguments = PrepareTaskArguments(
                            node->module, task, values, table);
                        operations.push_back(node->module.Launch(
                            task->symbol, arguments, stream));
                        break;
                    }
                    case TaskKind::kCopy: {
                        const NDArray source =
                            table->Get(task.input_value_ids()[0]);
                        const NDArray destination =
                            table->Get(task.output_value_ids()[0]);
                        operations.push_back(
                            destination.CopyFromAsync(source, stream));
                        break;
                    }
                    case TaskKind::kEvent:
                        break;
                    case TaskKind::kSync:
                        stream.Sync();
                        break;
                    case TaskKind::kShapeEval:
                        throw std::logic_error(
                            "validated RuntimeSession task plan contains ShapeEval");
                }
            },
            node->observer);

        Array<NDArray> outputs;
        for (int64_t value_id : node->task_plan.output_value_ids()) {
            outputs.push_back(table->Get(value_id));
        }
        AsyncOperation completion =
            operations.empty() ? AsyncOperation::Completed(stream)
                               : operations[operations.size() - 1];
        Array<AsyncOperation> prior_operations;
        for (size_t i = 0; i + 1 < operations.size(); ++i) {
            prior_operations.push_back(operations[i]);
        }
        auto state = std::make_shared<RuntimeExecutionState>(
            RuntimeExecutionState{node->module, ExecutablePlan(),
                                  node->task_plan, node->manifest, table,
                                  std::move(prior_operations)});
        completion.RetainDependencies(table->RetainedStorage(),
                                      std::move(state));
        return RunAsyncResult{std::move(outputs), std::move(completion)};
    }

    const Array<int64_t> input_ids = node->plan.input_value_ids();
    if (inputs.size() != input_ids.size()) {
        throw std::invalid_argument(
            "RuntimeSession input count expected " +
            std::to_string(input_ids.size()) + ", actual " +
            std::to_string(inputs.size()));
    }

    const auto values = IndexValues(node->plan);
    auto table = std::make_shared<internal::ValueTable>();
    for (size_t i = 0; i < inputs.size(); ++i) {
        const ValueSpec& spec =
            FindValue(values, input_ids[i], "RuntimeSession graph input");
        ValidateRuntimeValue(spec, inputs[i],
                             "RuntimeSession input[" + std::to_string(i) + "]");
        table->Bind(spec, inputs[i]);
    }

    const Map<String, NDArray>& constants =
        api::internal::BorrowCompiledModuleConstants(node->module);
    for (int64_t value_id : node->plan.constant_value_ids()) {
        const auto key = node->constant_keys_by_value.find(value_id);
        if (key == node->constant_keys_by_value.end() ||
            !constants.count(key->second)) {
            throw std::logic_error(
                "RuntimeSession validated constant binding disappeared");
        }
        const ValueSpec& spec =
            FindValue(values, value_id, "RuntimeSession constant");
        table->Bind(spec, constants.at(key->second));
    }
    ValidateBoundSourceArguments(node->module, node->plan, values, table);

    const Array<KernelCall> calls = node->plan.calls();
    Array<AsyncOperation> operations;
    for (const auto& call : calls) {
        Array<NDArray> arguments =
            PrepareCallArguments(node->module, call, values, table);
        operations.push_back(
            node->module.Launch(call->symbol, arguments, stream));
    }

    Array<NDArray> outputs;
    for (int64_t value_id : node->plan.output_value_ids()) {
        outputs.push_back(table->Get(value_id));
    }

    AsyncOperation completion = operations.empty()
                                    ? AsyncOperation::Completed(stream)
                                    : operations[operations.size() - 1];
    Array<AsyncOperation> prior_operations;
    for (size_t i = 0; i + 1 < operations.size(); ++i) {
        prior_operations.push_back(operations[i]);
    }
    auto state = std::make_shared<RuntimeExecutionState>(RuntimeExecutionState{
        node->module, node->plan, FrozenTaskPlan(), node->manifest, table,
        std::move(prior_operations)});
    completion.RetainDependencies(table->RetainedStorage(), std::move(state));
    return RunAsyncResult{std::move(outputs), std::move(completion)};
}

bool RuntimeSession::UsesTaskDAG() const {
    return operator->()->task_plan.defined();
}

TaskDAGSelectionResult RuntimeSession::TaskDAGSelection() const {
    return operator->()->selection;
}

SelectedArtifactManifest RuntimeSession::artifact_manifest() const {
    return operator->()->manifest;
}

const RuntimeSessionNode* RuntimeSession::operator->() const {
    const auto* node = As<RuntimeSessionNode>();
    if (!node) throw std::runtime_error("undefined or invalid RuntimeSession");
    return node;
}

}  // namespace kxc::runtime
