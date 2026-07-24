/*! \file src/runtime/session.cc
 * \brief Executes a validated multi-kernel graph plan on one device stream.
 */

#include "kxc/runtime/session.h"
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
    std::shared_ptr<internal::ValueTable> values;
    Array<AsyncOperation> prior_operations;
};

std::unordered_map<int64_t, ValueSpec> IndexValues(const ExecutablePlan& plan) {
    std::unordered_map<int64_t, ValueSpec> result;
    for (const auto& value : plan.values()) result.emplace(value->value_id, value);
    return result;
}

const ValueSpec& FindValue(const std::unordered_map<int64_t, ValueSpec>& values,
                           int64_t value_id, const std::string& context) {
    const auto it = values.find(value_id);
    if (it == values.end()) {
        throw std::invalid_argument(context + " references unknown value " +
                                    std::to_string(value_id));
    }
    return it->second;
}

void ValidateShape(const Array<int64_t>& expected, const Array<int64_t>& actual,
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
    if (!api::SameDType(value->dtype, argument->dtype) || value->device != argument->device) {
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
        throw std::invalid_argument(context + " device expected " +
                                    value->device.ToString() + ", actual " +
                                    array.device().ToString());
    }
    ValidateShape(value.shape(), array.shape(), context, value->is_input);
}

ValidatedPlanContract ValidateModuleAndPlan(const api::CompiledModule& module,
                                            const ExecutablePlan& plan) {
    if (!module.defined() || !module.IsReady()) {
        throw std::invalid_argument(
            "RuntimeSession requires a defined, ready CompiledModule");
    }
    if (!plan.defined()) {
        throw std::invalid_argument("RuntimeSession requires a defined ExecutablePlan");
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
    for (const auto& symbol : module_symbols) expected_symbols.insert(std::string(symbol));

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
                                    std::to_string(call_index) + "] '" + symbol + "'";
        if (!module.HasFunction(call->symbol) || expected_symbols.count(symbol) == 0 ||
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
            throw std::invalid_argument(context + " targets a different execution device");
        }
        const api::ModuleInvocationContract contract =
            module.invocation_contract(call->symbol);
        if (!contract.IsConstantShape(signature) ||
            !contract.runtime_extent_scalars().empty()) {
            throw std::invalid_argument(
                context + " requires unsupported nonstatic/scalar module invocation ABI");
        }

        Array<int64_t> regular_inputs;
        Array<int64_t> constant_inputs;
        for (int64_t value_id : call.input_value_ids()) {
            const ValueSpec& value = FindValue(values, value_id, context);
            (value->is_constant ? constant_inputs : regular_inputs).push_back(value_id);
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
                case codegen::KernelArgRole::kRuntimeExtent:
                    throw std::invalid_argument(
                        context + " cannot bind a generated runtime extent");
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
                                  context + " value " + std::to_string(value_id));
            if (argument->role == codegen::KernelArgRole::kConstant) {
                const std::string key = std::string(argument->constant_key);
                if (!module_constants.count(argument->constant_key)) {
                    throw std::invalid_argument(context + " has no module constant binding");
                }
                api::ValidateKernelArgument(signature, argument_index, argument,
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
            constant_index != constant_inputs.size() || output_index != outputs.size()) {
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
        ValidateRuntimeValue(FindValue(values, value_id, "RuntimeSession constant"),
                             module_constants.at(key_it->second),
                             "RuntimeSession constant value " +
                                 std::to_string(value_id));
    }
    return result;
}

void ValidateStoredSession(const RuntimeSessionNode& node) {
    const ValidatedPlanContract checked = ValidateModuleAndPlan(node.module, node.plan);
    if (checked.device != node.device ||
        checked.constant_keys_by_value.size() != node.constant_keys_by_value.size()) {
        throw std::invalid_argument("RuntimeSession stored plan or metadata is inconsistent");
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

AsyncOperation InvokeOrderedModuleEntry(const api::CompiledModule& module,
                                        const String& symbol,
                                        const Array<NDArray>& ordered,
                                        const DeviceStream& stream) {
    const codegen::KernelSignature module_signature = module.signature(symbol);
    const api::ModuleInvocationContract contract = module.invocation_contract(symbol);
    if (!contract.IsConstantShape(module_signature) ||
        !contract.runtime_extent_scalars().empty()) {
        throw std::logic_error(
            "RuntimeSession fails closed until dynamic graph memory planning exists");
    }
    Array<NDArray> inputs;
    Array<NDArray> outputs;
    const Array<codegen::KernelArgSpec> signature = module_signature.arguments();
    if (ordered.size() != signature.size()) {
        throw std::logic_error(
            "RuntimeSession ordered arguments do not match module ABI");
    }
    for (size_t i = 0; i < signature.size(); ++i) {
        if (signature[i]->role == codegen::KernelArgRole::kInput) inputs.push_back(ordered[i]);
        if (signature[i]->role == codegen::KernelArgRole::kOutput) outputs.push_back(ordered[i]);
    }
    return api::internal::InvokeCompiledModuleWithOutputs(
        module, symbol, inputs, outputs, stream);
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
        (value->is_constant ? constant_inputs : regular_inputs).push_back(value_id);
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
            case codegen::KernelArgRole::kRuntimeExtent:
                throw std::logic_error(
                    "RuntimeSession cannot bind generated runtime extents");
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
            (value->is_constant ? constant_inputs : regular_inputs).push_back(value_id);
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

RuntimeSession::RuntimeSession(api::CompiledModule module, ExecutablePlan plan) {
    ValidatedPlanContract contract = ValidateModuleAndPlan(module, plan);
    SetData(new RuntimeSessionNode(std::move(module), std::move(plan),
                                   std::move(contract.device),
                                   std::move(contract.constant_keys_by_value)));
}

RuntimeSession::RuntimeSession(const ObjectRef& ref) : ObjectRef(ref) {
    if (defined() && !As<RuntimeSessionNode>()) {
        SetData(nullptr);
        throw std::invalid_argument("ObjectRef does not contain RuntimeSessionNode");
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
        throw std::invalid_argument("RuntimeSession stream device expected " +
                                    node->device.ToString() + ", actual " +
                                    stream.device().ToString());
    }

    const Array<int64_t> input_ids = node->plan.input_value_ids();
    if (inputs.size() != input_ids.size()) {
        throw std::invalid_argument("RuntimeSession input count expected " +
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
        if (key == node->constant_keys_by_value.end() || !constants.count(key->second)) {
            throw std::logic_error(
                "RuntimeSession validated constant binding disappeared");
        }
        const ValueSpec& spec = FindValue(values, value_id, "RuntimeSession constant");
        table->Bind(spec, constants.at(key->second));
    }
    ValidateBoundSourceArguments(node->module, node->plan, values, table);

    const Array<KernelCall> calls = node->plan.calls();
    Array<AsyncOperation> operations;
    for (const auto& call : calls) {
        Array<NDArray> arguments =
            PrepareCallArguments(node->module, call, values, table);
        operations.push_back(InvokeOrderedModuleEntry(
            node->module, call->symbol, arguments, stream));
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
        node->module, node->plan, table, std::move(prior_operations)});
    completion.RetainDependencies(table->RetainedStorage(), std::move(state));
    return RunAsyncResult{std::move(outputs), std::move(completion)};
}

const RuntimeSessionNode* RuntimeSession::operator->() const {
    const auto* node = As<RuntimeSessionNode>();
    if (!node) throw std::runtime_error("undefined or invalid RuntimeSession");
    return node;
}

}  // namespace kxc::runtime
