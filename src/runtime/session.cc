/*! \file src/runtime/session.cc
 * \brief Executes a validated multi-kernel graph plan on one device stream.
 */

#include "kxc/runtime/session.h"
#include "kxc/support/object_registration.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
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
    std::unordered_map<int64_t, size_t> required_alignment_by_storage;
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
                          const std::string& context,
                          bool allow_dynamic_dimensions = false) {
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
    ValidateShape(value.shape(), array.shape(), context,
                  allow_dynamic_dimensions);
}

bool SameShapeExpression(const api::ModuleShapeExpr& lhs,
                         const api::ModuleShapeExpr& rhs) {
    std::string left;
    std::string right;
    lhs.AppendCanonical(left);
    rhs.AppendCanonical(right);
    return left == right;
}

void ValidateDynamicInvocationOutputs(
    const api::ModuleInvocationContract& contract,
    const std::string& context) {
    for (const auto& output : contract.outputs()) {
        for (size_t axis = 0; axis < output.logical.size(); ++axis) {
            if (!SameShapeExpression(output.logical[axis],
                                     output.physical[axis]) ||
                !SameShapeExpression(output.logical[axis],
                                     output.valid[axis])) {
                throw std::invalid_argument(
                    context +
                    " requires logical, physical, and valid output extents to match");
            }
        }
    }
}

/*! \brief One state's in-place append producer and its declared tokens input. */
struct StatefulAppendBinding final {
    size_t append_call_index{0};
    int64_t tokens_value_id{-1};
};

StatefulAppendBinding ResolveStatefulAppend(
    const ExecutablePlan& plan,
    const std::unordered_map<int64_t, ValueSpec>& values, int64_t state_id) {
    int64_t alias_value_id = -1;
    for (const auto& value : plan.values()) {
        if (value->alias_source_value_id == state_id) {
            alias_value_id = value->value_id;
            break;
        }
    }
    if (alias_value_id == -1) {
        throw std::logic_error("RuntimeSession stateful state has no append alias");
    }
    const Array<KernelCall> calls = plan.calls();
    for (size_t call_index = 0; call_index < calls.size(); ++call_index) {
        const KernelCall& call = calls[call_index];
        bool produces_alias = false;
        for (int64_t output_id : call.output_value_ids()) {
            produces_alias = produces_alias || output_id == alias_value_id;
        }
        if (!produces_alias) continue;
        int64_t tokens_value_id = -1;
        bool consumes_state = false;
        bool consumes_count = false;
        for (int64_t input_id : call.input_value_ids()) {
            const ValueSpec& input =
                FindValue(values, input_id, "RuntimeSession stateful append");
            if (input->is_constant) continue;
            if (input_id == state_id) {
                consumes_state = true;
            } else if (input_id == plan.state_count_input_value_id()) {
                consumes_count = true;
            } else if (tokens_value_id == -1) {
                tokens_value_id = input_id;
            } else {
                throw std::logic_error(
                    "RuntimeSession stateful append consumes multiple token tensors");
            }
        }
        if (!consumes_state || !consumes_count || tokens_value_id == -1) {
            throw std::logic_error(
                "RuntimeSession stateful append does not consume state, tokens, and count");
        }
        return StatefulAppendBinding{call_index, tokens_value_id};
    }
    throw std::logic_error("RuntimeSession stateful append producer disappeared");
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
    const bool dynamic =
        plan.mode() == ExecutablePlanMode::kDynamicFreshOutputV1;
    const bool stateful = plan.mode() == ExecutablePlanMode::kDynamicStatefulV1;
#if !KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI
    if (dynamic || stateful) {
        throw std::invalid_argument(
            "RuntimeSession dynamic modes require "
            "KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI=ON");
    }
#endif
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
        const api::ModuleInvocationContract& contract =
            api::internal::BorrowCompiledModuleInvocationContract(
                module, call->symbol);
        if (dynamic) {
            if (metadata->backend != codegen::CodeGenBackend::kLLVM ||
                metadata->device != Device::CPU()) {
                throw std::invalid_argument(
                    context + " dynamic fresh-output mode supports CPU/LLVM only");
            }
            if (contract.IsConstantShape(signature)) {
                throw std::invalid_argument(
                    context + " has no dynamic invocation contract");
            }
            ValidateDynamicInvocationOutputs(contract, context);
        } else if (stateful) {
            if (metadata->backend != codegen::CodeGenBackend::kLLVM ||
                metadata->device != Device::CPU()) {
                throw std::invalid_argument(
                    context + " dynamic stateful mode supports CPU/LLVM only");
            }
            if (contract.IsConstantShape(signature)) {
                throw std::invalid_argument(
                    context + " has no dynamic stateful invocation contract");
            }
            const auto& extent_scalars = contract.runtime_extent_scalars();
            if (extent_scalars.empty()) {
                throw std::invalid_argument(
                    context + " must declare state-sourced runtime extents");
            }
            for (const auto& scalar : extent_scalars) {
                if (scalar.source !=
                    api::ModuleRuntimeExtentScalar::Source::kStateExtent) {
                    throw std::invalid_argument(
                        context +
                        " dynamic stateful mode binds state-sourced runtime extents only");
                }
            }
            ValidateDynamicInvocationOutputs(contract, context);
        } else if (!contract.IsConstantShape(signature) ||
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
                    if (!dynamic && !stateful) {
                        throw std::invalid_argument(
                            context + " cannot bind a generated runtime extent");
                    }
                    continue;
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
            if (argument->alignment > std::numeric_limits<size_t>::max()) {
                throw std::invalid_argument(
                    context + " alignment exceeds the host size type");
            }
            size_t& required_alignment =
                result.required_alignment_by_storage[value->storage_id];
            required_alignment = std::max(
                required_alignment, static_cast<size_t>(argument->alignment));
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

    if (stateful) {
        const Array<int64_t> input_ids = plan.input_value_ids();
        const std::vector<std::vector<int64_t>> bindings =
            plan.state_extent_bindings();
        // v1 stateful contract fixes the state/tokens dtype to float32.
        for (int64_t state_id : plan.state_value_ids()) {
            const ValueSpec& spec =
                FindValue(values, state_id, "RuntimeSession stateful state");
            if (spec->dtype.code != kDLFloat || spec->dtype.bits != 32 ||
                spec->dtype.lanes != 1) {
                throw std::invalid_argument(
                    "dynamic stateful v1 requires float32 states");
            }
        }
        for (size_t call_index = 0; call_index < calls.size(); ++call_index) {
            const codegen::KernelSignature& signature =
                module.signature(calls[call_index]->symbol);
            size_t extent_args = 0;
            for (const auto& argument : signature.arguments()) {
                if (argument->role == codegen::KernelArgRole::kRuntimeExtent) {
                    ++extent_args;
                }
            }
            if (extent_args != bindings[call_index].size()) {
                throw std::invalid_argument(
                    "RuntimeSession stateful extent bindings do not match call[" +
                    std::to_string(call_index) + "] runtime extent ABI");
            }
        }
        for (int64_t state_id : plan.state_value_ids()) {
            const ValueSpec& state =
                FindValue(values, state_id, "RuntimeSession stateful state");
            const StatefulAppendBinding append =
                ResolveStatefulAppend(plan, values, state_id);
            const ValueSpec& tokens = FindValue(values, append.tokens_value_id,
                                                "RuntimeSession stateful tokens");
            const Array<int64_t> state_shape = state.shape();
            const Array<int64_t> tokens_shape = tokens.shape();
            if (tokens->dtype.code != kDLFloat || tokens->dtype.bits != 32 ||
                tokens->dtype.lanes != 1 || tokens->device != state->device ||
                tokens_shape.size() != state_shape.size()) {
                throw std::invalid_argument(
                    "RuntimeSession stateful tokens tensor must match the state layout");
            }
            const int64_t axis = state->state_extent_axis;
            for (int64_t dim = 0; dim < axis; ++dim) {
                if (tokens_shape[dim] != state_shape[dim]) {
                    throw std::invalid_argument(
                        "RuntimeSession stateful tokens tensor differs from the "
                        "state layout before the extent axis");
                }
            }
            if (tokens_shape[axis] < 1 ||
                tokens_shape[axis] > state_shape[axis]) {
                throw std::invalid_argument(
                    "RuntimeSession stateful tokens extent must fit the state capacity");
            }
            for (size_t dim = axis + 1; dim < state_shape.size(); ++dim) {
                if (tokens_shape[dim] != state_shape[dim]) {
                    throw std::invalid_argument(
                        "RuntimeSession stateful tokens tensor differs from the "
                        "state layout after the extent axis");
                }
            }
        }
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
                                 std::to_string(value_id), false);
    }
    return result;
}

void ValidateStoredSession(const RuntimeSessionNode& node) {
    const ValidatedPlanContract checked = ValidateModuleAndPlan(node.module, node.plan);
    if (checked.device != node.device ||
        checked.constant_keys_by_value.size() != node.constant_keys_by_value.size() ||
        checked.required_alignment_by_storage.size() !=
            node.required_alignment_by_storage.size() ||
        node.states_by_value.size() != node.plan.state_value_ids().size()) {
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
    for (const auto& item : checked.required_alignment_by_storage) {
        const auto stored = node.required_alignment_by_storage.find(item.first);
        if (stored == node.required_alignment_by_storage.end() ||
            stored->second != item.second) {
            throw std::invalid_argument(
                "RuntimeSession stored alignment mapping is inconsistent");
        }
    }
    const auto values = IndexValues(node.plan);
    for (int64_t value_id : node.plan.state_value_ids()) {
        const auto state = node.states_by_value.find(value_id);
        const ValueSpec& spec = FindValue(values, value_id, "RuntimeSession state");
        const auto alignment =
            node.required_alignment_by_storage.find(spec->storage_id);
        if (state == node.states_by_value.end() ||
            alignment == node.required_alignment_by_storage.end() ||
            state->second.storage()->alignment < alignment->second) {
            throw std::invalid_argument(
                "RuntimeSession stored state mapping is inconsistent");
        }
        ValidateRuntimeValue(spec, state->second,
                             "RuntimeSession state value " +
                                 std::to_string(value_id), false);
    }
    std::lock_guard<std::mutex> lock(node.state_mutex);
    if (node.state_completion.defined() &&
        node.state_completion.device() != node.device) {
        throw std::invalid_argument(
            "RuntimeSession stored state completion is inconsistent");
    }
}

AsyncOperation InvokeOrderedModuleEntry(const api::CompiledModule& module,
                                        const String& symbol,
                                        const Array<NDArray>& ordered,
                                        const DeviceStream& stream,
                                        const std::vector<uint64_t>* state_extents = nullptr) {
    const codegen::KernelSignature module_signature = module.signature(symbol);
    const api::ModuleInvocationContract& contract =
        api::internal::BorrowCompiledModuleInvocationContract(module, symbol);
    if (state_extents == nullptr &&
        (!contract.IsConstantShape(module_signature) ||
         !contract.runtime_extent_scalars().empty())) {
        throw std::logic_error(
            "RuntimeSession fails closed until dynamic graph memory planning exists");
    }
    if (state_extents != nullptr) {
        if (contract.runtime_extent_scalars().empty()) {
            throw std::logic_error(
                "RuntimeSession stateful invocation requires state-sourced extents");
        }
        for (const auto& scalar : contract.runtime_extent_scalars()) {
            if (scalar.source !=
                api::ModuleRuntimeExtentScalar::Source::kStateExtent) {
                throw std::logic_error(
                    "RuntimeSession stateful invocation binds state extents only");
            }
        }
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
        module, symbol, inputs, outputs, stream, 0, state_extents);
}

Array<NDArray> PrepareCallArguments(
    const api::CompiledModule& module, const KernelCall& call,
    const std::unordered_map<int64_t, ValueSpec>& values,
    const std::unordered_map<int64_t, size_t>& required_alignment_by_storage,
    const std::shared_ptr<internal::ValueTable>& table,
    const std::vector<uint64_t>* state_extent_values = nullptr) {
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
    size_t state_extent_index = 0;
    Array<NDArray> ordered;
    for (const auto& argument : signature.arguments()) {
        int64_t value_id = -1;
        switch (argument->role) {
            case codegen::KernelArgRole::kInput:
                value_id = regular_inputs[regular_index++];
                break;
            case codegen::KernelArgRole::kRuntimeExtent: {
                if (state_extent_values == nullptr) {
                    throw std::logic_error(
                        "RuntimeSession cannot bind generated runtime extents");
                }
                if (state_extent_index >= state_extent_values->size()) {
                    throw std::logic_error(
                        "RuntimeSession state extent values do not cover the ABI");
                }
                NDArray extent = NDArray::Empty({1}, argument->dtype,
                                                argument->device,
                                                static_cast<size_t>(argument->alignment));
                const uint64_t value = (*state_extent_values)[state_extent_index++];
                extent.CopyFromBytes(&value, sizeof(uint64_t));
                ordered.push_back(std::move(extent));
                continue;
            }
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
                if (value->write_mode == ValueWriteMode::kInPlace) {
                    table->Alias(value, value->alias_source_value_id);
                } else {
                    const auto alignment =
                        required_alignment_by_storage.find(value->storage_id);
                    if (alignment == required_alignment_by_storage.end()) {
                        throw std::logic_error(
                            "RuntimeSession output has no alignment contract");
                    }
                    table->Allocate(value, alignment->second);
                }
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
        const Array<int64_t> output_ids = call.output_value_ids();
        size_t regular_index = 0;
        size_t constant_index = 0;
        size_t output_index = 0;
        const Array<codegen::KernelArgSpec> arguments = signature.arguments();
        for (size_t argument_index = 0; argument_index < arguments.size();
             ++argument_index) {
            const auto& argument = arguments[argument_index];
            if (argument->role == codegen::KernelArgRole::kRuntimeExtent) {
                // Runtime extents are generated or injected per invocation;
                // they never consume a plan input slot.
                continue;
            }
            if (argument->role == codegen::KernelArgRole::kOutput) {
                const ValueSpec& output = FindValue(
                    values, output_ids[output_index++],
                    "RuntimeSession alias preflight");
                int64_t bound_source = output->alias_source_value_id;
                while (bound_source != -1 && !table->Contains(bound_source)) {
                    bound_source = FindValue(values, bound_source,
                                             "RuntimeSession alias preflight")
                                       ->alias_source_value_id;
                }
                if (bound_source != -1) {
                    api::ValidateKernelArgument(
                        signature, argument_index, argument,
                        table->Get(bound_source), constants);
                }
                continue;
            }
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

// 请求一次运行的观测关联；观测器缺失、被抑制或自身失败都返回空关联，
// 观测永远不能改变执行结果。
ExecutionRunCorrelation NotifyRunStart(ExecutionObserver* observer,
                                       const ExecutionRunStart& run) {
    if (observer == nullptr || ExecutionObservationHoldDepth() != 0) return {};
    const ExecutionObservationHold hold;
    try {
        return observer->OnRunStart(run);
    } catch (...) {
        return {};
    }
}

void PreflightDynamicGraphInputs(const ExecutablePlan& plan,
                                 const Array<NDArray>& inputs) {
    for (const GraphInputAxisGuard& guard : plan.graph_input_guards()) {
        const Array<int64_t> shape = inputs[guard.input_index].shape();
        const int64_t extent = shape[guard.axis];
        if (extent < guard.lower || extent > guard.upper ||
            extent % guard.divisible_by != 0) {
            throw std::invalid_argument(
                "RuntimeSession dynamic graph input guard rejected before launch");
        }
        if (guard.equal_to) {
            const Array<int64_t> reference_shape =
                inputs[guard.equal_to->input_index].shape();
            if (extent != reference_shape[guard.equal_to->axis]) {
                throw std::invalid_argument(
                    "RuntimeSession dynamic shared input axis rejected before launch");
            }
        }
    }
}

AsyncOperation InvokeDynamicCall(
    const api::CompiledModule& module, const KernelCall& call,
    const std::unordered_map<int64_t, ValueSpec>& values,
    const std::shared_ptr<internal::ValueTable>& table,
    const DeviceStream& stream) {
    Array<NDArray> inputs;
    for (int64_t value_id : call.input_value_ids()) {
        const ValueSpec& value =
            FindValue(values, value_id, "RuntimeSession dynamic execution");
        if (!value->is_constant) inputs.push_back(table->Get(value_id));
    }
    api::ModuleInvocationResult invoked =
        module.Invoke(call->symbol, inputs, stream);
    const Array<int64_t> output_ids = call.output_value_ids();
    if (invoked.outputs.size() != output_ids.size()) {
        throw std::logic_error(
            "RuntimeSession dynamic invocation returned the wrong output count");
    }
    for (size_t index = 0; index < output_ids.size(); ++index) {
        const api::ModuleInvocationOutput& output = invoked.outputs[index];
        if (output.logical != output.physical ||
            output.logical != output.valid) {
            throw std::logic_error(
                "RuntimeSession dynamic invocation returned non-logical extents");
        }
        const Array<int64_t> physical_shape = output.storage.shape();
        if (physical_shape.size() != output.physical.size()) {
            throw std::logic_error(
                "RuntimeSession dynamic invocation returned a dynamic rank");
        }
        for (size_t axis = 0; axis < physical_shape.size(); ++axis) {
            if (physical_shape[axis] < 0 ||
                static_cast<api::ModuleExtent>(physical_shape[axis]) !=
                    output.physical[axis]) {
                throw std::logic_error(
                    "RuntimeSession dynamic invocation output shape is inconsistent");
            }
        }
        const ValueSpec& spec = FindValue(
            values, output_ids[index], "RuntimeSession dynamic output");
        ValidateRuntimeValue(
            spec, output.storage,
            "RuntimeSession dynamic output " + std::to_string(spec->value_id),
            true);
        table->Bind(spec, output.storage);
    }
    return std::move(invoked.operation);
}

}  // namespace

namespace {

// 构造会话持有的 state 张量；声明了 state_fill 的动态有状态合同把
// 无效容量区填充为哨兵值，其余路径保持零初始化不变。
NDArray AllocateSessionState(const ValueSpec& spec, size_t alignment) {
    if (spec->state_fill == 0.0) {
        return NDArray::Zeros(spec.shape(), spec->dtype, spec->device, alignment);
    }
    NDArray value =
        NDArray::Empty(spec.shape(), spec->dtype, spec->device, alignment);
    // 动态有状态 v1 合同把 state dtype 固定为 float32（构造期校验）。
    size_t elements = 1;
    for (int64_t dimension : spec.shape()) elements *= static_cast<size_t>(dimension);
    std::vector<float> fill(elements, static_cast<float>(spec->state_fill));
    if (!fill.empty()) {
        value.CopyFromBytes(fill.data(), fill.size() * sizeof(float));
    }
    return value;
}

}  // namespace

RuntimeSession::RuntimeSession(api::CompiledModule module, ExecutablePlan plan) {
    ValidatedPlanContract contract = ValidateModuleAndPlan(module, plan);
    const auto values = IndexValues(plan);
    // 观测器从模块继承：模块未启用 profiling 时访问器返回空，行为不变。
    std::shared_ptr<ExecutionObserver> observer =
        api::internal::BorrowCompiledModuleExecutionObserver(module);
    std::unordered_map<int64_t, NDArray> states;
    {
        // 会话构造的 state 初始化分配也经过观测；此时没有运行上下文，
        // 分配事件不带 run 关联（run_id 留空）。
        std::optional<ExecutionObservationScope> observation;
        if (observer) observation.emplace(observer.get(), ExecutionRunCorrelation{});
        for (int64_t value_id : plan.state_value_ids()) {
            const ValueSpec& spec = FindValue(values, value_id, "RuntimeSession state");
            const auto alignment =
                contract.required_alignment_by_storage.find(spec->storage_id);
            if (alignment == contract.required_alignment_by_storage.end()) {
                throw std::logic_error(
                    "RuntimeSession state has no alignment contract");
            }
            states.emplace(value_id, AllocateSessionState(spec, alignment->second));
        }
    }
    auto* node = new RuntimeSessionNode(
        std::move(module), std::move(plan), std::move(contract.device),
        std::move(contract.constant_keys_by_value),
        std::move(contract.required_alignment_by_storage), std::move(states),
        std::move(observer));
    // 动态有状态合同：初始有效长度为 0，由 completion 在成功后提交。
    for (int64_t value_id : node->plan.state_value_ids()) {
        node->length_book->lengths.emplace(value_id, 0);
    }
    SetData(node);
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
    // 观测器为空时全部钩子只有一次空判断，行为与未装配时完全一致。
    ExecutionObserver* observer = node->observer.get();
    ExecutionRunCorrelation correlation;
    if (observer) {
        correlation = NotifyRunStart(observer,
                                     ExecutionRunStart{node->device, inputs.size(),
                                                       node->plan.calls().size()});
    }
    // 底层分配与拷贝路径经线程本地作用域查询当前观测器与运行关联。
    std::optional<ExecutionObservationScope> observation;
    if (observer) observation.emplace(observer, correlation);
    std::size_t submit_count = 0;
    try {
        RunAsyncResult result = [&] {
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
            if (observer) table->ObserveAllocations(observer, correlation);
            const bool dynamic =
                node->plan.mode() == ExecutablePlanMode::kDynamicFreshOutputV1;
            for (size_t i = 0; i < inputs.size(); ++i) {
                const ValueSpec& spec =
                    FindValue(values, input_ids[i], "RuntimeSession graph input");
                ValidateRuntimeValue(spec, inputs[i],
                                     "RuntimeSession input[" + std::to_string(i) + "]",
                                     dynamic);
                table->Bind(spec, inputs[i]);
            }

            const Array<KernelCall> calls = node->plan.calls();
            Array<AsyncOperation> operations;
            std::unique_lock<std::mutex> state_lock;
            const bool stateful =
                node->plan.mode() == ExecutablePlanMode::kDynamicStatefulV1;
            // 动态有状态运行待提交的有效长度；成功完成后经 completion 提交。
            std::unordered_map<int64_t, int64_t> staged_state_lengths;
            if (dynamic) {
                PreflightDynamicGraphInputs(node->plan, inputs);
                std::size_t call_index = 0;
                for (const auto& call : calls) {
                    if (observer) {
                        DispatchExecutionObservation(observer, [&](ExecutionObserver& sink) {
                            sink.OnKernelBegin(
                                KernelSubmitInfo{node->device, call_index,
                                                 std::string(call->symbol)},
                                correlation);
                        });
                    }
                    operations.push_back(InvokeDynamicCall(
                        node->module, call, values, table, stream));
                    ++submit_count;
                    if (observer) {
                        ExecutionCompletionCallback completion;
                        DispatchExecutionObservation(observer, [&](ExecutionObserver& sink) {
                            completion = sink.OnKernelSubmitted(
                                KernelSubmitInfo{node->device, call_index,
                                                 std::string(call->symbol)},
                                correlation);
                        });
                        if (completion && operations[operations.size() - 1].defined()) {
                            operations[operations.size() - 1].ObserveCompletion(
                                std::move(completion));
                        }
                    }
                    ++call_index;
                }
            } else if (stateful) {
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

                // 同一会话的有状态运行严格串行；未完成写入拒绝再次提交，
                // 已有 pending completion 在此结算并落地其长度提交。
                state_lock = std::unique_lock<std::mutex>(node->state_mutex);
                {
                    std::lock_guard<std::mutex> book_lock(node->length_book->mutex);
                    if (node->length_book->poisoned) {
                        throw std::runtime_error(
                            "RuntimeSession state write failed previously; "
                            "the session must be reconstructed");
                    }
                }
                if (node->state_completion.defined() &&
                    !node->state_completion.IsReady()) {
                    throw std::runtime_error(
                        "RuntimeSession state execution is still pending");
                }
                for (int64_t value_id : node->plan.state_value_ids()) {
                    const auto state = node->states_by_value.find(value_id);
                    if (state == node->states_by_value.end()) {
                        throw std::logic_error(
                            "RuntimeSession validated state binding disappeared");
                    }
                    const ValueSpec& spec =
                        FindValue(values, value_id, "RuntimeSession state");
                    table->Bind(spec, state->second);
                }
                ValidateBoundSourceArguments(node->module, node->plan, values, table);

                // Launch 前预检：读取声明的追加数量并对照每个 state 的已提交
                // 有效长度校验；任何拒绝都不改变长度与缓存。
                const Array<int64_t> input_ids = node->plan.input_value_ids();
                size_t count_position = input_ids.size();
                for (size_t index = 0; index < input_ids.size(); ++index) {
                    if (input_ids[index] ==
                        node->plan.state_count_input_value_id()) {
                        count_position = index;
                    }
                }
                if (count_position >= input_ids.size()) {
                    throw std::logic_error(
                        "RuntimeSession stateful append-count input binding disappeared");
                }
                uint64_t append_count = 0;
                inputs[count_position].CopyToBytes(&append_count,
                                                   sizeof(append_count));
                for (int64_t state_id : node->plan.state_value_ids()) {
                    const ValueSpec& spec =
                        FindValue(values, state_id, "RuntimeSession stateful state");
                    const StatefulAppendBinding append =
                        ResolveStatefulAppend(node->plan, values, state_id);
                    const ValueSpec& tokens =
                        FindValue(values, append.tokens_value_id,
                                  "RuntimeSession stateful tokens");
                    const Array<int64_t> tokens_shape = tokens.shape();
                    const int64_t tokens_extent =
                        tokens_shape[spec->state_extent_axis];
                    std::lock_guard<std::mutex> book_lock(node->length_book->mutex);
                    const auto length_it = node->length_book->lengths.find(state_id);
                    if (length_it == node->length_book->lengths.end()) {
                        throw std::logic_error(
                            "RuntimeSession stateful length book lost a state");
                    }
                    const int64_t length = length_it->second;
                    if (length < 0 || length > spec->state_capacity) {
                        throw std::runtime_error(
                            "RuntimeSession stateful length book is inconsistent");
                    }
                    // n <= capacity - length 以 uint64 全量比较：同时覆盖
                    // 负值回绕与 length+n 的整数溢出，全部发生在 launch 之前。
                    if (append_count >
                        static_cast<uint64_t>(spec->state_capacity - length)) {
                        throw std::invalid_argument(
                            "RuntimeSession stateful append exceeds the declared "
                            "capacity before launch");
                    }
                    if (append_count > static_cast<uint64_t>(tokens_extent)) {
                        throw std::invalid_argument(
                            "RuntimeSession stateful append exceeds the declared "
                            "tokens input extent before launch");
                    }
                    staged_state_lengths[state_id] =
                        length + static_cast<int64_t>(append_count);
                }

                std::size_t call_index = 0;
                for (const auto& call : calls) {
                    if (observer) {
                        DispatchExecutionObservation(observer, [&](ExecutionObserver& sink) {
                            sink.OnKernelBegin(
                                KernelSubmitInfo{node->device, call_index,
                                                 std::string(call->symbol)},
                                correlation);
                        });
                    }
                    // 注入经校验的已提交有效长度；读取 kernel 在计算内部推导
                    // 提交后长度（committed + n），会话只绑定一种 extent 表达。
                    std::vector<uint64_t> call_extents;
                    {
                        const std::vector<std::vector<int64_t>>& bindings =
                            node->plan.state_extent_bindings();
                        std::lock_guard<std::mutex> book_lock(node->length_book->mutex);
                        for (int64_t bound : bindings[call_index]) {
                            const auto length_it =
                                node->length_book->lengths.find(bound);
                            if (length_it == node->length_book->lengths.end()) {
                                throw std::logic_error(
                                    "RuntimeSession stateful extent binding is unbound");
                            }
                            call_extents.push_back(
                                static_cast<uint64_t>(length_it->second));
                        }
                    }
                    Array<NDArray> arguments = PrepareCallArguments(
                        node->module, call, values,
                        node->required_alignment_by_storage, table, &call_extents);
                    operations.push_back(InvokeOrderedModuleEntry(
                        node->module, call->symbol, arguments, stream,
                        &call_extents));
                    ++submit_count;
                    if (observer) {
                        ExecutionCompletionCallback completion;
                        DispatchExecutionObservation(observer, [&](ExecutionObserver& sink) {
                            completion = sink.OnKernelSubmitted(
                                KernelSubmitInfo{node->device, call_index,
                                                 std::string(call->symbol)},
                                correlation);
                        });
                        if (completion && operations[operations.size() - 1].defined()) {
                            operations[operations.size() - 1].ObserveCompletion(
                                std::move(completion));
                        }
                    }
                    ++call_index;
                }
            } else {
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

                if (!node->states_by_value.empty()) {
                    state_lock = std::unique_lock<std::mutex>(node->state_mutex);
                    if (node->state_completion.defined() &&
                        !node->state_completion.IsReady()) {
                        throw std::runtime_error(
                            "RuntimeSession state execution is still pending");
                    }
                    for (int64_t value_id : node->plan.state_value_ids()) {
                        const auto state = node->states_by_value.find(value_id);
                        if (state == node->states_by_value.end()) {
                            throw std::logic_error(
                                "RuntimeSession validated state binding disappeared");
                        }
                        const ValueSpec& spec =
                            FindValue(values, value_id, "RuntimeSession state");
                        table->Bind(spec, state->second);
                    }
                }
                ValidateBoundSourceArguments(node->module, node->plan, values, table);

                std::size_t call_index = 0;
                for (const auto& call : calls) {
                    if (observer) {
                        DispatchExecutionObservation(observer, [&](ExecutionObserver& sink) {
                            sink.OnKernelBegin(
                                KernelSubmitInfo{node->device, call_index,
                                                 std::string(call->symbol)},
                                correlation);
                        });
                    }
                    Array<NDArray> arguments = PrepareCallArguments(
                        node->module, call, values,
                        node->required_alignment_by_storage, table);
                    operations.push_back(InvokeOrderedModuleEntry(
                        node->module, call->symbol, arguments, stream));
                    ++submit_count;
                    if (observer) {
                        ExecutionCompletionCallback completion;
                        DispatchExecutionObservation(observer, [&](ExecutionObserver& sink) {
                            completion = sink.OnKernelSubmitted(
                                KernelSubmitInfo{node->device, call_index,
                                                 std::string(call->symbol)},
                                correlation);
                        });
                        // 完成观测在 Wait/IsReady/析构之间恰好结算一次；CPU 同步
                        // 后端注册即触发，异步后端在真实观测到完成时触发。
                        if (completion && operations[operations.size() - 1].defined()) {
                            operations[operations.size() - 1].ObserveCompletion(
                                std::move(completion));
                        }
                    }
                    ++call_index;
                }
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
            if (stateful) {
                // 长度更新只在全部 kernel 提交成功后注册到最终 completion：
                // CPU 同步后端注册即落地，异步后端在首次完成观测时落地。
                // 会话销毁后 book 仍由 completion 保活，提交不会悬空。
                auto book = node->length_book;
                const std::unordered_map<int64_t, int64_t> pending =
                    staged_state_lengths;
                completion.ObserveCompletion([book, pending](bool) {
                    std::lock_guard<std::mutex> book_lock(book->mutex);
                    for (const auto& entry : pending) {
                        book->lengths[entry.first] = entry.second;
                    }
                });
            }
            if (state_lock.owns_lock()) node->state_completion = completion;
            return RunAsyncResult{std::move(outputs), std::move(completion)};
        }();
        observation.reset();
        if (observer) {
            DispatchExecutionObservation(observer, [&](ExecutionObserver& sink) {
                sink.OnRunEnd(ExecutionRunEnd{node->device, inputs.size(),
                                              node->plan.calls().size(),
                                              submit_count, true, std::string()},
                              correlation);
            });
        }
        return result;
    } catch (...) {
        observation.reset();
        if (node->plan.mode() == ExecutablePlanMode::kDynamicStatefulV1 &&
            submit_count > 0) {
            // 写入开始后的失败：第一版不做事务回滚，会话标记为不能继续，
            // 要求显式重建；拒绝阶段的失败（launch 前）不置位该标记。
            std::lock_guard<std::mutex> book_lock(node->length_book->mutex);
            node->length_book->poisoned = true;
        }
        if (observer) {
            // 错误复用 run span 的 status="error"，message 为原始异常文本，
            // 不新增事件类型；校验失败发生在任何 kernel_submit 之前。
            std::string message = "unknown runtime error";
            try {
                throw;
            } catch (const std::exception& error) {
                message = error.what();
            } catch (...) {
            }
            DispatchExecutionObservation(observer, [&](ExecutionObserver& sink) {
                sink.OnRunEnd(ExecutionRunEnd{node->device, inputs.size(),
                                              node->plan.calls().size(),
                                              submit_count, false, message},
                              correlation);
            });
        }
        throw;
    }
}

int64_t RuntimeSession::StateExtent(int64_t state_value_id) const {
    const auto* node = operator->();
    bool declared = false;
    for (int64_t value_id : node->plan.state_value_ids()) {
        declared = declared || value_id == state_value_id;
    }
    if (!declared) {
        throw std::invalid_argument("RuntimeSession has no such state value");
    }
    if (node->plan.mode() != ExecutablePlanMode::kDynamicStatefulV1) {
        throw std::invalid_argument(
            "RuntimeSession state extents require the dynamic stateful mode");
    }
    std::lock_guard<std::mutex> book_lock(node->length_book->mutex);
    const auto it = node->length_book->lengths.find(state_value_id);
    if (it == node->length_book->lengths.end()) {
        throw std::logic_error("RuntimeSession stateful length book lost a state");
    }
    return it->second;
}

NDArray RuntimeSession::StateValue(int64_t state_value_id) const {
    const auto* node = operator->();
    bool declared = false;
    for (int64_t value_id : node->plan.state_value_ids()) {
        declared = declared || value_id == state_value_id;
    }
    if (!declared) {
        throw std::invalid_argument("RuntimeSession has no such state value");
    }
    const auto state = node->states_by_value.find(state_value_id);
    if (state == node->states_by_value.end()) {
        throw std::logic_error("RuntimeSession state binding disappeared");
    }
    return state->second;
}

const RuntimeSessionNode* RuntimeSession::operator->() const {
    const auto* node = As<RuntimeSessionNode>();
    if (!node) throw std::runtime_error("undefined or invalid RuntimeSession");
    return node;
}

}  // namespace kxc::runtime
