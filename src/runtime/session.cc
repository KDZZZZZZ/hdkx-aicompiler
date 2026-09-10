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

void ValidateSessionStream(const DeviceStream& stream, const Device& device) {
    if (!stream.defined() || !stream.As<DeviceStreamNode>()) {
        throw std::invalid_argument("RuntimeSession requires a defined DeviceStream");
    }
    if (stream.device() != device) {
        throw std::invalid_argument("RuntimeSession stream device expected " +
                                    device.ToString() + ", actual " + stream.device().ToString());
    }
}

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

struct PendingStateOutputUpdate final {
    StateOutputBinding binding;
    NDArray state;
    NDArray source;
    int64_t prior_extent{0};
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
    const bool bounded_stateful =
        plan.mode() == ExecutablePlanMode::kBoundedStatefulExternalV1;
    const bool dynamic =
        plan.mode() == ExecutablePlanMode::kDynamicFreshOutputV1 || bounded_stateful;
    const bool external_stateful =
        plan.mode() == ExecutablePlanMode::kStaticStatefulExternalV1 || bounded_stateful;
    const bool stateful =
        plan.mode() == ExecutablePlanMode::kDynamicStatefulV1 ||
        external_stateful;
#if !KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI
    if (dynamic || plan.mode() == ExecutablePlanMode::kDynamicStatefulV1) {
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
            const bool llvm = metadata->backend == codegen::CodeGenBackend::kLLVM &&
                              metadata->device == Device::CPU();
            const bool cuda = metadata->backend == codegen::CodeGenBackend::kCUDA &&
                              metadata->device.device_type() == kCUDA;
            if (!llvm && !cuda) {
                throw std::invalid_argument(
                    context + " bounded execution requires CPU:0/LLVM or CUDA");
            }
            // A bounded graph may contain statically shaped weight transforms.
            // Each call still uses its validated module-owned output contract.
            ValidateDynamicInvocationOutputs(contract, context);
        } else if (stateful && !external_stateful) {
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
        } else if (external_stateful) {
            const bool llvm = metadata->backend == codegen::CodeGenBackend::kLLVM &&
                              metadata->device == Device::CPU();
            const bool cuda = metadata->backend == codegen::CodeGenBackend::kCUDA &&
                              metadata->device.device_type() == kCUDA;
            if (!llvm && !cuda) {
                throw std::invalid_argument(
                    context + " static external stateful mode requires CPU/LLVM or CUDA");
            }
            if (!contract.IsConstantShape(signature) ||
                !contract.runtime_extent_scalars().empty()) {
                throw std::invalid_argument(
                    context + " static external stateful mode requires a constant-shape ABI");
            }
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
                    if (external_stateful && !bounded_stateful) {
                        throw std::invalid_argument(
                            context + " static external stateful mode has no runtime extent ABI");
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

    if (bounded_stateful) {
        for (const auto& binding : plan.state_output_bindings()) {
            const auto& state = values.at(binding.state_value_id);
            const auto& prefix = values.at(binding.input_value_id);
            const auto alignment = result.required_alignment_by_storage.find(prefix->storage_id);
            if (alignment == result.required_alignment_by_storage.end()) {
                throw std::invalid_argument("RuntimeSession bounded state prefix has no kernel consumer");
            }
            result.required_alignment_by_storage[state->storage_id] = alignment->second;
            // Bind the append to the module-owned output expression. A caller
            // cannot relabel P+1 as a different append count or data-derived
            // length; this proof runs before session storage allocation.
            for (const auto& call : calls) {
                const auto outputs = call.output_value_ids();
                for (size_t output = 0; output < outputs.size(); ++output) {
                    if (outputs[output] != binding.source_value_id) continue;
                    size_t input_index = 0, prefix_index = 0;
                    bool found = false;
                    for (int64_t id : call.input_value_ids()) {
                        if (values.at(id)->is_constant) continue;
                        if (id == binding.input_value_id) {
                            prefix_index = input_index;
                            found = true;
                        }
                        ++input_index;
                    }
                    const auto& contract = api::internal::BorrowCompiledModuleInvocationContract(module, call->symbol);
                    const auto expected = api::ModuleShapeExpr::Add(
                        api::ModuleShapeExpr::InputAxis(prefix_index, binding.source_extent_axis),
                        api::ModuleShapeExpr::Const(binding.append_count));
                    if (!found || !SameShapeExpression(
                        contract.outputs()[output].logical[binding.source_extent_axis], expected)) {
                        throw std::invalid_argument(
                            "RuntimeSession bounded state append differs from the module output extent contract");
                    }
                }
            }
        }
    }

    if (stateful && !external_stateful) {
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

void ValidateStoredMappings(const RuntimeSessionNode& node,
                            const ValidatedPlanContract& checked) {
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
}

void ValidateExecutionModule(const RuntimeSessionNode& node,
                             const api::CompiledModule& module) {
    ValidateStoredMappings(node, ValidateModuleAndPlan(module, node.plan));
    for (const auto& call : node.plan.calls()) {
        if (module.signature(call->symbol).CanonicalBytes() !=
                node.module.signature(call->symbol).CanonicalBytes() ||
            module.launch_metadata(call->symbol).CanonicalBytes() !=
                node.module.launch_metadata(call->symbol).CanonicalBytes() ||
            api::internal::BorrowCompiledModuleInvocationContract(module, call->symbol).CanonicalBytes() !=
                api::internal::BorrowCompiledModuleInvocationContract(node.module, call->symbol).CanonicalBytes()) {
            throw std::invalid_argument("RuntimeSession replacement module changed the executable contract");
        }
    }
}

void ValidateStoredSession(const RuntimeSessionNode& node) {
    ValidateStoredMappings(node, ValidateModuleAndPlan(node.module, node.plan));
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

bool StructuredPredicate(const NDArray& value) {
    std::uint8_t byte{0};
    value.CopyToBytes(&byte, sizeof(byte));
    return byte != 0;
}

const StructuredRegion& FindStructuredRegion(
    const std::unordered_map<int64_t, const StructuredRegion*>& regions,
    int64_t region_id) {
    const auto found = regions.find(region_id);
    if (found == regions.end()) {
        throw std::logic_error("RuntimeSession structured region is undefined");
    }
    return *found->second;
}

/*! \brief Walk a structured schedule on the main session machinery.
 *
 *  Each kernel task reuses the plan's own KernelCall, argument preparation and
 *  module invocation; only call selection and ordering differ from the linear
 *  path. Loop activations push a value frame so a static id may be re-bound per
 *  iteration while staying single-bound within a frame. */
class StructuredWalker final {
public:
    StructuredWalker(const api::CompiledModule& module,
                     const ExecutablePlan& plan,
                     const StructuredSchedule& schedule,
                     const std::unordered_map<int64_t, ValueSpec>& values,
                     const std::unordered_map<int64_t, size_t>& alignments,
                     const std::shared_ptr<internal::ValueTable>& table,
                     const DeviceStream& stream,
                     Array<AsyncOperation>* operations,
                     ExecutionObserver* observer,
                     const ExecutionRunCorrelation& correlation,
                     bool dynamic)
        : module_(module), schedule_(schedule), values_(values),
          alignments_(alignments), table_(table), stream_(stream),
          operations_(operations), observer_(observer), correlation_(correlation),
          device_(stream.device()), dynamic_(dynamic) {
        const Array<KernelCall> calls = plan.calls();
        for (const auto& region : schedule_.regions) {
            regions_.emplace(region.id, &region);
        }
        calls_.assign(calls.begin(), calls.end());
    }

    void Run() {
        ExecuteRegion(FindStructuredRegion(regions_, schedule_.entry_region));
    }

    std::size_t submit_count() const { return submit_count_; }

private:
    void WaitLast() {
        if (operations_ != nullptr && operations_->size() != 0) {
            (*operations_)[operations_->size() - 1].Wait();
        }
    }

    void ExecuteKernelTask(const StructuredTask& task) {
        if (task.call_index < 0 ||
            static_cast<std::size_t>(task.call_index) >= calls_.size()) {
            throw std::logic_error("structured kernel task call index is invalid");
        }
        const KernelCall& call = calls_[static_cast<std::size_t>(task.call_index)];
        const std::size_t call_index = static_cast<std::size_t>(task.call_index);
        std::optional<ExecutionObservationScope> kernel_observation;
        if (observer_) {
            const KernelSubmitInfo kernel{device_, call_index,
                                          std::string(call->symbol)};
            kernel_observation.emplace(observer_, correlation_, &kernel);
            DispatchExecutionObservation(observer_, [&](ExecutionObserver& sink) {
                sink.OnKernelBegin(kernel, correlation_);
            });
        }
        Array<AsyncOperation> submitted;
        if (dynamic_) {
            // Dynamic region kernels resolve their own runtime extents through
            // the module invocation contract, exactly like the linear path.
            operations_->push_back(
                InvokeDynamicCall(module_, call, values_, table_, stream_));
        } else {
            Array<NDArray> arguments =
                PrepareCallArguments(module_, call, values_, alignments_, table_);
            operations_->push_back(
                InvokeOrderedModuleEntry(module_, call->symbol, arguments, stream_));
        }
        ++submit_count_;
        if (observer_) {
            ExecutionCompletionCallback completion;
            DispatchExecutionObservation(observer_, [&](ExecutionObserver& sink) {
                completion = sink.OnKernelSubmitted(
                    KernelSubmitInfo{device_, call_index,
                                     std::string(call->symbol)},
                    correlation_);
            });
            if (completion && (*operations_)[operations_->size() - 1].defined()) {
                (*operations_)[operations_->size() - 1].ObserveCompletion(
                    std::move(completion));
            }
        }
    }

    void ExecuteBranchTask(const StructuredTask& task) {
        WaitLast();
        const bool selected_then =
            StructuredPredicate(table_->Get(task.branch.predicate));
        ExecuteRegion(FindStructuredRegion(
            regions_, selected_then ? task.branch.then_region
                                    : task.branch.else_region));
        for (const auto& phi : task.branch.phis) {
            const int64_t source =
                selected_then ? phi.then_value : phi.else_value;
            InstallResult(phi.result, table_->Get(source));
        }
    }

    void ExecuteLoopTask(const StructuredTask& task) {
        std::unordered_map<int64_t, NDArray> current;
        for (const auto& carried : task.loop.carried) {
            current.emplace(carried.body_argument, table_->Get(carried.initial));
        }
        int64_t iterations = 0;
        while (true) {
            table_->PushFrame();
            for (const auto& item : current) {
                table_->Bind(FindValue(values_, item.first,
                                       "structured loop argument"),
                             item.second);
            }
            ExecuteRegion(FindStructuredRegion(regions_, task.loop.condition_region));
            WaitLast();
            const bool keep_going =
                StructuredPredicate(table_->Get(task.loop.condition_value));
            table_->PopFrame();
            if (!keep_going) break;
            if (iterations >= task.loop.max_trip_count) {
                throw std::runtime_error(
                    "RuntimeSession structured loop exceeded max_trip_count");
            }
            table_->PushFrame();
            for (const auto& item : current) {
                table_->Bind(FindValue(values_, item.first,
                                       "structured loop argument"),
                             item.second);
            }
            ExecuteRegion(FindStructuredRegion(regions_, task.loop.body_region));
            WaitLast();
            // Collect every backedge before installing any, so a tuple swap
            // such as (a,b) <- (b,a) is not corrupted by ordered overwrite.
            std::vector<std::pair<int64_t, NDArray>> next;
            next.reserve(task.loop.carried.size());
            for (const auto& carried : task.loop.carried) {
                next.emplace_back(carried.body_argument,
                                  table_->Get(carried.backedge));
            }
            table_->PopFrame();
            for (auto& item : next) {
                current.insert_or_assign(item.first, std::move(item.second));
            }
            ++iterations;
        }
        for (const auto& carried : task.loop.carried) {
            InstallResult(carried.result, current.at(carried.body_argument));
        }
    }

    void ExecuteRegion(const StructuredRegion& region) {
        for (const auto& task : region.tasks) {
            switch (task.kind) {
                case StructuredTaskKind::kKernel:
                    ExecuteKernelTask(task);
                    break;
                case StructuredTaskKind::kBranch:
                    ExecuteBranchTask(task);
                    break;
                case StructuredTaskKind::kLoop:
                    ExecuteLoopTask(task);
                    break;
            }
        }
    }

    /*! \brief Materialize a topology-produced value (Phi result or loop result)
     *  in the current frame so later tasks and graph outputs can read it. */
    void InstallResult(int64_t value_id, NDArray value) {
        if (value_id < 0) {
            throw std::logic_error("structured result value id is invalid");
        }
        if (table_->ContainsInCurrentFrame(value_id)) return;
        table_->Bind(FindValue(values_, value_id, "structured result"), value);
    }

    const api::CompiledModule& module_;
    const StructuredSchedule& schedule_;
    const std::unordered_map<int64_t, ValueSpec>& values_;
    const std::unordered_map<int64_t, size_t>& alignments_;
    const std::shared_ptr<internal::ValueTable>& table_;
    const DeviceStream& stream_;
    Array<AsyncOperation>* operations_;
    ExecutionObserver* observer_{nullptr};
    ExecutionRunCorrelation correlation_;
    Device device_{};
    std::vector<KernelCall> calls_;
    std::unordered_map<int64_t, const StructuredRegion*> regions_;
    std::size_t submit_count_{0};
    bool dynamic_{false};
};

}  // namespace

namespace {
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

size_t Product(const Array<int64_t>& shape, size_t begin, size_t end,
               const char* context) {
    size_t result = 1;
    for (size_t index = begin; index < end; ++index) {
        if (shape[index] < 0 ||
            (shape[index] != 0 &&
             result > std::numeric_limits<size_t>::max() /
                          static_cast<size_t>(shape[index]))) {
            throw std::overflow_error(std::string(context) + " shape overflows size_t");
        }
        if (shape[index] == 0) return 0;
        result *= static_cast<size_t>(shape[index]);
    }
    return result;
}

size_t DTypeBytes(DLDataType dtype) {
    if (dtype.bits == 0 || dtype.bits % 8 != 0 || dtype.lanes == 0 ||
        static_cast<size_t>(dtype.bits / 8) >
            std::numeric_limits<size_t>::max() / dtype.lanes) {
        throw std::invalid_argument("state copy dtype is not byte-addressable");
    }
    return static_cast<size_t>(dtype.bits / 8) * dtype.lanes;
}

/*! \brief Copy a contiguous extent-axis range while preserving all outer
 *  dimensions.  This deliberately uses StorageCopySync: external stateful
 *  v1 waits for all kernels before this commit, including CUDA execution. */
void CopyStateRange(const NDArray& source, int64_t source_index,
                    const NDArray& destination, int64_t destination_index,
                    int64_t count, int64_t axis) {
    const Array<int64_t> source_shape = source.shape();
    const Array<int64_t> destination_shape = destination.shape();
    if (!source.IsContiguous() || !destination.IsContiguous() ||
        source_shape.size() != destination_shape.size() || axis < 0 ||
        axis >= static_cast<int64_t>(source_shape.size()) || count < 0 ||
        source_index < 0 || destination_index < 0 ||
        source_index > source_shape[axis] ||
        count > source_shape[axis] - source_index ||
        destination_index > destination_shape[axis] ||
        count > destination_shape[axis] - destination_index ||
        source.dtype().code != destination.dtype().code ||
        source.dtype().bits != destination.dtype().bits ||
        source.dtype().lanes != destination.dtype().lanes) {
        throw std::invalid_argument("state copy range violates tensor layout");
    }
    for (size_t index = 0; index < source_shape.size(); ++index) {
        if (static_cast<int64_t>(index) != axis &&
            source_shape[index] != destination_shape[index]) {
            throw std::invalid_argument("state copy range shapes differ");
        }
    }
    const size_t element_bytes = DTypeBytes(source.dtype());
    const size_t trailing = Product(source_shape, static_cast<size_t>(axis + 1),
                                    source_shape.size(), "state copy");
    if (trailing != 0 &&
        static_cast<size_t>(count) >
            std::numeric_limits<size_t>::max() / trailing) {
        throw std::overflow_error("state copy range bytes overflow");
    }
    const size_t row_elements = static_cast<size_t>(count) * trailing;
    if (row_elements != 0 &&
        element_bytes > std::numeric_limits<size_t>::max() / row_elements) {
        throw std::overflow_error("state copy range bytes overflow");
    }
    const size_t row_bytes = row_elements * element_bytes;
    const size_t outer = Product(source_shape, 0, static_cast<size_t>(axis),
                                 "state copy");
    if (outer == 0 || row_bytes == 0) return;
    const size_t source_axis = static_cast<size_t>(source_shape[axis]);
    const size_t destination_axis = static_cast<size_t>(destination_shape[axis]);
    const size_t source_stride = source_axis * trailing * element_bytes;
    const size_t destination_stride = destination_axis * trailing * element_bytes;
    const size_t source_offset = static_cast<size_t>(source_index) * trailing * element_bytes;
    const size_t destination_offset =
        static_cast<size_t>(destination_index) * trailing * element_bytes;
    for (size_t outer_index = 0; outer_index < outer; ++outer_index) {
        StorageCopySync(source.storage(), source->byte_offset +
                                             outer_index * source_stride + source_offset,
                        destination.storage(), destination->byte_offset +
                                                   outer_index * destination_stride +
                                                       destination_offset,
                        row_bytes);
    }
}

std::unique_lock<std::mutex> LockStateForRun(const RuntimeSessionNode* node) {
    std::unique_lock<std::mutex> lock(node->state_mutex);
    {
        std::lock_guard<std::mutex> book_lock(node->length_book->mutex);
        if (node->length_book->poisoned) {
            throw std::runtime_error(
                "RuntimeSession state write failed previously; the session must be reconstructed");
        }
    }
    if (node->state_completion.defined() && !node->state_completion.IsReady()) {
        throw std::runtime_error("RuntimeSession state execution is still pending");
    }
    return lock;
}

NDArray LeadingRow(const NDArray& value, size_t row) {
    auto shape = value.shape();
    if (shape.empty() || !value.IsContiguous() ||
        row >= static_cast<size_t>(shape[0])) {
        throw std::invalid_argument("Request batching row violates tensor layout");
    }
    const size_t bytes = value.NBytes() / static_cast<size_t>(shape[0]);
    shape[0] = 1;
    return value.CreateView(shape, {}, value->byte_offset + row * bytes);
}

class RequestAccess final {
public:
    explicit RequestAccess(const RuntimeSessionNode* node)
        : node_(node), lock_(node->request_mutex, std::try_to_lock) {
        if (!node->plan.request_batching()) {
            throw std::invalid_argument("RuntimeSession requires an explicit request batching contract");
        }
        // Recursive acquisition is defined, but nested operations are rejected.
        // This also prevents observer callbacks from deadlocking on this session.
        if (!lock_.owns_lock() || node->request_in_progress) {
            throw std::runtime_error("RuntimeSession request operation is already in progress");
        }
        std::lock_guard<std::mutex> book_lock(node->length_book->mutex);
        if (node->length_book->poisoned) {
            throw std::runtime_error("RuntimeSession state write failed previously; the session must be reconstructed");
        }
        node->request_in_progress = true;
    }
    ~RequestAccess() { node_->request_in_progress = false; }
    RequestAccess(const RequestAccess&) = delete;
    RequestAccess& operator=(const RequestAccess&) = delete;
private:
    const RuntimeSessionNode* node_;
    std::unique_lock<std::recursive_mutex> lock_;
};

size_t FindRequest(const RuntimeSessionNode* node, uint64_t request_id) {
    for (size_t index = 0; request_id != 0 && index < node->request_slots.size(); ++index) {
        if (node->request_slots[index].request_id == request_id) return index;
    }
    throw std::invalid_argument("RuntimeSession unknown or released request id");
}

void RequireUnbatchedState(const RuntimeSessionNode* node) {
    if (node->plan.request_batching()) {
        throw std::invalid_argument("RuntimeSession request batching requires the request API");
    }
}

}  // namespace

RuntimeSession::RuntimeSession(api::CompiledModule module, ExecutablePlan plan) {
    ValidatedPlanContract contract = ValidateModuleAndPlan(module, plan);
    const auto values = IndexValues(plan);
    // 观测器从模块继承：模块未启用 profiling 时访问器返回空，行为不变。
    std::shared_ptr<ExecutionObserver> observer =
        api::internal::BorrowCompiledModuleExecutionObserver(module);
    std::unordered_map<int64_t, NDArray> states;
    std::unordered_map<int64_t, NDArray> prefixes;
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
            if (plan.mode() == ExecutablePlanMode::kBoundedStatefulExternalV1) {
                for (const auto& binding : plan.state_output_bindings()) {
                    if (binding.state_value_id == value_id) {
                        prefixes.emplace(binding.input_value_id, NDArray::Empty(
                            spec.shape(), spec->dtype, spec->device, alignment->second));
                    }
                }
            }
        }
    }
    auto* node = new RuntimeSessionNode(
        std::move(module), std::move(plan), std::move(contract.device),
        std::move(contract.constant_keys_by_value),
        std::move(contract.required_alignment_by_storage), std::move(states),
        std::move(observer));
    node->state_prefixes_by_value = std::move(prefixes);
    if (node->plan.request_batching()) {
        node->request_slots.resize(static_cast<size_t>(
            values.at(node->plan.state_value_ids()[0]).shape()[0]));
    }
    // 动态有状态合同：初始有效长度为 0，由 completion 在成功后提交。
    for (int64_t value_id : node->plan.state_value_ids()) {
        node->length_book->lengths.emplace(value_id, 0);
    }
    SetData(node);
}

void RuntimeSession::Validate(const api::CompiledModule& module,
                              const ExecutablePlan& plan) {
    (void)ValidateModuleAndPlan(module, plan);
}

RuntimeSession::RuntimeSession(const ObjectRef& ref) : ObjectRef(ref) {
    if (defined() && !As<RuntimeSessionNode>()) {
        SetData(nullptr);
        throw std::invalid_argument("ObjectRef does not contain RuntimeSessionNode");
    }
    if (defined()) ValidateStoredSession(*operator->());
}

Array<NDArray> RuntimeSession::Run(const Array<NDArray>& inputs) const {
    return Run(inputs, ExecutionMetadata{});
}

Array<NDArray> RuntimeSession::Run(const Array<NDArray>& inputs,
                                   const ExecutionMetadata& metadata) const {
    const DeviceStream stream = DeviceStream::Default(operator->()->device);
    RunAsyncResult result = RunAsync(inputs, stream, metadata);
    result.completion.Wait();
    return result.outputs;
}

RunAsyncResult RuntimeSession::RunAsync(const Array<NDArray>& inputs,
                                        const DeviceStream& stream) const {
    return RunAsync(inputs, stream, ExecutionMetadata{});
}

RunAsyncResult RuntimeSession::RunAsync(const Array<NDArray>& inputs,
                                        const DeviceStream& stream,
                                        const ExecutionMetadata& metadata) const {
    RequireUnbatchedState(operator->());
    return RunAsyncImpl(operator->()->module, inputs, stream, metadata, nullptr);
}

RunAsyncResult RuntimeSession::RunAsyncWithModule(const api::CompiledModule& module,
    const Array<NDArray>& inputs, const DeviceStream& stream,
    const ExecutionMetadata& metadata) const {
    const auto* node = operator->();
    RequireUnbatchedState(node);
    ValidateExecutionModule(*node, module);
    return RunAsyncImpl(module, inputs, stream, metadata, nullptr);
}

RunAsyncResult RuntimeSession::RunAsyncImpl(const api::CompiledModule& module,
    const Array<NDArray>& inputs,
    const DeviceStream& stream, const ExecutionMetadata& metadata,
    const std::vector<size_t>* request_slots) const {
    const auto* node = operator->();
    // 观测器为空时全部钩子只有一次空判断，行为与未装配时完全一致。
    const auto observer_owner = api::internal::BorrowCompiledModuleExecutionObserver(module);
    ExecutionObserver* observer = observer_owner.get();
    ExecutionRunCorrelation correlation;
    if (observer) {
        correlation = NotifyRunStart(observer,
                                     ExecutionRunStart{node->device, inputs.size(),
                                                       node->plan.calls().size(), metadata});
    }
    // 底层分配与拷贝路径经线程本地作用域查询当前观测器与运行关联。
    std::optional<ExecutionObservationScope> observation;
    if (observer) observation.emplace(observer, correlation);
    std::size_t submit_count = 0;
    try {
        RunAsyncResult result = [&] {
            ValidateSessionStream(stream, node->device);

            const bool bounded_stateful =
                node->plan.mode() == ExecutablePlanMode::kBoundedStatefulExternalV1;
            const bool dynamic =
                node->plan.mode() == ExecutablePlanMode::kDynamicFreshOutputV1 || bounded_stateful;
            const bool external_stateful =
                node->plan.mode() == ExecutablePlanMode::kStaticStatefulExternalV1 || bounded_stateful;
            const bool stateful =
                node->plan.mode() == ExecutablePlanMode::kDynamicStatefulV1 || external_stateful;
            const Array<int64_t> input_ids = node->plan.input_value_ids();
            const auto values = IndexValues(node->plan);
            std::unordered_map<int64_t, StateOutputBinding> prefix_bindings;
            if (bounded_stateful) {
                for (const auto& binding : node->plan.state_output_bindings()) {
                    prefix_bindings.emplace(binding.input_value_id, binding);
                }
            }
            const size_t caller_count = input_ids.size() - prefix_bindings.size();
            if (inputs.size() != caller_count) {
                throw std::invalid_argument("RuntimeSession input count expected " +
                                            std::to_string(caller_count) + ", actual " +
                                            std::to_string(inputs.size()));
            }

            auto table = std::make_shared<internal::ValueTable>();
            if (observer) table->ObserveAllocations(observer, correlation);
            const Array<KernelCall> calls = node->plan.calls();
            const std::optional<StructuredSchedule> structured =
                node->plan.structured_schedule();
            Array<AsyncOperation> operations;
            std::unique_lock<std::mutex> state_lock;
            std::unordered_map<int64_t, int64_t> staged_state_lengths;
            std::vector<PendingStateOutputUpdate> pending_state_updates;
            std::unordered_map<int64_t, NDArray> prefixes;
            if (bounded_stateful) {
                state_lock = LockStateForRun(node);
                std::lock_guard<std::mutex> book_lock(node->length_book->mutex);
                for (const auto& binding : node->plan.state_output_bindings()) {
                    const auto& spec = values.at(binding.state_value_id);
                    const int64_t length = request_slots
                        ? node->request_slots.at(request_slots->front()).extent
                        : node->length_book->lengths.at(binding.state_value_id);
                    if (length < 0 || length > spec->state_capacity ||
                        binding.append_count > spec->state_capacity - length) {
                        throw std::invalid_argument("RuntimeSession bounded state append exceeds capacity before launch");
                    }
                    Array<int64_t> shape;
                    for (int64_t dimension : spec.shape()) shape.push_back(dimension);
                    if (request_slots) shape[0] = static_cast<int64_t>(request_slots->size());
                    shape[spec->state_extent_axis] = length;
                    prefixes.emplace(binding.input_value_id,
                        node->state_prefixes_by_value.at(binding.input_value_id).CreateView(shape, {}, 0));
                    staged_state_lengths.emplace(binding.state_value_id, length + binding.append_count);
                    pending_state_updates.push_back(PendingStateOutputUpdate{
                        binding, node->states_by_value.at(binding.state_value_id), NDArray(), length});
                }
            }
            Array<NDArray> graph_inputs;
            size_t caller_index = 0;
            for (int64_t input_id : input_ids) {
                const auto prefix = prefixes.find(input_id);
                const NDArray value = prefix == prefixes.end() ? inputs[caller_index++] : prefix->second;
                const ValueSpec& spec = FindValue(values, input_id, "RuntimeSession graph input");
                ValidateRuntimeValue(spec, value, "RuntimeSession graph input " + std::to_string(input_id), dynamic);
                table->Bind(spec, value);
                graph_inputs.push_back(value);
            }
            if (structured) {
                if (stateful) {
                    throw std::invalid_argument(
                        "RuntimeSession structured schedule does not combine with "
                        "persistent state");
                }
                if (dynamic) PreflightDynamicGraphInputs(node->plan, graph_inputs);
                const Map<String, NDArray>& constants =
                    api::internal::BorrowCompiledModuleConstants(module);
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
                ValidateBoundSourceArguments(module, node->plan, values, table);
                StructuredWalker walker(module, node->plan, *structured, values,
                                        node->required_alignment_by_storage,
                                        table, stream, &operations, observer,
                                        correlation, dynamic);
                walker.Run();
                submit_count += walker.submit_count();
            } else if (dynamic) {
                PreflightDynamicGraphInputs(node->plan, graph_inputs);
                // All caller shapes, shared B/P guards and capacities pass
                // before packing. Physical [B,C,...] rows become compact
                // [B,P,...] rows in reusable session storage.
                for (const auto& update : pending_state_updates) {
                    const auto prefix = prefixes.at(update.binding.input_value_id);
                    if (request_slots) {
                        for (size_t row = 0; row < request_slots->size(); ++row) {
                            CopyStateRange(LeadingRow(update.state, (*request_slots)[row]), 0,
                                LeadingRow(prefix, row), 0, update.prior_extent,
                                update.binding.source_extent_axis);
                        }
                    } else {
                        CopyStateRange(update.state, 0, prefix, 0,
                                       update.prior_extent, update.binding.source_extent_axis);
                    }
                }
                std::size_t call_index = 0;
                for (const auto& call : calls) {
                    std::optional<ExecutionObservationScope> kernel_observation;
                    if (observer) {
                        const KernelSubmitInfo kernel{node->device, call_index, std::string(call->symbol)};
                        kernel_observation.emplace(observer, correlation, &kernel);
                        DispatchExecutionObservation(observer, [&](ExecutionObserver& sink) {
                            sink.OnKernelBegin(
                                KernelSubmitInfo{node->device, call_index,
                                                 std::string(call->symbol)},
                                correlation);
                        });
                    }
                    operations.push_back(InvokeDynamicCall(
                        module, call, values, table, stream));
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
                // Validate every produced append before mutating any state.
                for (auto& update : pending_state_updates) {
                    update.source = table->Get(update.binding.source_value_id);
                    const auto actual = update.source.shape();
                    const auto physical = update.state.shape();
                    for (size_t axis = 0; axis < physical.size(); ++axis) {
                        const int64_t expected = request_slots && axis == 0
                            ? static_cast<int64_t>(request_slots->size())
                            : static_cast<int64_t>(axis) == update.binding.source_extent_axis
                                ? update.prior_extent + update.binding.append_count : physical[axis];
                        if (actual.size() != physical.size() || actual[axis] != expected) {
                            throw std::invalid_argument("RuntimeSession bounded state output does not match committed extent plus append");
                        }
                    }
                }
            } else if (stateful) {
                const Map<String, NDArray>& constants =
                    api::internal::BorrowCompiledModuleConstants(module);
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
                state_lock = LockStateForRun(node);
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
                ValidateBoundSourceArguments(module, node->plan, values, table);

                if (!external_stateful) {
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
                } else {
                    // Capacity decode 的静态 kernel ABI 没有 runtime extent 或
                    // append-count 参数；每次调用由 binding 声明固定追加段。
                    for (const StateOutputBinding& binding :
                         node->plan.state_output_bindings()) {
                        const ValueSpec& spec =
                            FindValue(values, binding.state_value_id,
                                      "RuntimeSession external stateful state");
                        std::lock_guard<std::mutex> book_lock(
                            node->length_book->mutex);
                        const auto length_it = node->length_book->lengths.find(
                            binding.state_value_id);
                        if (length_it == node->length_book->lengths.end()) {
                            throw std::logic_error(
                                "RuntimeSession external stateful length book lost a state");
                        }
                        const int64_t length = length_it->second;
                        if (length < 0 || length > spec->state_capacity ||
                            binding.append_count > spec->state_capacity - length) {
                            throw std::invalid_argument(
                                "RuntimeSession external stateful append exceeds capacity before launch");
                        }
                        staged_state_lengths[binding.state_value_id] =
                            length + binding.append_count;
                        pending_state_updates.push_back(PendingStateOutputUpdate{
                            binding, node->states_by_value.at(binding.state_value_id),
                            NDArray(), length});
                    }
                }

                std::size_t call_index = 0;
                for (const auto& call : calls) {
                    std::optional<ExecutionObservationScope> kernel_observation;
                    if (observer) {
                        const KernelSubmitInfo kernel{node->device, call_index, std::string(call->symbol)};
                        kernel_observation.emplace(observer, correlation, &kernel);
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
                    if (!external_stateful) {
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
                    const std::vector<uint64_t>* runtime_extents =
                        external_stateful ? nullptr : &call_extents;
                    Array<NDArray> arguments = PrepareCallArguments(
                        module, call, values,
                        node->required_alignment_by_storage, table, runtime_extents);
                    operations.push_back(InvokeOrderedModuleEntry(
                        module, call->symbol, arguments, stream,
                        runtime_extents));
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
                for (auto& update : pending_state_updates) {
                    update.source = table->Get(update.binding.source_value_id);
                }
            } else {
                const Map<String, NDArray>& constants =
                    api::internal::BorrowCompiledModuleConstants(module);
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
                ValidateBoundSourceArguments(module, node->plan, values, table);

                std::size_t call_index = 0;
                for (const auto& call : calls) {
                    std::optional<ExecutionObservationScope> kernel_observation;
                    if (observer) {
                        const KernelSubmitInfo kernel{node->device, call_index, std::string(call->symbol)};
                        kernel_observation.emplace(observer, correlation, &kernel);
                        DispatchExecutionObservation(observer, [&](ExecutionObserver& sink) {
                            sink.OnKernelBegin(
                                KernelSubmitInfo{node->device, call_index,
                                                 std::string(call->symbol)},
                                correlation);
                        });
                    }
                    Array<NDArray> arguments = PrepareCallArguments(
                        module, call, values,
                        node->required_alignment_by_storage, table);
                    operations.push_back(InvokeOrderedModuleEntry(
                        module, call->symbol, arguments, stream));
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
                const auto output = table->Get(value_id);
                if (request_slots && (output.shape().empty() ||
                    output.shape()[0] != static_cast<int64_t>(request_slots->size()))) {
                    throw std::invalid_argument("RuntimeSession output violates declared request batch axis");
                }
                outputs.push_back(output);
            }

            AsyncOperation completion = operations.empty()
                                            ? AsyncOperation::Completed(stream)
                                            : operations[operations.size() - 1];
            Array<AsyncOperation> prior_operations;
            for (size_t i = 0; i + 1 < operations.size(); ++i) {
                prior_operations.push_back(operations[i]);
            }
            auto state = std::make_shared<RuntimeExecutionState>(RuntimeExecutionState{
                module, node->plan, table, std::move(prior_operations)});
            completion.RetainDependencies(table->RetainedStorage(), std::move(state));
            if (external_stateful) {
                // External stateful graphs finish their state commit before
                // RunAsync returns.  Unlike observation callbacks this path
                // propagates copy failures to the caller and run receipt.
                completion.Wait();
                for (const auto& update : pending_state_updates) {
                    if (request_slots) {
                        for (size_t row = 0; row < request_slots->size(); ++row) {
                            CopyStateRange(LeadingRow(update.source, row), update.prior_extent,
                                LeadingRow(update.state, (*request_slots)[row]), update.prior_extent,
                                update.binding.append_count, update.binding.source_extent_axis);
                        }
                    } else {
                        CopyStateRange(update.source,
                                       bounded_stateful ? update.prior_extent : update.binding.source_slot,
                                       update.state, update.prior_extent,
                                       update.binding.append_count,
                                       update.binding.source_extent_axis);
                    }
                }
                std::lock_guard<std::mutex> book_lock(node->length_book->mutex);
                if (request_slots) {
                    const int64_t extent = pending_state_updates.front().prior_extent +
                        pending_state_updates.front().binding.append_count;
                    for (size_t slot : *request_slots) node->request_slots[slot].extent = extent;
                } else {
                    for (const auto& entry : staged_state_lengths) {
                        node->length_book->lengths[entry.first] = entry.second;
                    }
                }
            } else if (stateful) {
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
        if ((node->plan.mode() == ExecutablePlanMode::kDynamicStatefulV1 ||
             node->plan.mode() == ExecutablePlanMode::kStaticStatefulExternalV1 ||
             node->plan.mode() == ExecutablePlanMode::kBoundedStatefulExternalV1) &&
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
    RequireUnbatchedState(node);
    bool declared = false;
    for (int64_t value_id : node->plan.state_value_ids()) {
        declared = declared || value_id == state_value_id;
    }
    if (!declared) {
        throw std::invalid_argument("RuntimeSession has no such state value");
    }
    if (node->plan.mode() != ExecutablePlanMode::kDynamicStatefulV1 &&
        node->plan.mode() != ExecutablePlanMode::kStaticStatefulExternalV1 &&
        node->plan.mode() != ExecutablePlanMode::kBoundedStatefulExternalV1) {
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
    RequireUnbatchedState(node);
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

void RuntimeSession::InitializeState(int64_t state_value_id,
                                     const NDArray& contents,
                                     int64_t valid_extent) const {
    const auto* node = operator->();
    RequireUnbatchedState(node);
    if (node->plan.mode() != ExecutablePlanMode::kDynamicStatefulV1 &&
        node->plan.mode() != ExecutablePlanMode::kStaticStatefulExternalV1 &&
        node->plan.mode() != ExecutablePlanMode::kBoundedStatefulExternalV1) {
        throw std::invalid_argument(
            "RuntimeSession state initialization requires a stateful extent contract");
    }
    const auto values = IndexValues(node->plan);
    const ValueSpec& spec = FindValue(values, state_value_id,
                                      "RuntimeSession state initialization");
    const auto state_it = node->states_by_value.find(state_value_id);
    if (!spec->is_state || state_it == node->states_by_value.end()) {
        throw std::invalid_argument("RuntimeSession has no such state value");
    }
    if (!contents.defined() || !api::SameDType(spec->dtype, contents.dtype()) ||
        spec->device != contents.device() || !contents.IsContiguous() ||
        valid_extent < 0 || valid_extent > spec->state_capacity) {
        throw std::invalid_argument(
            "RuntimeSession initial state extent or layout is invalid");
    }
    const Array<int64_t> expected_shape = spec.shape();
    const Array<int64_t> source_shape = contents.shape();
    if (expected_shape.size() != source_shape.size() ||
        source_shape[spec->state_extent_axis] < valid_extent) {
        throw std::invalid_argument("RuntimeSession initial state shape is invalid");
    }
    for (size_t axis = 0; axis < expected_shape.size(); ++axis) {
        if (static_cast<int64_t>(axis) != spec->state_extent_axis &&
            expected_shape[axis] != source_shape[axis]) {
            throw std::invalid_argument("RuntimeSession initial state shape is invalid");
        }
    }
    std::unique_lock<std::mutex> state_lock(node->state_mutex);
    if (node->state_completion.defined() && !node->state_completion.IsReady()) {
        throw std::runtime_error("RuntimeSession state execution is still pending");
    }
    {
        std::lock_guard<std::mutex> book_lock(node->length_book->mutex);
        if (node->length_book->poisoned) {
            throw std::runtime_error(
                "RuntimeSession state write failed previously; the session must be reconstructed");
        }
        if (node->length_book->lengths.at(state_value_id) != 0) {
            throw std::invalid_argument(
                "RuntimeSession initial state may only be seeded before its first append");
        }
    }
    std::optional<ExecutionObservationScope> observation;
    if (node->observer) {
        observation.emplace(node->observer.get(), ExecutionRunCorrelation{});
    }
    try {
        CopyStateRange(contents, 0, state_it->second, 0, valid_extent,
                       spec->state_extent_axis);
    } catch (...) {
        std::lock_guard<std::mutex> book_lock(node->length_book->mutex);
        node->length_book->poisoned = true;
        throw;
    }
    std::lock_guard<std::mutex> book_lock(node->length_book->mutex);
    node->length_book->lengths[state_value_id] = valid_extent;
}

uint64_t RuntimeSession::AdmitRequest(const Array<NDArray>& initial_states,
                                     int64_t valid_extent) const {
    const auto* node = operator->();
    RequestAccess lock(node);
    const auto ids = node->plan.state_value_ids();
    const auto values = IndexValues(node->plan);
    if (valid_extent < 0 || (initial_states.empty() && valid_extent != 0) ||
        (!initial_states.empty() && initial_states.size() != ids.size())) {
        throw std::invalid_argument("Request admission requires all initial states and a valid extent");
    }
    size_t slot = 0;
    while (slot < node->request_slots.size() && node->request_slots[slot].request_id != 0) ++slot;
    if (slot == node->request_slots.size()) {
        throw std::runtime_error("RuntimeSession request slots are full");
    }
    if (node->next_request_id == 0) throw std::overflow_error("RuntimeSession request id exhausted");
    for (size_t index = 0; index < ids.size(); ++index) {
        const auto spec = values.at(ids[index]);
        if (valid_extent > spec->state_capacity) {
            throw std::invalid_argument("Request admission extent exceeds state capacity");
        }
        if (initial_states.empty()) continue;
        const auto& source = initial_states[index];
        auto shape = spec.shape();
        shape[0] = 1;
        shape[spec->state_extent_axis] = -1;
        if (!source.defined() || !api::SameDType(source.dtype(), spec->dtype) ||
            source.device() != spec->device || !source.IsContiguous()) {
            throw std::invalid_argument("Request admission initial state layout is invalid");
        }
        ValidateShape(shape, source.shape(), "Request admission initial state", true);
        if (source.shape()[spec->state_extent_axis] < valid_extent) {
            throw std::invalid_argument("Request admission initial state prefix is too short");
        }
    }
    std::optional<ExecutionObservationScope> observation;
    if (node->observer) observation.emplace(node->observer.get(), ExecutionRunCorrelation{});
    for (size_t index = 0; index < ids.size(); ++index) {
        const auto spec = values.at(ids[index]);
        const auto row = LeadingRow(node->states_by_value.at(ids[index]), slot);
        // A free slot is unpublished until every state has been initialized.
        std::vector<float> fill(row.NBytes() / sizeof(float), static_cast<float>(spec->state_fill));
        row.CopyFromBytes(fill.data(), row.NBytes());
        if (!initial_states.empty()) {
            CopyStateRange(initial_states[index], 0, row, 0, valid_extent, spec->state_extent_axis);
        }
    }
    const uint64_t id = node->next_request_id++;
    node->request_slots[slot].request_id = id;
    node->request_slots[slot].extent = valid_extent;
    return id;
}

void RuntimeSession::EnqueueRequest(uint64_t request_id, const Array<NDArray>& inputs) const {
    const auto* node = operator->();
    Array<NDArray> snapshots;
    RequestAccess lock(node);
    auto& slot = node->request_slots[FindRequest(node, request_id)];
    if (!slot.queued_inputs.empty()) {
        throw std::invalid_argument("RuntimeSession request already has a queued step");
    }
    const auto values = IndexValues(node->plan);
    const auto ids = node->plan.input_value_ids();
    std::unordered_map<int64_t, StateOutputBinding> prefixes;
    for (const auto& binding : node->plan.state_output_bindings()) {
        prefixes.emplace(binding.input_value_id, binding);
        const auto spec = values.at(binding.state_value_id);
        if (slot.extent > spec->state_capacity - binding.append_count) {
            throw std::invalid_argument("RuntimeSession request append exceeds capacity before launch");
        }
    }
    if (inputs.size() != ids.size() - prefixes.size()) {
        throw std::invalid_argument("RuntimeSession request input count does not match");
    }
    Array<NDArray> graph_inputs;
    std::vector<size_t> alignments;
    size_t caller = 0;
    for (int64_t id : ids) {
        const auto spec = values.at(id);
        const auto prefix = prefixes.find(id);
        NDArray value;
        if (prefix != prefixes.end()) {
            auto shape = values.at(prefix->second.state_value_id).shape();
            shape[0] = 1;
            shape[prefix->second.source_extent_axis] = slot.extent;
            value = node->state_prefixes_by_value.at(id).CreateView(shape, {}, 0);
        } else {
            value = inputs[caller++];
            alignments.push_back(node->required_alignment_by_storage.at(spec->storage_id));
        }
        ValidateRuntimeValue(spec, value, "RuntimeSession request input", true);
        if (value.shape().empty() || value.shape()[0] != 1) {
            throw std::invalid_argument("RuntimeSession individual request input batch must be one");
        }
        graph_inputs.push_back(value);
    }
    PreflightDynamicGraphInputs(node->plan, graph_inputs);
    std::optional<ExecutionObservationScope> observation;
    if (node->observer) observation.emplace(node->observer.get(), ExecutionRunCorrelation{});
    for (size_t index = 0; index < inputs.size(); ++index) {
        auto copy = NDArray::Empty(inputs[index].shape(), inputs[index].dtype(), node->device, alignments[index]);
        copy.CopyFrom(inputs[index]);
        snapshots.push_back(std::move(copy));
    }
    node->request_queue.push_back(request_id);
    slot.queued_inputs = std::move(snapshots);
}

std::vector<RequestResult> RuntimeSession::RunNextBatch(const ExecutionMetadata& metadata) const {
    return RunNextBatch(DeviceStream::Default(operator->()->device), metadata);
}

std::vector<RequestResult> RuntimeSession::RunNextBatch(const DeviceStream& stream,
    const ExecutionMetadata& metadata) const {
    return RunNextBatchImpl(operator->()->module, stream, metadata);
}

std::vector<RequestResult> RuntimeSession::RunNextBatchWithModule(
    const api::CompiledModule& module, const DeviceStream& stream,
    const ExecutionMetadata& metadata) const {
    ValidateExecutionModule(*operator->(), module);
    return RunNextBatchImpl(module, stream, metadata);
}

std::vector<RequestResult> RuntimeSession::RunNextBatchImpl(
    const api::CompiledModule& module, const DeviceStream& stream,
    const ExecutionMetadata& metadata) const {
    const auto* node = operator->();
    std::vector<Array<NDArray>> consumed;
    RequestAccess lock(node);
    ValidateSessionStream(stream, node->device);
    if (node->request_queue.empty()) return {};
    const auto& first = node->request_slots[FindRequest(node, node->request_queue.front())];
    std::vector<size_t> selected;
    const size_t maximum = static_cast<size_t>(node->plan.request_batching()->max_batch_size);
    for (uint64_t id : node->request_queue) {
        const size_t index = FindRequest(node, id);
        const auto& candidate = node->request_slots[index];
        bool compatible = candidate.extent == first.extent &&
            candidate.queued_inputs.size() == first.queued_inputs.size();
        for (size_t input = 0; compatible && input < first.queued_inputs.size(); ++input) {
            const auto a = first.queued_inputs[input].shape();
            const auto b = candidate.queued_inputs[input].shape();
            compatible = a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
        }
        if (compatible) selected.push_back(index);
        if (selected.size() == maximum) break;
    }
    Array<NDArray> inputs;
    const auto values = IndexValues(node->plan);
    std::unordered_set<int64_t> prefixes;
    for (const auto& binding : node->plan.state_output_bindings()) prefixes.insert(binding.input_value_id);
    ExecutionMetadata receipt = metadata;
    receipt["request_batch_size"] = std::to_string(selected.size());
    receipt["request_past_extent"] = std::to_string(first.extent);
    std::string request_ids;
    for (size_t slot : selected) {
        if (!request_ids.empty()) request_ids += ',';
        request_ids += std::to_string(node->request_slots[slot].request_id);
    }
    receipt["request_ids"] = std::move(request_ids);
    {
        const auto observer = api::internal::BorrowCompiledModuleExecutionObserver(module);
        std::optional<ExecutionObservationScope> observation;
        if (observer) observation.emplace(observer.get(), ExecutionRunCorrelation{});
        size_t caller = 0;
        for (int64_t id : node->plan.input_value_ids()) {
            if (prefixes.count(id)) continue;
            const auto spec = values.at(id);
            auto shape = first.queued_inputs[caller].shape();
            shape[0] = static_cast<int64_t>(selected.size());
            auto merged = NDArray::Empty(shape, spec->dtype, node->device,
                node->required_alignment_by_storage.at(spec->storage_id));
            for (size_t row = 0; row < selected.size(); ++row) {
                LeadingRow(merged, row).CopyFrom(node->request_slots[selected[row]].queued_inputs[caller]);
            }
            inputs.push_back(std::move(merged));
            ++caller;
        }
    }
    std::vector<RequestResult> results;
    results.reserve(selected.size());
    consumed.resize(selected.size());
    const auto run = RunAsyncImpl(module, inputs, stream, receipt, &selected);
    // Bounded external-state execution has already waited and committed here.
    try {
        for (size_t row = 0; row < selected.size(); ++row) {
            auto& slot = node->request_slots[selected[row]];
            RequestResult result{slot.request_id, {}};
            for (const auto& output : run.outputs) result.outputs.push_back(LeadingRow(output, row));
            results.push_back(std::move(result));
            std::swap(consumed[row], slot.queued_inputs);
            node->request_queue.erase(std::remove(node->request_queue.begin(),
                node->request_queue.end(), slot.request_id), node->request_queue.end());
        }
    } catch (...) {
        std::lock_guard<std::mutex> book_lock(node->length_book->mutex);
        node->length_book->poisoned = true;
        throw;
    }
    return results;
}

void RuntimeSession::ReleaseRequest(uint64_t request_id) const {
    const auto* node = operator->();
    Array<NDArray> discarded;
    RequestAccess lock(node);
    auto& slot = node->request_slots[FindRequest(node, request_id)];
    std::swap(discarded, slot.queued_inputs);
    node->request_queue.erase(std::remove(node->request_queue.begin(),
        node->request_queue.end(), request_id), node->request_queue.end());
    slot.request_id = 0;
    slot.extent = 0;
}

int64_t RuntimeSession::RequestExtent(uint64_t request_id) const {
    const auto* node = operator->();
    RequestAccess lock(node);
    return node->request_slots[FindRequest(node, request_id)].extent;
}

NDArray RuntimeSession::CopyRequestState(uint64_t request_id, int64_t state_value_id) const {
    const auto* node = operator->();
    RequestAccess lock(node);
    const size_t slot = FindRequest(node, request_id);
    const auto values = IndexValues(node->plan);
    const auto spec = FindValue(values, state_value_id, "RuntimeSession request state");
    if (!spec->is_state) throw std::invalid_argument("RuntimeSession request state id is not a state");
    auto shape = spec.shape();
    shape[0] = 1;
    shape[spec->state_extent_axis] = node->request_slots[slot].extent;
    std::optional<ExecutionObservationScope> observation;
    if (node->observer) observation.emplace(node->observer.get(), ExecutionRunCorrelation{});
    auto copy = NDArray::Empty(shape, spec->dtype, node->device,
        node->required_alignment_by_storage.at(spec->storage_id));
    CopyStateRange(LeadingRow(node->states_by_value.at(state_value_id), slot), 0,
                  copy, 0, shape[spec->state_extent_axis], spec->state_extent_axis);
    return copy;
}

const RuntimeSessionNode* RuntimeSession::operator->() const {
    const auto* node = As<RuntimeSessionNode>();
    if (!node) throw std::runtime_error("undefined or invalid RuntimeSession");
    return node;
}

}  // namespace kxc::runtime
