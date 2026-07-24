/*! \file src/runtime/compiled_module.cc
 * \brief 实现 opaque CompiledModule 的组装、校验与启动。
 */

#include "kxc/runtime/compiled_module.h"
#include "kxc/support/object_registration.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "internal/compiled_module_node.h"
#include "internal/kernel_argument_validation.h"
#include "kxc/runtime/device_api.h"

namespace kxc::api {

KXC_OBJECT_DEFINE_WITH_KEY(CompiledModuleNode, "kxc.api.CompiledModuleNode")

namespace {

Device TargetDevice(const Target& target) {
    return Device(target->device_type, target->device_id);
}

runtime::NDArray CloneConstantPayload(const runtime::NDArray& source,
                                      size_t alignment) {
    runtime::NDArray result = runtime::NDArray::Empty(
        source.shape(), source.dtype(), source.device(), alignment);
    result.CopyFrom(source);
    return result;
}

Map<String, runtime::NDArray> CloneConstantPayloads(
    const Map<String, runtime::NDArray>& source,
    const std::unordered_map<std::string, size_t>& alignments = {}) {
    Map<String, runtime::NDArray> result;
    for (const auto& item : source) {
        const auto alignment = alignments.find(std::string(item.first));
        const size_t required = alignment == alignments.end()
            ? item.second.storage()->alignment : alignment->second;
        result.Set(item.first, CloneConstantPayload(item.second, required));
    }
    return result;
}

void ValidateConstantValue(const String& constant_key,
                           const codegen::KernelArgSpec& spec,
                           const runtime::NDArray& value) {
    if (!value.defined()) {
        throw std::invalid_argument(
            "CompiledModule constant '" + std::string(constant_key) +
            "' is undefined");
    }
    if (!SameDType(value.dtype(), spec->dtype) ||
        value.device() != spec->device) {
        throw std::invalid_argument(
            "CompiledModule constant '" + std::string(constant_key) +
            "' dtype or device does not match its signature");
    }
    const Array<int64_t> expected_shape = spec.shape();
    const Array<int64_t> actual_shape = value.shape();
    if (expected_shape.size() != actual_shape.size()) {
        throw std::invalid_argument(
            "CompiledModule constant '" + std::string(constant_key) +
            "' rank does not match its signature");
    }
    for (size_t i = 0; i < expected_shape.size(); ++i) {
        if (expected_shape[i] != actual_shape[i]) {
            throw std::invalid_argument(
                "CompiledModule constant '" + std::string(constant_key) +
                "' shape does not match its signature");
        }
    }
}

bool SameConstantContract(const codegen::KernelArgSpec& lhs,
                          const codegen::KernelArgSpec& rhs) {
    if (!SameDType(lhs->dtype, rhs->dtype) || lhs->device != rhs->device) {
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

std::unordered_map<std::string, size_t> ValidateConstants(
    const std::vector<internal::CompiledModuleEntry>& entries,
    const Map<String, runtime::NDArray>& constants) {
    std::unordered_map<std::string, codegen::KernelArgSpec> required_specs;
    std::unordered_map<std::string, size_t> required_alignments;
    for (const auto& entry : entries) {
        for (const auto& spec : entry.signature.arguments()) {
            if (spec->role != codegen::KernelArgRole::kConstant) continue;
            const std::string key = std::string(spec->constant_key);
            auto inserted = required_specs.emplace(key, spec);
            if (!inserted.second &&
                !SameConstantContract(inserted.first->second, spec)) {
                throw std::invalid_argument(
                    "CompiledModule constant '" + key +
                    "' has conflicting signatures across entries");
            }
            if (spec->alignment > std::numeric_limits<size_t>::max()) {
                throw std::invalid_argument(
                    "CompiledModule constant '" + key +
                    "' alignment exceeds the host size type");
            }
            auto& alignment = required_alignments[key];
            alignment = std::max(
                alignment, static_cast<size_t>(spec->alignment));
        }
    }
    for (const auto& item : required_specs) {
        const String key(item.first);
        if (!constants.count(key)) {
            throw std::invalid_argument(
                "CompiledModule is missing constant '" + item.first + "'");
        }
        ValidateConstantValue(key, item.second, constants.at(key));
    }
    if (constants.size() != required_specs.size()) {
        throw std::invalid_argument(
            "CompiledModule constant table contains unexpected keys");
    }
    return required_alignments;
}

const CompiledModuleNode* CheckedNode(const CompiledModule& module) {
    const auto* node = module.As<CompiledModuleNode>();
    if (!node) throw std::runtime_error("undefined or invalid CompiledModule");
    return node;
}

const internal::CompiledModuleEntry& FindEntry(
    const CompiledModuleNode* node, const String& symbol) {
    const std::string wanted = std::string(symbol);
    const auto entry = node->entries_.find(wanted);
    if (entry != node->entries_.end()) return entry->second;
    throw std::out_of_range("CompiledModule has no function '" + wanted + "'");
}

std::shared_ptr<const ModuleInvocationContract> StaticContract(
    const codegen::KernelSignature& signature) {
    std::vector<ModuleInputContract> inputs;
    std::vector<ModuleTensorContract> outputs;
    for (const auto& arg : signature.arguments()) {
        if (arg->role == codegen::KernelArgRole::kInput) {
            ModuleInputContract input{arg->dtype, arg->device, arg.shape().size(), {}};
            const Array<int64_t> shape = arg.shape();
            for (size_t i = 0; i < shape.size(); ++i) {
                if (shape[i] != codegen::kDynamicDimension) {
                    input.axis_guards.push_back(ModuleAxisGuard{
                        i, static_cast<ModuleExtent>(shape[i]),
                        static_cast<ModuleExtent>(shape[i]), 1,
                        static_cast<ModuleExtent>(shape[i]), std::nullopt});
                }
            }
            inputs.push_back(std::move(input));
        } else if (arg->role == codegen::KernelArgRole::kOutput) {
            ModuleTensorContract output;
            output.dtype = arg->dtype; output.device = arg->device;
            output.alignment = static_cast<size_t>(arg->alignment);
            output.max_bytes = std::numeric_limits<size_t>::max();
            for (int64_t extent : arg.shape()) {
                if (extent == codegen::kDynamicDimension) {
                    throw std::invalid_argument("dynamic KernelSignature output requires an explicit module invocation contract");
                }
                const ModuleShapeExpr expr = ModuleShapeExpr::Const(static_cast<ModuleExtent>(extent));
                output.logical.push_back(expr); output.physical.push_back(expr); output.valid.push_back(expr);
            }
            outputs.push_back(std::move(output));
        }
    }
    return std::make_shared<ModuleInvocationContract>(std::move(inputs), std::move(outputs));
}

void ValidateEntry(const internal::CompiledModuleEntry& entry,
                   const Device& target_device) {
    if (!entry.signature.defined() || !entry.launch_metadata.defined()) {
        throw std::invalid_argument(
            "CompiledModule entry signature and launch metadata must be defined");
    }
    entry.signature.Validate();
    entry.launch_metadata.Validate();
    if (entry.launch_metadata->device != target_device) {
        throw std::invalid_argument(
            "CompiledModule target and launch metadata devices do not match");
    }
    if (!entry.executable.defined() || !entry.executable.IsReady()) {
        throw std::invalid_argument("CompiledModule executable must be ready");
    }
    if (entry.executable.signature().get() != entry.signature.get() ||
        entry.executable.launch_metadata().get() != entry.launch_metadata.get()) {
        throw std::invalid_argument(
            "CompiledModule executable contract does not match module metadata");
    }
    if (!entry.invocation_contract) {
        throw std::invalid_argument("CompiledModule entry has no invocation contract");
    }
    entry.invocation_contract->Validate(entry.signature);
    for (const auto& spec : entry.signature.arguments()) {
        if (spec->device != target_device) {
            throw std::invalid_argument(
                "CompiledModule signature device does not match target");
        }
    }
}

AsyncOperation LaunchEntry(
    const internal::CompiledModuleEntry& entry,
    const Map<String, runtime::NDArray>& constants,
    const Array<runtime::NDArray>& ordered_arguments,
    const DeviceStream& stream) {
    if (!entry.executable.IsReady()) {
        throw std::runtime_error(
            "kernel '" + std::string(entry.signature->symbol) +
            "' executable is not ready");
    }
    if (!stream.defined()) {
        throw std::invalid_argument(
            "kernel '" + std::string(entry.signature->symbol) +
            "' requires a defined DeviceStream");
    }
    if (stream.device() != entry.launch_metadata->device) {
        throw std::invalid_argument(
            "kernel '" + std::string(entry.signature->symbol) +
            "' stream device expected " +
            entry.launch_metadata->device.ToString() + ", actual " +
            stream.device().ToString());
    }

    const Array<codegen::KernelArgSpec> specs = entry.signature.arguments();
    if (ordered_arguments.size() != specs.size()) {
        throw std::invalid_argument(
            "kernel '" + std::string(entry.signature->symbol) +
            "' argument count expected " + std::to_string(specs.size()) +
            ", actual " + std::to_string(ordered_arguments.size()));
    }
    Array<runtime::NDArray> launch_arguments;
    for (size_t i = 0; i < specs.size(); ++i) {
        ValidateKernelArgument(entry.signature, i, specs[i],
                               ordered_arguments[i], constants);
        launch_arguments.push_back(
            specs[i]->role == codegen::KernelArgRole::kConstant
                ? constants.at(specs[i]->constant_key)
                : ordered_arguments[i]);
    }
    return entry.executable.Launch(launch_arguments, stream);
}

}  // namespace

CompiledModule internal::BuildCompiledModule(
    Target target,
    std::vector<CompiledModuleEntry> entries,
    Map<String, runtime::NDArray> constants,
    std::shared_ptr<profiling::ProfileContext> profile_context) {
    if (!target.defined()) {
        throw std::invalid_argument("CompiledModule target must be defined");
    }
    if (entries.empty()) {
        throw std::invalid_argument(
            "CompiledModule must contain at least one entry");
    }
    const Device target_device = TargetDevice(target);
    std::unordered_set<std::string> symbols;
    for (auto& entry : entries) {
        if (!entry.invocation_contract) entry.invocation_contract = StaticContract(entry.signature);
        ValidateEntry(entry, target_device);
        const std::string symbol = std::string(entry.signature->symbol);
        if (!symbols.insert(symbol).second) {
            throw std::invalid_argument(
                "CompiledModule contains duplicate symbol '" + symbol + "'");
        }
    }
    const auto constant_alignments = ValidateConstants(entries, constants);
    Map<String, runtime::NDArray> owned_constants;
    try {
        // No source stream/completion enters this boundary.  Drain CUDA's
        // nonblocking producer streams before taking the synchronous snapshot;
        // zero-byte constants require no backend interaction.
        if (target_device.device_type() == kCUDA) {
            for (const auto& item : constants) {
                if (item.second.NBytes() != 0) {
                    DeviceSynchronize(target_device);
                    break;
                }
            }
        }
        owned_constants = CloneConstantPayloads(constants, constant_alignments);
    } catch (const std::exception& error) {
        throw std::invalid_argument(
            std::string("CompiledModule failed to snapshot constants: ") +
            error.what());
    }

    return CompiledModule(ObjectRef(new CompiledModuleNode(
        std::move(target), std::move(entries), std::move(owned_constants),
        std::move(profile_context))));
}

CompiledModule::CompiledModule(const ObjectRef& ref) : ObjectRef(ref) {
    if (defined() && !As<CompiledModuleNode>()) {
        SetData(nullptr);
        throw std::invalid_argument(
            "ObjectRef does not contain CompiledModuleNode");
    }
}

namespace {
std::vector<ModuleExtent> Evaluate(const std::vector<ModuleShapeExpr>& expressions,
                                   const std::vector<std::vector<ModuleExtent>>& inputs) {
    std::vector<ModuleExtent> result;
    for (const auto& expression : expressions) result.push_back(expression.Evaluate(inputs));
    return result;
}
size_t CheckedBytes(const std::vector<ModuleExtent>& shape, DLDataType dtype) {
    ModuleExtent elements = 1;
    for (ModuleExtent extent : shape) {
        if (extent != 0 && elements > std::numeric_limits<ModuleExtent>::max() / extent)
            throw ModuleInvocationError(ModuleInvocationFailureKind::kResource, "module output element count overflow");
        elements *= extent;
    }
    const ModuleExtent item = static_cast<ModuleExtent>(dtype.bits / 8) * dtype.lanes;
    if (item && elements > std::numeric_limits<size_t>::max() / item)
        throw ModuleInvocationError(ModuleInvocationFailureKind::kResource, "module output byte count overflow");
    return static_cast<size_t>(elements * item);
}
Array<int64_t> ToShape(const std::vector<ModuleExtent>& shape) {
    Array<int64_t> result;
    for (ModuleExtent extent : shape) {
        if (extent > static_cast<ModuleExtent>(std::numeric_limits<int64_t>::max()))
            throw ModuleInvocationError(ModuleInvocationFailureKind::kResource, "module output extent exceeds NDArray range");
        result.push_back(static_cast<int64_t>(extent));
    }
    return result;
}
void ResolveInputs(const ModuleInvocationContract& contract, const Array<runtime::NDArray>& inputs,
                   std::vector<std::vector<ModuleExtent>>* dimensions) {
    if (inputs.size() != contract.inputs().size())
        throw ModuleInvocationError(ModuleInvocationFailureKind::kInvalidContract, "module invocation input count does not match contract");
    dimensions->clear();
    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& spec = contract.inputs()[i]; const auto& value = inputs[i];
        if (!value.defined() || !SameDType(value.dtype(), spec.dtype) || value.device() != spec.device ||
            value.shape().size() != spec.rank || !value.IsContiguous())
            throw ModuleInvocationError(ModuleInvocationFailureKind::kInvalidContract, "module invocation input dtype, device, rank, or layout does not match contract");
        std::vector<ModuleExtent> shape;
        for (int64_t extent : value.shape()) { if (extent < 0) throw ModuleInvocationError(ModuleInvocationFailureKind::kInvalidContract, "module invocation input has negative extent"); shape.push_back(static_cast<ModuleExtent>(extent)); }
        for (const auto& guard : spec.axis_guards) {
            const ModuleExtent value_extent = shape[guard.axis];
            if (value_extent < guard.lower || value_extent > guard.upper ||
                (guard.divisible_by && value_extent % guard.divisible_by) ||
                (guard.exact && value_extent != *guard.exact) ||
                (guard.equal_to && (guard.equal_to->input_index >= dimensions->size() ||
                 guard.equal_to->axis >= (*dimensions)[guard.equal_to->input_index].size() ||
                 value_extent != (*dimensions)[guard.equal_to->input_index][guard.equal_to->axis])))
                throw ModuleInvocationError(ModuleInvocationFailureKind::kGuard, "module invocation input guard rejected before allocation");
        }
        dimensions->push_back(std::move(shape));
    }
}
}  // namespace

AsyncOperation internal::InvokeCompiledModuleWithOutputs(
    const CompiledModule& module, const String& symbol, const Array<runtime::NDArray>& data_inputs,
    const Array<runtime::NDArray>& outputs, const DeviceStream& stream, std::size_t budget) {
    const auto* node = CheckedNode(module); const auto& entry = FindEntry(node, symbol);
    const ModuleInvocationContract& contract = *entry.invocation_contract;
#if !KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI
    if (!contract.IsConstantShape()) throw ModuleInvocationError(ModuleInvocationFailureKind::kDisabled, "dynamic compiled-module invocation ABI is disabled");
#endif
    std::vector<std::vector<ModuleExtent>> dimensions;
    ResolveInputs(contract, data_inputs, &dimensions);
    if (outputs.size() != contract.outputs().size()) throw ModuleInvocationError(ModuleInvocationFailureKind::kPreallocatedMismatch, "preallocated module output count does not match contract");
    size_t total = 0;
    for (size_t i = 0; i < outputs.size(); ++i) {
        const auto& tensor = contract.outputs()[i];
        const auto logical = Evaluate(tensor.logical, dimensions), physical = Evaluate(tensor.physical, dimensions), valid = Evaluate(tensor.valid, dimensions);
        if (logical.size()!=physical.size() || valid.size()!=logical.size()) throw ModuleInvocationError(ModuleInvocationFailureKind::kInvalidContract, "module output ranks differ");
        for (size_t d=0; d<logical.size(); ++d) if (valid[d] > logical[d] || logical[d] > physical[d]) throw ModuleInvocationError(ModuleInvocationFailureKind::kInvalidContract, "module output requires valid <= logical <= physical");
        const size_t bytes = CheckedBytes(physical, tensor.dtype);
        if (bytes > tensor.max_bytes || bytes > std::numeric_limits<size_t>::max() - total) throw ModuleInvocationError(ModuleInvocationFailureKind::kResource, "module output byte limit exceeded");
        total += bytes;
        if (!outputs[i].defined() || !SameDType(outputs[i].dtype(), tensor.dtype) || outputs[i].device()!=tensor.device || !outputs[i].IsContiguous() || outputs[i].shape().size()!=physical.size()) throw ModuleInvocationError(ModuleInvocationFailureKind::kPreallocatedMismatch, "preallocated module output does not match contract");
        const Array<int64_t> actual=outputs[i].shape(); for(size_t d=0;d<physical.size();++d) if(actual[d] < 0 || static_cast<ModuleExtent>(actual[d]) != physical[d]) throw ModuleInvocationError(ModuleInvocationFailureKind::kPreallocatedMismatch, "preallocated module output physical shape does not match contract");
    }
    const size_t limit = budget ? budget : contract.run_byte_budget();
    if (limit && total > limit) throw ModuleInvocationError(ModuleInvocationFailureKind::kResource, "module invocation run byte budget exceeded");
    Array<runtime::NDArray> ordered; size_t input=0, output=0;
    for (const auto& argument : entry.signature.arguments()) {
        if (argument->role == codegen::KernelArgRole::kInput) ordered.push_back(data_inputs[input++]);
        else if (argument->role == codegen::KernelArgRole::kOutput) ordered.push_back(outputs[output++]);
        else ordered.push_back(node->constants_.at(argument->constant_key));
    }
    AsyncOperation operation = LaunchEntry(entry, node->constants_, ordered, stream);
    Array<Storage> retained; for (const auto& value : ordered) retained.push_back(value.storage());
    operation.RetainDependencies(std::move(retained), std::make_shared<CompiledModule>(module));
    return operation;
}

ModuleInvocationResult CompiledModule::Invoke(const String& symbol,
                                               const Array<runtime::NDArray>& data_inputs,
                                               const DeviceStream& stream,
                                               std::size_t budget) const {
    const auto& entry = FindEntry(CheckedNode(*this), symbol); const auto& contract = *entry.invocation_contract;
#if !KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI
    if (!contract.IsConstantShape()) throw ModuleInvocationError(ModuleInvocationFailureKind::kDisabled, "dynamic compiled-module invocation ABI is disabled");
#endif
    std::vector<std::vector<ModuleExtent>> dimensions; ResolveInputs(contract, data_inputs, &dimensions);
    Array<runtime::NDArray> physical; std::vector<ModuleInvocationOutput> descriptors; size_t total=0;
    for (const auto& tensor : contract.outputs()) {
        const auto logical=Evaluate(tensor.logical,dimensions), allocated=Evaluate(tensor.physical,dimensions), valid=Evaluate(tensor.valid,dimensions);
        for(size_t d=0;d<logical.size();++d) if(valid[d]>logical[d]||logical[d]>allocated[d]) throw ModuleInvocationError(ModuleInvocationFailureKind::kInvalidContract,"module output requires valid <= logical <= physical");
        const size_t bytes=CheckedBytes(allocated,tensor.dtype); if(bytes>tensor.max_bytes||bytes>std::numeric_limits<size_t>::max()-total) throw ModuleInvocationError(ModuleInvocationFailureKind::kResource,"module output byte limit exceeded"); total+=bytes;
        physical.push_back(runtime::NDArray::Empty(ToShape(allocated),tensor.dtype,tensor.device,tensor.alignment)); descriptors.push_back(ModuleInvocationOutput{physical[physical.size()-1],logical,allocated,valid});
    }
    const size_t limit=budget?budget:contract.run_byte_budget(); if(limit&&total>limit) throw ModuleInvocationError(ModuleInvocationFailureKind::kResource,"module invocation run byte budget exceeded");
    AsyncOperation operation=internal::InvokeCompiledModuleWithOutputs(*this,symbol,data_inputs,physical,stream,budget);
    return ModuleInvocationResult{std::move(descriptors),std::move(operation)};
}

ModuleInvocationContract CompiledModule::invocation_contract(const String& symbol) const {
    return *FindEntry(CheckedNode(*this), symbol).invocation_contract;
}

codegen::KernelSignature CompiledModule::signature(const String& symbol) const {
    return FindEntry(CheckedNode(*this), symbol).signature;
}

codegen::KernelLaunchMetadata CompiledModule::launch_metadata(
    const String& symbol) const {
    return FindEntry(CheckedNode(*this), symbol).launch_metadata;
}

Map<String, runtime::NDArray> CompiledModule::constants() const {
    return CloneConstantPayloads(CheckedNode(*this)->constants_);
}

const Map<String, runtime::NDArray>&
internal::BorrowCompiledModuleConstants(const CompiledModule& module) {
    return CheckedNode(module)->constants_;
}

bool CompiledModule::HasFunction(const String& symbol) const {
    const auto* node = CheckedNode(*this);
    return node->entries_.count(std::string(symbol)) != 0;
}

size_t CompiledModule::entry_count() const {
    return CheckedNode(*this)->entries_.size();
}

Array<String> CompiledModule::symbols() const {
    const auto* node = CheckedNode(*this);
    Array<String> result;
    for (const auto& item : node->entries_) {
        result.push_back(item.second.signature->symbol);
    }
    return result;
}

bool CompiledModule::IsReady() const noexcept {
    const auto* node = As<CompiledModuleNode>();
    if (!node || node->entries_.empty()) return false;
    for (const auto& item : node->entries_) {
        if (!item.second.executable.IsReady()) return false;
    }
    return true;
}

String CompiledModule::GetStatus() const {
    return String(IsReady() ? "ready" : "not_ready");
}

String CompiledModule::GetProfileBundlePath() const {
    const auto* node = CheckedNode(*this);
    return String(node->profile_context_ ? node->profile_context_->bundle_dir() : "");
}

}  // namespace kxc::api
