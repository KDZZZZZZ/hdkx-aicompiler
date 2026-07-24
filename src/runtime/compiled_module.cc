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
            ModuleInputContract input;
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
        } else if (arg->role == codegen::KernelArgRole::kRuntimeExtent) {
            throw std::invalid_argument("KernelSignature runtime extent requires an explicit module invocation contract");
        } else if (arg->role == codegen::KernelArgRole::kOutput) {
            ModuleTensorContract output;
            size_t elements = 1;
            for (int64_t extent : arg.shape()) {
                if (extent == codegen::kDynamicDimension) {
                    throw std::invalid_argument("dynamic KernelSignature output requires an explicit module invocation contract");
                }
                if (extent != 0 && elements > std::numeric_limits<size_t>::max() /
                    static_cast<size_t>(extent)) {
                    throw std::invalid_argument("static KernelSignature output element count overflows size_t");
                }
                elements *= static_cast<size_t>(extent);
                const ModuleShapeExpr expr = ModuleShapeExpr::Const(static_cast<ModuleExtent>(extent));
                output.logical.push_back(expr); output.physical.push_back(expr); output.valid.push_back(expr);
            }
            const size_t item_bytes = static_cast<size_t>(arg->dtype.bits / 8) * arg->dtype.lanes;
            if (item_bytes != 0 && elements > std::numeric_limits<size_t>::max() / item_bytes) {
                throw std::invalid_argument("static KernelSignature output byte count overflows size_t");
            }
            output.max_bytes = elements * item_bytes;
            outputs.push_back(std::move(output));
        }
    }
    return std::make_shared<ModuleInvocationContract>(std::move(inputs), std::move(outputs), std::vector<ModuleRuntimeExtentScalar>{});
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
    try {
        std::vector<ModuleExtent> result; result.reserve(expressions.size());
        for (const auto& expression : expressions) result.push_back(expression.Evaluate(inputs));
        return result;
    } catch (const std::exception& error) {
        throw ModuleInvocationError(ModuleInvocationFailureKind::kInvalidContract,
                                    std::string("module shape evaluation failed: ") + error.what());
    }
}
size_t CheckedBytes(const std::vector<ModuleExtent>& shape, DLDataType dtype) {
    ModuleExtent elements = 1;
    for (ModuleExtent extent : shape) { if (extent && elements > std::numeric_limits<ModuleExtent>::max() / extent) throw ModuleInvocationError(ModuleInvocationFailureKind::kResource, "module output element count overflow"); elements *= extent; }
    const ModuleExtent item = static_cast<ModuleExtent>(dtype.bits / 8) * dtype.lanes;
    if (item && elements > std::numeric_limits<size_t>::max() / item) throw ModuleInvocationError(ModuleInvocationFailureKind::kResource, "module output byte count overflow");
    return static_cast<size_t>(elements * item);
}
Array<int64_t> ToShape(const std::vector<ModuleExtent>& shape) { Array<int64_t> result; for (ModuleExtent extent : shape) { if (extent > static_cast<ModuleExtent>(std::numeric_limits<int64_t>::max())) throw ModuleInvocationError(ModuleInvocationFailureKind::kResource, "module output extent exceeds NDArray range"); result.push_back(static_cast<int64_t>(extent)); } return result; }
struct ResolvedOutput { std::vector<ModuleExtent> logical, physical, valid; size_t bytes{0}; };
struct ResolvedInvocation { std::vector<std::vector<ModuleExtent>> dimensions; std::vector<ResolvedOutput> outputs; std::vector<ModuleExtent> scalars; };

ResolvedInvocation ResolveInvocation(const internal::CompiledModuleEntry& entry, const Map<String, runtime::NDArray>& constants, const Array<runtime::NDArray>& inputs, const DeviceStream& stream, size_t requested_budget) {
    const ModuleInvocationContract& contract=*entry.invocation_contract;
#if !KXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI
    if (!contract.IsConstantShape(entry.signature)) throw ModuleInvocationError(ModuleInvocationFailureKind::kDisabled, "dynamic compiled-module invocation ABI is disabled");
#endif
    if (!stream.defined() || stream.device()!=entry.launch_metadata->device) throw ModuleInvocationError(ModuleInvocationFailureKind::kInvalidContract, "module invocation stream is undefined or on the wrong device");
    if (inputs.size()!=contract.inputs().size()) throw ModuleInvocationError(ModuleInvocationFailureKind::kInvalidContract, "module invocation input count does not match contract");
    const auto args=entry.signature.arguments(); ResolvedInvocation result; result.dimensions.reserve(inputs.size()); size_t caller=0;
    for(size_t slot=0;slot<args.size();++slot) { const auto& arg=args[slot]; if(arg->role==codegen::KernelArgRole::kInput && caller<inputs.size()) { try { ValidateKernelArgument(entry.signature,slot,arg,inputs[caller],constants); } catch(const std::exception& e) { throw ModuleInvocationError(ModuleInvocationFailureKind::kInvalidContract,e.what()); } const auto& spec=contract.inputs()[caller]; std::vector<ModuleExtent> shape; for(auto extent:inputs[caller].shape()) shape.push_back(static_cast<ModuleExtent>(extent)); for(const auto& guard:spec.axis_guards) { const auto value=shape[guard.axis]; if(value<guard.lower||value>guard.upper||value%guard.divisible_by||(guard.exact&&value!=*guard.exact)||(guard.equal_to&&value!=result.dimensions[guard.equal_to->input_index][guard.equal_to->axis])) throw ModuleInvocationError(ModuleInvocationFailureKind::kGuard,"module invocation input guard rejected before allocation"); } result.dimensions.push_back(std::move(shape)); ++caller; } else if(arg->role==codegen::KernelArgRole::kConstant) { try { ValidateKernelArgument(entry.signature,slot,arg,constants.at(arg->constant_key),constants); } catch(const std::exception& e) { throw ModuleInvocationError(ModuleInvocationFailureKind::kInvalidContract,e.what()); } } }
    std::vector<codegen::KernelArgSpec> output_specs; for(const auto& arg:args) if(arg->role==codegen::KernelArgRole::kOutput) output_specs.push_back(arg);
    size_t total=0; result.outputs.reserve(contract.outputs().size());
    size_t output_index=0; for(const auto& tensor:contract.outputs()) { const auto& output_spec=output_specs[output_index++]; ResolvedOutput output{Evaluate(tensor.logical,result.dimensions),Evaluate(tensor.physical,result.dimensions),Evaluate(tensor.valid,result.dimensions)}; if(output.logical.size()!=output.physical.size()||output.valid.size()!=output.logical.size()) throw ModuleInvocationError(ModuleInvocationFailureKind::kInvalidContract,"module output ranks differ"); for(size_t d=0;d<output.logical.size();++d) if(output.valid[d]>output.logical[d]||output.logical[d]>output.physical[d]) throw ModuleInvocationError(ModuleInvocationFailureKind::kInvalidContract,"module output requires valid <= logical <= physical"); output.bytes=CheckedBytes(output.physical,output_spec->dtype); if(output.bytes>tensor.max_bytes||output.bytes>std::numeric_limits<size_t>::max()-total) throw ModuleInvocationError(ModuleInvocationFailureKind::kResource,"module output byte limit exceeded"); total+=output.bytes; result.outputs.push_back(std::move(output)); }
    const size_t contract_limit=contract.run_byte_budget(); const size_t limit=!requested_budget ? contract_limit : !contract_limit ? requested_budget : std::min(requested_budget,contract_limit); if(limit&&total>limit) throw ModuleInvocationError(ModuleInvocationFailureKind::kResource,"module invocation run byte budget exceeded");
    result.scalars.reserve(contract.runtime_extent_scalars().size()); try { for(const auto& scalar:contract.runtime_extent_scalars()) result.scalars.push_back(scalar.expression.Evaluate(result.dimensions)); } catch(const std::exception& error) { throw ModuleInvocationError(ModuleInvocationFailureKind::kInvalidContract,std::string("module shape evaluation failed: ")+error.what()); }
    return result;
}
void ValidatePreallocated(const internal::CompiledModuleEntry& entry, const Map<String, runtime::NDArray>& constants, const Array<runtime::NDArray>& outputs, const ResolvedInvocation& resolved) {
    const auto& contract=*entry.invocation_contract; if(outputs.size()!=resolved.outputs.size()) throw ModuleInvocationError(ModuleInvocationFailureKind::kPreallocatedMismatch,"preallocated module output count does not match contract"); const auto args=entry.signature.arguments(); size_t output=0;
    for(size_t slot=0;slot<args.size();++slot) if(args[slot]->role==codegen::KernelArgRole::kOutput) { try { ValidateKernelArgument(entry.signature,slot,args[slot],outputs[output],constants); } catch(const std::exception& e) { throw ModuleInvocationError(ModuleInvocationFailureKind::kPreallocatedMismatch,e.what()); } const auto& shape=outputs[output].shape(); const auto& expected=resolved.outputs[output].physical; if(shape.size()!=expected.size()) throw ModuleInvocationError(ModuleInvocationFailureKind::kPreallocatedMismatch,"preallocated output rank differs from resolved physical extent"); for(size_t d=0;d<shape.size();++d) if(shape[d]<0||static_cast<ModuleExtent>(shape[d])!=expected[d]) throw ModuleInvocationError(ModuleInvocationFailureKind::kPreallocatedMismatch,"preallocated output physical shape differs from resolved contract"); ++output; }
}
struct Quarantine { AsyncOperation operation; Array<Storage> storage; std::shared_ptr<CompiledModule> module; };
[[noreturn]] void FailedPostLaunch(AsyncOperation operation, Array<Storage> storage, std::shared_ptr<CompiledModule> module, const std::string& message) {
    try { if(operation.defined() && operation->stream.defined()) operation.Wait(); } catch (...) { (void)new Quarantine{std::move(operation),std::move(storage),std::move(module)}; }
    throw ModuleInvocationError(ModuleInvocationFailureKind::kLaunch,message);
}
AsyncOperation LaunchResolved(const CompiledModule& module, const internal::CompiledModuleEntry& entry, const Map<String, runtime::NDArray>& constants, const Array<runtime::NDArray>& inputs, const Array<runtime::NDArray>& outputs, const DeviceStream& stream, const ResolvedInvocation& resolved) {
    Array<runtime::NDArray> scalar_buffers; const auto& descriptors=entry.invocation_contract->runtime_extent_scalars(); const auto signature_args=entry.signature.arguments(); size_t scalar_slot=0;
    for(const auto& arg:signature_args) if(arg->role==codegen::KernelArgRole::kRuntimeExtent) { auto scalar=runtime::NDArray::Empty({1},arg->dtype,arg->device,static_cast<size_t>(arg->alignment)); scalar.CopyFromBytes(&resolved.scalars[scalar_slot++],sizeof(ModuleExtent)); scalar_buffers.push_back(std::move(scalar)); }
    Array<runtime::NDArray> ordered; size_t input=0, scalar=0, output=0; for(const auto& arg:entry.signature.arguments()) { if(arg->role==codegen::KernelArgRole::kInput) ordered.push_back(inputs[input++]); else if(arg->role==codegen::KernelArgRole::kRuntimeExtent) ordered.push_back(scalar_buffers[scalar++]); else if(arg->role==codegen::KernelArgRole::kOutput) ordered.push_back(outputs[output++]); else if(arg->role==codegen::KernelArgRole::kConstant) ordered.push_back(constants.at(arg->constant_key)); else throw ModuleInvocationError(ModuleInvocationFailureKind::kInvalidContract,"unknown kernel argument role"); }
    AsyncOperation operation; try { operation=LaunchEntry(entry,constants,ordered,stream); if(!operation.defined()||!operation->stream.defined()||operation.device()!=stream.device()||operation->completed==(operation->backend_event!=nullptr)) FailedPostLaunch(std::move(operation),{},std::make_shared<CompiledModule>(module),"kernel returned an invalid completion"); Array<Storage> retained; for(const auto& value:ordered) retained.push_back(value.storage()); try { operation.RetainDependencies(retained,std::make_shared<CompiledModule>(module)); } catch(...) { FailedPostLaunch(std::move(operation),std::move(retained),std::make_shared<CompiledModule>(module),"failed to retain launched module dependencies"); } } catch(const ModuleInvocationError&) { throw; } catch(const std::exception& e) { throw ModuleInvocationError(ModuleInvocationFailureKind::kLaunch,e.what()); } return operation;
}
}  // namespace

AsyncOperation internal::InvokeCompiledModuleWithOutputs(const CompiledModule& module, const String& symbol, const Array<runtime::NDArray>& data_inputs, const Array<runtime::NDArray>& outputs, const DeviceStream& stream, std::size_t budget) {
    const auto* node=CheckedNode(module); const auto& entry=FindEntry(node,symbol); const auto resolved=ResolveInvocation(entry,node->constants_,data_inputs,stream,budget); ValidatePreallocated(entry,node->constants_,outputs,resolved); return LaunchResolved(module,entry,node->constants_,data_inputs,outputs,stream,resolved);
}

ModuleInvocationResult CompiledModule::Invoke(const String& symbol, const Array<runtime::NDArray>& data_inputs, const DeviceStream& stream, std::size_t budget) const {
    const auto* node=CheckedNode(*this); const auto& entry=FindEntry(node,symbol); const auto resolved=ResolveInvocation(entry,node->constants_,data_inputs,stream,budget);
    Array<runtime::NDArray> outputs; std::vector<ModuleInvocationOutput> descriptors; descriptors.reserve(resolved.outputs.size());
    const auto args=entry.signature.arguments(); size_t output_slot=0; for(size_t i=0;i<resolved.outputs.size();++i) { while(args[output_slot]->role!=codegen::KernelArgRole::kOutput) ++output_slot; const auto output_spec=args[output_slot++]; const auto& output=resolved.outputs[i]; outputs.push_back(runtime::NDArray::Empty(ToShape(output.physical),output_spec->dtype,output_spec->device,static_cast<size_t>(output_spec->alignment))); descriptors.push_back(ModuleInvocationOutput{outputs[outputs.size()-1],output.logical,output.physical,output.valid}); }
    AsyncOperation operation=LaunchResolved(*this,entry,node->constants_,data_inputs,outputs,stream,resolved); return ModuleInvocationResult{std::move(descriptors),std::move(operation)};
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

const ModuleInvocationContract&
internal::BorrowCompiledModuleInvocationContract(
    const CompiledModule& module, const String& symbol) {
    return *FindEntry(CheckedNode(module), symbol).invocation_contract;
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
