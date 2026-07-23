/*! \file src/runtime/control_execution_plan.cc */

#include "kxc/runtime/control_execution_plan.h"

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "internal/compiled_module_node.h"
#include "internal/kernel_argument_validation.h"

namespace kxc::runtime {
namespace {

#ifndef KXC_ENABLE_CONTROL_RUNTIME
#define KXC_ENABLE_CONTROL_RUNTIME 0
#endif

[[noreturn]] void Fail(const std::string& detail) {
    throw std::invalid_argument("ControlExecutionPlan: " + detail);
}

bool IsKnownDType(const std::string& dtype) {
    static const std::set<std::string> types{
        "bool", "int8", "int16", "int32", "int64", "uint8", "uint16",
        "uint32", "uint64", "float16", "float32", "float64", "bfloat16"};
    return types.count(dtype) != 0;
}

DLDataType DType(const std::string& dtype) {
    if (dtype == "bool") return {kDLBool, 8, 1};
    if (dtype == "int8") return {kDLInt, 8, 1};
    if (dtype == "int16") return {kDLInt, 16, 1};
    if (dtype == "int32") return {kDLInt, 32, 1};
    if (dtype == "int64") return {kDLInt, 64, 1};
    if (dtype == "uint8") return {kDLUInt, 8, 1};
    if (dtype == "uint16") return {kDLUInt, 16, 1};
    if (dtype == "uint32") return {kDLUInt, 32, 1};
    if (dtype == "uint64") return {kDLUInt, 64, 1};
    if (dtype == "float16") return {kDLFloat, 16, 1};
    if (dtype == "float32") return {kDLFloat, 32, 1};
    if (dtype == "float64") return {kDLFloat, 64, 1};
    if (dtype == "bfloat16") return {kDLBfloat, 16, 1};
    Fail("unknown dtype");
}

bool IsCpu0(const Device& device) {
    return device.defined() && device == Device::CPU();
}

template <typename Id>
void RequireUnique(const std::vector<Id>& ids, const char* what) {
    std::unordered_set<Id> seen;
    for (const Id id : ids) {
        if (id < 0 || !seen.insert(id).second) {
            Fail(std::string(what) + " must be unique and non-negative");
        }
    }
}

template <typename Id>
bool SameSet(std::vector<Id> lhs, std::vector<Id> rhs) {
    std::sort(lhs.begin(), lhs.end());
    std::sort(rhs.begin(), rhs.end());
    return lhs == rhs;
}

bool SameShape(const std::vector<std::int64_t>& value,
               const Array<std::int64_t>& argument) {
    if (value.size() != argument.size()) return false;
    for (std::size_t i = 0; i < value.size(); ++i) {
        if (value[i] != argument[i]) return false;
    }
    return true;
}

bool SameContract(const ControlExecutionValueSpec& lhs,
                  const ControlExecutionValueSpec& rhs) {
    return lhs.dtype == rhs.dtype && lhs.shape == rhs.shape &&
           lhs.device == rhs.device;
}

bool SamePayload(const NDArray& lhs, const NDArray& rhs) {
    if (!lhs.defined() || !rhs.defined() ||
        !api::SameDType(lhs.dtype(), rhs.dtype()) ||
        lhs.device() != rhs.device()) {
        return false;
    }
    const Array<std::int64_t> lhs_shape = lhs.shape();
    const Array<std::int64_t> rhs_shape = rhs.shape();
    if (lhs_shape.size() != rhs_shape.size()) return false;
    for (std::size_t i = 0; i < lhs_shape.size(); ++i) {
        if (lhs_shape[i] != rhs_shape[i]) return false;
    }
    const std::size_t nbytes = lhs.NBytes();
    if (nbytes != rhs.NBytes()) return false;
    std::vector<std::uint8_t> lhs_bytes(nbytes);
    std::vector<std::uint8_t> rhs_bytes(nbytes);
    lhs.CopyToBytes(lhs_bytes.data(), nbytes);
    rhs.CopyToBytes(rhs_bytes.data(), nbytes);
    return lhs_bytes == rhs_bytes;
}

std::size_t CheckedAlignment(const codegen::KernelArgSpec& argument) {
    if (argument->alignment > std::numeric_limits<std::size_t>::max()) {
        Fail("kernel constant alignment exceeds host size_t");
    }
    return static_cast<std::size_t>(argument->alignment);
}

std::size_t ConstantAlignment(const codegen::KernelSignature& signature,
                              const String& key) {
    for (const auto& argument : signature.arguments()) {
        if (argument->role == codegen::KernelArgRole::kConstant &&
            argument->constant_key == key) {
            return CheckedAlignment(argument);
        }
    }
    Fail("bound kernel constant key is not present in its signature");
}

NDArray CopyCpuPayload(const NDArray& source, std::size_t alignment) {
    if (!source.defined() || source.device() != Device::CPU() ||
        !source.IsContiguous()) {
        Fail("constant snapshot requires a contiguous CPU payload");
    }
    NDArray snapshot = NDArray::Empty(source.shape(), source.dtype(),
                                      Device::CPU(), alignment);
    snapshot.CopyFrom(source);
    return snapshot;
}

Map<String, NDArray> SnapshotCpuConstants(
    const api::CompiledModule& module,
    const codegen::KernelSignature& signature) {
    const Map<String, NDArray> source_constants = module.constants();
    Map<String, NDArray> snapshots;
    const Array<codegen::KernelArgSpec> arguments = signature.arguments();
    for (std::size_t i = 0; i < arguments.size(); ++i) {
        const auto& argument = arguments[i];
        if (argument->role != codegen::KernelArgRole::kConstant) continue;
        if (argument->device != Device::CPU() ||
            !source_constants.count(argument->constant_key)) {
            Fail("constant snapshot requires a bound CPU fixture payload");
        }
        const NDArray& source = source_constants.at(argument->constant_key);
        api::ValidateKernelArgument(signature, i, argument, source,
                                    source_constants);
        snapshots.Set(argument->constant_key,
                      CopyCpuPayload(source, CheckedAlignment(argument)));
    }
    return snapshots;
}

struct State final {
    explicit State(const ControlExecutionPlanSpec& plan) : plan(plan) {}

    const ControlExecutionPlanSpec& plan;
    std::unordered_map<ControlExecutionValueId, const ControlExecutionValueSpec*> values;
    std::unordered_map<ControlExecutionRegionId, const ControlExecutionRegion*> regions;
    std::unordered_map<ControlExecutionTaskId, const ControlExecutionTask*> tasks;
    std::unordered_map<ControlExecutionValueId, ControlExecutionTaskId> producers;
    std::unordered_set<ControlExecutionValueId> sources;
    std::unordered_set<ControlExecutionValueId> constants;
    std::unordered_set<ControlExecutionValueId> body_arguments;
    std::unordered_map<ControlExecutionValueId,
                       std::pair<ControlExecutionRegionId, ControlExecutionRegionId>>
        body_argument_regions;
    std::unordered_map<ControlExecutionValueId, NDArray> constant_payloads;
    std::unordered_map<std::string, ControlExecutionValueId> constant_values_by_key;
    std::unordered_set<ControlExecutionRegionId> active;
    std::unordered_set<ControlExecutionRegionId> visited;
};

const ControlExecutionValueSpec& Value(const State& state,
                                       ControlExecutionValueId id,
                                       const char* where) {
    const auto found = state.values.find(id);
    if (found == state.values.end()) {
        Fail(std::string(where) + " references an unknown value");
    }
    return *found->second;
}

const ControlExecutionRegion& Region(const State& state,
                                     ControlExecutionRegionId id,
                                     const char* where) {
    const auto found = state.regions.find(id);
    if (found == state.regions.end()) {
        Fail(std::string(where) + " references an unknown region");
    }
    return *found->second;
}

bool Empty(const ControlExecutionBranchSpec& spec) {
    return spec.predicate < 0 && spec.then_region < 0 && spec.else_region < 0 &&
           spec.phis.empty();
}

bool Empty(const ControlExecutionLoopSpec& spec) {
    return spec.condition_region < 0 && spec.body_region < 0 &&
           spec.condition_value < 0 && spec.carried.empty() &&
           spec.max_trip_count < 0;
}

void ValidateArray(const ControlExecutionValueSpec& value, const NDArray& array,
                   const std::string& where) {
    if (!array.defined() || !array.storage().defined() || !array.IsContiguous()) {
        Fail(where + " requires a defined contiguous NDArray");
    }
    const DLDataType expected = DType(value.dtype);
    const DLDataType actual = array.dtype();
    if (expected.code != actual.code || expected.bits != actual.bits ||
        expected.lanes != actual.lanes || array.device() != value.device) {
        Fail(where + " dtype or device does not match its value spec");
    }
    const Array<std::int64_t> shape = array.shape();
    if (shape.size() != value.shape.size()) Fail(where + " rank does not match");
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] != value.shape[i]) Fail(where + " shape does not match");
    }
    try {
        const std::size_t nbytes = array.NBytes();
        array.storage().ValidateRange(array->byte_offset, nbytes);
        if (nbytes != 0 && array.storage().data() == nullptr) {
            Fail(where + " has null storage data");
        }
    } catch (const std::invalid_argument&) {
        throw;
    } catch (const std::exception& error) {
        Fail(where + " storage range is invalid: " + error.what());
    }
}

void ValidateKernel(State& state, const ControlExecutionTask& task) {
    if (!task.kernel.defined() || !Empty(task.branch) || !Empty(task.loop)) {
        Fail("kernel task has an invalid kind-specific contract");
    }
    task.kernel.Validate();
    if (task.device != Device::CPU() || task.kernel.device() != task.device ||
        task.stream != "default") {
        Fail("control runtime v1 kernel must target CPU:0 default stream");
    }
    const codegen::KernelSignature signature = task.kernel.signature();
    const codegen::KernelLaunchMetadata metadata = task.kernel.launch_metadata();
    signature.Validate();
    metadata.Validate();
    const Array<codegen::KernelArgSpec> arguments = signature.arguments();
    if (arguments.size() != task.argument_values.size() || task.outputs.empty()) {
        Fail("kernel ABI argument_values or outputs do not match its signature");
    }

    std::vector<ControlExecutionValueId> boundary;
    std::vector<ControlExecutionValueId> outputs;
    for (std::size_t i = 0; i < arguments.size(); ++i) {
        const auto& argument = arguments[i];
        const ControlExecutionValueId value_id = task.argument_values[i];
        const ControlExecutionValueSpec& value = Value(state, value_id, "kernel ABI");
        if (argument->device != Device::CPU() ||
            (argument->role == codegen::KernelArgRole::kOutput &&
             !argument->mutable_data) ||
            value.dtype.empty() || !SameShape(value.shape, argument.shape())) {
            Fail("kernel ABI value does not exactly match its signature");
        }
        const DLDataType dtype = DType(value.dtype);
        if (dtype.code != argument->dtype.code || dtype.bits != argument->dtype.bits ||
            dtype.lanes != argument->dtype.lanes || value.device != argument->device) {
            Fail("kernel ABI dtype/device does not exactly match its signature");
        }
        for (const std::int64_t dimension : argument.shape()) {
            if (dimension < 0) Fail("dynamic kernel signature shapes are unsupported");
        }
        switch (argument->role) {
            case codegen::KernelArgRole::kInput:
                if (state.constants.count(value_id)) {
                    Fail("kernel input ABI cannot bind a graph constant");
                }
                boundary.push_back(value_id);
                break;
            case codegen::KernelArgRole::kConstant: {
                if (!state.constants.count(value_id)) {
                    Fail("kernel constant ABI must bind a graph constant");
                }
                boundary.push_back(value_id);
                const NDArray payload =
                    task.kernel.Constant(argument->constant_key);
                ValidateArray(value, payload, "kernel constant binding");
                if (payload.NBytes() != 0) {
                    const uintptr_t base =
                        reinterpret_cast<uintptr_t>(payload.storage().data());
                    if (payload->byte_offset >
                            std::numeric_limits<uintptr_t>::max() - base ||
                        (base + payload->byte_offset) % argument->alignment != 0) {
                        Fail("kernel constant binding alignment does not match ABI");
                    }
                }
                const auto existing = state.constant_payloads.find(value_id);
                if (existing != state.constant_payloads.end() &&
                    !SamePayload(existing->second, payload)) {
                    Fail("one logical constant has different bound payloads");
                }
                const std::string key = std::string(argument->constant_key);
                const auto reverse = state.constant_values_by_key.emplace(key, value_id);
                if (!reverse.second && reverse.first->second != value_id) {
                    Fail("one module constant key maps to different logical values");
                }
                state.constant_payloads[value_id] = payload;
                break;
            }
            case codegen::KernelArgRole::kOutput:
                outputs.push_back(value_id);
                break;
        }
    }
    std::sort(boundary.begin(), boundary.end());
    boundary.erase(std::unique(boundary.begin(), boundary.end()), boundary.end());
    if (!SameSet(boundary, task.inputs) || outputs != task.outputs) {
        Fail("kernel inputs or ordered outputs do not exactly map signature ABI roles");
    }
}

void ValidateRegion(State& state, ControlExecutionRegionId id,
                    const std::unordered_set<ControlExecutionValueId>& parent_available);

void ValidateBranch(State& state, const ControlExecutionTask& task,
                    const std::unordered_set<ControlExecutionValueId>& available) {
    const auto& spec = task.branch;
    const auto& predicate = Value(state, spec.predicate, "branch predicate");
    if (predicate.dtype != "bool" || !predicate.shape.empty() ||
        predicate.device != Device::CPU() || spec.then_region == spec.else_region ||
        spec.then_region < 0 || spec.else_region < 0) {
        Fail("branch requires a CPU scalar bool and distinct child regions");
    }
    const auto& then_region = Region(state, spec.then_region, "branch");
    const auto& else_region = Region(state, spec.else_region, "branch");
    std::vector<ControlExecutionValueId> required{spec.predicate};
    required.insert(required.end(), then_region.live_ins.begin(), then_region.live_ins.end());
    required.insert(required.end(), else_region.live_ins.begin(), else_region.live_ins.end());
    std::sort(required.begin(), required.end());
    required.erase(std::unique(required.begin(), required.end()), required.end());
    if (!SameSet(required, task.inputs) || spec.phis.empty()) {
        Fail("branch task inputs do not close children or has no Phi bindings");
    }
    std::unordered_set<ControlExecutionValueId> then_out(then_region.live_outs.begin(), then_region.live_outs.end());
    std::unordered_set<ControlExecutionValueId> else_out(else_region.live_outs.begin(), else_region.live_outs.end());
    std::vector<ControlExecutionValueId> results;
    for (const auto& phi : spec.phis) {
        const auto& result = Value(state, phi.result, "Phi");
        const auto& then_value = Value(state, phi.then_value, "Phi");
        const auto& else_value = Value(state, phi.else_value, "Phi");
        if (!then_out.count(phi.then_value) || !else_out.count(phi.else_value) ||
            !SameContract(result, then_value) || !SameContract(result, else_value)) {
            Fail("Phi must logically forward matching selected child live-outs");
        }
        results.push_back(phi.result);
    }
    RequireUnique(results, "Phi results");
    if (!SameSet(results, task.outputs) || !task.argument_values.empty() ||
        task.kernel.defined() || !Empty(task.loop)) {
        Fail("branch task has an invalid kind-specific contract");
    }
    ValidateRegion(state, spec.then_region, available);
    ValidateRegion(state, spec.else_region, available);
}

void ValidateLoop(State& state, const ControlExecutionTask& task,
                  const std::unordered_set<ControlExecutionValueId>& available) {
    const auto& spec = task.loop;
    if (spec.max_trip_count < 0 || spec.condition_region < 0 || spec.body_region < 0 ||
        spec.condition_region == spec.body_region || spec.carried.empty()) {
        Fail("loop requires bounded, distinct condition/body regions and carried values");
    }
    const auto& condition = Region(state, spec.condition_region, "loop");
    const auto& body = Region(state, spec.body_region, "loop");
    const auto& condition_value = Value(state, spec.condition_value, "loop condition");
    if (condition_value.dtype != "bool" || !condition_value.shape.empty() ||
        condition_value.device != Device::CPU() ||
        std::find(condition.live_outs.begin(), condition.live_outs.end(),
                  spec.condition_value) == condition.live_outs.end()) {
        Fail("loop condition must be a CPU scalar bool condition-region live-out");
    }
    std::unordered_set<ControlExecutionValueId> arguments;
    std::vector<ControlExecutionValueId> initial;
    std::vector<ControlExecutionValueId> results;
    for (const auto& carried : spec.carried) {
        const auto& result = Value(state, carried.result, "loop carried");
        const auto& initial_value = Value(state, carried.initial, "loop carried");
        const auto& argument = Value(state, carried.body_argument, "loop carried");
        const auto& backedge = Value(state, carried.backedge, "loop carried");
        if (!arguments.insert(carried.body_argument).second ||
            !SameContract(result, initial_value) || !SameContract(result, argument) ||
            !SameContract(result, backedge) ||
            std::find(body.live_outs.begin(), body.live_outs.end(), carried.backedge) == body.live_outs.end() ||
            std::find(condition.live_ins.begin(), condition.live_ins.end(), carried.body_argument) == condition.live_ins.end() ||
            std::find(body.live_ins.begin(), body.live_ins.end(), carried.body_argument) == body.live_ins.end()) {
            Fail("loop carried contract is not static-exact");
        }
        initial.push_back(carried.initial);
        results.push_back(carried.result);
    }
    std::vector<ControlExecutionValueId> required = initial;
    for (const auto value : condition.live_ins) if (!arguments.count(value)) required.push_back(value);
    for (const auto value : body.live_ins) if (!arguments.count(value)) required.push_back(value);
    std::sort(required.begin(), required.end());
    required.erase(std::unique(required.begin(), required.end()), required.end());
    if (!SameSet(required, task.inputs) || !SameSet(results, task.outputs) ||
        !task.argument_values.empty() || task.kernel.defined() || !Empty(task.branch)) {
        Fail("loop task has an invalid kind-specific contract");
    }
    ValidateRegion(state, spec.condition_region, available);
    ValidateRegion(state, spec.body_region, available);
}

void ValidateRegion(State& state, ControlExecutionRegionId id,
                    const std::unordered_set<ControlExecutionValueId>& parent_available) {
    if (state.active.count(id) || state.visited.count(id)) {
        Fail("regions must form one reachable structured tree");
    }
    const auto& region = Region(state, id, "region");
    if (region.source_locator.empty()) Fail("region source_locator is required");
    RequireUnique(region.live_ins, "region live_ins");
    RequireUnique(region.live_outs, "region live_outs");
    state.active.insert(id);
    for (const auto value : region.live_ins) {
        Value(state, value, "region live_in");
        if (!parent_available.count(value)) {
            const auto body = state.body_argument_regions.find(value);
            if (body == state.body_argument_regions.end() ||
                (body->second.first != id && body->second.second != id)) {
                Fail("region live_in is unavailable from its parent");
            }
        }
    }
    std::unordered_set<ControlExecutionValueId> available(region.live_ins.begin(), region.live_ins.end());
    std::unordered_set<ControlExecutionTaskId> prior;
    std::unordered_map<ControlExecutionValueId, ControlExecutionTaskId> local_producers;
    for (const auto& task : region.tasks) {
        if (task.source_locator.empty()) Fail("task source_locator is required");
        RequireUnique(task.inputs, "task inputs");
        RequireUnique(task.outputs, "task outputs");
        RequireUnique(task.dependencies, "task dependencies");
        if (!IsCpu0(task.device) || task.stream != "default") {
            Fail("control runtime v1 requires CPU:0 and one default stream");
        }
        for (const auto dependency : task.dependencies) {
            if (!prior.count(dependency)) Fail("task dependency is forward or outside its region");
        }
        for (const auto input : task.inputs) {
            Value(state, input, "task input");
            if (!available.count(input)) Fail("task input is unavailable");
            const auto producer = local_producers.find(input);
            if (producer != local_producers.end() &&
                std::find(task.dependencies.begin(), task.dependencies.end(), producer->second) == task.dependencies.end()) {
                Fail("task omits its local input producer dependency");
            }
        }
        switch (task.kind) {
            case ControlExecutionTaskKind::kKernel:
                ValidateKernel(state, task);
                break;
            case ControlExecutionTaskKind::kBranch:
                ValidateBranch(state, task, available);
                break;
            case ControlExecutionTaskKind::kLoop:
                ValidateLoop(state, task, available);
                break;
            default:
                Fail("unknown control execution task kind");
        }
        for (const auto output : task.outputs) {
            Value(state, output, "task output");
            if (state.sources.count(output) || state.body_arguments.count(output) || available.count(output)) {
                Fail("task output is not a fresh logical value");
            }
            available.insert(output);
            local_producers.emplace(output, task.id);
        }
        prior.insert(task.id);
    }
    for (const auto output : region.live_outs) {
        Value(state, output, "region live_out");
        if (!available.count(output)) Fail("region live_out is unavailable");
    }
    state.active.erase(id);
    state.visited.insert(id);
}

}  // namespace

struct BoundControlKernel::State final {
    State(api::CompiledModule module, codegen::KernelSignature signature,
          codegen::KernelLaunchMetadata metadata,
          codegen::CompiledKernel executable, Map<String, NDArray> constants,
          std::uint64_t binding_revision, std::uint64_t authority_generation,
          std::string authority_lease, std::shared_ptr<const void> retention_lease)
        : module(std::move(module)),
          signature(std::move(signature)),
          metadata(std::move(metadata)),
          executable(std::move(executable)),
          constants(std::move(constants)),
          binding_revision(binding_revision),
          authority_generation(authority_generation),
          authority_lease(std::move(authority_lease)),
          retention_lease(std::move(retention_lease)) {}

    api::CompiledModule module;
    codegen::KernelSignature signature;
    codegen::KernelLaunchMetadata metadata;
    codegen::CompiledKernel executable;
    Map<String, NDArray> constants;
    const std::uint64_t binding_revision{0};
    const std::uint64_t authority_generation{0};
    const std::string authority_lease;
    // Keeps compiler-owned immutable pins alive without exposing compiler API.
    const std::shared_ptr<const void> retention_lease;
};

BoundControlKernel::BoundControlKernel(api::CompiledModule module,
                                       String entry_symbol,
                                       std::uint64_t binding_revision,
                                       std::uint64_t authority_generation,
                                       std::string authority_lease,
                                       std::shared_ptr<const void> retention_lease) {
    const bool production = authority_generation != 0;
    if (!module.defined() || !module.IsReady() || entry_symbol == "" ||
        !module.HasFunction(entry_symbol) ||
        (production == authority_lease.empty()) ||
        (production == (binding_revision != 0)) ||
        (production && !retention_lease)) {
        throw std::invalid_argument(
            "BoundControlKernel requires a ready module entry and exactly one fixture revision or retained production authority");
    }
    const auto* node = module.As<api::CompiledModuleNode>();
    const auto entry = node->entries_.find(std::string(entry_symbol));
    if (entry == node->entries_.end()) {
        throw std::invalid_argument("BoundControlKernel module entry disappeared");
    }
    const codegen::KernelSignature signature = entry->second.signature;
    const codegen::KernelLaunchMetadata metadata = entry->second.launch_metadata;
    const codegen::CompiledKernel executable = entry->second.executable;
    signature.Validate();
    metadata.Validate();
    if (metadata->device != Device::CPU()) {
        throw std::invalid_argument(
            "BoundControlKernel fixture binding requires CPU:0 metadata");
    }
    Map<String, NDArray> constants = SnapshotCpuConstants(module, signature);
    state_ = std::make_shared<State>(
        std::move(module), signature, metadata, executable,
        std::move(constants), binding_revision, authority_generation,
        std::move(authority_lease), std::move(retention_lease));
    Validate();
}

void BoundControlKernel::Validate() const {
    if (!state_ ||
        (state_->binding_revision == 0 && state_->authority_generation == 0) ||
        !state_->module.defined() ||
        !state_->executable.defined() || !state_->executable.IsReady() ||
        state_->executable.signature().get() != state_->signature.get() ||
        state_->executable.launch_metadata().get() != state_->metadata.get()) {
        throw std::invalid_argument(
            "BoundControlKernel immutable entry is no longer ready");
    }
    state_->signature.Validate();
    state_->metadata.Validate();
    if (state_->metadata->device != Device::CPU()) {
        throw std::invalid_argument("BoundControlKernel v1 requires CPU:0 metadata");
    }
}

AsyncOperation BoundControlKernel::Launch(
    const Array<NDArray>& ordered_arguments,
    const DeviceStream& stream) const {
#if KXC_ENABLE_CONTROL_RUNTIME
    Validate();
    if (!stream.defined() || stream.device() != state_->metadata->device ||
        !stream.is_default()) {
        throw std::invalid_argument(
            "BoundControlKernel requires its immutable entry's default stream");
    }
    const Array<codegen::KernelArgSpec> arguments = state_->signature.arguments();
    if (ordered_arguments.size() != arguments.size()) {
        throw std::invalid_argument(
            "BoundControlKernel argument count does not match its signature");
    }
    Array<NDArray> launch_arguments;
    for (std::size_t i = 0; i < arguments.size(); ++i) {
        const auto& spec = arguments[i];
        if (spec->role == codegen::KernelArgRole::kConstant) {
            Map<String, NDArray> supplied;
            supplied.Set(spec->constant_key, ordered_arguments[i]);
            api::ValidateKernelArgument(state_->signature, i, spec,
                                        ordered_arguments[i], supplied);
            const NDArray& snapshot = state_->constants.at(spec->constant_key);
            if (!SamePayload(snapshot, ordered_arguments[i])) {
                throw std::invalid_argument(
                    "BoundControlKernel constant argument differs from its private snapshot");
            }
            launch_arguments.push_back(snapshot);
        } else {
            api::ValidateKernelArgument(state_->signature, i, spec,
                                        ordered_arguments[i], state_->constants);
            launch_arguments.push_back(ordered_arguments[i]);
        }
    }
    AsyncOperation operation =
        state_->executable.Launch(launch_arguments, stream);
    if (!operation.defined() || !operation->stream.defined() ||
        operation.device() != stream.device() || !operation->stream.is_default() ||
        operation->completed == (operation->backend_event != nullptr)) {
        throw std::invalid_argument(
            "BoundControlKernel launcher returned an invalid completion");
    }
    return operation;
#else
    (void)ordered_arguments;
    (void)stream;
    throw std::runtime_error(
        "BoundControlKernel launch is disabled by KXC_ENABLE_CONTROL_RUNTIME");
#endif
}

codegen::KernelSignature BoundControlKernel::signature() const {
    Validate();
    return state_->signature;
}

codegen::KernelLaunchMetadata BoundControlKernel::launch_metadata() const {
    Validate();
    return state_->metadata;
}

NDArray BoundControlKernel::Constant(const String& key) const {
    Validate();
    if (!state_->constants.count(key)) {
        throw std::invalid_argument("BoundControlKernel constant is missing");
    }
    return CopyCpuPayload(state_->constants.at(key),
                          ConstantAlignment(state_->signature, key));
}

bool BoundControlKernel::MatchesConstant(
    const String& key, const NDArray& candidate) const {
    Validate();
    if (!state_->constants.count(key)) {
        throw std::invalid_argument("BoundControlKernel constant is missing");
    }
    return SamePayload(state_->constants.at(key), candidate);
}

std::uint64_t BoundControlKernel::binding_revision() const {
    if (!state_) throw std::runtime_error("undefined BoundControlKernel");
    return state_->binding_revision;
}

std::uint64_t BoundControlKernel::authority_generation() const {
    if (!state_) throw std::runtime_error("undefined BoundControlKernel");
    return state_->authority_generation;
}

const std::string& BoundControlKernel::authority_lease() const {
    if (!state_) throw std::runtime_error("undefined BoundControlKernel");
    return state_->authority_lease;
}

Device BoundControlKernel::device() const { return launch_metadata()->device; }
bool BoundControlKernel::defined() const noexcept { return static_cast<bool>(state_); }

void VerifyControlExecutionPlan(const ControlExecutionPlanSpec& plan) {
    if (plan.schema_version != ControlExecutionPlanSpec::kSchemaVersion ||
        plan.source_control_plan_version != 2 ||
        plan.effect_model !=
            ControlExecutionEffectModel::kPureFreshKernelOutputsV1) {
        Fail("requires schema v1, ControlPlan v2 provenance, and kPureFreshKernelOutputsV1");
    }
    if (plan.values.empty() || plan.regions.empty() || plan.graph_outputs.empty()) {
        Fail("values, regions, and graph outputs are required");
    }
    State state(plan);
    for (const auto& value : plan.values) {
        if (value.id < 0 || !state.values.emplace(value.id, &value).second ||
            !IsKnownDType(value.dtype) || !IsCpu0(value.device) ||
            value.source_locator.empty()) {
            Fail("value ids/specs must be unique, static CPU:0 contracts with locators");
        }
        for (const auto dimension : value.shape) {
            if (dimension < 0) Fail("dynamic shapes are unsupported everywhere");
        }
    }
    RequireUnique(plan.graph_inputs, "graph inputs");
    RequireUnique(plan.constant_values, "constant values");
    RequireUnique(plan.graph_outputs, "graph outputs");
    for (const auto value : plan.graph_inputs) {
        Value(state, value, "graph input");
        state.sources.insert(value);
    }
    for (const auto value : plan.constant_values) {
        Value(state, value, "constant value");
        if (!state.sources.insert(value).second) Fail("graph inputs and constants overlap");
        state.constants.insert(value);
    }
    for (const auto value : plan.graph_outputs) Value(state, value, "graph output");
    RequireUnique(plan.region_order, "region order");
    if (plan.region_order.size() != plan.regions.size()) Fail("region_order must list each region once");
    for (const auto& region : plan.regions) {
        if (region.id < 0 || !state.regions.emplace(region.id, &region).second) {
            Fail("region ids must be unique and non-negative");
        }
        for (const auto& task : region.tasks) {
            if (task.id < 0 || !state.tasks.emplace(task.id, &task).second) {
                Fail("task ids must be unique and non-negative");
            }
            for (const auto output : task.outputs) {
                Value(state, output, "task output");
                if (!state.producers.emplace(output, task.id).second) Fail("value has duplicate producers");
            }
            if (task.kind == ControlExecutionTaskKind::kLoop) {
                for (const auto& carried : task.loop.carried) {
                    Value(state, carried.body_argument, "loop body_argument");
                    if (!state.body_arguments.insert(carried.body_argument).second) {
                        Fail("loop body_argument must be unique");
                    }
                    state.body_argument_regions.emplace(carried.body_argument,
                        std::make_pair(task.loop.condition_region, task.loop.body_region));
                }
            }
        }
    }
    if (!state.regions.count(plan.entry_region)) Fail("entry region is missing");
    for (const auto region : plan.region_order) if (!state.regions.count(region)) Fail("region_order has unknown region");
    for (const auto& item : state.values) {
        const auto value = item.first;
        if (state.sources.count(value) || state.body_arguments.count(value)) {
            if (state.sources.count(value) && state.body_arguments.count(value)) Fail("loop body argument cannot be a source");
            if (state.producers.count(value)) Fail("source/body argument has a producer");
        } else if (!state.producers.count(value)) {
            Fail("non-source value has no producer");
        }
    }
    ValidateRegion(state, plan.entry_region, state.sources);
    if (state.visited.size() != plan.regions.size()) Fail("all regions must be reachable");
    const auto& entry = Region(state, plan.entry_region, "entry");
    std::vector<ControlExecutionValueId> sources = plan.graph_inputs;
    sources.insert(sources.end(), plan.constant_values.begin(), plan.constant_values.end());
    if (!SameSet(entry.live_ins, sources) || !SameSet(entry.live_outs, plan.graph_outputs)) {
        Fail("entry region boundaries do not match graph boundary");
    }
    if (state.constant_payloads.size() != state.constants.size()) {
        Fail("every graph constant requires one exact module binding");
    }
}

struct ControlExecutionPlan::Impl final {
    explicit Impl(ControlExecutionPlanSpec spec) : spec(std::move(spec)) {}
    const ControlExecutionPlanSpec spec;
};

ControlExecutionPlan::ControlExecutionPlan(ControlExecutionPlanSpec spec) {
    VerifyControlExecutionPlan(spec);
    impl_ = std::make_shared<Impl>(std::move(spec));
}

bool ControlExecutionPlan::defined() const noexcept { return static_cast<bool>(impl_); }
void ControlExecutionPlan::Validate() const {
    if (!impl_) throw std::invalid_argument("ControlExecutionPlan is undefined");
    VerifyControlExecutionPlan(impl_->spec);
}
const ControlExecutionPlanSpec& ControlExecutionPlan::spec() const {
    if (!impl_) throw std::runtime_error("undefined ControlExecutionPlan");
    return impl_->spec;
}
const std::vector<ControlExecutionValueSpec>& ControlExecutionPlan::values() const { return spec().values; }
const std::vector<ControlExecutionRegion>& ControlExecutionPlan::regions() const { return spec().regions; }
const std::vector<ControlExecutionValueId>& ControlExecutionPlan::graph_inputs() const { return spec().graph_inputs; }
const std::vector<ControlExecutionValueId>& ControlExecutionPlan::constant_values() const { return spec().constant_values; }
const std::vector<ControlExecutionValueId>& ControlExecutionPlan::graph_outputs() const { return spec().graph_outputs; }

}  // namespace kxc::runtime
