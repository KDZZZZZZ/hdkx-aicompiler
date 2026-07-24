/*! \file src/runtime/control_session.cc */

#include "kxc/runtime/control_session.h"

#include "internal/bound_control_kernel_access.h"

#include <cstdint>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace kxc::runtime {
namespace {

#ifndef KXC_ENABLE_CONTROL_RUNTIME
#define KXC_ENABLE_CONTROL_RUNTIME 0
#endif

[[noreturn]] void Fail(const std::string& detail) {
    throw std::invalid_argument("ControlRuntimeSession: " + detail);
}

bool SameDType(DLDataType lhs, DLDataType rhs) {
    return lhs.code == rhs.code && lhs.bits == rhs.bits && lhs.lanes == rhs.lanes;
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

void ValidateArray(const ControlExecutionValueSpec& spec, const NDArray& value,
                   const std::string& where) {
    if (!value.defined() || !value.storage().defined() || !value.IsContiguous()) {
        Fail(where + " requires a defined contiguous NDArray");
    }
    if (!SameDType(DType(spec.dtype), value.dtype()) ||
        value.device() != spec.device) {
        Fail(where + " dtype or device does not match");
    }
    const Array<std::int64_t> shape = value.shape();
    if (shape.size() != spec.shape.size()) Fail(where + " rank does not match");
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] != spec.shape[i]) Fail(where + " shape does not match");
    }
    try {
        const std::size_t nbytes = value.NBytes();
        value.storage().ValidateRange(value->byte_offset, nbytes);
        if (nbytes != 0 && value.storage().data() == nullptr) {
            Fail(where + " has null storage data");
        }
    } catch (const std::invalid_argument&) {
        throw;
    } catch (const std::exception& error) {
        Fail(where + " storage range is invalid: " + error.what());
    }
}

std::string Event(const char* kind, ControlExecutionTaskId task,
                  ControlExecutionValueId value) {
    return std::string(kind) + ":" + std::to_string(task) + ":" +
           std::to_string(value);
}

struct RunState final {
    explicit RunState(ControlExecutionPlan frozen_plan) : plan(std::move(frozen_plan)) {}

    ControlExecutionPlan plan;
    std::unordered_map<ControlExecutionValueId, NDArray> current;
    std::vector<NDArray> retained_values;
    Array<AsyncOperation> operations;
    std::vector<std::string> events;
    std::vector<ControlLoopIterationCount> loop_iterations;

    void Bind(ControlExecutionValueId id, NDArray value) {
        if (!value.defined()) throw std::logic_error("Control runtime bound undefined value");
        current.insert_or_assign(id, value);
        retained_values.push_back(std::move(value));
    }

    NDArray Get(ControlExecutionValueId id) const {
        const auto found = current.find(id);
        if (found == current.end()) {
            throw std::logic_error("Control runtime value is not bound: " + std::to_string(id));
        }
        return found->second;
    }

    Array<Storage> RetainedStorage() const {
        Array<Storage> result;
        std::unordered_set<const Object*> seen;
        for (const auto& value : retained_values) {
            const Storage storage = value.storage();
            if (seen.insert(storage.get()).second) result.push_back(storage);
        }
        return result;
    }
};

struct PlanIndex final {
    std::unordered_map<ControlExecutionValueId, const ControlExecutionValueSpec*> values;
    std::unordered_map<ControlExecutionRegionId, const ControlExecutionRegion*> regions;
};

[[maybe_unused]] PlanIndex Index(const ControlExecutionPlan& plan) {
    PlanIndex index;
    for (const auto& value : plan.values()) index.values.emplace(value.id, &value);
    for (const auto& region : plan.regions()) index.regions.emplace(region.id, &region);
    return index;
}

const ControlExecutionValueSpec& Value(const PlanIndex& index,
                                       ControlExecutionValueId id) {
    return *index.values.at(id);
}

const ControlExecutionRegion& Region(const PlanIndex& index,
                                     ControlExecutionRegionId id) {
    return *index.regions.at(id);
}

void CollectConstantBindings(const ControlExecutionRegion& region,
                             const PlanIndex& index,
                             std::unordered_map<ControlExecutionValueId, NDArray>* constants,
                             std::unordered_set<ControlExecutionRegionId>* visited) {
    if (!visited->insert(region.id).second) return;
    for (const auto& task : region.tasks) {
        if (task.kind == ControlExecutionTaskKind::kKernel) {
            const Array<codegen::KernelArgSpec> arguments = task.kernel.signature().arguments();
            for (std::size_t i = 0; i < arguments.size(); ++i) {
                if (arguments[i]->role != codegen::KernelArgRole::kConstant) continue;
                const auto value_id = task.argument_values[i];
                const NDArray payload =
                    task.kernel.Constant(arguments[i]->constant_key);
                ValidateArray(Value(index, value_id), payload, "constant preflight");
                const auto previous = constants->find(value_id);
                if (previous != constants->end() &&
                    !task.kernel.MatchesConstant(
                        arguments[i]->constant_key, previous->second)) {
                    Fail("one logical constant has inconsistent fixture payloads");
                }
                if (previous == constants->end() ||
                    previous->second.storage()->alignment <
                        arguments[i]->alignment) {
                    constants->insert_or_assign(value_id, payload);
                }
            }
        } else if (task.kind == ControlExecutionTaskKind::kBranch) {
            CollectConstantBindings(Region(index, task.branch.then_region), index, constants, visited);
            CollectConstantBindings(Region(index, task.branch.else_region), index, constants, visited);
        } else {
            CollectConstantBindings(Region(index, task.loop.condition_region), index, constants, visited);
            CollectConstantBindings(Region(index, task.loop.body_region), index, constants, visited);
        }
    }
}

void PreflightRegion(const ControlExecutionRegion& region, const PlanIndex& index,
                     const RunState& state,
                     std::unordered_set<ControlExecutionRegionId>* visited) {
    if (!visited->insert(region.id).second) return;
    for (const auto& task : region.tasks) {
        if (task.kind == ControlExecutionTaskKind::kKernel) {
            const Array<codegen::KernelArgSpec> arguments =
                task.kernel.signature().arguments();
            for (std::size_t i = 0; i < arguments.size(); ++i) {
                if (arguments[i]->role == codegen::KernelArgRole::kOutput) continue;
                const auto value_id = task.argument_values[i];
                const auto found = state.current.find(value_id);
                if (found == state.current.end()) continue;
                ValidateArray(Value(index, value_id), found->second,
                              "kernel source preflight");
                if (found->second.NBytes() != 0) {
                    const uintptr_t base = reinterpret_cast<uintptr_t>(
                        found->second.storage().data());
                    if (found->second->byte_offset >
                        std::numeric_limits<uintptr_t>::max() - base ||
                        (base + found->second->byte_offset) %
                            arguments[i]->alignment != 0) {
                        Fail("kernel source preflight alignment does not match ABI");
                    }
                }
                if (arguments[i]->role == codegen::KernelArgRole::kConstant &&
                    !task.kernel.MatchesConstant(
                        arguments[i]->constant_key, found->second)) {
                    Fail("kernel source preflight constant payload does not match ABI");
                }
            }
        } else if (task.kind == ControlExecutionTaskKind::kBranch) {
            PreflightRegion(Region(index, task.branch.then_region), index, state, visited);
            PreflightRegion(Region(index, task.branch.else_region), index, state, visited);
        } else {
            // Loop body arguments are scoped logical forwardings.  Seed a probe
            // only when their initial source is already available, so graph
            // inputs receive ABI preflight before the first launch.
            RunState loop_state = state;
            for (const auto& carried : task.loop.carried) {
                const auto initial = state.current.find(carried.initial);
                if (initial != state.current.end()) {
                    loop_state.Bind(carried.body_argument, initial->second);
                }
            }
            std::unordered_set<ControlExecutionRegionId> loop_visited;
            PreflightRegion(Region(index, task.loop.condition_region), index,
                            loop_state, &loop_visited);
            PreflightRegion(Region(index, task.loop.body_region), index,
                            loop_state, &loop_visited);
        }
    }
}

bool Predicate(const NDArray& value) {
    std::uint8_t byte{0};
    value.CopyToBytes(&byte, sizeof(byte));
    return byte != 0;
}

void ExecuteRegion(const ControlExecutionRegion& region, const PlanIndex& index,
                   const DeviceStream& stream, RunState* state);

void ExecuteKernel(const ControlExecutionTask& task, const PlanIndex& index,
                   const DeviceStream& stream, RunState* state) {
    const Array<codegen::KernelArgSpec> signature = task.kernel.signature().arguments();
    Array<NDArray> ordered;
    for (std::size_t i = 0; i < signature.size(); ++i) {
        const auto value_id = task.argument_values[i];
        if (signature[i]->role == codegen::KernelArgRole::kOutput) {
            const auto& value = Value(index, value_id);
            Array<std::int64_t> shape;
            for (const auto dimension : value.shape) shape.push_back(dimension);
            NDArray output = NDArray::Empty(shape, DType(value.dtype),
                                             value.device, signature[i]->alignment);
            state->Bind(value_id, std::move(output));
        }
        ordered.push_back(state->Get(value_id));
    }
    state->operations.push_back(
        internal::BoundControlKernelAccess::Launch(task.kernel, ordered, stream));
}

void ExecuteBranch(const ControlExecutionTask& task, const PlanIndex& index,
                   const DeviceStream& stream, RunState* state) {
    if (!state->operations.empty()) {
        state->operations[state->operations.size() - 1].Wait();
    }
    const bool selected_then = Predicate(state->Get(task.branch.predicate));
    state->events.push_back("branch:" + std::to_string(task.id) + ":" +
                            (selected_then ? "then" : "else"));
    ExecuteRegion(Region(index, selected_then ? task.branch.then_region : task.branch.else_region),
                  index, stream, state);
    for (const auto& phi : task.branch.phis) {
        state->Bind(phi.result, state->Get(selected_then ? phi.then_value : phi.else_value));
    }
}

void ExecuteLoop(const ControlExecutionTask& task, const PlanIndex& index,
                 const DeviceStream& stream, RunState* state) {
    std::unordered_map<ControlExecutionValueId, NDArray> current;
    for (const auto& carried : task.loop.carried) {
        current.emplace(carried.body_argument, state->Get(carried.initial));
    }
    std::int64_t iterations = 0;
    while (true) {
        for (const auto& item : current) state->Bind(item.first, item.second);
        ExecuteRegion(Region(index, task.loop.condition_region), index, stream, state);
        if (!state->operations.empty()) {
            state->operations[state->operations.size() - 1].Wait();
        }
        if (!Predicate(state->Get(task.loop.condition_value))) break;
        if (iterations >= task.loop.max_trip_count) {
            throw std::runtime_error("ControlRuntimeSession loop exceeded max_trip_count");
        }
        state->events.push_back("loop:" + std::to_string(task.id) + ":iteration:" +
                                std::to_string(iterations));
        for (const auto& item : current) state->Bind(item.first, item.second);
        ExecuteRegion(Region(index, task.loop.body_region), index, stream, state);
        for (const auto& carried : task.loop.carried) {
            current.insert_or_assign(carried.body_argument, state->Get(carried.backedge));
        }
        ++iterations;
    }
    for (const auto& carried : task.loop.carried) {
        state->Bind(carried.result, current.at(carried.body_argument));
    }
    state->loop_iterations.push_back(ControlLoopIterationCount{task.id, iterations});
}

void ExecuteRegion(const ControlExecutionRegion& region, const PlanIndex& index,
                   const DeviceStream& stream, RunState* state) {
    for (const auto& task : region.tasks) {
        for (const auto input : task.inputs) {
            (void)state->Get(input);
            state->events.push_back(Event("read", task.id, input));
        }
        state->events.push_back("task:" + std::to_string(task.id));
        switch (task.kind) {
            case ControlExecutionTaskKind::kKernel:
                ExecuteKernel(task, index, stream, state);
                break;
            case ControlExecutionTaskKind::kBranch:
                ExecuteBranch(task, index, stream, state);
                break;
            case ControlExecutionTaskKind::kLoop:
                ExecuteLoop(task, index, stream, state);
                break;
            default:
                Fail("validated plan contains an unknown task kind");
        }
        for (const auto output : task.outputs) {
            state->events.push_back(Event("write", task.id, output));
        }
    }
}

[[maybe_unused]] void WaitPrior(const Array<AsyncOperation>& operations) noexcept {
    for (const auto& operation : operations) {
        try {
            operation.Wait();
        } catch (...) {
            // AsyncOperation destructor retains an operation that cannot be proven complete.
        }
    }
}

}  // namespace

struct ControlRuntimeSession::ConstantState final {
    std::unordered_map<ControlExecutionValueId, NDArray> values;
};

ControlRuntimeSession::ControlRuntimeSession(ControlExecutionPlan plan)
    : plan_(std::move(plan)) {
#if KXC_ENABLE_CONTROL_RUNTIME
    plan_.Validate();
    const PlanIndex index = Index(plan_);
    auto constants = std::make_shared<ConstantState>();
    std::unordered_set<ControlExecutionRegionId> visited;
    CollectConstantBindings(Region(index, plan_.entry_region()), index,
                            &constants->values, &visited);
    for (const auto value_id : plan_.constant_values()) {
        if (!constants->values.count(value_id)) {
            Fail("validated constant has no module binding");
        }
    }
    constants_ = std::move(constants);
#else
    throw std::runtime_error(
        "ControlRuntimeSession is disabled by KXC_ENABLE_CONTROL_RUNTIME");
#endif
}

ControlRunResult ControlRuntimeSession::Run(const Array<NDArray>& inputs) const {
#if KXC_ENABLE_CONTROL_RUNTIME
    ControlRunAsyncResult result = RunAsync(inputs, DeviceStream::Default(Device::CPU()));
    result.completion.Wait();
    return ControlRunResult{std::move(result.outputs), std::move(result.events),
                            std::move(result.loop_iterations)};
#else
    (void)inputs;
    throw std::runtime_error(
        "ControlRuntimeSession is disabled by KXC_ENABLE_CONTROL_RUNTIME");
#endif
}

ControlRunAsyncResult ControlRuntimeSession::RunAsync(const Array<NDArray>& inputs,
                                                       const DeviceStream& stream) const {
#if KXC_ENABLE_CONTROL_RUNTIME
    plan_.Validate();
    if (!stream.defined() || stream.device() != Device::CPU() || !stream.is_default()) {
        Fail("v1 requires the CPU:0 default DeviceStream");
    }
    const auto& graph_inputs = plan_.graph_inputs();
    if (inputs.size() != graph_inputs.size()) {
        Fail("input count expected " + std::to_string(graph_inputs.size()) +
             ", actual " + std::to_string(inputs.size()));
    }
    const PlanIndex index = Index(plan_);
    auto state = std::make_shared<RunState>(plan_);
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const auto value_id = graph_inputs[i];
        ValidateArray(Value(index, value_id), inputs[i], "input[" + std::to_string(i) + "]");
        state->Bind(value_id, inputs[i]);
    }
    if (!constants_) Fail("session constant cache is unavailable");
    for (const auto value_id : plan_.constant_values()) {
        state->Bind(value_id, constants_->values.at(value_id));
    }
    std::unordered_set<ControlExecutionRegionId> preflight_visited;
    PreflightRegion(Region(index, plan_.entry_region()), index, *state,
                    &preflight_visited);

    try {
        ExecuteRegion(Region(index, plan_.entry_region()), index, stream, state.get());
    } catch (...) {
        WaitPrior(state->operations);
        throw;
    }
    Array<NDArray> outputs;
    for (const auto value_id : plan_.graph_outputs()) outputs.push_back(state->Get(value_id));
    AsyncOperation completion;
    if (state->operations.empty()) {
        completion = AsyncOperation::Completed(stream);
    } else {
        completion = state->operations[state->operations.size() - 1];
        // The completion owns state; state must retain only earlier operations
        // to avoid an ownership cycle through the final operation itself.
        state->operations.erase(state->operations.end() - 1);
    }
    completion.RetainDependencies(state->RetainedStorage(), state);
    ControlRunAsyncResult result;
    result.outputs = std::move(outputs);
    result.events = std::move(state->events);
    result.loop_iterations = std::move(state->loop_iterations);
    result.completion = std::move(completion);
    return result;
#else
    (void)inputs;
    (void)stream;
    throw std::runtime_error(
        "ControlRuntimeSession is disabled by KXC_ENABLE_CONTROL_RUNTIME");
#endif
}

const ControlExecutionPlan& ControlRuntimeSession::plan() const noexcept { return plan_; }

}  // namespace kxc::runtime
