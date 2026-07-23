/*! \file src/runtime/control_plan.cc */

#include "kxc/runtime/control_plan.h"

#include <algorithm>
#include <functional>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace kxc::runtime {
namespace {

[[noreturn]] void Fail(const std::string& message) {
    throw std::invalid_argument("ControlPlan: " + message);
}

bool IsDType(const std::string& dtype) {
    static const std::set<std::string> kTypes{
        "bool", "int8", "int16", "int32", "int64", "uint8", "uint16",
        "uint32", "uint64", "float16", "float32", "float64", "bfloat16"};
    return kTypes.count(dtype) != 0;
}

bool IsDevice(const Device& device) {
    return device.defined() &&
           (device.device_type() == kCPU || device.device_type() == kCUDA);
}

bool SameIds(std::vector<ValueId> left, std::vector<ValueId> right) {
    std::sort(left.begin(), left.end());
    std::sort(right.begin(), right.end());
    return left == right;
}

template <typename Id>
void RequireUnique(const std::vector<Id>& ids, const char* what) {
    std::unordered_set<Id> seen;
    for (const Id id : ids) {
        if (id < 0 || !seen.insert(id).second) Fail(std::string(what) + " must be unique and non-negative");
    }
}

struct State {
    explicit State(const ControlPlan& source) : plan(source) {}

    const ControlPlan& plan;
    std::unordered_map<ValueId, const ControlValueSpec*> values;
    std::unordered_map<RegionId, const ControlRegion*> regions;
    std::unordered_map<TaskId, const ControlTask*> tasks;
    std::unordered_set<ValueId> graph_inputs;
    std::unordered_set<ValueId> constant_values;
    std::unordered_set<ValueId> source_values;
    std::unordered_set<ValueId> body_arguments;
    std::unordered_map<ValueId, std::pair<RegionId, RegionId>> body_argument_regions;
    std::unordered_map<ValueId, TaskId> producers;
    std::unordered_set<RegionId> visited;
    std::unordered_set<RegionId> active;
};

const ControlValueSpec& Value(const State& state, ValueId id, const char* where) {
    const auto it = state.values.find(id);
    if (it == state.values.end()) Fail(std::string(where) + " references an unknown value");
    return *it->second;
}

const ControlRegion& Region(const State& state, RegionId id, const char* where) {
    const auto it = state.regions.find(id);
    if (it == state.regions.end()) Fail(std::string(where) + " references an unknown region");
    return *it->second;
}

bool SameContract(const ControlValueSpec& left, const ControlValueSpec& right) {
    return left.dtype == right.dtype && left.shape == right.shape && left.device == right.device;
}

void ValidateEffects(const EffectSummary& effect, const std::vector<ValueId>& expected_reads,
                     const char* where) {
    RequireUnique(effect.reads, "effect reads");
    RequireUnique(effect.writes, "effect writes");
    RequireUnique(effect.allocates, "effect allocates");
    if (!SameIds(effect.reads, expected_reads)) Fail(std::string(where) + " effect reads do not match inputs");
    if (!effect.writes.empty() || !effect.allocates.empty() || effect.host_callback || effect.device_sync) {
        Fail(std::string(where) + " has an illegal v1 effect");
    }
}

void ValidateAlias(const AliasSummary& alias, const State& state, const char* where) {
    if (!alias.must_alias.empty() || !alias.may_alias.empty()) {
        Fail(std::string(where) + " permits aliasing in v1");
    }
    std::set<std::pair<ValueId, ValueId>> seen;
    for (const AliasPair pair : alias.no_alias) {
        Value(state, pair.first, where);
        Value(state, pair.second, where);
        if (pair.first < 0 || pair.second < 0 || pair.first == pair.second) {
            Fail(std::string(where) + " has an invalid no_alias pair");
        }
        const auto canonical = std::minmax(pair.first, pair.second);
        if (!seen.insert(canonical).second) Fail(std::string(where) + " repeats a no_alias pair");
    }
}

void ValidateRegion(State& state, RegionId id,
                    const std::unordered_set<ValueId>& parent_available);

bool IsEmpty(const BranchSpec& branch) {
    return branch.predicate < 0 && branch.then_region < 0 &&
           branch.else_region < 0 && branch.phis.empty();
}

bool IsEmpty(const LoopSpec& loop) {
    return loop.condition_region < 0 && loop.body_region < 0 &&
           loop.condition_value < 0 && loop.carried.empty() &&
           loop.max_trip_count < 0;
}

void ValidateBranch(State& state, const ControlTask& task,
                    const std::unordered_set<ValueId>& available) {
    const BranchSpec& spec = task.branch;
    Value(state, spec.predicate, "branch predicate");
    const ControlValueSpec& predicate = Value(state, spec.predicate, "branch predicate");
    if (predicate.dtype != "bool" || !predicate.shape.empty() ||
        predicate.device != Device::CPU()) {
        Fail("branch predicate must be a CPU scalar bool");
    }
    if (spec.then_region == spec.else_region || spec.then_region < 0 || spec.else_region < 0) {
        Fail("branch requires distinct then and else regions");
    }
    const ControlRegion& then_region = Region(state, spec.then_region, "branch");
    const ControlRegion& else_region = Region(state, spec.else_region, "branch");
    std::vector<ValueId> required{spec.predicate};
    required.insert(required.end(), then_region.live_ins.begin(), then_region.live_ins.end());
    required.insert(required.end(), else_region.live_ins.begin(), else_region.live_ins.end());
    std::sort(required.begin(), required.end());
    required.erase(std::unique(required.begin(), required.end()), required.end());
    if (!SameIds(task.inputs, required)) Fail("branch task inputs do not close child live-ins");
    RequireUnique(task.outputs, "branch outputs");
    std::vector<ValueId> results;
    std::unordered_set<ValueId> then_values;
    std::unordered_set<ValueId> else_values;
    for (ValueId value : then_region.live_outs) then_values.insert(value);
    for (ValueId value : else_region.live_outs) else_values.insert(value);
    for (const PhiBinding& phi : spec.phis) {
        Value(state, phi.result, "phi");
        const ControlValueSpec& then_value = Value(state, phi.then_value, "phi");
        const ControlValueSpec& else_value = Value(state, phi.else_value, "phi");
        const ControlValueSpec& result = Value(state, phi.result, "phi");
        if (!then_values.count(phi.then_value) || !else_values.count(phi.else_value) ||
            !SameContract(result, then_value) || !SameContract(result, else_value)) {
            Fail("phi sources/results must be matching child live-outs");
        }
        results.push_back(phi.result);
    }
    RequireUnique(results, "phi results");
    if (spec.phis.empty() || !SameIds(task.outputs, results)) Fail("branch outputs must be exactly phi results");
    ValidateRegion(state, spec.then_region, available);
    ValidateRegion(state, spec.else_region, available);
}

void ValidateLoop(State& state, const ControlTask& task,
                  const std::unordered_set<ValueId>& available) {
    const LoopSpec& spec = task.loop;
    if (spec.max_trip_count < 0) Fail("loop max_trip_count must be non-negative");
    if (spec.condition_region == spec.body_region || spec.condition_region < 0 || spec.body_region < 0) {
        Fail("loop requires distinct condition and body regions");
    }
    const ControlRegion& condition = Region(state, spec.condition_region, "loop");
    const ControlRegion& body = Region(state, spec.body_region, "loop");
    const ControlValueSpec& condition_value = Value(state, spec.condition_value, "loop condition");
    if (condition_value.dtype != "bool" || !condition_value.shape.empty() ||
        condition_value.device != Device::CPU()) {
        Fail("loop condition must be a CPU scalar bool");
    }
    if (std::find(condition.live_outs.begin(), condition.live_outs.end(), spec.condition_value) == condition.live_outs.end()) {
        Fail("loop condition value must be a condition-region live-out");
    }
    std::unordered_set<ValueId> arguments;
    std::vector<ValueId> results;
    std::vector<ValueId> initial_values;
    for (const LoopCarriedBinding& binding : spec.carried) {
        const ControlValueSpec& result = Value(state, binding.result, "loop carried");
        const ControlValueSpec& initial = Value(state, binding.initial, "loop carried");
        const ControlValueSpec& argument = Value(state, binding.body_argument, "loop carried");
        const ControlValueSpec& backedge = Value(state, binding.backedge, "loop carried");
        if (!arguments.insert(binding.body_argument).second ||
            !SameContract(result, initial) || !SameContract(result, argument) || !SameContract(result, backedge)) {
            Fail("loop carried bindings must be unique and static-exact");
        }
        if (std::find(body.live_outs.begin(), body.live_outs.end(), binding.backedge) == body.live_outs.end()) {
            Fail("loop backedge must be a body-region live-out");
        }
        results.push_back(binding.result);
        initial_values.push_back(binding.initial);
    }
    if (spec.carried.empty()) Fail("loop requires at least one carried binding");
    RequireUnique(results, "loop results");
    if (!SameIds(task.outputs, results)) Fail("loop outputs must be exactly carried results");
    std::vector<ValueId> required = initial_values;
    for (ValueId value : condition.live_ins) if (!arguments.count(value)) required.push_back(value);
    for (ValueId value : body.live_ins) if (!arguments.count(value)) required.push_back(value);
    std::sort(required.begin(), required.end());
    required.erase(std::unique(required.begin(), required.end()), required.end());
    if (!SameIds(task.inputs, required)) Fail("loop task inputs do not close child live-ins");
    for (ValueId argument : arguments) {
        if (std::find(condition.live_ins.begin(), condition.live_ins.end(), argument) == condition.live_ins.end() ||
            std::find(body.live_ins.begin(), body.live_ins.end(), argument) == body.live_ins.end()) {
            Fail("each loop body_argument must be live into condition and body regions");
        }
    }
    ValidateRegion(state, spec.condition_region, available);
    ValidateRegion(state, spec.body_region, available);
}

void ValidateRegion(State& state, RegionId id, const std::unordered_set<ValueId>& parent_available) {
    if (state.active.count(id) || state.visited.count(id)) Fail("regions must form one structured tree");
    const ControlRegion& region = Region(state, id, "region");
    state.active.insert(id);
    if (region.source_locator.empty()) Fail("region source_locator is required");
    RequireUnique(region.live_ins, "region live_ins");
    RequireUnique(region.live_outs, "region live_outs");
    for (ValueId value : region.live_ins) {
        Value(state, value, "region live_in");
        if (!parent_available.count(value)) {
            const auto body_argument = state.body_argument_regions.find(value);
            if (body_argument == state.body_argument_regions.end() ||
                (body_argument->second.first != id && body_argument->second.second != id)) {
                Fail("region live_in is not available from its parent");
            }
        }
    }
    ValidateEffects(region.effect, region.live_ins, "region");
    ValidateAlias(region.alias, state, "region");
    std::unordered_set<ValueId> available(region.live_ins.begin(), region.live_ins.end());
    std::unordered_set<TaskId> previous_tasks;
    std::unordered_map<ValueId, TaskId> local_producers;
    for (const ControlTask& task : region.tasks) {
        if (task.source_locator.empty()) Fail("task source_locator is required");
        RequireUnique(task.inputs, "task inputs");
        RequireUnique(task.outputs, "task outputs");
        RequireUnique(task.dependencies, "task dependencies");
        for (TaskId dependency : task.dependencies) {
            if (!previous_tasks.count(dependency)) {
                Fail("task dependency is missing, forward, or outside its region");
            }
        }
        for (ValueId input : task.inputs) {
            Value(state, input, "task input");
            if (!available.count(input)) {
                Fail("task input is not available before its task");
            }
            const auto producer = local_producers.find(input);
            if (producer != local_producers.end() &&
                std::find(task.dependencies.begin(), task.dependencies.end(),
                          producer->second) == task.dependencies.end()) {
                Fail("task omits the dependency for a local input producer");
            }
        }
        if (!IsDevice(task.device) || task.stream != "default") {
            Fail("task device/stream is unsupported by ControlPlan v1");
        }
        ValidateEffects(task.effect, task.inputs, "task");
        ValidateAlias(task.alias, state, "task");
        if (task.kind == ControlTaskKind::kKernel) {
            std::vector<ValueId> unique_arguments = task.argument_values;
            std::sort(unique_arguments.begin(), unique_arguments.end());
            unique_arguments.erase(
                std::unique(unique_arguments.begin(), unique_arguments.end()),
                unique_arguments.end());
            if (task.kernel_ref.empty() || task.outputs.empty() ||
                !IsEmpty(task.branch) || !IsEmpty(task.loop) ||
                !SameIds(task.inputs, unique_arguments)) {
                Fail("kernel task has an invalid kind-specific contract");
            }
            for (ValueId input : task.inputs) {
                if (Value(state, input, "kernel input").device != task.device) {
                    Fail("kernel input device differs from its task device");
                }
            }
            for (ValueId output : task.outputs) {
                if (Value(state, output, "kernel output").device != task.device) {
                    Fail("kernel output device differs from its task device");
                }
            }
        } else if (task.kind == ControlTaskKind::kBranch) {
            if (!task.kernel_ref.empty() || !task.argument_values.empty() ||
                !IsEmpty(task.loop) || task.device != Device::CPU()) {
                Fail("branch task has an invalid kind-specific contract");
            }
            ValidateBranch(state, task, available);
        } else if (task.kind == ControlTaskKind::kLoop) {
            if (!task.kernel_ref.empty() || !task.argument_values.empty() ||
                !IsEmpty(task.branch) || task.device != Device::CPU()) {
                Fail("loop task has an invalid kind-specific contract");
            }
            ValidateLoop(state, task, available);
        } else {
            Fail("unknown task kind");
        }
        for (ValueId output : task.outputs) {
            Value(state, output, "task output");
            if (state.source_values.count(output) || state.body_arguments.count(output) ||
                available.count(output)) {
                Fail("task output is not a fresh value");
            }
            available.insert(output);
            local_producers.emplace(output, task.id);
        }
        previous_tasks.insert(task.id);
    }
    for (ValueId output : region.live_outs) {
        Value(state, output, "region live_out");
        if (!available.count(output)) Fail("region live_out is unavailable");
    }
    state.active.erase(id);
    state.visited.insert(id);
}

std::string Quote(const std::string& value) {
    std::ostringstream out;
    out << std::quoted(value);
    return out.str();
}

template <typename Id>
void PrintIds(std::ostringstream& out, const std::vector<Id>& ids) {
    out << '[';
    for (std::size_t i = 0; i < ids.size(); ++i) out << (i == 0 ? "" : ",") << ids[i];
    out << ']';
}

void PrintEffect(std::ostringstream& out, const EffectSummary& effect) {
    out << "read="; PrintIds(out, effect.reads);
    out << " write="; PrintIds(out, effect.writes);
    out << " allocate="; PrintIds(out, effect.allocates);
    out << " callback=" << effect.host_callback << " sync=" << effect.device_sync;
}

void PrintAlias(std::ostringstream& out, const AliasSummary& alias) {
    const auto print = [&out](const std::vector<AliasPair>& pairs) {
        out << '[';
        for (std::size_t i = 0; i < pairs.size(); ++i) out << (i == 0 ? "" : ",") << pairs[i].first << ':' << pairs[i].second;
        out << ']';
    };
    out << "must="; print(alias.must_alias); out << " may="; print(alias.may_alias); out << " no="; print(alias.no_alias);
}

}  // namespace

void VerifyControlPlan(const ControlPlan& plan) {
    if (plan.schema_version != ControlPlan::kSchemaVersion) {
        Fail("schema_version must be 1");
    }
    if (plan.values.empty() || plan.regions.empty()) Fail("values and regions are required");
    State state(plan);
    for (const ControlValueSpec& value : plan.values) {
        if (value.id < 0 || !state.values.emplace(value.id, &value).second) {
            Fail("value ids must be unique and non-negative");
        }
        if (!IsDType(value.dtype) || !IsDevice(value.device)) {
            Fail("value dtype/device is malformed or unsupported");
        }
        if (value.source_locator.empty()) Fail("value source_locator is required");
        for (const std::int64_t dim : value.shape) {
            if (dim < 0) Fail("value shape must be static and non-negative");
        }
    }
    RequireUnique(plan.graph_inputs, "graph inputs");
    RequireUnique(plan.constant_values, "constant values");
    RequireUnique(plan.graph_outputs, "graph outputs");
    if (plan.graph_outputs.empty()) Fail("graph outputs are required");
    for (ValueId id : plan.graph_inputs) {
        Value(state, id, "graph input");
        state.graph_inputs.insert(id);
        state.source_values.insert(id);
    }
    for (ValueId id : plan.constant_values) {
        Value(state, id, "constant value");
        state.constant_values.insert(id);
        if (!state.source_values.insert(id).second) {
            Fail("graph inputs and constant values must be disjoint");
        }
    }
    for (ValueId id : plan.graph_outputs) Value(state, id, "graph output");
    RequireUnique(plan.region_order, "region order");
    if (plan.region_order.size() != plan.regions.size()) {
        Fail("region_order must list every region exactly once");
    }
    for (const ControlRegion& region : plan.regions) {
        if (region.id < 0 || !state.regions.emplace(region.id, &region).second) Fail("region ids must be unique and non-negative");
        for (const ControlTask& task : region.tasks) {
            if (task.id < 0 || !state.tasks.emplace(task.id, &task).second) Fail("task ids must be unique and non-negative");
            for (ValueId output : task.outputs) {
                Value(state, output, "task output");
                if (!state.producers.emplace(output, task.id).second) Fail("value has duplicate producers");
            }
            if (task.kind == ControlTaskKind::kLoop) {
                for (const LoopCarriedBinding& binding : task.loop.carried) {
                    Value(state, binding.body_argument, "loop body_argument");
                    if (!state.body_arguments.insert(binding.body_argument).second) Fail("loop body_argument is not unique");
                    state.body_argument_regions.emplace(
                        binding.body_argument,
                        std::make_pair(task.loop.condition_region, task.loop.body_region));
                }
            }
        }
    }
    if (state.regions.count(plan.entry_region) == 0) Fail("entry region is missing");
    for (RegionId id : plan.region_order) {
        if (state.regions.count(id) == 0) {
            Fail("region_order references an unknown region");
        }
    }
    for (const auto& value : state.values) {
        const ValueId id = value.first;
        if (state.source_values.count(id) || state.body_arguments.count(id)) {
            if (state.source_values.count(id) && state.body_arguments.count(id)) {
                Fail("loop body_argument must be local to its loop");
            }
            if (state.producers.count(id)) Fail("source value has a producer");
        } else if (!state.producers.count(id)) {
            Fail("non-source value has no producer");
        }
    }
    std::unordered_set<ValueId> root_available = state.source_values;
    ValidateRegion(state, plan.entry_region, root_available);
    if (state.visited.size() != state.regions.size()) Fail("all regions must be reachable from the entry region");
    const ControlRegion& entry = Region(state, plan.entry_region, "entry");
    std::vector<ValueId> entry_sources = plan.graph_inputs;
    entry_sources.insert(entry_sources.end(), plan.constant_values.begin(),
                         plan.constant_values.end());
    if (!SameIds(entry.live_ins, entry_sources) ||
        !SameIds(entry.live_outs, plan.graph_outputs)) {
        Fail("entry region boundaries must match graph sources/outputs");
    }
}

void ControlPlan::ValidateStaticExact() const { VerifyControlPlan(*this); }

std::string ControlPlan::CanonicalText() const {
    Validate();
    std::ostringstream out;
    out << "ControlPlan/v1\nvalues\n";
    std::vector<const ControlValueSpec*> values_by_id;
    values_by_id.reserve(values.size());
    for (const ControlValueSpec& value : values) values_by_id.push_back(&value);
    std::sort(values_by_id.begin(), values_by_id.end(),
              [](const ControlValueSpec* lhs, const ControlValueSpec* rhs) {
                  return lhs->id < rhs->id;
              });
    for (const ControlValueSpec* value : values_by_id) {
        out << "  v" << value->id << " " << value->dtype << " ";
        PrintIds(out, value->shape);
        out << " " << Quote(value->device.ToString()) << " loc="
            << Quote(value->source_locator) << "\n";
    }
    out << "entry=" << entry_region << " regions=";
    PrintIds(out, region_order);
    out << "\ninputs=";
    PrintIds(out, graph_inputs);
    out << " constants=";
    PrintIds(out, constant_values);
    out << " outputs=";
    PrintIds(out, graph_outputs);
    out << "\n";
    std::unordered_map<RegionId, const ControlRegion*> regions_by_id;
    for (const ControlRegion& region : regions) regions_by_id.emplace(region.id, &region);
    for (RegionId region_id : region_order) {
        const ControlRegion& region = *regions_by_id.at(region_id);
        out << "region " << region.id << " in="; PrintIds(out, region.live_ins); out << " out="; PrintIds(out, region.live_outs);
        out << " loc=" << Quote(region.source_locator) << " "; PrintEffect(out, region.effect); out << " "; PrintAlias(out, region.alias); out << "\n";
        for (const ControlTask& task : region.tasks) {
            out << "  task " << task.id << " kind=" << static_cast<int>(task.kind) << " in="; PrintIds(out, task.inputs); out << " args="; PrintIds(out, task.argument_values); out << " out="; PrintIds(out, task.outputs); out << " dep="; PrintIds(out, task.dependencies);
            out << " ref=" << Quote(task.kernel_ref) << " loc=" << Quote(task.source_locator)
                << " device=" << Quote(task.device.ToString())
                << " stream=" << Quote(task.stream) << " "; PrintEffect(out, task.effect); out << " "; PrintAlias(out, task.alias);
            if (task.kind == ControlTaskKind::kBranch) {
                out << " branch=" << task.branch.predicate << ':' << task.branch.then_region << ':' << task.branch.else_region;
                for (const PhiBinding& phi : task.branch.phis) out << " phi=" << phi.result << ':' << phi.then_value << ':' << phi.else_value;
            } else if (task.kind == ControlTaskKind::kLoop) {
                out << " loop=" << task.loop.condition_region << ':' << task.loop.body_region << ':' << task.loop.condition_value << ':' << task.loop.max_trip_count;
                for (const LoopCarriedBinding& binding : task.loop.carried) out << " carry=" << binding.result << ':' << binding.initial << ':' << binding.body_argument << ':' << binding.backedge;
            }
            out << "\n";
        }
    }
    return out.str();
}

}  // namespace kxc::runtime
