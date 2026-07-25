/*! \file src/compiler/control_flow/production_control_flow.cc
 * \brief Explicit, default-OFF real-artifact resolution for static Relay If.
 */

#include "kxc/compiler/compiler.h"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "internal_lowering.h"

namespace kxc::api {

struct CompiledControlFlowGraph::State final {
    explicit State(runtime::ControlExecutionPlan plan)
        : plan(std::move(plan)) {
        this->plan.Validate();
    }
    const runtime::ControlExecutionPlan plan;
};

CompiledControlFlowGraph::CompiledControlFlowGraph(
    std::shared_ptr<const State> state) : state_(std::move(state)) {}

bool CompiledControlFlowGraph::defined() const noexcept {
    return static_cast<bool>(state_);
}

const runtime::ControlExecutionPlan& CompiledControlFlowGraph::plan() const {
    if (!state_) throw std::logic_error("CompiledControlFlowGraph is undefined");
    return state_->plan;
}

namespace {

#ifndef KXC_ENABLE_CONTROL_RUNTIME
#define KXC_ENABLE_CONTROL_RUNTIME 0
#endif

[[noreturn]] void Fail(const std::string& detail) {
    throw std::invalid_argument("CompileControlFlowExact: " + detail);
}

std::vector<runtime::ValueId> AbiNonOutputs(const runtime::ControlTask& task,
                                            const runtime::ControlPlan& plan) {
    std::unordered_set<runtime::ValueId> constants(plan.constant_values.begin(),
                                                    plan.constant_values.end());
    std::unordered_set<runtime::ValueId> seen;
    std::vector<runtime::ValueId> unique;
    unique.reserve(task.argument_values.size());
    for (const auto value : task.argument_values) {
        if (seen.insert(value).second) unique.push_back(value);
    }
    std::stable_partition(unique.begin(), unique.end(), [&constants](const auto value) {
        return !constants.count(value);
    });
    return unique;
}

void RequireProductionSubset(const runtime::ControlPlan& plan,
                             const CompileConfig& config) {
#if !KXC_USE_LLVM
    (void)plan;
    (void)config;
    Fail("requires a build with KXC_ENABLE_LLVM=ON for real CPU artifacts");
#else
    if (config->target->device_type != kCPU || config->target->device_id != 0 ||
        config->target->kind != "llvm") {
        Fail("requires the available LLVM CPU:0 backend; CUDA and non-default devices are not enabled");
    }
    for (const auto& value : plan.values) {
        if (value.device != Device::CPU()) {
            Fail("requires static CPU:0 values; implicit device copies are unsupported");
        }
    }
    for (const auto& region : plan.regions) {
        for (const auto& task : region.tasks) {
            if (task.device != Device::CPU() || task.stream != "default") {
                Fail("requires CPU:0/default stream tasks");
            }
        }
    }
#endif
}

struct ResolvedBinding final {
    internal::PrimitiveUnitId primitive_unit_id{-1};
    CompiledModule module;
    String entry_symbol;
    std::vector<runtime::ValueId> abi_non_output_value_ids;
    ArtifactPin artifact_pin;
};

Function BuildCompatibilityFunction(const internal::PrimitiveUnit& unit) {
    const auto* call = unit.call.call.As<CallNode>();
    if (!call) Fail("PrimitiveUnit does not contain a Relay Call");
    Array<Var> parameters;
    Array<Expr> arguments;
    std::unordered_map<const Object*, Expr> substitutions;
    for (std::size_t index = 0; index < call->args.size(); ++index) {
        const Expr& argument = call->args[index];
        if (argument.As<ConstantNode>()) {
            arguments.push_back(argument);
            continue;
        }
        const auto* variable = argument.As<VarNode>();
        if (!variable || !argument.checked_type().defined()) {
            Fail("PrimitiveUnit compatibility compile requires typed ANF atomic arguments");
        }
        auto found = substitutions.find(argument.get());
        if (found == substitutions.end()) {
            Var fresh(
                "primitive_unit_" + std::to_string(unit.id) + "_arg_" +
                    std::to_string(parameters.size()),
                argument.checked_type());
            found =
                substitutions.emplace(argument.get(), Expr(fresh)).first;
            parameters.push_back(std::move(fresh));
        }
        arguments.push_back(found->second);
    }
    return Function(
        std::move(parameters),
        Call(call->op, std::move(arguments), call->attrs));
}

}  // namespace

CompiledControlFlowGraph Compiler::CompileControlFlowExact(
    Function function, CompileConfig config) {
#if !KXC_ENABLE_CONTROL_RUNTIME
    (void)function;
    (void)config;
    throw std::runtime_error(
        "CompileControlFlowExact is disabled by KXC_ENABLE_CONTROL_RUNTIME");
#else
    config.Validate();
    internal::PreparedRelayProgram prepared =
        internal::PrepareRelayProgram(
            std::move(function), config,
            internal::ControlFlowPolicy::NativeExact());
    const internal::PreparedProgramPlan program_plan =
        internal::PlanRelayProgram(prepared);
    if (!std::holds_alternative<internal::PreparedControlPlan>(
            program_plan)) {
        Fail("has no residual control topology after preparation; use "
             "Compiler::Compile for the static fast path");
    }
    internal::ControlPlanLowering lowered =
        internal::LowerPreparedRelayToControlPlanWithSidecar(prepared);
    RequireProductionSubset(lowered.plan, config);

    std::vector<ResolvedBinding> resolved;
    resolved.reserve(lowered.primitive_units.size());
    for (const auto& region : lowered.plan.regions) {
        for (const auto& task : region.tasks) {
            if (task.kind != runtime::ControlTaskKind::kKernel) continue;
            if (task.primitive_unit_id < 0 ||
                static_cast<std::size_t>(task.primitive_unit_id) >=
                    lowered.primitive_units.size()) {
                Fail("control task references an unknown PrimitiveUnit");
            }
            const internal::PrimitiveUnit& unit =
                lowered.primitive_units[
                    static_cast<std::size_t>(task.primitive_unit_id)];
            if (unit.id != task.primitive_unit_id) {
                Fail("control PrimitiveUnit ids must be dense and ordered");
            }
            // This is the unchanged real Compiler path on an If-free branch
            // fragment. Phase 5 replaces this compatibility adapter with the
            // shared primitive batch compiler.
            CompiledGraph compiled =
                Compiler::Compile(BuildCompatibilityFunction(unit), config);
            const auto& pins = compiled.artifact_pins();
            const auto& calls = compiled.plan().calls();
            if (!compiled.module().IsReady() || pins.size() != 1 ||
                calls.size() != 1 || !pins.front().defined() ||
                !compiled.module().HasFunction(calls[0]->symbol)) {
                Fail("branch Call must resolve to exactly one real immutable compiler artifact");
            }
            resolved.push_back(ResolvedBinding{
                unit.id, compiled.module(), calls[0]->symbol,
                AbiNonOutputs(task, lowered.plan), pins.front()});
        }
    }
    if (resolved.empty()) {
        Fail("requires at least one real branch kernel; a pure structural If has no production artifact");
    }

    std::vector<ArtifactPin> pins;
    pins.reserve(resolved.size());
    for (const ResolvedBinding& binding : resolved) {
        pins.push_back(binding.artifact_pin);
    }
    const std::shared_ptr<const void> retention_owner =
        std::make_shared<const std::vector<ArtifactPin>>(std::move(pins));

    std::vector<internal::ControlKernelBinding> bindings;
    bindings.reserve(resolved.size());
    for (ResolvedBinding& binding : resolved) {
        bindings.push_back(internal::ControlKernelBinding{
            binding.primitive_unit_id, std::move(binding.module),
            std::move(binding.entry_symbol),
            std::move(binding.abi_non_output_value_ids), retention_owner});
    }
    runtime::ControlExecutionPlan plan =
        internal::BindControlPlanForRuntime(lowered.plan, bindings);
    return CompiledControlFlowGraph(
        std::make_shared<const CompiledControlFlowGraph::State>(
            std::move(plan)));
#endif
}

}  // namespace kxc::api
