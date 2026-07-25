/*! \file src/compiler/control_flow/production_control_flow.cc
 * \brief Explicit, default-OFF real-artifact resolution for static Relay If.
 */

#include "kxc/compiler/compiler.h"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "internal_lowering.h"
#include "../internal/primitive_compiler.h"
#include "kxc/pass/context.h"
#include "kxc/relay/visitor.h"

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
    const PassContext pass_context = PassContext::MergeTarget(
        relay::PassContextFromRelay(prepared.typed_anf()), config->target);
    PassContext::Scope pass_scope(pass_context);
    internal::CompiledPrimitiveBatch compiled =
        internal::CompilePrimitiveUnits(
            lowered.primitive_units, lowered.plan.values, config,
            prepared.execution_contract());
    if (compiled.primitives.empty()) {
        Fail("requires at least one real branch kernel; a pure structural If has no production artifact");
    }
    CompiledModule module = internal::AssemblePrimitiveModule(
        compiled, config->target);
    std::vector<ArtifactPin> pins;
    pins.reserve(compiled.primitives.size());
    for (const internal::CompiledPrimitive& primitive : compiled.primitives) {
        pins.push_back(primitive.pin);
    }
    const std::shared_ptr<const void> retention_owner =
        std::make_shared<const std::vector<ArtifactPin>>(std::move(pins));

    std::vector<internal::ControlKernelBinding> bindings;
    bindings.reserve(compiled.primitives.size());
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
            const internal::CompiledPrimitive& primitive =
                compiled.primitives[
                    static_cast<std::size_t>(task.primitive_unit_id)];
            if (primitive.unit_id != unit.id ||
                !(primitive.symbol == unit.symbol) ||
                !primitive.pin.defined() ||
                !module.HasFunction(primitive.symbol)) {
                Fail("PrimitiveUnit did not resolve to its immutable compiler artifact");
            }
            bindings.push_back(internal::ControlKernelBinding{
                unit.id, module, primitive.symbol,
                AbiNonOutputs(task, lowered.plan), retention_owner});
        }
    }
    if (bindings.size() != compiled.primitives.size()) {
        Fail("control topology and compiled PrimitiveUnit cardinality differ");
    }
    runtime::ControlExecutionPlan plan =
        internal::BindControlPlanForRuntime(lowered.plan, bindings);
    return CompiledControlFlowGraph(
        std::make_shared<const CompiledControlFlowGraph::State>(
            std::move(plan)));
#endif
}

}  // namespace kxc::api
