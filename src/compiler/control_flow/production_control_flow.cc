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
    runtime::TaskId task_id{-1};
    CompiledModule module;
    String entry_symbol;
    std::vector<runtime::ValueId> abi_non_output_value_ids;
    ArtifactPin artifact_pin;
};

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
    internal::ControlPlanLowering lowered =
        internal::LowerRelayToControlPlanWithSidecar(std::move(function));
    RequireProductionSubset(lowered.plan, config);

    std::vector<ResolvedBinding> resolved;
    resolved.reserve(lowered.kernel_functions.size());
    for (const auto& region : lowered.plan.regions) {
        for (const auto& task : region.tasks) {
            if (task.kind != runtime::ControlTaskKind::kKernel) continue;
            const auto frozen = lowered.kernel_functions.find(task.id);
            if (frozen == lowered.kernel_functions.end()) {
                Fail("malformed compiler-private lowering sidecar: missing frozen Call for task " +
                     std::to_string(task.id));
            }
            // This is the unchanged real Compiler path on an If-free branch
            // fragment, therefore it uses normal lowering, codegen, cache, and pins.
            CompiledGraph compiled = Compiler::Compile(frozen->second, config);
            const auto& pins = compiled.artifact_pins();
            const auto& calls = compiled.plan().calls();
            if (!compiled.module().IsReady() || pins.size() != 1 ||
                calls.size() != 1 || !pins.front().defined() ||
                !compiled.module().HasFunction(calls[0]->symbol)) {
                Fail("branch Call must resolve to exactly one real immutable compiler artifact");
            }
            resolved.push_back(ResolvedBinding{
                task.id, compiled.module(), calls[0]->symbol,
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
            binding.task_id, std::move(binding.module),
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
