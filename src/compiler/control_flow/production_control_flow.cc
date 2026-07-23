/*! \file src/compiler/control_flow/production_control_flow.cc
 * \brief Explicit, default-OFF real-artifact resolution for static Relay If.
 */

#include "kxc/compiler/compiler.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "internal_lowering.h"

namespace kxc::api {
namespace {

#ifndef KXC_ENABLE_RELAY_CONTROL_FLOW_PRODUCTION
#define KXC_ENABLE_RELAY_CONTROL_FLOW_PRODUCTION 0
#endif

[[noreturn]] void Fail(const std::string& detail) {
    throw std::invalid_argument("CompileControlFlowExact: " + detail);
}

std::vector<runtime::ValueId> AbiNonOutputs(const runtime::ControlTask& task,
                                            const runtime::ControlPlan& plan) {
    std::vector<runtime::ValueId> result;
    result.reserve(task.argument_values.size());
    const auto is_constant = [&plan](runtime::ValueId id) {
        for (const auto constant : plan.constant_values) {
            if (constant == id) return true;
        }
        return false;
    };
    for (const auto value : task.argument_values) {
        if (!is_constant(value)) result.push_back(value);
    }
    for (const auto value : task.argument_values) {
        if (is_constant(value)) result.push_back(value);
    }
    return result;
}

void RequireProductionSubset(const runtime::ControlPlan& plan,
                             const CompileConfig& config) {
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
            if (task.kind == runtime::ControlTaskKind::kLoop) {
                Fail("Relay source has no Loop node; generic Relay Loop/recursion is not supported");
            }
            if (task.device != Device::CPU() || task.stream != "default") {
                Fail("requires CPU:0/default stream tasks");
            }
        }
    }
}

}  // namespace

CompiledControlFlowGraph Compiler::CompileControlFlowExact(
    Function function, CompileConfig config, ControlFlowArtifactAuthority authority) {
#if !KXC_ENABLE_RELAY_CONTROL_FLOW_PRODUCTION
    (void)function;
    (void)config;
    (void)authority;
    throw std::runtime_error(
        "CompileControlFlowExact is disabled by KXC_ENABLE_RELAY_CONTROL_FLOW_PRODUCTION");
#else
    if (authority.generation == 0 || authority.lease_id.empty()) {
        Fail("requires nonzero externally supplied authority generation and nonempty lease_id");
    }
    config.Validate();
    internal::ControlPlanLowering lowered =
        internal::LowerRelayToControlPlanWithSidecar(std::move(function));
    RequireProductionSubset(lowered.plan, config);

    std::vector<ControlKernelBinding> bindings;
    std::vector<ArtifactPin> selected_pins;
    bindings.reserve(lowered.kernel_functions.size());
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
            if (!compiled.module.IsReady() || compiled.artifact_pins.size() != 1 ||
                compiled.artifact_plan_bindings.size() != 1 ||
                !compiled.artifact_pins.front().defined()) {
                Fail("branch Call must resolve to exactly one real immutable compiler artifact");
            }
            const ArtifactPlanBinding& artifact =
                compiled.artifact_plan_bindings.front();
            if (!artifact.artifact_pin.defined() ||
                !compiled.module.HasFunction(String(artifact.link_symbol.value))) {
                Fail("real branch artifact lacks its selected module entry");
            }
            auto pin_lease = std::make_shared<const std::vector<ArtifactPin>>(
                compiled.artifact_pins);
            // binding_revision remains zero: it is only an old fixture label.
            bindings.push_back(ControlKernelBinding{
                task.id, std::move(compiled.module),
                String(artifact.link_symbol.value), 0,
                AbiNonOutputs(task, lowered.plan), authority.generation,
                authority.lease_id, std::move(pin_lease)});
            selected_pins.push_back(artifact.artifact_pin);
        }
    }
    if (bindings.empty()) {
        Fail("requires at least one real branch kernel; a pure structural If has no production artifact");
    }
    runtime::ControlExecutionPlan plan =
        BindControlPlanForRuntime(lowered.plan, bindings);
    return CompiledControlFlowGraph{std::move(plan), std::move(selected_pins),
                                    std::move(authority)};
#endif
}

}  // namespace kxc::api
