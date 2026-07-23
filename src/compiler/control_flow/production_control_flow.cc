/*! \file src/compiler/control_flow/production_control_flow.cc
 * \brief Explicit, default-OFF real-artifact resolution for static Relay If.
 */

#include "kxc/compiler/compiler.h"

#include <algorithm>
#include <atomic>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "internal_lowering.h"
#include "kxc/profiling/profiling.h"

namespace kxc::api {

struct ControlFlowArtifactLease::Entry final {
    runtime::TaskId task_id{-1};
    String entry_symbol;
    std::string signature_digest;
    std::string launch_metadata_digest;
};

struct ControlFlowArtifactLease::State final {
    std::uint64_t generation{0};
    std::vector<Entry> entries;
    // The lease is the strong owner of exactly the pins selected below.
    std::vector<ArtifactPin> artifact_pins;
};

ControlFlowArtifactLease::ControlFlowArtifactLease(
    std::shared_ptr<const State> state) : state_(std::move(state)) {}

std::uint64_t ControlFlowArtifactLease::generation() const noexcept {
    return state_ ? state_->generation : 0;
}

bool ControlFlowArtifactLease::Covers(
    runtime::TaskId task_id, const String& entry_symbol,
    const std::string& signature_digest,
    const std::string& launch_metadata_digest) const {
    if (!state_ || state_->generation == 0) return false;
    for (const Entry& entry : state_->entries) {
        if (entry.task_id == task_id && entry.entry_symbol == entry_symbol &&
            entry.signature_digest == signature_digest &&
            entry.launch_metadata_digest == launch_metadata_digest) {
            return true;
        }
    }
    return false;
}

namespace {

#ifndef KXC_ENABLE_RELAY_CONTROL_FLOW_PRODUCTION
#define KXC_ENABLE_RELAY_CONTROL_FLOW_PRODUCTION 0
#endif

std::atomic<std::uint64_t> next_lease_generation{1};

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

struct ResolvedBinding final {
    runtime::TaskId task_id{-1};
    CompiledModule module;
    String entry_symbol;
    std::vector<runtime::ValueId> abi_non_output_value_ids;
    ArtifactPin artifact_pin;
    std::string signature_digest;
    std::string launch_metadata_digest;
};

}  // namespace

CompiledControlFlowGraph Compiler::CompileControlFlowExact(
    Function function, CompileConfig config) {
#if !KXC_ENABLE_RELAY_CONTROL_FLOW_PRODUCTION
    (void)function;
    (void)config;
    throw std::runtime_error(
        "CompileControlFlowExact is disabled by KXC_ENABLE_RELAY_CONTROL_FLOW_PRODUCTION");
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
            if (!compiled.module.IsReady() || compiled.artifact_pins.size() != 1 ||
                compiled.artifact_plan_bindings.size() != 1 ||
                !compiled.artifact_pins.front().defined()) {
                Fail("branch Call must resolve to exactly one real immutable compiler artifact");
            }
            const ArtifactPlanBinding& artifact = compiled.artifact_plan_bindings.front();
            if (!artifact.artifact_pin.defined() ||
                !compiled.module.HasFunction(String(artifact.link_symbol.value))) {
                Fail("real branch artifact lacks its selected module entry");
            }
            resolved.push_back(ResolvedBinding{
                task.id, std::move(compiled.module), String(artifact.link_symbol.value),
                AbiNonOutputs(task, lowered.plan), artifact.artifact_pin,
                artifact.signature_digest, artifact.launch_metadata_digest});
        }
    }
    if (resolved.empty()) {
        Fail("requires at least one real branch kernel; a pure structural If has no production artifact");
    }

    const std::uint64_t generation = next_lease_generation.fetch_add(
        1, std::memory_order_relaxed);
    if (generation == 0) Fail("process-local artifact lease generation overflowed");
    auto state = std::make_shared<ControlFlowArtifactLease::State>();
    state->generation = generation;
    state->entries.reserve(resolved.size());
    state->artifact_pins.reserve(resolved.size());
    for (const ResolvedBinding& binding : resolved) {
        state->entries.push_back(ControlFlowArtifactLease::Entry{
            binding.task_id, binding.entry_symbol, binding.signature_digest,
            binding.launch_metadata_digest});
        state->artifact_pins.push_back(binding.artifact_pin);
    }
    std::vector<ArtifactPin> selected_pins = state->artifact_pins;
    const auto lease = std::shared_ptr<const ControlFlowArtifactLease>(
        new ControlFlowArtifactLease(std::move(state)));

    std::vector<ControlKernelBinding> bindings;
    bindings.reserve(resolved.size());
    for (ResolvedBinding& binding : resolved) {
        bindings.push_back(ControlKernelBinding{
            binding.task_id, std::move(binding.module), std::move(binding.entry_symbol),
            0, std::move(binding.abi_non_output_value_ids), lease});
    }
    runtime::ControlExecutionPlan plan =
        BindControlPlanForRuntime(lowered.plan, bindings);
    return CompiledControlFlowGraph{std::move(plan), std::move(selected_pins), lease};
#endif
}

}  // namespace kxc::api
