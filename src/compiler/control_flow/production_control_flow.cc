/*! \file src/compiler/control_flow/production_control_flow.cc
 * \brief Explicit, default-OFF real-artifact resolution for static Relay If.
 */

#include "kxc/compiler/compiler.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "internal_lowering.h"
#include "production_control_flow_test.h"
#include "kxc/profiling/profiling.h"
#include "kxc/support/hash.h"

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

// This stores the most recently issued generation.  Keeping the terminal
// value makes overflow a permanent fail-closed state instead of wrapping.
std::atomic<std::uint64_t> last_lease_generation{0};

[[noreturn]] void Fail(const std::string& detail) {
    throw std::invalid_argument("CompileControlFlowExact: " + detail);
}

std::uint64_t MintLeaseGeneration() {
    std::uint64_t previous = last_lease_generation.load(std::memory_order_relaxed);
    for (;;) {
        if (previous == std::numeric_limits<std::uint64_t>::max()) {
            Fail("process-local artifact lease generation overflowed");
        }
        const std::uint64_t generation = previous + 1;
        if (last_lease_generation.compare_exchange_weak(
                previous, generation, std::memory_order_relaxed,
                std::memory_order_relaxed)) {
            return generation;
        }
    }
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
    std::string signature_digest;
    std::string launch_metadata_digest;
};

}  // namespace

namespace internal {

std::uint64_t MintControlFlowLeaseGenerationForTest() {
    return MintLeaseGeneration();
}

void SetControlFlowLeaseGenerationForTest(std::uint64_t last_generation) {
    last_lease_generation.store(last_generation, std::memory_order_relaxed);
}

}  // namespace internal

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
            const auto& pins = compiled.artifact_pins();
            const auto& calls = compiled.plan().calls();
            if (!compiled.module().IsReady() || pins.size() != 1 ||
                calls.size() != 1 || !pins.front().defined() ||
                !compiled.module().HasFunction(calls[0]->symbol)) {
                Fail("branch Call must resolve to exactly one real immutable compiler artifact");
            }
            const auto signature = compiled.module().signature(calls[0]->symbol);
            const auto metadata = compiled.module().launch_metadata(calls[0]->symbol);
            resolved.push_back(ResolvedBinding{
                task.id, compiled.module(), calls[0]->symbol,
                AbiNonOutputs(task, lowered.plan), pins.front(),
                support::HashText(signature.CanonicalBytes()),
                support::HashText(metadata.CanonicalBytes())});
        }
    }
    if (resolved.empty()) {
        Fail("requires at least one real branch kernel; a pure structural If has no production artifact");
    }

    const std::uint64_t generation = MintLeaseGeneration();
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
