/*! \file include/kxc/runtime/control_execution_plan.h
 * \brief Immutable resolved control-runtime execution schema v1.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kxc/runtime/compiled_module.h"

namespace kxc::runtime {

using ControlExecutionValueId = std::int64_t;
using ControlExecutionRegionId = std::int64_t;
using ControlExecutionTaskId = std::int64_t;

/*! \brief Experimental entry snapshot bound to one ready fixture module. */
class BoundControlKernel final {
public:
    BoundControlKernel() = default;
    BoundControlKernel(api::CompiledModule module, String entry_symbol,
                       std::uint64_t binding_revision);

    void Validate() const;
    AsyncOperation Launch(const Array<NDArray>& ordered_arguments,
                          const DeviceStream& stream) const;
    codegen::KernelSignature signature() const;
    codegen::KernelLaunchMetadata launch_metadata() const;
    /*! \brief Returns an independent deep copy; execution retains a private snapshot. */
    NDArray Constant(const String& key) const;
    /*! \brief Compares bytes/contracts without exposing the execution snapshot. */
    bool MatchesConstant(const String& key, const NDArray& candidate) const;
    /*! \brief Caller label only, with no authority, freshness, or hot-swap proof. */
    std::uint64_t binding_revision() const;
    Device device() const;
    bool defined() const noexcept;

private:
    struct State;
    std::shared_ptr<const State> state_;
};

struct ControlExecutionValueSpec {
    ControlExecutionValueId id{-1};
    std::string dtype;
    std::vector<std::int64_t> shape;
    Device device{Device::CPU()};
    std::string source_locator;
};

struct ControlExecutionPhiBinding {
    ControlExecutionValueId result{-1};
    ControlExecutionValueId then_value{-1};
    ControlExecutionValueId else_value{-1};
};

struct ControlExecutionBranchSpec {
    ControlExecutionValueId predicate{-1};
    ControlExecutionRegionId then_region{-1};
    ControlExecutionRegionId else_region{-1};
    std::vector<ControlExecutionPhiBinding> phis;
};

struct ControlExecutionLoopCarriedBinding {
    ControlExecutionValueId result{-1};
    ControlExecutionValueId initial{-1};
    ControlExecutionValueId body_argument{-1};
    ControlExecutionValueId backedge{-1};
};

struct ControlExecutionLoopSpec {
    ControlExecutionRegionId condition_region{-1};
    ControlExecutionRegionId body_region{-1};
    ControlExecutionValueId condition_value{-1};
    std::vector<ControlExecutionLoopCarriedBinding> carried;
    std::int64_t max_trip_count{-1};
};

enum class ControlExecutionTaskKind { kKernel, kBranch, kLoop };
/*! \brief Read-only sources and fresh kernel outputs; not a physical no-alias proof.
 *
 * Distinct read-only graph inputs may alias.  The control executor freshly
 * allocates each kernel output, while Phi and loop forwarding preserve selected
 * storage.  The executor performs no internal storage reuse.
 */
enum class ControlExecutionEffectModel { kPureFreshKernelOutputsV1 };

struct ControlExecutionTask {
    ControlExecutionTaskId id{-1};
    ControlExecutionTaskKind kind{ControlExecutionTaskKind::kKernel};
    /*! \brief Unique task boundary values, unlike argument_values. */
    std::vector<ControlExecutionValueId> inputs;
    /*! \brief Full ABI-order value ids, including outputs; duplicates matter. */
    std::vector<ControlExecutionValueId> argument_values;
    std::vector<ControlExecutionValueId> outputs;
    std::vector<ControlExecutionTaskId> dependencies;
    std::string source_locator;
    Device device{Device::CPU()};
    std::string stream{"default"};
    BoundControlKernel kernel;
    ControlExecutionBranchSpec branch;
    ControlExecutionLoopSpec loop;
};

struct ControlExecutionRegion {
    ControlExecutionRegionId id{-1};
    std::vector<ControlExecutionValueId> live_ins;
    std::vector<ControlExecutionValueId> live_outs;
    std::vector<ControlExecutionTask> tasks;
    std::string source_locator;
};

struct ControlExecutionPlanSpec {
    static constexpr std::int64_t kSchemaVersion = 1;
    std::int64_t schema_version{kSchemaVersion};
    std::int64_t source_control_plan_version{2};
    ControlExecutionEffectModel effect_model{
        ControlExecutionEffectModel::kPureFreshKernelOutputsV1};
    std::vector<ControlExecutionValueSpec> values;
    ControlExecutionRegionId entry_region{-1};
    std::vector<ControlExecutionRegionId> region_order;
    std::vector<ControlExecutionRegion> regions;
    std::vector<ControlExecutionValueId> graph_inputs;
    std::vector<ControlExecutionValueId> constant_values;
    std::vector<ControlExecutionValueId> graph_outputs;
};

/*! \brief Frozen, runtime-only resolved control graph. */
class ControlExecutionPlan final {
public:
    static constexpr std::int64_t kSchemaVersion = 1;

    ControlExecutionPlan() = default;
    explicit ControlExecutionPlan(ControlExecutionPlanSpec spec);

    bool defined() const noexcept;
    void Validate() const;
    const ControlExecutionPlanSpec& spec() const;
    const std::vector<ControlExecutionValueSpec>& values() const;
    const std::vector<ControlExecutionRegion>& regions() const;
    const std::vector<ControlExecutionValueId>& graph_inputs() const;
    const std::vector<ControlExecutionValueId>& constant_values() const;
    const std::vector<ControlExecutionValueId>& graph_outputs() const;

private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};

void VerifyControlExecutionPlan(const ControlExecutionPlanSpec& plan);

}  // namespace kxc::runtime
