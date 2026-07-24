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

namespace internal {
struct BoundControlKernelAccess;
struct ControlExecutionPlanAccess;
struct ControlExecutionPlanSpec;
}  // namespace internal

using ControlExecutionValueId = std::int64_t;
using ControlExecutionRegionId = std::int64_t;
using ControlExecutionTaskId = std::int64_t;

/*! \brief Immutable entry snapshot bound by Compiler to one ready module artifact. */
class BoundControlKernel final {
public:
    BoundControlKernel() = default;

    void Validate() const;
    codegen::KernelSignature signature() const;
    codegen::KernelLaunchMetadata launch_metadata() const;
    /*! \brief Returns an independent deep copy; execution retains a private snapshot. */
    NDArray Constant(const String& key) const;
    /*! \brief Compares bytes/contracts without exposing the execution snapshot. */
    bool MatchesConstant(const String& key, const NDArray& candidate) const;
    Device device() const;
    bool defined() const noexcept;

private:
    friend struct internal::BoundControlKernelAccess;
    friend struct internal::ControlExecutionPlanAccess;

    AsyncOperation Launch(const Array<NDArray>& ordered_arguments,
                          const DeviceStream& stream) const;
    BoundControlKernel(api::CompiledModule module, String entry_symbol,
                       std::shared_ptr<const void> retention_owner);

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
    /*! \brief Physical ABI-order value ids, including outputs; unique per role. */
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

/*! \brief Frozen runtime execution typestate minted only by Compiler. */
class ControlExecutionPlan final {
public:
    static constexpr std::int64_t kSchemaVersion = 1;

    ControlExecutionPlan() = default;

    bool defined() const noexcept;
    void Validate() const;
    std::int64_t source_control_plan_version() const;
    ControlExecutionEffectModel effect_model() const;
    ControlExecutionRegionId entry_region() const;
    const std::vector<ControlExecutionRegionId>& region_order() const;
    const std::vector<ControlExecutionValueSpec>& values() const;
    const std::vector<ControlExecutionRegion>& regions() const;
    const std::vector<ControlExecutionValueId>& graph_inputs() const;
    const std::vector<ControlExecutionValueId>& constant_values() const;
    const std::vector<ControlExecutionValueId>& graph_outputs() const;

private:
    friend struct internal::ControlExecutionPlanAccess;
    explicit ControlExecutionPlan(internal::ControlExecutionPlanSpec spec);

    struct Impl;
    std::shared_ptr<const Impl> impl_;
};

}  // namespace kxc::runtime
