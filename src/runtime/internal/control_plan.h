/*! \file src/runtime/internal/control_plan.h
 * \brief Static-exact, runtime-neutral structured control-flow plan v2.
 */
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "kxc/runtime/device.h"

namespace kxc::runtime {

using ValueId = std::int64_t;
using RegionId = std::int64_t;
using TaskId = std::int64_t;

/*! \brief Static-exact logical value contract for ControlPlan v2. */
struct ControlValueSpec {
    ValueId id{-1};
    std::string dtype;
    std::vector<std::int64_t> shape;
    Device device{Device::CPU()};
    std::string source_locator;
};

struct EffectSummary {
    std::vector<ValueId> reads;
    std::vector<ValueId> writes;
    std::vector<ValueId> allocates;
    bool host_callback{false};
    bool device_sync{false};
};

struct AliasPair {
    ValueId first{-1};
    ValueId second{-1};
};

struct AliasSummary {
    std::vector<AliasPair> must_alias;
    std::vector<AliasPair> may_alias;
    std::vector<AliasPair> no_alias;
};

struct PhiBinding {
    ValueId result{-1};
    ValueId then_value{-1};
    ValueId else_value{-1};
};

struct BranchSpec {
    ValueId predicate{-1};
    RegionId then_region{-1};
    RegionId else_region{-1};
    std::vector<PhiBinding> phis;
};

struct LoopCarriedBinding {
    ValueId result{-1};
    ValueId initial{-1};
    ValueId body_argument{-1};
    ValueId backedge{-1};
};

struct LoopSpec {
    RegionId condition_region{-1};
    RegionId body_region{-1};
    ValueId condition_value{-1};
    std::vector<LoopCarriedBinding> carried;
    std::int64_t max_trip_count{-1};
};

enum class ControlTaskKind { kKernel, kBranch, kLoop };

/*! \brief Preparation-time kernel binding state; v2 has no executable artifact. */
enum class KernelBindingState {
    kNotApplicable,
    kUnresolvedRelayKernel,
};

struct ControlTask {
    TaskId id{-1};
    ControlTaskKind kind{ControlTaskKind::kKernel};
    /*! \brief Runtime must not execute kUnresolvedRelayKernel as an artifact. */
    KernelBindingState binding_state{KernelBindingState::kNotApplicable};
    std::vector<ValueId> inputs;
    /*! \brief Ordered logical kernel operands; duplicates are significant. */
    std::vector<ValueId> argument_values;
    std::vector<ValueId> outputs;
    std::vector<TaskId> dependencies;
    std::string kernel_ref;
    std::string source_locator;
    Device device{Device::CPU()};
    std::string stream{"default"};
    EffectSummary effect;
    AliasSummary alias;
    BranchSpec branch;
    LoopSpec loop;
};

struct ControlRegion {
    RegionId id{-1};
    std::vector<ValueId> live_ins;
    std::vector<ValueId> live_outs;
    std::vector<ControlTask> tasks;
    EffectSummary effect;
    AliasSummary alias;
    std::string source_locator;
};

/*! \brief Frozen v2 vocabulary. IDs and locators identify plan locations only. */
struct ControlPlan {
    static constexpr std::int64_t kSchemaVersion = 2;

    std::int64_t schema_version{kSchemaVersion};
    std::vector<ControlValueSpec> values;
    RegionId entry_region{-1};
    std::vector<RegionId> region_order;
    std::vector<ControlRegion> regions;
    std::vector<ValueId> graph_inputs;
    std::vector<ValueId> constant_values;
    std::vector<ValueId> graph_outputs;

    /*! \throws std::invalid_argument if this is not a static-exact v2 plan. */
    void ValidateStaticExact() const;
    /*! \brief Compatibility shorthand for ValidateStaticExact. */
    void Validate() const { ValidateStaticExact(); }
    /*! \brief Stable text for diagnostics; locators are rendered, never interpreted. */
    std::string CanonicalText() const;
};

void VerifyControlPlan(const ControlPlan& plan);

}  // namespace kxc::runtime
