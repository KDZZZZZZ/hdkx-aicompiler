#pragma once

#include "kxc/runtime/executable_plan.h"

namespace kxc::runtime::internal {

// Bump whenever PlanMemory can assign a different physical storage contract.
inline constexpr char kStaticMemoryPlanVersion[] =
    "last-use-sequential-single-stream-state-alias-v2";
inline constexpr char kDynamicFreshOutputMemoryPlanVersion[] =
    "dynamic-fresh-output-single-stream-no-reuse-v1";

/*! \brief Assigns deterministic reusable storage ids to safe intermediates. */
ExecutablePlan PlanMemory(const ExecutablePlan& plan);

}  // namespace kxc::runtime::internal
