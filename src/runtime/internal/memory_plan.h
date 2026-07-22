#pragma once

#include "kxc/runtime/executable_plan.h"

namespace kxc::runtime::internal {

/*! \brief Assigns deterministic reusable storage ids to safe intermediates. */
ExecutablePlan PlanMemory(const ExecutablePlan& plan);

}  // namespace kxc::runtime::internal
