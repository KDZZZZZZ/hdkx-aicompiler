/*! \file src/runtime/internal/executable_plan_validation.h
 * \brief Internal whole-plan validation entry point.
 */

#pragma once

#include "kxc/runtime/executable_plan.h"

namespace kxc::runtime::internal {

/*! \brief Validate cross-value producer, availability, and ordering invariants. */
void ValidateExecutablePlan(const ExecutablePlan& plan);

}  // namespace kxc::runtime::internal
