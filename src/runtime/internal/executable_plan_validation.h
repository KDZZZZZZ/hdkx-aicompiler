/*! \file src/runtime/internal/executable_plan_validation.h
 * \brief Internal whole-plan validation entry point.
 */

#pragma once

#include "kxc/runtime/executable_plan.h"

namespace kxc::runtime::internal {

/*! \brief Return whether two values require the same physical tensor layout. */
bool SameValueStorageContract(const ValueSpec& lhs, const ValueSpec& rhs);

/*! \brief Return whether last-use planning may assign a value to a reused slot. */
bool IsValueStorageReusable(const ValueSpec& value);

/*! \brief Validate cross-value producer, availability, and ordering invariants. */
void ValidateExecutablePlan(const ExecutablePlan& plan);

}  // namespace kxc::runtime::internal
