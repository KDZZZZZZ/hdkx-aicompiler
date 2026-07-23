/*! \file include/kxc/compiler/control_flow.h
 * \brief Static-exact Relay preparation for ControlPlan v1.
 */
#pragma once

#include "kxc/relay/relay.h"
#include "kxc/runtime/control_plan.h"

namespace kxc::api {

/*! \brief Lowers checked static Relay, including Relay If, into a ControlPlan v1.
 *
 * This is compiler preparation/reference semantics only.  It neither creates a
 * runtime session nor performs eager execution, tracing, or backend execution.
 */
runtime::ControlPlan LowerRelayToControlPlan(Function function);

}  // namespace kxc::api
