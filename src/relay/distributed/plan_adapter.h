/*! \file src/relay/distributed/plan_adapter.h
 * \brief Private Relay-to-distributed execution plan attribute adapter.
 */

#pragma once

#include "kxc/distributed/execution_plan.h"
#include "kxc/relay/relay.h"

namespace kxc::relay::distributed_internal {

CommExecAttrs AdaptCommExecAttrs(const std::string& op_name,
                                 const ObjectRef& attrs);

}  // namespace kxc::relay::distributed_internal
