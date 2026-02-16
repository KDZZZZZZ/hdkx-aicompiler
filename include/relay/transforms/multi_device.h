#pragma once

#include "base/execution_plan.h"
#include "base/pass.h"
#include "relay/relay.h"
#include "tir/stmt.h"

namespace kxc {
namespace relay {

PassContext BuildDiscoPlacementPass(const Function& func);
Function InsertDeviceCommunicationPass(const Function& func);
tir::PrimFunc LowerRelayComputeToTIRPass(const Function& func);
ExecutionPlan LowerRelayToExecPlanPass(const Function& func);

}  // namespace relay
}  // namespace kxc

