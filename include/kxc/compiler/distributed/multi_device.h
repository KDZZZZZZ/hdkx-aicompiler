/*! \file include/kxc/compiler/distributed/multi_device.h
 * \brief 声明 Relay 优化 pass、multi-device 处理和 lowering 入口。
 */

#pragma once

#include "kxc/distributed/execution_plan.h"
#include "kxc/relay/visitor.h"
#include "kxc/relay/relay.h"
#include "kxc/compiler/lowering/relay_to_tir.h"

namespace kxc {
namespace relay {

Function InsertDeviceCommunicationPass(const Function& func);
DiscoPlacement BuildDiscoPlacementPass(const Function& func);
ExecutionPlan LowerRelayToExecPlanPass(const Function& func);

}  // namespace relay
}  // namespace kxc
