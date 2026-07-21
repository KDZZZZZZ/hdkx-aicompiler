/*! \file include/relay/transforms/multi_device.h
 * \brief 声明 Relay 优化 pass、multi-device 处理和 lowering 入口。
 */

#pragma once

#include "base/execution_plan.h"
#include "base/pass.h"
#include "relay/relay.h"
#include "relay/transforms/lower.h"

namespace kxc {
namespace relay {

Function InsertDeviceCommunicationPass(const Function& func);
ExecutionPlan LowerRelayToExecPlanPass(const Function& func);

}  // namespace relay
}  // namespace kxc

