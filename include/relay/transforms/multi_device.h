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

PassContext BuildDiscoPlacementPass(const Function& func);
Function InsertDeviceCommunicationPass(const Function& func);
/*! \brief 降低 Relay 计算并保留 codegen 所需的常量绑定。 */
LoweredFunction LowerRelayComputeToTIRPass(const Function& func);
ExecutionPlan LowerRelayToExecPlanPass(const Function& func);

}  // namespace relay
}  // namespace kxc

