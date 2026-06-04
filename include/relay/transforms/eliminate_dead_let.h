/*! \file include/relay/transforms/eliminate_dead_let.h
 * \brief 声明 Relay 优化 pass、multi-device 处理和 lowering 入口。
 */

#pragma once

#include "relay/relay.h"

namespace kxc {
namespace relay {

Function EliminateDeadLetPass(const Function& func);

}  // namespace relay
}  // namespace kxc
