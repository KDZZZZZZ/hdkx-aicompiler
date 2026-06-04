/*! \file include/relay/transforms/simplify_expr.h
 * \brief 声明 Relay 优化 pass、multi-device 处理和 lowering 入口。
 */

#pragma once

#include "relay/relay.h"

namespace kxc {
namespace relay {

Function SimplifyExprPass(const Function& func);

}  // namespace relay
}  // namespace kxc
