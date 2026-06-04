/*! \file include/relay/transforms/pipeline.h
 * \brief 声明 Relay 优化 pass、multi-device 处理和 lowering 入口。
 */

#pragma once

#include "base/container.h"
#include "relay/relay.h"

namespace kxc {
namespace relay {

Function RunRelayPassPipeline(const Function& func, const Array<String>& pass_names);

}  // namespace relay
}  // namespace kxc
