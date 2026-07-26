/*! \file include/kxc/relay/transforms/pipeline.h
 * \brief 声明 Relay 优化 pass、multi-device 处理和 lowering 入口。
 */

#pragma once

#include "kxc/pass/context.h"
#include "kxc/pass/pass.h"
#include "kxc/support/container.h"
#include "kxc/relay/relay.h"

namespace kxc {
namespace relay {

Function RunRelayPassPipeline(const Function& func, const Array<String>& pass_names);
Function RunRelayPassPipeline(const Function& func, const Array<String>& pass_names,
                              const PassContext& pass_ctx);
Array<String> RelayDefaultPassOrder();
Array<PassSpec> RelayRegisteredPassSpecs();

}  // namespace relay
}  // namespace kxc
