/*! \file include/kxc/relay/transforms/fold_constant.h
 * \brief 声明 Relay 优化 pass、multi-device 处理和 lowering 入口。
 */

#pragma once

#include "kxc/relay/relay.h"

namespace kxc {
namespace relay {

Function FoldConstantPass(const Function& func);

}  // namespace relay
}  // namespace kxc
