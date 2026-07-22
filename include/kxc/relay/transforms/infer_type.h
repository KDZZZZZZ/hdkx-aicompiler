/*! \file include/kxc/relay/transforms/infer_type.h
 * \brief 声明 Relay 静态类型和 shape 推导 pass。
 */

#pragma once

#include "kxc/relay/relay.h"

namespace kxc {
namespace relay {

Function InferTypePass(const Function& func);

}  // namespace relay
}  // namespace kxc
