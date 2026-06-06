/*! \file include/relay/transforms/infer_type.h
 * \brief Declares Relay static type and shape inference.
 */

#pragma once

#include "relay/relay.h"

namespace kxc {
namespace relay {

Function InferTypePass(const Function& func);

}  // namespace relay
}  // namespace kxc
