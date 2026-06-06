/*! \file include/relay/op_attr_types.h
 * \brief Defines typed Relay operator attributes used by compiler passes.
 */

#pragma once

#include <functional>

#include "base/container.h"
#include "base/expr.h"
#include "relay/op.h"
#include "te/te.h"

namespace kxc {
namespace relay {

using FInferType = std::function<Type(const Attrs&, const Array<Type>&)>;

using FRelayToTE =
    std::function<te::Tensor(const Attrs&, const Array<te::Tensor>&, const kxc::Type&)>;

}  // namespace relay
}  // namespace kxc
