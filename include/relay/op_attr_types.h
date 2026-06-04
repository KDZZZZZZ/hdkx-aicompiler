/*! \file include/relay/op_attr_types.h
 * \brief 定义 Relay IR 节点、算子注册、attrs 和 Relay 到 TE lowering 属性。
 */

#pragma once
#include "te/te.h"
#include "relay/op.h" // For Attrs
#include "base/expr.h" // For Type
#include "base/container.h"
#include <vector>
#include <functional>
#include <any>

namespace kxc {
namespace relay {

// FRelayToTE: The function signature for lowering a Relay Op to TE Tensors.
// Args:
//   attrs: The operator attributes.
//   inputs: The input TE Tensors.
//   out_type: The output type (for shape/dtype info).
// Returns:
//   The output TE Tensor (or Tensors). For simplicity, assume single output for now.
using FRelayToTE = std::function<te::Tensor(const Attrs&, const Array<te::Tensor>&, const kxc::Type&)>;

}
}
