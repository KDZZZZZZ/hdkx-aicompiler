#pragma once
#include "te/te.h"
#include "relay/relay.h"
#include "relay/op.h" // For Attrs
#include "base/expr.h" // For Type
#include <vector>
#include <functional>
#include <any>

namespace kxc {
namespace relay {

// FTVMCompute: The function signature for lowering a Relay Op to TE Tensors.
// Args:
//   attrs: The operator attributes.
//   inputs: The input TE Tensors.
//   out_type: The output type (for shape/dtype info).
// Returns:
//   The output TE Tensor (or Tensors). For simplicity, assume single output for now.
using FTVMCompute = std::function<te::Tensor(const Attrs&, const std::vector<te::Tensor>&, const kxc::Type&)>;

}
}
