/*! \file include/kxc/te/topi/transform.h
 * \brief 定义 TOPI 风格的 tensor compute helper。
 */

#pragma once
#include "kxc/te/te.h"
#include "kxc/te/topi/tags.h"
#include "kxc/te/topi/utils.h"
#include "kxc/support/container.h"
#include <vector>
#include <numeric>
#include <set>
#include <stdexcept>

namespace kxc {
namespace te {
namespace topi {

// Transpose
Tensor transpose(const Tensor& x, Array<int> axes, std::string name = "transpose", std::string tag = kInjective);

// Expand Dims
Tensor expand_dims(const Tensor& x, int axis, int num_newaxis = 1, std::string name = "expand_dims", std::string tag = kBroadcast);

// Squeeze
Tensor squeeze(const Tensor& x, Array<int> axes = {}, std::string name = "squeeze", std::string tag = kInjective);

// Concatenate
Tensor concatenate(const Array<Tensor>& inputs, int axis = 0, std::string name = "concatenate", std::string tag = kInjective);

} // namespace topi
} // namespace te
} // namespace kxc
