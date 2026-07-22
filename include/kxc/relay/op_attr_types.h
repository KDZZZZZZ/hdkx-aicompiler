/*! \file include/kxc/relay/op_attr_types.h
 * \brief 定义编译 pass 使用的 Relay 算子属性函数类型。
 */

#pragma once

#include <functional>

#include "kxc/support/container.h"
#include "kxc/ir/expr.h"
#include "kxc/relay/op.h"
#include "kxc/te/te.h"

namespace kxc {
namespace relay {

using FInferType = std::function<Type(const Attrs&, const Array<Type>&)>;

using FRelayToTE =
    std::function<te::Tensor(const Attrs&, const Array<te::Tensor>&, const kxc::Type&)>;

using FRelayToTEMulti =
    std::function<Array<te::Tensor>(const Attrs&, const Array<te::Tensor>&, const kxc::Type&)>;

}  // namespace relay
}  // namespace kxc
