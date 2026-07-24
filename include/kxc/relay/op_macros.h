/*! \file include/kxc/relay/op_macros.h
 * \brief 定义 Relay IR 节点、算子注册、attrs 和 Relay 到 TE lowering 属性。
 */

#pragma once

#include <any>
#include <functional>
#include <string>
#include <type_traits>

#include "op_attr_types.h"

namespace kxc {
namespace relay {

class OpRegEntry {
public:
    explicit OpRegEntry(Op op);

    OpRegEntry& describe(const std::string& descr);
    OpRegEntry& set_num_inputs(int n);
    OpRegEntry& set_input_arity_range(int min_inputs, int max_inputs);
    OpRegEntry& add_argument(const std::string& name, const std::string& type,
                             const std::string& description, bool is_optional = false,
                             const std::string& default_val = "");
    template <typename ValueType>
    OpRegEntry& set_attr(const std::string& attr_name, const ValueType& value) {
        OpNode* node = const_cast<OpNode*>(op_.operator->());
        if constexpr (std::is_same_v<std::decay_t<ValueType>, std::string>) {
            ValidateSemanticBinding(node, attr_name, value);
        } else {
            ValidateSemanticBinding(node, attr_name, "");
        }
        node->attrs[attr_name] = std::any(value);
        return *this;
    }

private:
    static void ValidateSemanticBinding(OpNode* node, const std::string& attr_name,
                                        const std::string& string_value);

    Op op_;
};

#define KXC_REGISTER_OP(OpName) \
    static ::kxc::relay::OpRegEntry __make_OpEntry_##OpName##__ = \
        ::kxc::relay::OpRegEntry(::kxc::relay::Op::Register(#OpName))

}  // namespace relay
}  // namespace kxc
