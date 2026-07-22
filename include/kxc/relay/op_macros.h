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
    OpRegEntry& set_spec(OperatorSpec spec);
    OpRegEntry& add_argument(const std::string& name, const std::string& type,
                             const std::string& description, bool is_optional = false,
                             const std::string& default_val = "");
    template <typename ValueType>
    OpRegEntry& set_attr(const std::string& attr_name, const ValueType& value) {
        OpNode* node = const_cast<OpNode*>(op_.operator->());
        node->attrs[attr_name] = std::any(value);
        node->has_spec = true;
        node->spec.name = node->name;
        using Decayed = std::decay_t<ValueType>;
        if constexpr (std::is_same_v<Decayed, std::string>) {
            if (attr_name == "TAttrs") {
                node->spec.attrs_type_key = value;
            }
        } else if constexpr (std::is_same_v<Decayed, FInferType>) {
            if (attr_name == "FInferType") {
                node->spec.type_relation_key = attr_name;
            }
        } else if constexpr (std::is_same_v<Decayed, FRelayToTE>) {
            if (attr_name == "FRelayToTE") {
                node->spec.lowering_kind = OperatorLoweringKind::kSingleTE;
                node->spec.lowering_key = attr_name;
            }
        } else if constexpr (std::is_same_v<Decayed, FRelayToTEMulti>) {
            if (attr_name == "FRelayToTEMulti") {
                node->spec.lowering_kind = OperatorLoweringKind::kMultiTE;
                node->spec.lowering_key = attr_name;
            }
        }
        return *this;
    }

private:
    Op op_;
};

#define KXC_REGISTER_OP(OpName) \
    static ::kxc::relay::OpRegEntry __make_OpEntry_##OpName##__ = \
        ::kxc::relay::OpRegEntry(::kxc::relay::Op::Register(#OpName))

}  // namespace relay
}  // namespace kxc
