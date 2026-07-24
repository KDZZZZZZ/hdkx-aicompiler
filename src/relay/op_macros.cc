/*! \file src/relay/op_macros.cc
 * \brief 实现 Relay 节点、算子元数据、pass 工具和公共注册。
 */

#include "kxc/relay/op_macros.h"

#include <stdexcept>
#include <utility>

namespace kxc {
namespace relay {

OpRegEntry::OpRegEntry(Op op) : op_(std::move(op)) {}

void OpRegEntry::ValidateSemanticBinding(OpNode* node, const std::string& attr_name,
                                         const std::string& string_value) {
    if (!node || !node->has_spec) {
        throw std::runtime_error("operator binding requires an OperatorSpec");
    }
    if (attr_name == "TAttrs" && node->spec.attrs_type_key != string_value) {
        throw std::runtime_error("TAttrs binding conflicts with OperatorSpec for op: " +
                                 node->name);
    }
    if (attr_name == "FInferType" && node->spec.type_relation_key != attr_name) {
        throw std::runtime_error("FInferType binding conflicts with OperatorSpec for op: " +
                                 node->name);
    }
    if (attr_name == "FRelayToTE" &&
        (node->spec.lowering_kind != OperatorLoweringKind::kSingleTE ||
         node->spec.lowering_key != attr_name)) {
        throw std::runtime_error("FRelayToTE binding conflicts with OperatorSpec for op: " +
                                 node->name);
    }
    if (attr_name == "FRelayToTEMulti" &&
        (node->spec.lowering_kind != OperatorLoweringKind::kMultiTE ||
         node->spec.lowering_key != attr_name)) {
        throw std::runtime_error("FRelayToTEMulti binding conflicts with OperatorSpec for op: " +
                                 node->name);
    }
}

OpRegEntry& OpRegEntry::describe(const std::string& descr) {
    const_cast<OpNode*>(op_.operator->())->description = descr;
    return *this;
}

OpRegEntry& OpRegEntry::set_num_inputs(int n) {
    const OpNode* node = op_.operator->();
    if (node->spec.input_arity.num_inputs != n) {
        throw std::runtime_error("input arity conflicts with OperatorSpec for op: " +
                                 node->name);
    }
    return *this;
}

OpRegEntry& OpRegEntry::set_input_arity_range(int min_inputs, int max_inputs) {
    const OpNode* node = op_.operator->();
    const InputArity& arity = node->spec.input_arity;
    if (min_inputs < 0 || max_inputs < min_inputs || arity.num_inputs != -1 ||
        arity.min_inputs != min_inputs || arity.max_inputs != max_inputs) {
        throw std::runtime_error("input arity range conflicts with OperatorSpec for op: " +
                                 node->name);
    }
    return *this;
}

OpRegEntry& OpRegEntry::add_argument(const std::string& name, const std::string& type,
                                     const std::string& description, bool is_optional,
                                     const std::string& default_val) {
    OpNode* node = const_cast<OpNode*>(op_.operator->());
    node->arguments.push_back({name, type, description, is_optional, default_val});
    return *this;
}

}  // namespace relay
}  // namespace kxc
