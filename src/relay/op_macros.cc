/*! \file src/relay/op_macros.cc
 * \brief 实现 Relay 节点、算子元数据、pass 工具和公共注册。
 */

#include "kxc/relay/op_macros.h"

#include <stdexcept>
#include <utility>

namespace kxc {
namespace relay {

OpRegEntry::OpRegEntry(Op op) : op_(std::move(op)) {}

OpRegEntry& OpRegEntry::describe(const std::string& descr) {
    OpNode* node = const_cast<OpNode*>(op_.operator->());
    node->description = descr;
    node->has_spec = true;
    node->spec.name = node->name;
    return *this;
}

OpRegEntry& OpRegEntry::set_num_inputs(int n) {
    OpNode* node = const_cast<OpNode*>(op_.operator->());
    node->num_inputs = n;
    node->has_spec = true;
    node->spec.name = node->name;
    node->spec.input_arity.num_inputs = n;
    return *this;
}

OpRegEntry& OpRegEntry::set_input_arity_range(int min_inputs, int max_inputs) {
    if (min_inputs < 0 || max_inputs < min_inputs) {
        throw std::runtime_error("operator input arity range is invalid");
    }
    OpNode* node = const_cast<OpNode*>(op_.operator->());
    node->num_inputs = -1;
    node->has_spec = true;
    node->spec.name = node->name;
    node->spec.input_arity.num_inputs = -1;
    node->spec.input_arity.min_inputs = min_inputs;
    node->spec.input_arity.max_inputs = max_inputs;
    return *this;
}

OpRegEntry& OpRegEntry::set_spec(OperatorSpec spec) {
    OpNode* node = const_cast<OpNode*>(op_.operator->());
    if (spec.name.empty()) {
        spec.name = node->name;
    }
    if (spec.name != node->name) {
        throw std::runtime_error("OperatorSpec name mismatch for op: " + node->name);
    }
    node->spec = std::move(spec);
    node->has_spec = true;
    return *this;
}

OpRegEntry& OpRegEntry::add_argument(const std::string& name, const std::string& type,
                                     const std::string& description, bool is_optional,
                                     const std::string& default_val) {
    OpNode* node = const_cast<OpNode*>(op_.operator->());
    ArgumentInfo arg;
    arg.name = name;
    arg.type = type;
    arg.description = description;
    arg.is_optional = is_optional;
    arg.default_value_desc = default_val;
    node->arguments.push_back(arg);
    node->has_spec = true;
    node->spec.name = node->name;
    node->spec.arguments.push_back(arg);
    return *this;
}

}  // namespace relay
}  // namespace kxc
