#include "relay/op_macros.h"

namespace kxc {
namespace relay {

OpRegEntry::OpRegEntry(Op op) : op_(std::move(op)) {}

OpRegEntry& OpRegEntry::describe(const std::string& descr) {
    OpNode* node = const_cast<OpNode*>(op_.operator->());
    node->description = descr;
    return *this;
}

OpRegEntry& OpRegEntry::set_num_inputs(int n) {
    OpNode* node = const_cast<OpNode*>(op_.operator->());
    node->num_inputs = n;
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
    return *this;
}

OpRegEntry& OpRegEntry::set_support_level(int level) {
    (void)level;
    return *this;
}

}  // namespace relay
}  // namespace kxc

