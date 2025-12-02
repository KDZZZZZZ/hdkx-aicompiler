#pragma once
#include "../base/op.h"
#include <string>
#include <functional>
#include <any>

namespace kxc {

// Helper class for fluent operator registration
class OpRegEntry {
public:
    explicit OpRegEntry(Op op) : op_(op) {}

    OpRegEntry& describe(const std::string& descr) {
        OpNode* node = const_cast<OpNode*>(op_.operator->());
        node->description = descr;
        return *this;
    }

    OpRegEntry& set_num_inputs(int n) {
        // In a real implementation, we would set this in OpNode
        // op_->num_inputs = n;
        return *this;
    }

    OpRegEntry& set_support_level(int level) {
        // op_->support_level = level;
        return *this;
    }
    
    template<typename ValueType>
    OpRegEntry& set_attr(const std::string& attr_name, const ValueType& value) {
        OpNode* node = const_cast<OpNode*>(op_.operator->());
        node->attrs[attr_name] = std::any(value);
        return *this;
    }

private:
    Op op_;
};

// Macro for operator registration
#define KXC_REGISTER_OP(OpName) \
    static ::kxc::OpRegEntry __make_OpEntry_##OpName##__ = \
        ::kxc::OpRegEntry(::kxc::Op::Get(#OpName))

} // namespace kxc
