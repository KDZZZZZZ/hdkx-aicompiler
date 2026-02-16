#include "base/disco/dref.h"

namespace kxc {
namespace disco {

DRef::DRef(int reg_id, ObjectRef session) {
    DRefNode* node = new DRefNode();
    node->reg_id = reg_id;
    node->session = std::move(session);
    SetData(node);
}

const DRefNode* DRef::operator->() const {
    return static_cast<const DRefNode*>(object_);
}

int DRef::reg_id() const {
    if (!defined()) {
        return -1;
    }
    return operator->()->reg_id;
}

bool DRef::valid() const {
    return defined() && operator->()->reg_id >= 0 && operator->()->session.defined();
}

}  // namespace disco
}  // namespace kxc

