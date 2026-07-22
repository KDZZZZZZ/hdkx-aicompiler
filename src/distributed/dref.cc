/*! \file src/distributed/dref.cc
 * \brief 实现 Disco 线程会话、执行计划解释器和 CPU/NCCL 通信后端。
 */

#include "kxc/distributed/dref.h"
#include "kxc/support/object_registration.h"

namespace kxc {
namespace disco {

KXC_OBJECT_DEFINE(DRefNode)

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
