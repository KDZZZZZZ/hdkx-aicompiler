/*! \file src/distributed/dref.cc
 * \brief 实现 DRef 对所属会话寄存器的引用生命周期。
 */

#include "kxc/distributed/dref.h"
#include "kxc/support/object_registration.h"
#include "kxc/distributed/session.h"

#include <memory>
#include <stdexcept>

namespace kxc {
namespace disco {

KXC_OBJECT_DEFINE(DRefNode)

DRef::DRef(int reg_id, ObjectRef session) {
    auto* owner = const_cast<DiscoSessionNode*>(session.As<DiscoSessionNode>());
    if (!owner || reg_id < 0) throw std::invalid_argument("DRef requires an allocated register and its DiscoSession");
    auto node = std::make_unique<DRefNode>();
    owner->RetainRegister(reg_id);
    node->session = std::move(session);
    node->reg_id = reg_id;
    SetData(node.release());
}

DRefNode::~DRefNode() {
    if (reg_id >= 0) {
        if (const auto* owner = session.As<DiscoSessionNode>()) {
            const_cast<DiscoSessionNode*>(owner)->ReleaseRegister(reg_id);
        }
    }
}

const DRefNode* DRef::operator->() const {
    const auto* node = As<DRefNode>();
    if (!node) throw std::invalid_argument("expected DRef");
    return node;
}

int DRef::reg_id() const {
    if (!defined()) {
        return -1;
    }
    return operator->()->reg_id;
}

bool DRef::valid() const {
    const auto* node = As<DRefNode>();
    return node && node->reg_id >= 0 && node->session.As<DiscoSessionNode>();
}

}  // namespace disco
}  // namespace kxc
