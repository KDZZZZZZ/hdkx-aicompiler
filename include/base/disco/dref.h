#pragma once

#include "base/object.h"

namespace kxc {
namespace disco {

class DRefNode : public Object {
public:
    int reg_id{-1};
    ObjectRef session;

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(DRefNode)

class DRef : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    explicit DRef(const ObjectRef& ref) : ObjectRef(ref) {}
    DRef(int reg_id, ObjectRef session);

    const DRefNode* operator->() const;
    int reg_id() const;
    bool valid() const;
};

}  // namespace disco
}  // namespace kxc

