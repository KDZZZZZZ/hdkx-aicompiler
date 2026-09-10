/*! \file include/kxc/distributed/dref.h
 * \brief 定义 Disco 分布式会话、DRef、执行器和通信后端接口。
 */

#pragma once

#include "kxc/support/object.h"

namespace kxc {
namespace disco {

class DRefNode : public Object {
public:
    ~DRefNode() override;
    int reg_id{-1};
    ObjectRef session;

    KXC_OBJECT_DECLARE
};

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
