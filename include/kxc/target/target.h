/*! \file include/kxc/target/target.h
 * \brief 定义基础对象系统、容器、设备、NDArray、Target、PassContext 和 profiling 公共类型。
 */

#pragma once

#include <string>

#include "kxc/runtime/device_info.h"

namespace kxc {

class TargetNode : public Object {
public:
    std::string kind;
    DeviceTypeCode device_type{kUnknown};
    int device_id{-1};
    DeviceAttributes attrs;

    KXC_OBJECT_DECLARE
};


class Target : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    Target(const ObjectRef& ref) : ObjectRef(ref) {}

    const TargetNode* operator->() const;
    std::string ToString() const;
    // Canonical capability snapshot, shared by compiler identity and placement.
    // Volatile available memory is excluded.
    std::string CanonicalBytes() const;
};

Target BuildTarget(const Device& device);

}  // namespace kxc
