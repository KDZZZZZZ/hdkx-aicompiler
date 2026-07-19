/*! \file include/base/target.h
 * \brief 定义基础对象系统、容器、设备、NDArray、Target、PassContext 和 profiling 公共类型。
 */

#pragma once

#include <string>

#include "base/device_api.h"

namespace kxc {

class TargetNode : public Object {
public:
    std::string kind;
    DeviceTypeCode device_type{kUnknown};
    int device_id{-1};
    DeviceAttributes attrs;

    KXC_OBJECT_DECLARE
};

KXC_OBJECT_DEFINE(TargetNode)

class Target : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    Target(const ObjectRef& ref) : ObjectRef(ref) {}

    const TargetNode* operator->() const;
    std::string ToString() const;
};

Target BuildTarget(const Device& device);

}  // namespace kxc

