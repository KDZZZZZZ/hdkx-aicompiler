/*! \file include/base/virtual_device.h
 * \brief 定义基础对象系统、容器、设备、NDArray、Target、PassContext 和 profiling 公共类型。
 */

#pragma once

#include <string>
#include <functional>

#include "base/target.h"

namespace kxc {

constexpr int kInvalidVirtualDeviceId = -1;

class VirtualDeviceNode : public Object {
public:
    ObjectRef device_obj;
    Target target;
    std::string memory_scope;
    int virtual_device_id{kInvalidVirtualDeviceId};

    KXC_OBJECT_DECLARE

    bool has_device() const;
    bool has_target() const;
    bool IsFullyUnconstrained() const;
    bool IsFullyConstrained() const;
    class Device device() const;
};

KXC_OBJECT_DEFINE(VirtualDeviceNode)

class VirtualDevice : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    VirtualDevice(const ObjectRef& ref) : ObjectRef(ref) {}

    VirtualDevice(const class Device& device, Target target = Target(),
                  std::string memory_scope = "",
                  int virtual_device_id = kInvalidVirtualDeviceId);
    VirtualDevice(Target target, std::string memory_scope = "",
                  int virtual_device_id = kInvalidVirtualDeviceId);

    const VirtualDeviceNode* operator->() const;

    bool IsFullyUnconstrained() const;
    bool IsFullyConstrained() const;
    class Device device() const;
    std::string ToString() const;

    static VirtualDevice FullyUnconstrained();
    static VirtualDevice ForDevice(const class Device& device);
    static VirtualDevice ForTarget(const Target& target);
    static VirtualDevice ForDeviceAndTarget(const class Device& device, const Target& target);
};

}  // namespace kxc

namespace std {
template <>
struct hash<kxc::VirtualDevice> {
    size_t operator()(const kxc::VirtualDevice& vd) const noexcept {
        return std::hash<const kxc::Object*>()(vd.get());
    }
};
}  // namespace std
