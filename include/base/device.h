/*! \file include/base/device.h
 * \brief 定义基础对象系统、容器、设备、NDArray、Target、PassContext 和 profiling 公共类型。
 */

#pragma once

#include <sstream>
#include <string>
#include <unordered_map>

#include "object.h"

namespace kxc {

class Device;  // Forward declaration.

// Keep source compatibility with legacy type macro.
#define kKXC_DEVICE_TYPE (Device::_type_index)

enum DeviceTypeCode : int {
    kCPU = 0,
    kGPU = 1,
    kOpenCL = 2,
    kMetal = 3,
    kUnknown = 99,
};

#ifdef Device
#undef Device
#endif
class Device : public Object {
public:
    Device(DeviceTypeCode type, int id);

    KXC_OBJECT_DECLARE

    DeviceTypeCode device_type() const;
    int device_id() const;
    std::string ToString() const;
    bool operator==(const Device& other) const;

private:
    DeviceTypeCode device_type_;
    int device_id_;
};

KXC_OBJECT_DEFINE(Device)

class DeviceManager {
public:
    static DeviceManager* Global();
    ObjectRef GetOrCreate(DeviceTypeCode type, int id);

private:
    DeviceManager() = default;
    DeviceManager(const DeviceManager&) = delete;
    DeviceManager& operator=(const DeviceManager&) = delete;

    std::unordered_map<std::string, ObjectRef> devices_;
};

ObjectRef Device(DeviceTypeCode type, int id);

}  // namespace kxc