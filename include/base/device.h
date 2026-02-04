#pragma once

#include <string>
#include <unordered_map>
#include "object.h" // 包含 ObjectRef 的定义
#include <sstream> // 用于 ToString

namespace kxc {

class Device; // Forward declaration

// Device will use automatic registration.
// Use a macro to maintain source compatibility for kKXC_DEVICE_TYPE
#define kKXC_DEVICE_TYPE (Device::_type_index)

/**
 * @brief 枚举类型：DeviceTypeCode，用于表示设备类型
 */
enum DeviceTypeCode : int {
    kCPU = 0,
    kGPU = 1,
    kOpenCL = 2,
    kMetal = 3,
    kUnknown = 99, // 未知设备类型
};

/**
 * @brief Device 对象，继承自 Object
 */
// Avoid conflict with Windows Device macro if present
#ifdef Device
#undef Device
#endif
class Device : public Object {
public:
    Device(DeviceTypeCode type, int id) : device_type_(type), device_id_(id) {}

    KXC_OBJECT_DECLARE

    // 获取设备类型
    DeviceTypeCode device_type() const { return device_type_; }

    // 获取设备 ID
    int device_id() const { return device_id_; }

    // 转换为字符串表示
    std::string ToString() const {
        std::stringstream ss;
        ss << "Device(";
        switch (device_type_) {
            case kCPU: ss << "CPU"; break;
            case kGPU: ss << "GPU"; break;
            case kOpenCL: ss << "OpenCL"; break;
            case kMetal: ss << "Metal"; break;
            case kUnknown: ss << "Unknown"; break;
        }
        ss << ", id=" << device_id_ << ")";
        return ss.str();
    }

    // 可能需要重载 == 和 hash 函数以便在 unordered_map 中使用
    bool operator==(const Device& other) const {
        return device_type_ == other.device_type_ && device_id_ == other.device_id_;
    }

private:
    DeviceTypeCode device_type_;
    int device_id_;
};

KXC_OBJECT_DEFINE(Device)

/**
 * @brief DeviceManager 管理 Device 实例的创建和缓存
 */
class DeviceManager {
public:
    static DeviceManager* Global() {
        static DeviceManager manager;
        return &manager;
    }

    /**
     * @brief 获取或创建一个 Device 实例的 ObjectRef
     * @param type 设备类型码
     * @param id 设备 ID
     * @return 对应的 Device 的 ObjectRef
     */
    ObjectRef GetOrCreate(DeviceTypeCode type, int id) {
        std::string key = std::to_string(type) + "_" + std::to_string(id);
        
        // 尝试从缓存中查找
        auto it = devices_.find(key);
        if (it != devices_.end()) {
            return it->second; // 返回缓存中的 ObjectRef
        }

        // 如果不存在，则创建新的 Device 实例并包装成 ObjectRef
        // 注意：这里 new Device(type, id) 会调用 Object::operator new 进行内存分配
        ObjectRef new_device_ref(new class Device(type, id));
        devices_[key] = new_device_ref; // 存储 ObjectRef，增加引用计数
        return new_device_ref;
    }

private:
    DeviceManager() = default; // 单例模式，私有构造函数
    // 禁用拷贝构造和拷贝赋值
    DeviceManager(const DeviceManager&) = delete;
    DeviceManager& operator=(const DeviceManager&) = delete;

    // 存储 Device 实例的 ObjectRef。
    // ObjectRef 会负责其持有的 Device 对象的生命周期管理。
    // unordered_map 的键是字符串，格式如 "0_0" (CPU, id=0)。
    std::unordered_map<std::string, ObjectRef> devices_;
};

/**
 * @brief 方便的工厂函数，用于获取 Device 的 ObjectRef
 * @param type 设备类型码
 * @param id 设备 ID
 * @return 对应的 Device 的 ObjectRef
 */
inline ObjectRef Device(DeviceTypeCode type, int id) {
    return DeviceManager::Global()->GetOrCreate(type, id);
}

} // namespace kxc
