/*! \file include/kxc/runtime/device.h
 * \brief 定义不可变的物理设备身份、设备驻留表以及 DLPack 映射。
 */

#pragma once

#include <cstddef>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>

#include "kxc/support/container.h"
#include "kxc/support/object.h"
#include "dlpack/dlpack.h"

namespace kxc {

/*! \brief 运行时支持的物理设备类型；数值不直接等同于 DLPack 枚举。 */
enum DeviceTypeCode : int {
    kCPU = 0,
    kCUDA = 1,
    kOpenCL = 2,
    kMetal = 3,
    kUnknown = 99,
};

/*! \brief 保存设备类型和设备序号的不可变对象节点。 */
class DeviceNode final : public Object {
public:
    /*! \brief 物理设备的后端类型。 */
    DeviceTypeCode device_type{kUnknown};
    /*! \brief 该后端命名空间内的设备序号。 */
    int device_id{-1};

    KXC_OBJECT_DECLARE
};

/*! \brief 设备发现记录，完整定义位于 device_api.h。 */
struct DeviceInfo;


#ifdef Device
#undef Device
#endif
/*! \brief 具有值语义的物理设备句柄，例如 cpu:0 或 cuda:1。 */
class Device : public ObjectRef {
public:
    /*! \brief 构造未定义的设备句柄。 */
    Device() = default;
    /*! \brief 获取指定物理设备的规范驻留对象。 */
    Device(DeviceTypeCode type, int id);
    /*! \brief 从 ObjectRef 恢复 Device，并校验节点类型。 */
    explicit Device(const ObjectRef& ref);

    /*! \brief 获取指定编号的 CPU 设备；当前仅 cpu:0 可用。 */
    static Device CPU(int id = 0);
    /*! \brief 获取指定编号的 CUDA 设备；构造身份时不检查硬件可用性。 */
    static Device CUDA(int id = 0);

    /*! \brief 返回设备后端类型。 */
    DeviceTypeCode device_type() const;
    /*! \brief 返回后端命名空间内的设备序号。 */
    int device_id() const;
    /*! \brief 返回 cpu:0、cuda:1 形式的稳定字符串。 */
    std::string ToString() const;

    /*! \brief 按设备类型和序号比较两个设备身份。 */
    bool operator==(const Device& other) const;
    /*! \brief 按设备类型和序号判断两个设备身份不同。 */
    bool operator!=(const Device& other) const;

    /*! \brief 返回由当前 Device 共享持有的只读节点。 */
    const DeviceNode* operator->() const;
};

/*! \brief 线程安全地驻留 Device，并统一提供设备发现与信息查询。 */
class DeviceManager {
public:
    /*! \brief 返回进程级 DeviceManager 单例。 */
    static DeviceManager* Global();

    /*! \brief 获取规范 Device；相同 type/id 始终返回同一节点。 */
    Device Get(DeviceTypeCode type, int id);
    /*! \brief 枚举当前构建和硬件环境中实际可用的设备。 */
    Array<Device> ListAvailableDevices() const;
    /*! \brief 返回可用及不可用后端的结构化诊断信息。 */
    Array<DeviceInfo> GetAllDeviceInfo() const;

private:
    /*! \brief DeviceManager 驻留表使用的物理设备值键。 */
    struct Key {
        DeviceTypeCode type;
        int id;

        /*! \brief 按设备类型和序号比较两个驻留键。 */
        bool operator==(const Key& other) const {
            return type == other.type && id == other.id;
        }
    };

    /*! \brief 为驻留键组合设备类型与序号的哈希值。 */
    struct KeyHash {
        /*! \brief 计算适用于 unordered_map 的驻留键哈希。 */
        size_t operator()(const Key& key) const noexcept {
            const size_t type_hash = std::hash<int>{}(static_cast<int>(key.type));
            const size_t id_hash = std::hash<int>{}(key.id);
            return type_hash ^ (id_hash + static_cast<size_t>(0x9e3779b9) +
                                (type_hash << 6) + (type_hash >> 2));
        }
    };

    /*! \brief 仅允许 Global() 构造进程级管理器，禁止复制其驻留状态。 */
    DeviceManager() = default;
    DeviceManager(const DeviceManager&) = delete;
    DeviceManager& operator=(const DeviceManager&) = delete;

    std::mutex mutex_;
    std::unordered_map<Key, Device, KeyHash> devices_;
};

/*! \brief 在项目设备类型与 DLPack 设备类型之间做显式映射。 */
DLDeviceType ToDLDeviceType(DeviceTypeCode type);
/*! \brief 将受支持的 DLPack 设备类型映射为项目设备类型。 */
DeviceTypeCode FromDLDeviceType(DLDeviceType type);
/*! \brief 在 Device 与 DLDevice 之间转换，并保留 device_id。 */
DLDevice ToDLDevice(const Device& device);
/*! \brief 从 DLDevice 获取对应的规范驻留 Device。 */
Device FromDLDevice(const DLDevice& device);

}  // namespace kxc

namespace std {
/*! \brief 按设备类型和序号计算 Device 的值哈希。 */
template <>
struct hash<kxc::Device> {
    /*! \brief 计算 Device 的值哈希；未定义句柄稳定映射为零。 */
    size_t operator()(const kxc::Device& device) const noexcept {
        if (!device.defined()) return 0;
        const size_t type_hash =
            std::hash<int>{}(static_cast<int>(device.device_type()));
        const size_t id_hash = std::hash<int>{}(device.device_id());
        return type_hash ^ (id_hash + static_cast<size_t>(0x9e3779b9) +
                            (type_hash << 6) + (type_hash >> 2));
    }
};
}  // namespace std
