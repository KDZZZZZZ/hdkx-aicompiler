#pragma once
#include "device.h"
#include <string>

namespace kxc {

/**
 * @brief DeviceAPI 是一个纯虚基类，定义了与特定设备交互的接口。
 * 每个后端（CPU、CUDA、Metal等）都应该有一个具体的实现。
 */
class DeviceAPI {
public:
    virtual ~DeviceAPI() = default;

    /**
     * @brief 设置当前线程工作的设备。
     * @param device 要激活的设备。
     */
    virtual void SetDevice(const class Device& device) = 0;

    /**
     * @brief 在指定的设备上分配一块内存。
     * @param device 分配内存的目标设备。
     * @param nbytes 分配的字节数。
     * @param alignment 内存对齐要求。
     * @return 指向分配内存的指针。
     */
    virtual void* AllocDataSpace(const class Device& device, size_t nbytes, size_t alignment) = 0;

    /**
     * @brief 释放指定设备上的内存。
     * @param device 内存所在的设备。
     * @param ptr 要释放的内存指针。
     */
    virtual void FreeDataSpace(const class Device& device, void* ptr) = 0;

    /**
     * @brief 在不同设备间拷贝数据。
     * @param from_dev 源设备。
     * @param from_ptr 源数据指针。
     * @param to_dev 目标设备。
     * @param to_ptr 目标数据指针。
     * @param nbytes 拷贝的字节数。
     */
    virtual void CopyDataFromTo(const class Device& from_dev, const void* from_ptr,
                                const class Device& to_dev, void* to_ptr,
                                size_t nbytes) = 0;
};
extern DeviceAPI* GetCPUDeviceAPI();
extern DeviceAPI* GetCUDADeviceAPI();
extern DeviceAPI* GetMetalDeviceAPI();
extern DeviceAPI* GetOpenCLDeviceAPI();
class DeviceAPIManager {
public:
    static DeviceAPIManager* Global() {
        static DeviceAPIManager instance;
        return &instance;
    }
    DeviceAPI* GetAPI(DeviceTypeCode type){
        if (apis_.size() <= static_cast<size_t>(type)) {
            apis_.resize(static_cast<size_t>(type) + 1, nullptr);
        }
        if (apis_[type] == nullptr) {
            if (type == kCPU) {
                apis_[type] = GetCPUDeviceAPI();
            } else if (type == kGPU) {
                apis_[type] = GetCUDADeviceAPI();
            } else {
                throw std::runtime_error("Device API not supported");
            }
        }
        return apis_[type];
    }
private:
    std::vector<DeviceAPI*> apis_;
    DeviceAPIManager() = default;
};
inline DeviceAPI* GetDeviceAPI(DeviceTypeCode type) {
    return DeviceAPIManager::Global()->GetAPI(type);
}
} // namespace kxc
