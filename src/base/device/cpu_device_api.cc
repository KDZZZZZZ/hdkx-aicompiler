#pragma once
#include "base/device_api.h"
#include <cstdlib>
#include <cstring>
namespace kxc{
class CPUDeviceAPI : public DeviceAPI{
public:
    void SetDevice(const class Device& device) override {
        // CPU 只有一个设备，不需要切换
    }

    void* AllocDataSpace(const class Device& device, size_t nbytes, size_t alignment) override {
        void* ptr = nullptr;
#if defined(_MSC_VER)
        ptr = _aligned_malloc(nbytes, alignment);
#else
        if (posix_memalign(&ptr, alignment, nbytes) != 0) {
            ptr = nullptr;
        }
#endif
        return ptr;
    }

    void FreeDataSpace(const class Device& device, void* ptr) override {
#if defined(_MSC_VER)
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }

    void CopyDataFromTo(const class Device& from_dev, const void* from_ptr,
                        const class Device& to_dev, void* to_ptr,
                        size_t nbytes) override {
        if (from_dev.device_type() == kCPU && to_dev.device_type() == kCPU) {
            std::memcpy(to_ptr, from_ptr, nbytes);
        } else {
            // TODO: 支持 CPU <-> 其他设备的拷贝
            throw std::runtime_error("Copy between CPU and other devices not implemented yet");
        }
    }
};

DeviceAPI* GetCPUDeviceAPI() {
    static CPUDeviceAPI inst;
    return &inst;
}

}
