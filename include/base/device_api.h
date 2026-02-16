#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "device.h"

namespace kxc {

class RetValue;
using StreamHandle = void*;

enum class DeviceAttrKind : int {
    kExist = 0,
    kMaxThreadsPerBlock = 1,
    kWarpSize = 2,
    kMaxSharedMemoryPerBlock = 3,
    kComputeVersion = 4,
    kDeviceName = 5,
    kMaxClockRate = 6,
    kMultiProcessorCount = 7,
    kMaxRegistersPerBlock = 8,
    kApiVersion = 9,
    kDriverVersion = 10,
    kL2CacheSizeBytes = 11,
    kTotalGlobalMemory = 12,
    kAvailableGlobalMemory = 13,
    kMaxSharedMemoryPerMultiprocessor = 14,
    kMaxRegistersPerMultiprocessor = 15,
    kMaxThreadsPerMultiprocessor = 16,
    kComputeVersionMajor = 17,
    kComputeVersionMinor = 18,
};

struct DeviceAttributes {
    int exists{0};
    int64_t max_threads_per_block{1};
    int64_t warp_size{1};
    int64_t max_shared_memory_per_block{0};
    std::string compute_version{"0.0"};
    std::string device_name;
    int64_t max_clock_rate_khz{0};
    int64_t max_registers_per_block{0};
    int64_t api_version{0};
    int64_t driver_version{0};
    int64_t l2_cache_size_bytes{0};
    int64_t total_global_memory{0};
    int64_t available_global_memory{0};
    int64_t max_shared_memory_per_multiprocessor{0};
    int64_t max_registers_per_multiprocessor{0};
    int64_t max_threads_per_multiprocessor{0};
    int64_t compute_version_major{0};
    int64_t compute_version_minor{0};
    int64_t multi_processor_count{1};
    std::string arch;
};

class DeviceAPI {
public:
    virtual ~DeviceAPI() = default;

    virtual void SetDevice(const class Device& device) = 0;
    virtual void* AllocDataSpace(const class Device& device, size_t nbytes, size_t alignment) = 0;
    virtual void FreeDataSpace(const class Device& device, void* ptr) = 0;
    virtual void CopyDataFromTo(const class Device& from_dev, const void* from_ptr,
                                const class Device& to_dev, void* to_ptr,
                                size_t nbytes) = 0;
    virtual void CopyDataFromTo(const void* from_ptr, size_t from_offset,
                                void* to_ptr, size_t to_offset, size_t nbytes,
                                const class Device& from_dev, const class Device& to_dev,
                                StreamHandle stream);
    virtual DeviceAttributes GetDeviceAttributes(const class Device& device) = 0;
    virtual std::string GetTargetKind(const class Device& device) const = 0;
    virtual void GetAttr(const class Device& device, DeviceAttrKind kind, RetValue* rv);
    virtual StreamHandle CreateStream(const class Device& device);
    virtual void FreeStream(const class Device& device, StreamHandle stream);
    virtual void SetStream(const class Device& device, StreamHandle stream);
    virtual StreamHandle GetCurrentStream(const class Device& device);
    virtual void StreamSync(const class Device& device, StreamHandle stream);
    virtual void SyncStreamFromTo(const class Device& device, StreamHandle src, StreamHandle dst);
    virtual void* AllocWorkspace(const class Device& device, size_t nbytes, size_t alignment);
    virtual void FreeWorkspace(const class Device& device, void* ptr);
    virtual bool SupportsDevicePointerArithmeticsOnHost(const class Device& device) const;

    static bool NeedSetDevice(DeviceTypeCode type);
};

DeviceAPI* GetCPUDeviceAPI();
DeviceAPI* GetCUDADeviceAPI();
DeviceAPI* GetMetalDeviceAPI();
DeviceAPI* GetOpenCLDeviceAPI();

class DeviceAPIManager {
public:
    static DeviceAPIManager* Global();
    DeviceAPI* GetAPI(DeviceTypeCode type);

private:
    DeviceAPIManager() = default;
    std::vector<DeviceAPI*> apis_;
};

DeviceAPI* GetDeviceAPI(DeviceTypeCode type);

DeviceAttributes CollectDeviceAttributes(const class Device& device);
void GetDeviceAttr(const class Device& device, DeviceAttrKind kind, RetValue* rv);
int64_t GetDeviceAttr(const class Device& device, DeviceAttrKind kind);

}  // namespace kxc
