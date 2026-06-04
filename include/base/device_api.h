/*! \file include/base/device_api.h
 * \brief 定义基础对象系统、容器、设备、NDArray、Target、PassContext 和 profiling 公共类型。
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "device.h"

namespace kxc {

class RetValue;
using StreamHandle = void*;

/*! \brief DeviceAPI::GetAttr 支持查询的设备属性类型。 */
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

/*! \brief 后端设备能力快照，用于 target 推导、调度和诊断输出。 */
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

/*! \brief 设备后端抽象，统一 CPU/CUDA/Metal/OpenCL 的内存、拷贝和 stream 操作。 */
class DeviceAPI {
public:
    virtual ~DeviceAPI() = default;

    /*! \brief 切换当前线程或后端上下文到指定设备。 */
    virtual void SetDevice(const class Device& device) = 0;
    /*! \brief 在指定设备上分配数据空间。 */
    virtual void* AllocDataSpace(const class Device& device, size_t nbytes, size_t alignment) = 0;
    /*! \brief 释放指定设备上的数据空间。 */
    virtual void FreeDataSpace(const class Device& device, void* ptr) = 0;
    /*! \brief 在两个设备指针之间执行同步数据拷贝。 */
    virtual void CopyDataFromTo(const class Device& from_dev, const void* from_ptr,
                                const class Device& to_dev, void* to_ptr,
                                size_t nbytes) = 0;
    /*! \brief 在两个设备指针之间按 offset 和 stream 执行数据拷贝。 */
    virtual void CopyDataFromTo(const void* from_ptr, size_t from_offset,
                                void* to_ptr, size_t to_offset, size_t nbytes,
                                const class Device& from_dev, const class Device& to_dev,
                                StreamHandle stream);
    /*! \brief 查询设备能力属性快照。 */
    virtual DeviceAttributes GetDeviceAttributes(const class Device& device) = 0;
    /*! \brief 返回设备对应的 target kind 名称。 */
    virtual std::string GetTargetKind(const class Device& device) const = 0;
    /*! \brief 按 DeviceAttrKind 查询单个设备属性。 */
    virtual void GetAttr(const class Device& device, DeviceAttrKind kind, RetValue* rv);
    /*! \brief 创建后端 stream；不支持 stream 的后端可返回 nullptr。 */
    virtual StreamHandle CreateStream(const class Device& device);
    /*! \brief 释放后端 stream。 */
    virtual void FreeStream(const class Device& device, StreamHandle stream);
    /*! \brief 设置当前后端 stream。 */
    virtual void SetStream(const class Device& device, StreamHandle stream);
    /*! \brief 获取当前后端 stream。 */
    virtual StreamHandle GetCurrentStream(const class Device& device);
    /*! \brief 同步指定 stream。 */
    virtual void StreamSync(const class Device& device, StreamHandle stream);
    /*! \brief 建立两个 stream 之间的同步依赖。 */
    virtual void SyncStreamFromTo(const class Device& device, StreamHandle src, StreamHandle dst);
    /*! \brief 分配临时 workspace。 */
    virtual void* AllocWorkspace(const class Device& device, size_t nbytes, size_t alignment);
    /*! \brief 释放临时 workspace。 */
    virtual void FreeWorkspace(const class Device& device, void* ptr);
    /*! \brief 返回该后端是否允许 host 端对设备指针做地址算术。 */
    virtual bool SupportsDevicePointerArithmeticsOnHost(const class Device& device) const;

    /*! \brief 判断访问该设备类型前是否需要调用 SetDevice。 */
    static bool NeedSetDevice(DeviceTypeCode type);
};

DeviceAPI* GetCPUDeviceAPI();
DeviceAPI* GetCUDADeviceAPI();
DeviceAPI* GetMetalDeviceAPI();
DeviceAPI* GetOpenCLDeviceAPI();

/*! \brief DeviceAPI 全局注册表，按 DeviceTypeCode 返回对应后端实现。 */
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
