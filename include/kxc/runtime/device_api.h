/*! \file include/kxc/runtime/device_api.h
 * \brief 定义设备后端接口、公共路由函数和结构化设备信息。
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "kxc/runtime/device.h"
#include "kxc/runtime/device_info.h"

namespace kxc {

/*! \brief PackedFunc 单项属性查询使用的返回值类型。 */
class RetValue;
/*! \brief 设备物理存储对象，完整定义位于 storage.h。 */
class Storage;
/*! \brief 设备执行流对象，完整定义位于 device_stream.h。 */
class DeviceStream;
/*! \brief 异步完成对象，完整定义位于 device_stream.h。 */
class AsyncOperation;

/*! \brief 仅在 DeviceAPI 后端边界内使用的非拥有型原生句柄。 */
using StreamHandle = void*;
using EventHandle = void*;

/*! \brief PackedFunc 兼容查询所使用的设备属性编号。 */
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

/*! \brief CPU/CUDA 后端必须实现的内存、复制和同步原语。 */
class DeviceAPI {
public:
    /*! \brief 允许通过后端接口安全销毁具体实现。 */
    virtual ~DeviceAPI() = default;

    /*! \brief 选择当前设备；CPU 后端用它校验只能访问 cpu:0。 */
    virtual void SetDevice(const Device& device) = 0;
    /*! \brief 分配和释放后端内存；零字节分配由公共路由直接处理。 */
    virtual void* AllocDataSpace(const Device& device, size_t nbytes,
                                 size_t alignment) = 0;
    /*! \brief 释放由 AllocDataSpace 在同一设备上返回的地址。 */
    virtual void FreeDataSpace(const Device& device, void* ptr) = 0;
    /*! \brief 从 ptr 的字节 offset 开始清零 nbytes。 */
    virtual void ZeroData(const Device& device, void* ptr, size_t offset,
                          size_t nbytes) = 0;
    /*! \brief 同步复制；两个 offset 和 nbytes 的单位均为字节。 */
    virtual void CopyDataSync(const Device& from_device, const void* from,
                              size_t from_offset, const Device& to_device,
                              void* to, size_t to_offset, size_t nbytes) = 0;
    /*! \brief 将复制排入 stream；完成状态由调用层用 event 跟踪。 */
    virtual void CopyDataAsync(const Device& from_device, const void* from,
                               size_t from_offset, const Device& to_device,
                               void* to, size_t to_offset, size_t nbytes,
                               StreamHandle stream) = 0;

    /*! \brief 等待设备上所有已提交工作完成；CPU 后端为校验后的无操作。 */
    virtual void DeviceSync(const Device& device) = 0;
    /*! \brief 创建、销毁和同步后端 stream；空句柄表示默认 stream。 */
    virtual StreamHandle CreateStream(const Device& device) = 0;
    /*! \brief 释放 CreateStream 创建的非默认 stream。 */
    virtual void FreeStream(const Device& device, StreamHandle stream) = 0;
    /*! \brief 等待 stream 中此前提交的任务完成。 */
    virtual void StreamSync(const Device& device, StreamHandle stream) = 0;
    /*! \brief event 由创建它的后端负责记录、查询、等待和销毁。 */
    virtual EventHandle CreateEvent(const Device& device) = 0;
    /*! \brief 在 stream 中记录 event；空 stream 表示默认 stream。 */
    virtual void RecordEvent(const Device& device, EventHandle event,
                             StreamHandle stream) = 0;
    /*! \brief 非阻塞查询 event 是否已经完成。 */
    virtual bool QueryEvent(const Device& device, EventHandle event) = 0;
    /*! \brief 阻塞等待 event 完成。 */
    virtual void WaitEvent(const Device& device, EventHandle event) = 0;
    /*! \brief 释放由 CreateEvent 创建的 event。 */
    virtual void FreeEvent(const Device& device, EventHandle event) = 0;

    /*! \brief 查询设备能力及其编译目标类型。 */
    virtual DeviceAttributes GetDeviceAttributes(const Device& device) = 0;
    /*! \brief 返回用于编译 Target 的后端 kind 名称。 */
    virtual std::string GetTargetKind(const Device& device) const = 0;
    /*! \brief 将指定属性写入 PackedFunc 返回值。 */
    virtual void GetAttr(const Device& device, DeviceAttrKind kind, RetValue* rv);
};

/*! \brief 返回进程级 CPU 后端实现。 */
DeviceAPI* GetCPUDeviceAPI();
/*! \brief 返回 CUDA 后端实现；未启用 CUDA 时操作会显式失败。 */
DeviceAPI* GetCUDADeviceAPI();

/*! \brief 按设备类型缓存进程级后端实例。 */
class DeviceAPIManager {
public:
    /*! \brief 返回进程级后端注册表。 */
    static DeviceAPIManager* Global();
    /*! \brief 按设备类型取得后端实现；不支持的类型会抛出异常。 */
    DeviceAPI* GetAPI(DeviceTypeCode type);

private:
    std::mutex mutex_;
    std::unordered_map<int, DeviceAPI*> apis_;
};

/*! \brief 公共强类型路由；调用方不直接持有后端实现。 */
DeviceAPI* GetDeviceAPI(DeviceTypeCode type);
/*! \brief 分配设备内存；alignment 为 0 或二的幂，零字节返回 nullptr。 */
void* DeviceAlloc(const Device& device, size_t nbytes, size_t alignment = 0);
/*! \brief 释放 DeviceAlloc 返回的内存；空指针是无操作。 */
void DeviceFree(const Device& device, void* ptr);
/*! \brief 对设备内存中的字节区间执行清零。 */
void DeviceZero(const Device& device, void* ptr, size_t offset, size_t nbytes);
/*! \brief 根据源和目标 Device 自动选择 H2H、H2D、D2H 或 D2D 路由。 */
void DeviceCopySync(const Device& from_device, const void* from, size_t from_offset,
                    const Device& to_device, void* to, size_t to_offset,
                    size_t nbytes);
/*! \brief 等待指定设备上所有先前提交的异步工作完成。 */
void DeviceSynchronize(const Device& device);

/*! \brief 设备能力查询以及面向工具的枚举/JSON 接口。 */
DeviceAttributes CollectDeviceAttributes(const Device& device);
/*! \brief 以 PackedFunc 返回值形式查询单项设备属性。 */
void GetDeviceAttr(const Device& device, DeviceAttrKind kind, RetValue* rv);
/*! \brief 查询可表示为整数的单项设备属性。 */
int64_t GetDeviceAttr(const Device& device, DeviceAttrKind kind);
/*! \brief 枚举当前实际可用的物理设备。 */
std::vector<Device> ListDevices();
/*! \brief 返回已知后端的结构化设备信息和不可用原因。 */
std::vector<DeviceInfo> GetAllDeviceInfo();
/*! \brief 将可用设备列表序列化为 JSON。 */
std::string ListDevicesJSON();
/*! \brief 将完整设备信息序列化为 JSON。 */
std::string GetAllDeviceInfoJSON();

}  // namespace kxc
