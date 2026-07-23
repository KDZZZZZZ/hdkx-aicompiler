/*! \file src/runtime/device/cpu_device_api.cc
 * \brief 实现 cpu:0 的内存、复制、同步和设备属性后端。
 */

#include "kxc/runtime/device_api.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <thread>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <malloc.h>
#include <windows.h>
#ifdef CreateEvent
#undef CreateEvent
#endif
#else
#include <unistd.h>
#endif

namespace kxc {
namespace {

// 限定 CPU 后端只处理当前唯一建模的 cpu:0。
void ValidateCPU(const Device& device) {
    if (device.device_type() != kCPU || device.device_id() != 0) {
        throw std::invalid_argument("CPU backend only accepts cpu:0");
    }
}

// 从编译器目标宏推断稳定的主机架构名称。
std::string DetectHostCPUArch() {
#if defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
    return "aarch64";
#elif defined(__arm__) || defined(_M_ARM)
    return "arm";
#elif defined(__i386__) || defined(_M_IX86)
    return "x86";
#else
    return "unknown";
#endif
}

// 使用平台 API 填充主机物理内存总量与当前可用量。
void FillHostMemoryInfo(DeviceAttributes* attrs) {
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        attrs->total_global_memory = static_cast<int64_t>(status.ullTotalPhys);
        attrs->available_global_memory = static_cast<int64_t>(status.ullAvailPhys);
    }
#elif defined(_SC_PHYS_PAGES) && defined(_SC_AVPHYS_PAGES) && defined(_SC_PAGESIZE)
    const long page_size = sysconf(_SC_PAGESIZE);
    const long total_pages = sysconf(_SC_PHYS_PAGES);
    const long available_pages = sysconf(_SC_AVPHYS_PAGES);
    if (page_size > 0 && total_pages > 0) {
        attrs->total_global_memory = static_cast<int64_t>(page_size) * total_pages;
    }
    if (page_size > 0 && available_pages > 0) {
        attrs->available_global_memory = static_cast<int64_t>(page_size) * available_pages;
    }
#endif
}

// 实现 cpu:0 的内存、复制及统一 stream/event 空句柄语义。
class CPUDeviceAPI final : public DeviceAPI {
public:
    // 验证当前请求确实属于 CPU 后端。
    void SetDevice(const Device& device) override { ValidateCPU(device); }

    // 使用平台对齐分配器申请主机内存。
    void* AllocDataSpace(const Device& device, size_t nbytes,
                         size_t alignment) override {
        ValidateCPU(device);
        if (nbytes == 0) return nullptr;
        if (alignment != 0 && (alignment & (alignment - 1)) != 0) {
            throw std::invalid_argument("alignment must be zero or a power of two");
        }
        // Windows 和 POSIX 对齐分配都要求至少满足指针及平台最大基础对齐。
        alignment = std::max(alignment == 0 ? alignof(std::max_align_t) : alignment,
                             alignof(void*));
#if defined(_WIN32)
        void* ptr = _aligned_malloc(nbytes, alignment);
#else
        void* ptr = nullptr;
        const int error = posix_memalign(&ptr, alignment, nbytes);
        if (error != 0) ptr = nullptr;
#endif
        if (!ptr) throw std::bad_alloc();
        return ptr;
    }

    // 使用与分配路径配对的平台 API 释放主机内存。
    void FreeDataSpace(const Device& device, void* ptr) override {
        ValidateCPU(device);
        if (!ptr) return;
#if defined(_WIN32)
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
    }

    // 清零指定主机字节区间。
    void ZeroData(const Device& device, void* ptr, size_t offset,
                  size_t nbytes) override {
        ValidateCPU(device);
        if (nbytes == 0) return;
        std::memset(static_cast<unsigned char*>(ptr) + offset, 0, nbytes);
    }

    // 同步复制主机区间，并为重叠视图提供 memmove 语义。
    void CopyDataSync(const Device& from, const void* from_ptr,
                      size_t from_offset, const Device& to, void* to_ptr,
                      size_t to_offset, size_t nbytes) override {
        ValidateCPU(from);
        ValidateCPU(to);
        if (nbytes == 0) return;
        // 同一 Storage 的视图可能重叠，CPU 路径必须提供 memmove 语义。
        std::memmove(static_cast<unsigned char*>(to_ptr) + to_offset,
                     static_cast<const unsigned char*>(from_ptr) + from_offset,
                     nbytes);
    }

    // CPU 没有独立队列，因此立即同步复制并要求空 stream。
    void CopyDataAsync(const Device& from, const void* from_ptr,
                       size_t from_offset, const Device& to, void* to_ptr,
                       size_t to_offset, size_t nbytes,
                       StreamHandle stream) override {
        if (stream != nullptr) throw std::invalid_argument("CPU stream handle must be null");
        CopyDataSync(from, from_ptr, from_offset, to, to_ptr, to_offset, nbytes);
    }

    // CPU 操作同步完成；设备级同步只需校验设备身份。
    void DeviceSync(const Device& device) override { ValidateCPU(device); }

    // CPU 工作同步完成，创建 stream 返回代表完成态的空句柄。
    StreamHandle CreateStream(const Device& device) override {
        ValidateCPU(device);
        return nullptr;
    }
    // 验证并释放 CPU 空 stream；没有后端资源需要回收。
    void FreeStream(const Device& device, StreamHandle stream) override {
        ValidateCPU(device);
        if (stream) throw std::invalid_argument("CPU stream handle must be null");
    }
    // CPU 空 stream 始终已同步。
    void StreamSync(const Device& device, StreamHandle stream) override {
        ValidateCPU(device);
        if (stream) throw std::invalid_argument("CPU stream handle must be null");
    }
    // CPU 操作已完成，创建 event 返回完成态空句柄。
    EventHandle CreateEvent(const Device& device) override {
        ValidateCPU(device);
        return nullptr;
    }
    // 验证 CPU event/stream 均为空，记录操作本身无需工作。
    void RecordEvent(const Device& device, EventHandle event,
                     StreamHandle stream) override {
        ValidateCPU(device);
        if (event || stream) throw std::invalid_argument("CPU event/stream must be null");
    }
    // CPU 空 event 始终处于完成状态。
    bool QueryEvent(const Device& device, EventHandle event) override {
        ValidateCPU(device);
        if (event) throw std::invalid_argument("CPU event must be null");
        return true;
    }
    // CPU 空 event 无需等待，仅检查句柄契约。
    void WaitEvent(const Device& device, EventHandle event) override {
        ValidateCPU(device);
        if (event) throw std::invalid_argument("CPU event must be null");
    }
    // CPU 空 event 没有后端资源，仅检查句柄契约。
    void FreeEvent(const Device& device, EventHandle event) override {
        ValidateCPU(device);
        if (event) throw std::invalid_argument("CPU event must be null");
    }

    // 将主机线程数、架构和内存信息投影到统一设备属性结构。
    DeviceAttributes GetDeviceAttributes(const Device& device) override {
        ValidateCPU(device);
        DeviceAttributes attrs;
        attrs.exists = 1;
        const unsigned int threads = std::thread::hardware_concurrency();
        // 这些统一字段是主机能力近似值，不表示 CPU 具有 CUDA block/warp 语义。
        attrs.max_threads_per_block = threads == 0 ? 1 : threads;
        attrs.max_threads_per_multiprocessor = attrs.max_threads_per_block;
        attrs.multi_processor_count = attrs.max_threads_per_block;
        attrs.warp_size = 1;
        attrs.arch = DetectHostCPUArch();
        attrs.device_name = "cpu/" + attrs.arch + "/" +
                            std::to_string(attrs.max_threads_per_block) + "-threads";
        FillHostMemoryInfo(&attrs);
        return attrs;
    }

    // 返回 CPU 代码生成所使用的 target kind。
    std::string GetTargetKind(const Device& device) const override {
        ValidateCPU(device);
        return "llvm";
    }
};

}  // namespace

// 返回进程级 CPU 后端单例。
DeviceAPI* GetCPUDeviceAPI() {
    static CPUDeviceAPI api;
    return &api;
}

}  // namespace kxc
