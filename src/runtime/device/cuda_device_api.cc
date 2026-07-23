/*! \file src/runtime/device/cuda_device_api.cc
 * \brief 实现 CUDA 内存、复制、stream/event 和设备能力查询后端。
 */

#include "kxc/runtime/device_api.h"

#include <stdexcept>
#include <string>

#if KXC_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace kxc {
namespace {

#if KXC_USE_CUDA

// 将 CUDA 错误码转换为带操作名的 C++ 异常。
void CheckCUDA(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(error));
    }
}

// 验证请求具有合法的 CUDA 物理设备身份。
void ValidateCUDA(const Device& device) {
    if (device.device_type() != kCUDA || device.device_id() < 0) {
        throw std::invalid_argument("CUDA backend requires cuda:N");
    }
}

// 根据源和目标设备选择 CUDA Runtime 复制方向。
cudaMemcpyKind CopyKind(const Device& from, const Device& to) {
    if (from.device_type() == kCPU && to.device_type() == kCUDA) {
        return cudaMemcpyHostToDevice;
    }
    if (from.device_type() == kCUDA && to.device_type() == kCPU) {
        return cudaMemcpyDeviceToHost;
    }
    if (from.device_type() == kCUDA && to.device_type() == kCUDA) {
        if (from.device_id() != to.device_id()) {
            throw std::invalid_argument("cross-device CUDA copy is not supported");
        }
        return cudaMemcpyDeviceToDevice;
    }
    throw std::invalid_argument("invalid CUDA copy direction");
}

// 选择执行本次复制所需的 CUDA 当前设备上下文。
int CopyDeviceId(const Device& from, const Device& to) {
    // H2D 在目标卡上下文执行，D2H/D2D 在源卡或同一张卡上下文执行。
    return to.device_type() == kCUDA ? to.device_id() : from.device_id();
}

// 实现 CUDA 设备内存、复制、非阻塞 stream 和完成 event。
class CUDADeviceAPI final : public DeviceAPI {
public:
    // 校验并切换当前线程的 CUDA 设备上下文。
    void SetDevice(const Device& device) override {
        ValidateCUDA(device);
        CheckCUDA(cudaSetDevice(device.device_id()), "cudaSetDevice");
    }

    // 在指定 CUDA 设备上分配满足 CUDA 基础对齐保证的内存。
    void* AllocDataSpace(const Device& device, size_t nbytes,
                         size_t alignment) override {
        ValidateCUDA(device);
        if (nbytes == 0) return nullptr;
        if (alignment != 0 && (alignment & (alignment - 1)) != 0) {
            throw std::invalid_argument("alignment must be zero or a power of two");
        }
        // cudaMalloc 保证至少 256 字节对齐，更高对齐需单独的分配策略。
        if (alignment > 256) {
            throw std::invalid_argument("CUDA allocation alignment above 256 is unsupported");
        }
        SetDevice(device);
        void* ptr = nullptr;
        CheckCUDA(cudaMalloc(&ptr, nbytes), "cudaMalloc");
        return ptr;
    }

    // 切换到所有者设备后释放 CUDA 地址。
    void FreeDataSpace(const Device& device, void* ptr) override {
        ValidateCUDA(device);
        if (!ptr) return;
        SetDevice(device);
        CheckCUDA(cudaFree(ptr), "cudaFree");
    }

    // 在指定设备上同步清零一段 CUDA 内存。
    void ZeroData(const Device& device, void* ptr, size_t offset,
                  size_t nbytes) override {
        if (nbytes == 0) return;
        SetDevice(device);
        CheckCUDA(cudaMemset(static_cast<unsigned char*>(ptr) + offset, 0, nbytes),
                  "cudaMemset");
    }

    // 按 H2D、D2H 或同设备 D2D 方向执行同步复制。
    void CopyDataSync(const Device& from, const void* from_ptr,
                      size_t from_offset, const Device& to, void* to_ptr,
                      size_t to_offset, size_t nbytes) override {
        if (nbytes == 0) return;
        CheckCUDA(cudaSetDevice(CopyDeviceId(from, to)), "cudaSetDevice(copy)");
        const auto* src = static_cast<const unsigned char*>(from_ptr) + from_offset;
        auto* dst = static_cast<unsigned char*>(to_ptr) + to_offset;
        CheckCUDA(cudaMemcpy(dst, src, nbytes, CopyKind(from, to)), "cudaMemcpy");
    }

    // 将复制提交到给定 CUDA stream；空句柄表示 CUDA 默认流。
    void CopyDataAsync(const Device& from, const void* from_ptr,
                       size_t from_offset, const Device& to, void* to_ptr,
                       size_t to_offset, size_t nbytes,
                       StreamHandle stream) override {
        if (nbytes == 0) return;
        CheckCUDA(cudaSetDevice(CopyDeviceId(from, to)), "cudaSetDevice(copy async)");
        const auto* src = static_cast<const unsigned char*>(from_ptr) + from_offset;
        auto* dst = static_cast<unsigned char*>(to_ptr) + to_offset;
        CheckCUDA(cudaMemcpyAsync(dst, src, nbytes, CopyKind(from, to),
                                  reinterpret_cast<cudaStream_t>(stream)),
                  "cudaMemcpyAsync");
    }

    // 等待包括非阻塞 stream 在内的设备全部既有工作完成。
    void DeviceSync(const Device& device) override {
        SetDevice(device);
        CheckCUDA(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }

    // 创建不与默认流隐式同步的非阻塞 CUDA stream，并由 DeviceStream 独占销毁。
    StreamHandle CreateStream(const Device& device) override {
        SetDevice(device);
        cudaStream_t stream = nullptr;
        CheckCUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                  "cudaStreamCreateWithFlags");
        return reinterpret_cast<void*>(stream);
    }

    // 在所属设备上下文中销毁显式 CUDA stream。
    void FreeStream(const Device& device, StreamHandle stream) override {
        ValidateCUDA(device);
        if (!stream) return;
        SetDevice(device);
        CheckCUDA(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream)),
                  "cudaStreamDestroy");
    }

    // 等待指定 stream 中此前提交的工作完成。
    void StreamSync(const Device& device, StreamHandle stream) override {
        SetDevice(device);
        CheckCUDA(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)),
                  "cudaStreamSynchronize");
    }

    // 创建仅用于完成检测、关闭计时功能的 CUDA event。
    EventHandle CreateEvent(const Device& device) override {
        SetDevice(device);
        cudaEvent_t event = nullptr;
        // event 只承担完成栅栏和生命周期跟踪，不采集计时数据。
        CheckCUDA(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
                  "cudaEventCreateWithFlags");
        return reinterpret_cast<void*>(event);
    }

    // 将 event 记录到 stream 尾部，用作异步资源释放栅栏。
    void RecordEvent(const Device& device, EventHandle event,
                     StreamHandle stream) override {
        SetDevice(device);
        CheckCUDA(cudaEventRecord(reinterpret_cast<cudaEvent_t>(event),
                                  reinterpret_cast<cudaStream_t>(stream)),
                  "cudaEventRecord");
    }

    // 非阻塞查询 event 是否完成，并保留真实后端错误。
    bool QueryEvent(const Device& device, EventHandle event) override {
        SetDevice(device);
        cudaError_t result = cudaEventQuery(reinterpret_cast<cudaEvent_t>(event));
        if (result == cudaErrorNotReady) return false;
        CheckCUDA(result, "cudaEventQuery");
        return true;
    }

    // 阻塞等待 event 对应的 stream 工作完成。
    void WaitEvent(const Device& device, EventHandle event) override {
        SetDevice(device);
        CheckCUDA(cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(event)),
                  "cudaEventSynchronize");
    }

    // 在所属设备上下文中销毁完成 event。
    void FreeEvent(const Device& device, EventHandle event) override {
        ValidateCUDA(device);
        if (!event) return;
        SetDevice(device);
        CheckCUDA(cudaEventDestroy(reinterpret_cast<cudaEvent_t>(event)),
                  "cudaEventDestroy");
    }

    // 查询 CUDA 设备存在性、体系结构、执行上限及当前内存状态。
    DeviceAttributes GetDeviceAttributes(const Device& device) override {
        ValidateCUDA(device);
        DeviceAttributes attrs;
        int count = 0;
        cudaError_t count_error = cudaGetDeviceCount(&count);
        if (count_error != cudaSuccess || device.device_id() >= count) {
            attrs.device_name = count_error == cudaSuccess
                                    ? "CUDA device does not exist"
                                    : cudaGetErrorString(count_error);
            return attrs;
        }
        SetDevice(device);
        cudaDeviceProp prop{};
        CheckCUDA(cudaGetDeviceProperties(&prop, device.device_id()),
                  "cudaGetDeviceProperties");
        attrs.exists = 1;
        attrs.max_threads_per_block = prop.maxThreadsPerBlock;
        attrs.warp_size = prop.warpSize;
        attrs.max_shared_memory_per_block = prop.sharedMemPerBlock;
        attrs.compute_version_major = prop.major;
        attrs.compute_version_minor = prop.minor;
        attrs.compute_version = std::to_string(prop.major) + "." + std::to_string(prop.minor);
        attrs.arch = "sm_" + std::to_string(prop.major) + std::to_string(prop.minor);
        attrs.device_name = prop.name;
        attrs.max_clock_rate_khz = prop.clockRate;
        attrs.multi_processor_count = prop.multiProcessorCount;
        attrs.max_registers_per_block = prop.regsPerBlock;
        attrs.l2_cache_size_bytes = prop.l2CacheSize;
        attrs.total_global_memory = static_cast<int64_t>(prop.totalGlobalMem);
        attrs.max_shared_memory_per_multiprocessor = prop.sharedMemPerMultiprocessor;
        attrs.max_registers_per_multiprocessor = prop.regsPerMultiprocessor;
        attrs.max_threads_per_multiprocessor = prop.maxThreadsPerMultiProcessor;
        int runtime_version = 0;
        int driver_version = 0;
        CheckCUDA(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
        CheckCUDA(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
        attrs.api_version = runtime_version;
        attrs.driver_version = driver_version;
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        CheckCUDA(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
        attrs.available_global_memory = static_cast<int64_t>(free_bytes);
        attrs.total_global_memory = static_cast<int64_t>(total_bytes);
        return attrs;
    }

    // 返回 CUDA 代码生成对应的 target kind。
    std::string GetTargetKind(const Device& device) const override {
        ValidateCUDA(device);
        return "cuda";
    }
};

#else

// CPU-only 构建保留可诊断的占位后端，所有执行操作明确失败且不回退 CPU。
class CUDADeviceAPI final : public DeviceAPI {
    // 为所有不可执行操作生成一致的禁用诊断。
    [[noreturn]] static void Disabled() {
        throw std::runtime_error("CUDA support is disabled (KXC_USE_CUDA=0)");
    }

public:
    // 禁止切换未编译的 CUDA 后端。
    void SetDevice(const Device&) override { Disabled(); }
    // 禁止在未编译的 CUDA 后端分配内存。
    void* AllocDataSpace(const Device&, size_t, size_t) override { Disabled(); }
    // 禁止在未编译的 CUDA 后端释放内存。
    void FreeDataSpace(const Device&, void*) override { Disabled(); }
    // 禁止在未编译的 CUDA 后端清零内存。
    void ZeroData(const Device&, void*, size_t, size_t) override { Disabled(); }
    // 禁止通过未编译的 CUDA 后端同步复制。
    void CopyDataSync(const Device&, const void*, size_t, const Device&, void*,
                      size_t, size_t) override { Disabled(); }
    // 禁止通过未编译的 CUDA 后端异步复制。
    void CopyDataAsync(const Device&, const void*, size_t, const Device&, void*,
                       size_t, size_t, StreamHandle) override { Disabled(); }
    // 禁止同步未编译的 CUDA 后端。
    void DeviceSync(const Device&) override { Disabled(); }
    // 禁止创建 CUDA stream。
    StreamHandle CreateStream(const Device&) override { Disabled(); }
    // 禁止释放不存在的 CUDA stream。
    void FreeStream(const Device&, StreamHandle) override { Disabled(); }
    // 禁止同步不存在的 CUDA stream。
    void StreamSync(const Device&, StreamHandle) override { Disabled(); }
    // 禁止创建 CUDA event。
    EventHandle CreateEvent(const Device&) override { Disabled(); }
    // 禁止记录 CUDA event。
    void RecordEvent(const Device&, EventHandle, StreamHandle) override { Disabled(); }
    // 禁止查询 CUDA event。
    bool QueryEvent(const Device&, EventHandle) override { Disabled(); }
    // 禁止等待 CUDA event。
    void WaitEvent(const Device&, EventHandle) override { Disabled(); }
    // 禁止释放 CUDA event。
    void FreeEvent(const Device&, EventHandle) override { Disabled(); }
    // 返回 exists=0 的诊断属性，使枚举接口仍可解释禁用原因。
    DeviceAttributes GetDeviceAttributes(const Device&) override {
        DeviceAttributes attrs;
        attrs.device_name = "CUDA support is disabled (KXC_USE_CUDA=0)";
        return attrs;
    }
    // 保留 cuda target 名称用于诊断，不表示后端可执行。
    std::string GetTargetKind(const Device&) const override { return "cuda"; }
};

#endif

}  // namespace

// 返回编译配置对应的进程级 CUDA 后端单例。
DeviceAPI* GetCUDADeviceAPI() {
    static CUDADeviceAPI api;
    return &api;
}

}  // namespace kxc
