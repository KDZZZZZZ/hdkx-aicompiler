/*! \file src/base/device/cuda_device_api.cc
 * \brief 实现具体 CPU/CUDA DeviceAPI 后端。
 */

#include "base/device_api.h"

#include <cstdint>
#include <stdexcept>
#include <string>

#include "base/profiling.h"

#if KXC_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace kxc {

#if KXC_USE_CUDA

namespace {

thread_local cudaStream_t tls_cuda_stream = nullptr;

std::string DeviceTag(const class Device& device) {
    return "cuda:" + std::to_string(device.device_id());
}

profiling::EventSpec MakeDeviceSpec(const std::string& event_type, const class Device& device) {
    profiling::EventSpec spec;
    spec.component = "device_api";
    spec.event_type = event_type;
    spec.device = DeviceTag(device);
    return spec;
}

cudaMemcpyKind ResolveCopyKind(const class Device& from_dev, const class Device& to_dev) {
    if (from_dev.device_type() == kCPU && to_dev.device_type() == kGPU) {
        return cudaMemcpyHostToDevice;
    }
    if (from_dev.device_type() == kGPU && to_dev.device_type() == kCPU) {
        return cudaMemcpyDeviceToHost;
    }
    if (from_dev.device_type() == kGPU && to_dev.device_type() == kGPU) {
        return cudaMemcpyDeviceToDevice;
    }
    throw std::runtime_error("Unsupported CUDA copy direction");
}

int64_t ReadAvailableMemory(int device_id) {
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        return 0;
    }
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) {
        return 0;
    }
    return static_cast<int64_t>(free_bytes);
}

DeviceAttributes UnavailableCUDAAttributes(const std::string& reason) {
    DeviceAttributes attrs;
    attrs.exists = 0;
    attrs.device_name = reason;
    attrs.arch = "";
    attrs.compute_version = "0.0";
    return attrs;
}

}  // namespace

class CUDADeviceAPI : public DeviceAPI {
public:
    void SetDevice(const class Device& device) override {
        profiling::ScopedSpan span(profiling::CurrentContext(),
                                   MakeDeviceSpec("set_device", device));
        cudaError_t err = cudaSetDevice(device.device_id());
        if (err != cudaSuccess) {
            span.SetStatus("error");
            span.SetMessage(cudaGetErrorString(err));
            throw std::runtime_error("CUDA SetDevice failed: " +
                                     std::string(cudaGetErrorString(err)));
        }
    }

    void* AllocDataSpace(const class Device& device, size_t nbytes, size_t alignment) override {
        profiling::ScopedSpan span(profiling::CurrentContext(),
                                   MakeDeviceSpec("alloc", device));
        span.AddMetric("bytes", static_cast<double>(nbytes));
        span.AddMetric("alignment", static_cast<double>(alignment));
        (void)alignment;
        void* ptr = nullptr;
        cudaSetDevice(device.device_id());
        cudaError_t err = cudaMalloc(&ptr, nbytes);
        if (err != cudaSuccess) {
            span.SetStatus("error");
            span.SetMessage(cudaGetErrorString(err));
            throw std::runtime_error("CUDA AllocDataSpace failed: " +
                                     std::string(cudaGetErrorString(err)));
        }
        return ptr;
    }

    void FreeDataSpace(const class Device& device, void* ptr) override {
        profiling::ScopedSpan span(profiling::CurrentContext(),
                                   MakeDeviceSpec("free", device));
        span.AddField("ptr", ptr ? "nonnull" : "null");
        cudaSetDevice(device.device_id());
        cudaFree(ptr);
    }

    void CopyDataFromTo(const class Device& from_dev, const void* from_ptr,
                        const class Device& to_dev, void* to_ptr,
                        size_t nbytes) override {
        CopyDataFromTo(from_ptr, 0, to_ptr, 0, nbytes, from_dev, to_dev, nullptr);
    }

    void CopyDataFromTo(const void* from_ptr, size_t from_offset,
                        void* to_ptr, size_t to_offset, size_t nbytes,
                        const class Device& from_dev, const class Device& to_dev,
                        StreamHandle stream) override {
        profiling::EventSpec spec;
        spec.component = "device_api";
        spec.event_type = "copy";
        spec.device = DeviceTag(from_dev) + "->" + DeviceTag(to_dev);
        profiling::ScopedSpan span(profiling::CurrentContext(), std::move(spec));
        span.AddMetric("bytes", static_cast<double>(nbytes));
        cudaMemcpyKind kind = ResolveCopyKind(from_dev, to_dev);
        if (to_dev.device_type() == kGPU) {
            cudaSetDevice(to_dev.device_id());
        } else {
            cudaSetDevice(from_dev.device_id());
        }

        const uint8_t* src = static_cast<const uint8_t*>(from_ptr) + from_offset;
        uint8_t* dst = static_cast<uint8_t*>(to_ptr) + to_offset;
        cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
        if (cuda_stream == nullptr) {
            cuda_stream = tls_cuda_stream;
        }

        cudaError_t err = cudaSuccess;
        if (cuda_stream == nullptr) {
            err = cudaMemcpy(dst, src, nbytes, kind);
        } else {
            err = cudaMemcpyAsync(dst, src, nbytes, kind, cuda_stream);
        }
        if (err != cudaSuccess) {
            span.SetStatus("error");
            span.SetMessage(cudaGetErrorString(err));
            throw std::runtime_error("CUDA CopyDataFromTo failed: " +
                                     std::string(cudaGetErrorString(err)));
        }
    }

    DeviceAttributes GetDeviceAttributes(const class Device& device) override {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess) {
            return UnavailableCUDAAttributes("CUDA device query failed: " +
                                             std::string(cudaGetErrorString(err)));
        }
        if (device.device_id() < 0 || device.device_id() >= device_count) {
            return UnavailableCUDAAttributes("CUDA device id is not available: " +
                                             std::to_string(device.device_id()));
        }

        err = cudaSetDevice(device.device_id());
        if (err != cudaSuccess) {
            return UnavailableCUDAAttributes("CUDA SetDevice failed: " +
                                             std::string(cudaGetErrorString(err)));
        }

        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, device.device_id());
        if (err != cudaSuccess) {
            return UnavailableCUDAAttributes("CUDA GetDeviceProperties failed: " +
                                             std::string(cudaGetErrorString(err)));
        }

        DeviceAttributes attrs;
        attrs.exists = 1;
        attrs.max_threads_per_block = prop.maxThreadsPerBlock;
        attrs.warp_size = prop.warpSize;
        attrs.max_shared_memory_per_block = static_cast<int64_t>(prop.sharedMemPerBlock);
        attrs.compute_version = std::to_string(prop.major) + "." + std::to_string(prop.minor);
        int clock_rate_khz = 0;
        if (cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, device.device_id()) ==
            cudaSuccess) {
            attrs.max_clock_rate_khz = clock_rate_khz;
        }
        attrs.max_registers_per_block = prop.regsPerBlock;
        int runtime_version = 0;
        int driver_version = 0;
        if (cudaRuntimeGetVersion(&runtime_version) == cudaSuccess) {
            attrs.api_version = runtime_version;
        }
        if (cudaDriverGetVersion(&driver_version) == cudaSuccess) {
            attrs.driver_version = driver_version;
        }
        attrs.l2_cache_size_bytes = prop.l2CacheSize;
        attrs.total_global_memory = static_cast<int64_t>(prop.totalGlobalMem);
        attrs.available_global_memory = ReadAvailableMemory(device.device_id());
        attrs.max_shared_memory_per_multiprocessor = static_cast<int64_t>(prop.sharedMemPerMultiprocessor);
        attrs.max_registers_per_multiprocessor = prop.regsPerMultiprocessor;
        attrs.max_threads_per_multiprocessor = prop.maxThreadsPerMultiProcessor;
        attrs.compute_version_major = prop.major;
        attrs.compute_version_minor = prop.minor;
        attrs.multi_processor_count = prop.multiProcessorCount;
        attrs.device_name = prop.name;
        attrs.arch = "sm_" + std::to_string(prop.major) + std::to_string(prop.minor);
        return attrs;
    }

    std::string GetTargetKind(const class Device& device) const override {
        (void)device;
        return "cuda";
    }

    StreamHandle CreateStream(const class Device& device) override {
        profiling::ScopedSpan span(profiling::CurrentContext(),
                                   MakeDeviceSpec("create_stream", device));
        cudaError_t err = cudaSetDevice(device.device_id());
        if (err != cudaSuccess) {
            span.SetStatus("error");
            span.SetMessage(cudaGetErrorString(err));
            throw std::runtime_error("CUDA SetDevice failed: " +
                                     std::string(cudaGetErrorString(err)));
        }

        cudaStream_t stream = nullptr;
        err = cudaStreamCreate(&stream);
        if (err != cudaSuccess) {
            span.SetStatus("error");
            span.SetMessage(cudaGetErrorString(err));
            throw std::runtime_error("CUDA StreamCreate failed: " +
                                     std::string(cudaGetErrorString(err)));
        }
        return reinterpret_cast<StreamHandle>(stream);
    }

    void FreeStream(const class Device& device, StreamHandle stream) override {
        profiling::ScopedSpan span(profiling::CurrentContext(),
                                   MakeDeviceSpec("free_stream", device));
        cudaError_t err = cudaSetDevice(device.device_id());
        if (err != cudaSuccess) {
            span.SetStatus("error");
            span.SetMessage(cudaGetErrorString(err));
            throw std::runtime_error("CUDA SetDevice failed: " +
                                     std::string(cudaGetErrorString(err)));
        }

        cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
        if (cuda_stream == nullptr) {
            return;
        }

        err = cudaStreamDestroy(cuda_stream);
        if (err != cudaSuccess) {
            span.SetStatus("error");
            span.SetMessage(cudaGetErrorString(err));
            throw std::runtime_error("CUDA StreamDestroy failed: " +
                                     std::string(cudaGetErrorString(err)));
        }
        if (tls_cuda_stream == cuda_stream) {
            tls_cuda_stream = nullptr;
        }
    }

    void SetStream(const class Device& device, StreamHandle stream) override {
        profiling::ScopedSpan span(profiling::CurrentContext(),
                                   MakeDeviceSpec("set_stream", device));
        span.AddField("stream", stream ? "nonnull" : "null");
        cudaError_t err = cudaSetDevice(device.device_id());
        if (err != cudaSuccess) {
            span.SetStatus("error");
            span.SetMessage(cudaGetErrorString(err));
            throw std::runtime_error("CUDA SetDevice failed: " +
                                     std::string(cudaGetErrorString(err)));
        }
        tls_cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    }

    StreamHandle GetCurrentStream(const class Device& device) override {
        profiling::ScopedSpan span(profiling::CurrentContext(),
                                   MakeDeviceSpec("get_current_stream", device));
        cudaError_t err = cudaSetDevice(device.device_id());
        if (err != cudaSuccess) {
            span.SetStatus("error");
            span.SetMessage(cudaGetErrorString(err));
            throw std::runtime_error("CUDA SetDevice failed: " +
                                     std::string(cudaGetErrorString(err)));
        }
        return reinterpret_cast<StreamHandle>(tls_cuda_stream);
    }

    void StreamSync(const class Device& device, StreamHandle stream) override {
        profiling::ScopedSpan span(profiling::CurrentContext(),
                                   MakeDeviceSpec("stream_sync", device));
        span.AddField("stream", stream ? "nonnull" : "null");
        cudaError_t err = cudaSetDevice(device.device_id());
        if (err != cudaSuccess) {
            span.SetStatus("error");
            span.SetMessage(cudaGetErrorString(err));
            throw std::runtime_error("CUDA SetDevice failed: " +
                                     std::string(cudaGetErrorString(err)));
        }

        cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
        if (cuda_stream == nullptr) {
            cuda_stream = tls_cuda_stream;
        }
        if (cuda_stream == nullptr) {
            err = cudaDeviceSynchronize();
        } else {
            err = cudaStreamSynchronize(cuda_stream);
        }
        if (err != cudaSuccess) {
            span.SetStatus("error");
            span.SetMessage(cudaGetErrorString(err));
            throw std::runtime_error("CUDA StreamSync failed: " +
                                     std::string(cudaGetErrorString(err)));
        }
    }
};

DeviceAPI* GetCUDADeviceAPI() {
    static CUDADeviceAPI inst;
    return &inst;
}

#else

namespace {

class DisabledCUDADeviceAPI : public DeviceAPI {
public:
    void SetDevice(const class Device& device) override {
        (void)device;
        throw std::runtime_error("CUDA support is disabled (KXC_USE_CUDA=0)");
    }

    void* AllocDataSpace(const class Device& device, size_t nbytes, size_t alignment) override {
        (void)device;
        (void)nbytes;
        (void)alignment;
        throw std::runtime_error("CUDA support is disabled (KXC_USE_CUDA=0)");
    }

    void FreeDataSpace(const class Device& device, void* ptr) override {
        (void)device;
        (void)ptr;
        throw std::runtime_error("CUDA support is disabled (KXC_USE_CUDA=0)");
    }

    void CopyDataFromTo(const class Device& from_dev, const void* from_ptr,
                        const class Device& to_dev, void* to_ptr, size_t nbytes) override {
        (void)from_dev;
        (void)from_ptr;
        (void)to_dev;
        (void)to_ptr;
        (void)nbytes;
        throw std::runtime_error("CUDA support is disabled (KXC_USE_CUDA=0)");
    }

    DeviceAttributes GetDeviceAttributes(const class Device& device) override {
        (void)device;
        DeviceAttributes attrs;
        attrs.exists = 0;
        attrs.device_name = "CUDA support is disabled (KXC_USE_CUDA=0)";
        attrs.arch = "";
        attrs.compute_version = "0.0";
        return attrs;
    }

    std::string GetTargetKind(const class Device& device) const override {
        (void)device;
        return "cuda";
    }

    StreamHandle CreateStream(const class Device& device) override {
        (void)device;
        throw std::runtime_error("CUDA support is disabled (KXC_USE_CUDA=0)");
    }

    void FreeStream(const class Device& device, StreamHandle stream) override {
        (void)device;
        (void)stream;
        throw std::runtime_error("CUDA support is disabled (KXC_USE_CUDA=0)");
    }

    void SetStream(const class Device& device, StreamHandle stream) override {
        (void)device;
        (void)stream;
        throw std::runtime_error("CUDA support is disabled (KXC_USE_CUDA=0)");
    }

    StreamHandle GetCurrentStream(const class Device& device) override {
        (void)device;
        return nullptr;
    }

    void StreamSync(const class Device& device, StreamHandle stream) override {
        (void)device;
        (void)stream;
        throw std::runtime_error("CUDA support is disabled (KXC_USE_CUDA=0)");
    }
};

}  // namespace

DeviceAPI* GetCUDADeviceAPI() {
    static DisabledCUDADeviceAPI inst;
    return &inst;
}

#endif

}  // namespace kxc
