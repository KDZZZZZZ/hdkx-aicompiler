#include "base/device_api.h"

#include <cstdint>
#include <stdexcept>
#include <string>

#if KXC_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace kxc {

#if KXC_USE_CUDA

namespace {

thread_local cudaStream_t tls_cuda_stream = nullptr;

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

}  // namespace

class CUDADeviceAPI : public DeviceAPI {
public:
    void SetDevice(const class Device& device) override {
        cudaError_t err = cudaSetDevice(device.device_id());
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA SetDevice failed: " +
                                     std::string(cudaGetErrorString(err)));
        }
    }

    void* AllocDataSpace(const class Device& device, size_t nbytes, size_t alignment) override {
        (void)alignment;
        void* ptr = nullptr;
        cudaSetDevice(device.device_id());
        cudaError_t err = cudaMalloc(&ptr, nbytes);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA AllocDataSpace failed: " +
                                     std::string(cudaGetErrorString(err)));
        }
        return ptr;
    }

    void FreeDataSpace(const class Device& device, void* ptr) override {
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
            throw std::runtime_error("CUDA CopyDataFromTo failed: " +
                                     std::string(cudaGetErrorString(err)));
        }
    }

    DeviceAttributes GetDeviceAttributes(const class Device& device) override {
        cudaError_t err = cudaSetDevice(device.device_id());
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA SetDevice failed: " +
                                     std::string(cudaGetErrorString(err)));
        }

        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, device.device_id());
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA GetDeviceProperties failed: " +
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
        cudaError_t err = cudaSetDevice(device.device_id());
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA SetDevice failed: " +
                                     std::string(cudaGetErrorString(err)));
        }

        cudaStream_t stream = nullptr;
        err = cudaStreamCreate(&stream);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA StreamCreate failed: " +
                                     std::string(cudaGetErrorString(err)));
        }
        return reinterpret_cast<StreamHandle>(stream);
    }

    void FreeStream(const class Device& device, StreamHandle stream) override {
        cudaError_t err = cudaSetDevice(device.device_id());
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA SetDevice failed: " +
                                     std::string(cudaGetErrorString(err)));
        }

        cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
        if (cuda_stream == nullptr) {
            return;
        }

        err = cudaStreamDestroy(cuda_stream);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA StreamDestroy failed: " +
                                     std::string(cudaGetErrorString(err)));
        }
        if (tls_cuda_stream == cuda_stream) {
            tls_cuda_stream = nullptr;
        }
    }

    void SetStream(const class Device& device, StreamHandle stream) override {
        cudaError_t err = cudaSetDevice(device.device_id());
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA SetDevice failed: " +
                                     std::string(cudaGetErrorString(err)));
        }
        tls_cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    }

    StreamHandle GetCurrentStream(const class Device& device) override {
        cudaError_t err = cudaSetDevice(device.device_id());
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA SetDevice failed: " +
                                     std::string(cudaGetErrorString(err)));
        }
        return reinterpret_cast<StreamHandle>(tls_cuda_stream);
    }

    void StreamSync(const class Device& device, StreamHandle stream) override {
        cudaError_t err = cudaSetDevice(device.device_id());
        if (err != cudaSuccess) {
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

DeviceAPI* GetCUDADeviceAPI() {
    throw std::runtime_error("CUDA support is disabled (KXC_USE_CUDA=0)");
}

#endif

}  // namespace kxc
