#pragma once
#include "base/device_api.h"
#include <cuda_runtime.h>

namespace kxc{
class CUDADeviceAPI : public DeviceAPI{
public:
    void SetDevice(const class Device& device) override {
        cudaError_t err = cudaSetDevice(device.device_id());
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA SetDevice failed: " + std::string(cudaGetErrorString(err)));
        }
    }

    void* AllocDataSpace(const class Device& device, size_t nbytes, size_t alignment) override {
        void* ptr = nullptr;
        cudaSetDevice(device.device_id());
        cudaError_t err = cudaMalloc(&ptr, nbytes);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA AllocDataSpace failed: " + std::string(cudaGetErrorString(err)));
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
        cudaMemcpyKind kind;
        if (from_dev.device_type() == kCPU && to_dev.device_type() == kGPU) {
            kind = cudaMemcpyHostToDevice;
        } else if (from_dev.device_type() == kGPU && to_dev.device_type() == kCPU) {
            kind = cudaMemcpyDeviceToHost;
        } else if (from_dev.device_type() == kGPU && to_dev.device_type() == kGPU) {
            kind = cudaMemcpyDeviceToDevice;
        } else {
            throw std::runtime_error("Unsupported CUDA copy direction");
        }

        // 简单起见，假设 current device 是操作相关的设备
        // 更严谨的实现可能需要 SetDevice 到 src 或 dst
        if (to_dev.device_type() == kGPU) {
            cudaSetDevice(to_dev.device_id());
        } else {
            cudaSetDevice(from_dev.device_id());
        }

        cudaError_t err = cudaMemcpy(to_ptr, from_ptr, nbytes, kind);
        if (err != cudaSuccess) {
             throw std::runtime_error("CUDA CopyDataFromTo failed: " + std::string(cudaGetErrorString(err)));
        }
    }
};

DeviceAPI* GetCUDADeviceAPI() {
    static CUDADeviceAPI inst;
    return &inst;
}

}