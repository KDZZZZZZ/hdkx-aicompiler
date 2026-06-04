/*! \file src/base/device/cpu_device_api.cc
 * \brief 实现具体 CPU/CUDA DeviceAPI 后端。
 */

#include "base/device_api.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>

#include "base/profiling.h"

namespace kxc {

namespace {

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

std::string DeviceTag(const class Device& device) {
    return "cpu:" + std::to_string(device.device_id());
}

profiling::EventSpec MakeDeviceSpec(const std::string& event_type, const class Device& device) {
    profiling::EventSpec spec;
    spec.component = "device_api";
    spec.event_type = event_type;
    spec.device = DeviceTag(device);
    return spec;
}

}  // namespace

class CPUDeviceAPI : public DeviceAPI {
public:
    void SetDevice(const class Device& device) override { (void)device; }

    void* AllocDataSpace(const class Device& device, size_t nbytes, size_t alignment) override {
        profiling::ScopedSpan span(profiling::CurrentContext(),
                                   MakeDeviceSpec("alloc", device));
        span.AddMetric("bytes", static_cast<double>(nbytes));
        span.AddMetric("alignment", static_cast<double>(alignment));
        void* ptr = nullptr;
#if defined(_MSC_VER)
        ptr = _aligned_malloc(nbytes, alignment);
#else
        if (posix_memalign(&ptr, alignment, nbytes) != 0) {
            ptr = nullptr;
        }
#endif
        if (ptr == nullptr) {
            span.SetStatus("error");
            span.SetMessage("CPU allocation returned null");
        }
        return ptr;
    }

    void FreeDataSpace(const class Device& device, void* ptr) override {
        profiling::ScopedSpan span(profiling::CurrentContext(),
                                   MakeDeviceSpec("free", device));
        span.AddField("ptr", ptr ? "nonnull" : "null");
#if defined(_MSC_VER)
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }

    void CopyDataFromTo(const class Device& from_dev, const void* from_ptr,
                        const class Device& to_dev, void* to_ptr,
                        size_t nbytes) override {
        profiling::EventSpec spec;
        spec.component = "device_api";
        spec.event_type = "copy";
        spec.device = DeviceTag(from_dev) + "->" + DeviceTag(to_dev);
        profiling::ScopedSpan span(profiling::CurrentContext(), std::move(spec));
        span.AddMetric("bytes", static_cast<double>(nbytes));
        if (from_dev.device_type() == kCPU && to_dev.device_type() == kCPU) {
            std::memcpy(to_ptr, from_ptr, nbytes);
            return;
        }
        span.SetStatus("error");
        span.SetMessage("Copy between CPU and non-CPU devices is not implemented");
        throw std::runtime_error("Copy between CPU and non-CPU devices is not implemented");
    }

    DeviceAttributes GetDeviceAttributes(const class Device& device) override {
        (void)device;
        DeviceAttributes attrs;
        attrs.exists = 1;
        unsigned int hw_threads = std::thread::hardware_concurrency();
        attrs.max_threads_per_block = hw_threads == 0 ? 1 : static_cast<int64_t>(hw_threads);
        attrs.warp_size = 1;
        attrs.max_shared_memory_per_block = 0;
        attrs.compute_version = "0.0";
        attrs.max_clock_rate_khz = 0;
        attrs.max_registers_per_block = 0;
        attrs.api_version = 0;
        attrs.driver_version = 0;
        attrs.l2_cache_size_bytes = 0;
        attrs.total_global_memory = 0;
        attrs.available_global_memory = 0;
        attrs.max_shared_memory_per_multiprocessor = 0;
        attrs.max_registers_per_multiprocessor = 0;
        attrs.max_threads_per_multiprocessor = attrs.max_threads_per_block;
        attrs.compute_version_major = 0;
        attrs.compute_version_minor = 0;
        attrs.multi_processor_count = attrs.max_threads_per_block;
        attrs.device_name = "cpu";
        attrs.arch = DetectHostCPUArch();
        return attrs;
    }

    std::string GetTargetKind(const class Device& device) const override {
        (void)device;
        return "llvm";
    }

    bool SupportsDevicePointerArithmeticsOnHost(const class Device& device) const override {
        (void)device;
        return true;
    }
};

DeviceAPI* GetCPUDeviceAPI() {
    static CPUDeviceAPI inst;
    return &inst;
}

}  // namespace kxc
