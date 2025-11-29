#include "base/device_api.h"
#include "base/registry.h"
#include "base/packedfunc.h"
#include "base/object.h" // Include for current_arena
#include <cuda_runtime.h>
#include <string>
#include <iostream>

namespace kxc {

thread_local Arena* current_arena = nullptr;

// ArgConverter specialization for Device
template<>
struct ArgConverter<class Device> {
    static class Device From(const Value& v, TypeCode t) {
        if (t == kObjectRef) {
            const Object* obj = v.v_object;
            if (!obj) throw std::runtime_error("Null device argument");
            // Assuming TypeIndex check or dynamic cast
            // kKXC_DEVICE_TYPE is defined in device.h
            if (obj->GetTypeId() != kKXC_DEVICE_TYPE) {
                 throw std::runtime_error("Argument is not a Device");
            }
            return *static_cast<const class Device*>(obj);
        }
        throw std::runtime_error("Type mismatch: expected Device");
    }
};

// 自动识别设备并注册的函数
// 这里我们实现一个简单的版本，检测 CUDA 设备并注册其属性
void DetectAndRegisterDevices() {
    // 检测 CUDA 设备
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        // std::cout << "cudaGetDeviceCount returned " << (int)error_id << "\n-> " << cudaGetErrorString(error_id) << std::endl;
        // 如果没有 CUDA 设备，这里可能不是 fatal error，只是没有检测到
    } else {
        // std::cout << "Detected " << deviceCount << " CUDA Capable devices." << std::endl;
        for (int dev = 0; dev < deviceCount; ++dev) {
            cudaSetDevice(dev);
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, dev);
            
            std::string name = deviceProp.name;
            int major = deviceProp.major;
            int minor = deviceProp.minor;
            size_t totalGlobalMem = deviceProp.totalGlobalMem;

            // std::cout << "Device " << dev << ": \"" << name << "\"" << std::endl;
            // std::cout << "  CUDA Capability Major/Minor version number:    " << major << "." << minor << std::endl;
            // std::cout << "  Total amount of global memory:                 " << (float)totalGlobalMem/1048576.0f << " MBytes" << std::endl;

            // 注册设备信息到全局注册表，供上层查询
            // 格式: device_info.cuda.<id>.name -> string
            //       device_info.cuda.<id>.memory -> int64
            
            std::string prefix = "device_info.cuda." + std::to_string(dev);
            
            KXC_REGISTER_GLOBAL(prefix + ".name")
            .set_body(PackedFunc(std::function<void(Args, RetValue*)>([name](Args args, RetValue* rv) {
                *rv = name;
            })));
            
            KXC_REGISTER_GLOBAL(prefix + ".memory")
            .set_body(PackedFunc(std::function<void(Args, RetValue*)>([totalGlobalMem](Args args, RetValue* rv) {
                *rv = (int64_t)totalGlobalMem;
            })));
            
             KXC_REGISTER_GLOBAL(prefix + ".compute_capability")
            .set_body(PackedFunc(std::function<void(Args, RetValue*)>([major, minor](Args args, RetValue* rv) {
                std::string cap = std::to_string(major) + "." + std::to_string(minor);
                *rv = cap;
            })));
        }
    }
}

// 注册全局函数，暴露 DeviceAPI 功能给 PackedFunc 系统
// 自动识别设备并调用对应的 DeviceAPI 实现

KXC_REGISTER_GLOBAL("device_api.AllocDataSpace")
.set_body(PackedFunc(std::function<void(Args, RetValue*)>([](Args args, RetValue* rv) {
    class Device dev = ArgConverter<class Device>::From(args[0], args.type_code(0));
    int64_t nbytes = ArgConverter<int64_t>::From(args[1], args.type_code(1));
    int64_t alignment = ArgConverter<int64_t>::From(args[2], args.type_code(2));
    
    DeviceAPI* api = DeviceAPIManager::Global()->GetAPI(dev.device_type());
    void* ptr = api->AllocDataSpace(dev, (size_t)nbytes, (size_t)alignment);
    
    // 返回指针地址作为 int64
    *rv = (int64_t)(uintptr_t)ptr;
})));

KXC_REGISTER_GLOBAL("device_api.FreeDataSpace")
.set_body(PackedFunc(std::function<void(Args, RetValue*)>([](Args args, RetValue* rv) {
    class Device dev = ArgConverter<class Device>::From(args[0], args.type_code(0));
    int64_t ptr_val = ArgConverter<int64_t>::From(args[1], args.type_code(1));
    void* ptr = (void*)(uintptr_t)ptr_val;
    
    DeviceAPI* api = DeviceAPIManager::Global()->GetAPI(dev.device_type());
    api->FreeDataSpace(dev, ptr);
})));

KXC_REGISTER_GLOBAL("device_api.CopyDataFromTo")
.set_body(PackedFunc(std::function<void(Args, RetValue*)>([](Args args, RetValue* rv) {
    class Device from_dev = ArgConverter<class Device>::From(args[0], args.type_code(0));
    int64_t from_ptr_val = ArgConverter<int64_t>::From(args[1], args.type_code(1));
    void* from_ptr = (void*)(uintptr_t)from_ptr_val;
    
    class Device to_dev = ArgConverter<class Device>::From(args[2], args.type_code(2));
    int64_t to_ptr_val = ArgConverter<int64_t>::From(args[3], args.type_code(3));
    void* to_ptr = (void*)(uintptr_t)to_ptr_val;
    
    int64_t nbytes = ArgConverter<int64_t>::From(args[4], args.type_code(4));
    
    DeviceAPI* api = nullptr;
    if (from_dev.device_type() == kCPU && to_dev.device_type() != kCPU) {
        api = DeviceAPIManager::Global()->GetAPI(to_dev.device_type());
    } else {
        api = DeviceAPIManager::Global()->GetAPI(from_dev.device_type());
    }
    
    api->CopyDataFromTo(from_dev, from_ptr, to_dev, to_ptr, (size_t)nbytes);
})));

// 暴露给 Python 的检测函数
KXC_REGISTER_GLOBAL("device.DetectAndRegister")
.set_body(PackedFunc(std::function<void(Args, RetValue*)>([](Args args, RetValue* rv) {
    DetectAndRegisterDevices();
})));

} // namespace kxc
