/*! \file src/base/device_api.cc
 * \brief 实现基础对象、设备、NDArray、Target、执行计划、PassContext 和 profiling 支撑逻辑。
 */

#include "base/device_api.h"

#include "base/object.h"
#include "base/packedfunc.h"
#include "base/registry.h"
#include "base/target.h"

#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <stdexcept>
#include <vector>

#if KXC_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace kxc {

thread_local Arena* current_arena = nullptr;

namespace {

const char* DeviceTypeName(DeviceTypeCode type) {
    switch (type) {
        case kCPU:
            return "cpu";
        case kGPU:
            return "cuda";
        case kOpenCL:
            return "opencl";
        case kMetal:
            return "metal";
        case kUnknown:
            return "unknown";
    }
    return "unknown";
}

std::string EscapeJson(const std::string& value) {
    std::ostringstream os;
    for (char ch : value) {
        switch (ch) {
            case '\\':
                os << "\\\\";
                break;
            case '"':
                os << "\\\"";
                break;
            case '\n':
                os << "\\n";
                break;
            case '\r':
                os << "\\r";
                break;
            case '\t':
                os << "\\t";
                break;
            default:
                os << ch;
                break;
        }
    }
    return os.str();
}

void WriteAttrJson(std::ostream& os, const DeviceAttributes& attrs) {
    os << "{";
    os << "\"exists\":" << attrs.exists << ",";
    os << "\"max_threads_per_block\":" << attrs.max_threads_per_block << ",";
    os << "\"warp_size\":" << attrs.warp_size << ",";
    os << "\"max_shared_memory_per_block\":" << attrs.max_shared_memory_per_block << ",";
    os << "\"compute_version\":\"" << EscapeJson(attrs.compute_version) << "\",";
    os << "\"device_name\":\"" << EscapeJson(attrs.device_name) << "\",";
    os << "\"max_clock_rate_khz\":" << attrs.max_clock_rate_khz << ",";
    os << "\"multi_processor_count\":" << attrs.multi_processor_count << ",";
    os << "\"max_registers_per_block\":" << attrs.max_registers_per_block << ",";
    os << "\"api_version\":" << attrs.api_version << ",";
    os << "\"driver_version\":" << attrs.driver_version << ",";
    os << "\"l2_cache_size_bytes\":" << attrs.l2_cache_size_bytes << ",";
    os << "\"total_global_memory\":" << attrs.total_global_memory << ",";
    os << "\"available_global_memory\":" << attrs.available_global_memory << ",";
    os << "\"max_shared_memory_per_multiprocessor\":"
       << attrs.max_shared_memory_per_multiprocessor << ",";
    os << "\"max_registers_per_multiprocessor\":"
       << attrs.max_registers_per_multiprocessor << ",";
    os << "\"max_threads_per_multiprocessor\":"
       << attrs.max_threads_per_multiprocessor << ",";
    os << "\"compute_version_major\":" << attrs.compute_version_major << ",";
    os << "\"compute_version_minor\":" << attrs.compute_version_minor << ",";
    os << "\"arch\":\"" << EscapeJson(attrs.arch) << "\"";
    os << "}";
}

void WriteDeviceInfoJson(std::ostream& os, const DeviceInfo& info) {
    os << "{";
    os << "\"device_type\":" << static_cast<int>(info.device_type) << ",";
    os << "\"device_type_name\":\"" << EscapeJson(info.device_type_name) << "\",";
    os << "\"device_id\":" << info.device_id << ",";
    os << "\"target_kind\":\"" << EscapeJson(info.target_kind) << "\",";
    os << "\"available\":" << (info.available ? "true" : "false") << ",";
    os << "\"status\":\"" << EscapeJson(info.status) << "\",";
    os << "\"attrs\":";
    WriteAttrJson(os, info.attrs);
    os << "}";
}

DeviceInfo QueryDeviceInfo(DeviceTypeCode type, int device_id,
                           const std::string& unavailable_status = "") {
    DeviceInfo info;
    info.device_type = type;
    info.device_id = device_id;
    info.device_type_name = DeviceTypeName(type);
    info.status = unavailable_status;

    try {
        class Device device(type, device_id);
        DeviceAPI* api = GetDeviceAPI(type);
        info.target_kind = api->GetTargetKind(device);
        info.attrs = api->GetDeviceAttributes(device);
        info.available = info.attrs.exists != 0;
        if (info.status.empty()) {
            info.status = info.available ? "ok" : "unavailable";
        }
    } catch (const std::exception& e) {
        info.available = false;
        if (info.status.empty()) {
            info.status = e.what();
        }
        if (info.target_kind.empty()) {
            info.target_kind = DeviceTypeName(type);
        }
        info.attrs.exists = 0;
        info.attrs.device_name = info.status;
        info.attrs.arch = "";
        info.attrs.compute_version = "0.0";
    }
    return info;
}

std::string DeviceInfosToJSON(const std::vector<DeviceInfo>& infos) {
    std::ostringstream os;
    os << "{\"devices\":[";
    for (size_t i = 0; i < infos.size(); ++i) {
        if (i) os << ",";
        WriteDeviceInfoJson(os, infos[i]);
    }
    os << "]}";
    return os.str();
}

}  // namespace

DeviceAPIManager* DeviceAPIManager::Global() {
    static DeviceAPIManager instance;
    return &instance;
}

DeviceAPI* DeviceAPIManager::GetAPI(DeviceTypeCode type) {
    if (apis_.size() <= static_cast<size_t>(type)) {
        apis_.resize(static_cast<size_t>(type) + 1, nullptr);
    }
    if (apis_[type] == nullptr) {
        if (type == kCPU) {
            apis_[type] = GetCPUDeviceAPI();
        } else if (type == kGPU) {
            apis_[type] = GetCUDADeviceAPI();
        } else {
            throw std::runtime_error("Device API not supported");
        }
    }
    return apis_[type];
}

DeviceAPI* GetDeviceAPI(DeviceTypeCode type) {
    return DeviceAPIManager::Global()->GetAPI(type);
}

bool DeviceAPI::NeedSetDevice(DeviceTypeCode type) {
    return type != kCPU;
}

void DeviceAPI::CopyDataFromTo(const void* from_ptr, size_t from_offset,
                               void* to_ptr, size_t to_offset, size_t nbytes,
                               const class Device& from_dev, const class Device& to_dev,
                               StreamHandle stream) {
    (void)stream;
    const uint8_t* src = static_cast<const uint8_t*>(from_ptr) + from_offset;
    uint8_t* dst = static_cast<uint8_t*>(to_ptr) + to_offset;
    CopyDataFromTo(from_dev, src, to_dev, dst, nbytes);
}

void DeviceAPI::GetAttr(const class Device& device, DeviceAttrKind kind, RetValue* rv) {
    DeviceAttributes attrs = GetDeviceAttributes(device);
    switch (kind) {
        case DeviceAttrKind::kExist:
            *rv = attrs.exists;
            return;
        case DeviceAttrKind::kMaxThreadsPerBlock:
            *rv = attrs.max_threads_per_block;
            return;
        case DeviceAttrKind::kWarpSize:
            *rv = attrs.warp_size;
            return;
        case DeviceAttrKind::kMaxSharedMemoryPerBlock:
            *rv = attrs.max_shared_memory_per_block;
            return;
        case DeviceAttrKind::kComputeVersion:
            *rv = attrs.compute_version;
            return;
        case DeviceAttrKind::kDeviceName:
            *rv = attrs.device_name;
            return;
        case DeviceAttrKind::kMaxClockRate:
            *rv = attrs.max_clock_rate_khz;
            return;
        case DeviceAttrKind::kMultiProcessorCount:
            *rv = attrs.multi_processor_count;
            return;
        case DeviceAttrKind::kMaxRegistersPerBlock:
            *rv = attrs.max_registers_per_block;
            return;
        case DeviceAttrKind::kApiVersion:
            *rv = attrs.api_version;
            return;
        case DeviceAttrKind::kDriverVersion:
            *rv = attrs.driver_version;
            return;
        case DeviceAttrKind::kL2CacheSizeBytes:
            *rv = attrs.l2_cache_size_bytes;
            return;
        case DeviceAttrKind::kTotalGlobalMemory:
            *rv = attrs.total_global_memory;
            return;
        case DeviceAttrKind::kAvailableGlobalMemory:
            *rv = attrs.available_global_memory;
            return;
        case DeviceAttrKind::kMaxSharedMemoryPerMultiprocessor:
            *rv = attrs.max_shared_memory_per_multiprocessor;
            return;
        case DeviceAttrKind::kMaxRegistersPerMultiprocessor:
            *rv = attrs.max_registers_per_multiprocessor;
            return;
        case DeviceAttrKind::kMaxThreadsPerMultiprocessor:
            *rv = attrs.max_threads_per_multiprocessor;
            return;
        case DeviceAttrKind::kComputeVersionMajor:
            *rv = attrs.compute_version_major;
            return;
        case DeviceAttrKind::kComputeVersionMinor:
            *rv = attrs.compute_version_minor;
            return;
        default:
            break;
    }
    throw std::runtime_error("Unknown DeviceAttrKind: " + std::to_string(static_cast<int>(kind)));
}

StreamHandle DeviceAPI::CreateStream(const class Device& device) {
    (void)device;
    throw std::runtime_error("CreateStream is not supported on this device backend");
}

void DeviceAPI::FreeStream(const class Device& device, StreamHandle stream) {
    (void)device;
    (void)stream;
    throw std::runtime_error("FreeStream is not supported on this device backend");
}

void DeviceAPI::SetStream(const class Device& device, StreamHandle stream) {
    (void)device;
    (void)stream;
    throw std::runtime_error("SetStream is not supported on this device backend");
}

StreamHandle DeviceAPI::GetCurrentStream(const class Device& device) {
    (void)device;
    return nullptr;
}

void DeviceAPI::StreamSync(const class Device& device, StreamHandle stream) {
    (void)device;
    (void)stream;
}

void DeviceAPI::SyncStreamFromTo(const class Device& device, StreamHandle src, StreamHandle dst) {
    (void)dst;
    StreamSync(device, src);
}

void* DeviceAPI::AllocWorkspace(const class Device& device, size_t nbytes, size_t alignment) {
    return AllocDataSpace(device, nbytes, alignment);
}

void DeviceAPI::FreeWorkspace(const class Device& device, void* ptr) {
    FreeDataSpace(device, ptr);
}

bool DeviceAPI::SupportsDevicePointerArithmeticsOnHost(const class Device& device) const {
    return device.device_type() == kCPU;
}

DeviceAttributes CollectDeviceAttributes(const class Device& device) {
    DeviceAPI* api = GetDeviceAPI(device.device_type());
    return api->GetDeviceAttributes(device);
}

void GetDeviceAttr(const class Device& device, DeviceAttrKind kind, RetValue* rv) {
    DeviceAPI* api = GetDeviceAPI(device.device_type());
    api->GetAttr(device, kind, rv);
}

int64_t GetDeviceAttr(const class Device& device, DeviceAttrKind kind) {
    RetValue rv;
    GetDeviceAttr(device, kind, &rv);
    if (rv.type_code() == kInt) {
        return rv.As<int64_t>();
    }
    throw std::runtime_error("GetDeviceAttr(int64) got non-integer attr kind: " +
                             std::to_string(static_cast<int>(kind)));
}

std::vector<class Device> ListDevices() {
    std::vector<class Device> devices;
    for (const auto& info : GetAllDeviceInfo()) {
        if (info.available) {
            devices.emplace_back(info.device_type, info.device_id);
        }
    }
    return devices;
}

std::vector<DeviceInfo> GetAllDeviceInfo() {
    std::vector<DeviceInfo> infos;
    infos.push_back(QueryDeviceInfo(kCPU, 0));

#if KXC_USE_CUDA
    int device_count = 0;
    cudaError_t error_id = cudaGetDeviceCount(&device_count);
    if (error_id == cudaSuccess && device_count > 0) {
        for (int dev = 0; dev < device_count; ++dev) {
            infos.push_back(QueryDeviceInfo(kGPU, dev));
        }
    } else {
        std::string status = "no CUDA device detected";
        if (error_id != cudaSuccess) {
            status = "CUDA device query failed: " + std::string(cudaGetErrorString(error_id));
        }
        infos.push_back(QueryDeviceInfo(kGPU, 0, status));
    }
#else
    infos.push_back(QueryDeviceInfo(kGPU, 0, "CUDA support is disabled (KXC_USE_CUDA=0)"));
#endif

    return infos;
}

std::string ListDevicesJSON() {
    std::vector<DeviceInfo> available;
    for (const auto& info : GetAllDeviceInfo()) {
        if (info.available) {
            available.push_back(info);
        }
    }
    return DeviceInfosToJSON(available);
}

std::string GetAllDeviceInfoJSON() {
    return DeviceInfosToJSON(GetAllDeviceInfo());
}

template <>
struct ArgConverter<class Device> {
    static class Device From(const Value& v, TypeCode t) {
        if (t == kObjectRef) {
            const Object* obj = v.v_object;
            if (!obj) throw std::runtime_error("Null device argument");
            if (obj->GetTypeId() != kKXC_DEVICE_TYPE) {
                throw std::runtime_error("Argument is not a Device");
            }
            return *static_cast<const class Device*>(obj);
        }
        throw std::runtime_error("Type mismatch: expected Device");
    }
};

void DetectAndRegisterDevices() {
#if KXC_USE_CUDA
    int device_count = 0;
    cudaError_t error_id = cudaGetDeviceCount(&device_count);

    if (error_id == cudaSuccess) {
        for (int dev = 0; dev < device_count; ++dev) {
            cudaSetDevice(dev);

            cudaDeviceProp prop;
            cudaError_t prop_err = cudaGetDeviceProperties(&prop, dev);
            if (prop_err != cudaSuccess) {
                continue;
            }

            std::string name = prop.name;
            int major = prop.major;
            int minor = prop.minor;
            size_t total_global_mem = prop.totalGlobalMem;

            std::string prefix = "device_info.cuda." + std::to_string(dev);

            KXC_REGISTER_GLOBAL(prefix + ".name")
                .set_body(PackedFunc(std::function<void(Args, RetValue*)>(
                    [name](Args args, RetValue* rv) {
                        (void)args;
                        *rv = name;
                    })));

            KXC_REGISTER_GLOBAL(prefix + ".memory")
                .set_body(PackedFunc(std::function<void(Args, RetValue*)>(
                    [total_global_mem](Args args, RetValue* rv) {
                        (void)args;
                        *rv = static_cast<int64_t>(total_global_mem);
                    })));

            KXC_REGISTER_GLOBAL(prefix + ".compute_capability")
                .set_body(PackedFunc(std::function<void(Args, RetValue*)>(
                    [major, minor](Args args, RetValue* rv) {
                        (void)args;
                        std::string cap = std::to_string(major) + "." + std::to_string(minor);
                        *rv = cap;
                    })));
        }
    }
#else
    // CUDA disabled at compile time.
#endif
}

KXC_REGISTER_GLOBAL("device_api.AllocDataSpace")
    .set_body(PackedFunc(std::function<void(Args, RetValue*)>([](Args args, RetValue* rv) {
        class Device dev = ArgConverter<class Device>::From(args[0], args.type_code(0));
        int64_t nbytes = ArgConverter<int64_t>::From(args[1], args.type_code(1));
        int64_t alignment = ArgConverter<int64_t>::From(args[2], args.type_code(2));

        DeviceAPI* api = DeviceAPIManager::Global()->GetAPI(dev.device_type());
        void* ptr = api->AllocDataSpace(dev, static_cast<size_t>(nbytes),
                                        static_cast<size_t>(alignment));
        *rv = static_cast<int64_t>(reinterpret_cast<uintptr_t>(ptr));
    })));

KXC_REGISTER_GLOBAL("device_api.FreeDataSpace")
    .set_body(PackedFunc(std::function<void(Args, RetValue*)>([](Args args, RetValue* rv) {
        class Device dev = ArgConverter<class Device>::From(args[0], args.type_code(0));
        int64_t ptr_val = ArgConverter<int64_t>::From(args[1], args.type_code(1));
        void* ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(ptr_val));

        DeviceAPI* api = DeviceAPIManager::Global()->GetAPI(dev.device_type());
        api->FreeDataSpace(dev, ptr);
        (void)rv;
    })));

KXC_REGISTER_GLOBAL("device_api.AllocWorkspace")
    .set_body(PackedFunc(std::function<void(Args, RetValue*)>([](Args args, RetValue* rv) {
        class Device dev = ArgConverter<class Device>::From(args[0], args.type_code(0));
        int64_t nbytes = ArgConverter<int64_t>::From(args[1], args.type_code(1));
        int64_t alignment = ArgConverter<int64_t>::From(args[2], args.type_code(2));

        DeviceAPI* api = DeviceAPIManager::Global()->GetAPI(dev.device_type());
        void* ptr = api->AllocWorkspace(dev, static_cast<size_t>(nbytes),
                                        static_cast<size_t>(alignment));
        *rv = static_cast<int64_t>(reinterpret_cast<uintptr_t>(ptr));
    })));

KXC_REGISTER_GLOBAL("device_api.FreeWorkspace")
    .set_body(PackedFunc(std::function<void(Args, RetValue*)>([](Args args, RetValue* rv) {
        class Device dev = ArgConverter<class Device>::From(args[0], args.type_code(0));
        int64_t ptr_val = ArgConverter<int64_t>::From(args[1], args.type_code(1));
        void* ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(ptr_val));

        DeviceAPI* api = DeviceAPIManager::Global()->GetAPI(dev.device_type());
        api->FreeWorkspace(dev, ptr);
        (void)rv;
    })));

KXC_REGISTER_GLOBAL("device_api.CopyDataFromTo")
    .set_body(PackedFunc(std::function<void(Args, RetValue*)>([](Args args, RetValue* rv) {
        class Device from_dev = ArgConverter<class Device>::From(args[0], args.type_code(0));
        int64_t from_ptr_val = ArgConverter<int64_t>::From(args[1], args.type_code(1));
        void* from_ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(from_ptr_val));

        class Device to_dev = ArgConverter<class Device>::From(args[2], args.type_code(2));
        int64_t to_ptr_val = ArgConverter<int64_t>::From(args[3], args.type_code(3));
        void* to_ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(to_ptr_val));

        int64_t nbytes = ArgConverter<int64_t>::From(args[4], args.type_code(4));

        DeviceAPI* api = nullptr;
        if (from_dev.device_type() == kCPU && to_dev.device_type() != kCPU) {
            api = DeviceAPIManager::Global()->GetAPI(to_dev.device_type());
        } else {
            api = DeviceAPIManager::Global()->GetAPI(from_dev.device_type());
        }

        api->CopyDataFromTo(from_dev, from_ptr, to_dev, to_ptr, static_cast<size_t>(nbytes));
        (void)rv;
    })));

KXC_REGISTER_GLOBAL("device_api.CopyDataFromToEx")
    .set_body(PackedFunc(std::function<void(Args, RetValue*)>([](Args args, RetValue* rv) {
        class Device from_dev = ArgConverter<class Device>::From(args[0], args.type_code(0));
        int64_t from_ptr_val = ArgConverter<int64_t>::From(args[1], args.type_code(1));
        int64_t from_offset = ArgConverter<int64_t>::From(args[2], args.type_code(2));
        class Device to_dev = ArgConverter<class Device>::From(args[3], args.type_code(3));
        int64_t to_ptr_val = ArgConverter<int64_t>::From(args[4], args.type_code(4));
        int64_t to_offset = ArgConverter<int64_t>::From(args[5], args.type_code(5));
        int64_t nbytes = ArgConverter<int64_t>::From(args[6], args.type_code(6));
        int64_t stream_val = ArgConverter<int64_t>::From(args[7], args.type_code(7));

        const void* from_ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(from_ptr_val));
        void* to_ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(to_ptr_val));
        StreamHandle stream = reinterpret_cast<StreamHandle>(static_cast<uintptr_t>(stream_val));

        DeviceAPI* api = nullptr;
        if (from_dev.device_type() == kCPU && to_dev.device_type() != kCPU) {
            api = DeviceAPIManager::Global()->GetAPI(to_dev.device_type());
        } else {
            api = DeviceAPIManager::Global()->GetAPI(from_dev.device_type());
        }
        api->CopyDataFromTo(from_ptr, static_cast<size_t>(from_offset), to_ptr,
                            static_cast<size_t>(to_offset), static_cast<size_t>(nbytes),
                            from_dev, to_dev, stream);
        (void)rv;
    })));

KXC_REGISTER_GLOBAL("device_api.GetAttr")
    .set_body(PackedFunc(std::function<void(Args, RetValue*)>([](Args args, RetValue* rv) {
        class Device dev = ArgConverter<class Device>::From(args[0], args.type_code(0));
        int attr_kind = ArgConverter<int>::From(args[1], args.type_code(1));
        GetDeviceAttr(dev, static_cast<DeviceAttrKind>(attr_kind), rv);
    })));

KXC_REGISTER_GLOBAL("device_api.CreateStream")
    .set_body(PackedFunc(std::function<void(Args, RetValue*)>([](Args args, RetValue* rv) {
        class Device dev = ArgConverter<class Device>::From(args[0], args.type_code(0));
        DeviceAPI* api = DeviceAPIManager::Global()->GetAPI(dev.device_type());
        StreamHandle stream = api->CreateStream(dev);
        *rv = static_cast<int64_t>(reinterpret_cast<uintptr_t>(stream));
    })));

KXC_REGISTER_GLOBAL("device_api.FreeStream")
    .set_body(PackedFunc(std::function<void(Args, RetValue*)>([](Args args, RetValue* rv) {
        class Device dev = ArgConverter<class Device>::From(args[0], args.type_code(0));
        int64_t stream_val = ArgConverter<int64_t>::From(args[1], args.type_code(1));
        StreamHandle stream = reinterpret_cast<StreamHandle>(static_cast<uintptr_t>(stream_val));
        DeviceAPI* api = DeviceAPIManager::Global()->GetAPI(dev.device_type());
        api->FreeStream(dev, stream);
        (void)rv;
    })));

KXC_REGISTER_GLOBAL("device_api.StreamSync")
    .set_body(PackedFunc(std::function<void(Args, RetValue*)>([](Args args, RetValue* rv) {
        class Device dev = ArgConverter<class Device>::From(args[0], args.type_code(0));
        int64_t stream_val = ArgConverter<int64_t>::From(args[1], args.type_code(1));
        StreamHandle stream = reinterpret_cast<StreamHandle>(static_cast<uintptr_t>(stream_val));
        DeviceAPI* api = DeviceAPIManager::Global()->GetAPI(dev.device_type());
        api->StreamSync(dev, stream);
        (void)rv;
    })));

KXC_REGISTER_GLOBAL("device_api.SyncStreamFromTo")
    .set_body(PackedFunc(std::function<void(Args, RetValue*)>([](Args args, RetValue* rv) {
        class Device dev = ArgConverter<class Device>::From(args[0], args.type_code(0));
        int64_t src_val = ArgConverter<int64_t>::From(args[1], args.type_code(1));
        int64_t dst_val = ArgConverter<int64_t>::From(args[2], args.type_code(2));
        StreamHandle src = reinterpret_cast<StreamHandle>(static_cast<uintptr_t>(src_val));
        StreamHandle dst = reinterpret_cast<StreamHandle>(static_cast<uintptr_t>(dst_val));
        DeviceAPI* api = DeviceAPIManager::Global()->GetAPI(dev.device_type());
        api->SyncStreamFromTo(dev, src, dst);
        (void)rv;
    })));

KXC_REGISTER_GLOBAL("device_api.SetStream")
    .set_body(PackedFunc(std::function<void(Args, RetValue*)>([](Args args, RetValue* rv) {
        class Device dev = ArgConverter<class Device>::From(args[0], args.type_code(0));
        int64_t stream_val = ArgConverter<int64_t>::From(args[1], args.type_code(1));
        StreamHandle stream = reinterpret_cast<StreamHandle>(static_cast<uintptr_t>(stream_val));
        DeviceAPI* api = DeviceAPIManager::Global()->GetAPI(dev.device_type());
        api->SetStream(dev, stream);
        (void)rv;
    })));

KXC_REGISTER_GLOBAL("device_api.GetCurrentStream")
    .set_body(PackedFunc(std::function<void(Args, RetValue*)>([](Args args, RetValue* rv) {
        class Device dev = ArgConverter<class Device>::From(args[0], args.type_code(0));
        DeviceAPI* api = DeviceAPIManager::Global()->GetAPI(dev.device_type());
        StreamHandle stream = api->GetCurrentStream(dev);
        *rv = static_cast<int64_t>(reinterpret_cast<uintptr_t>(stream));
    })));

KXC_REGISTER_GLOBAL("device_api.NeedSetDevice")
    .set_body(PackedFunc(std::function<void(Args, RetValue*)>([](Args args, RetValue* rv) {
        int type_code = ArgConverter<int>::From(args[0], args.type_code(0));
        *rv = static_cast<int64_t>(DeviceAPI::NeedSetDevice(static_cast<DeviceTypeCode>(type_code)));
    })));

KXC_REGISTER_GLOBAL("device_api.ListDevices")
    .set_body(PackedFunc(std::function<void(Args, RetValue*)>([](Args args, RetValue* rv) {
        (void)args;
        *rv = ListDevicesJSON();
    })));

KXC_REGISTER_GLOBAL("device_api.ListDevicesJSON")
    .set_body(PackedFunc(std::function<void(Args, RetValue*)>([](Args args, RetValue* rv) {
        (void)args;
        *rv = ListDevicesJSON();
    })));

KXC_REGISTER_GLOBAL("device_api.GetAllDeviceInfo")
    .set_body(PackedFunc(std::function<void(Args, RetValue*)>([](Args args, RetValue* rv) {
        (void)args;
        *rv = GetAllDeviceInfoJSON();
    })));

KXC_REGISTER_GLOBAL("device_api.GetAllDeviceInfoJSON")
    .set_body(PackedFunc(std::function<void(Args, RetValue*)>([](Args args, RetValue* rv) {
        (void)args;
        *rv = GetAllDeviceInfoJSON();
    })));

KXC_REGISTER_GLOBAL("device_api.GetTargetKind")
    .set_body(PackedFunc(std::function<void(Args, RetValue*)>([](Args args, RetValue* rv) {
        class Device dev = ArgConverter<class Device>::From(args[0], args.type_code(0));
        Target target = BuildTarget(dev);
        *rv = target->kind;
    })));

KXC_REGISTER_GLOBAL("device_api.GetTargetArch")
    .set_body(PackedFunc(std::function<void(Args, RetValue*)>([](Args args, RetValue* rv) {
        class Device dev = ArgConverter<class Device>::From(args[0], args.type_code(0));
        Target target = BuildTarget(dev);
        *rv = target->attrs.arch;
    })));

KXC_REGISTER_GLOBAL("target.Build")
    .set_body(PackedFunc(std::function<void(Args, RetValue*)>([](Args args, RetValue* rv) {
        class Device dev = ArgConverter<class Device>::From(args[0], args.type_code(0));
        *rv = ObjectRef(BuildTarget(dev));
    })));

KXC_REGISTER_GLOBAL("device.DetectAndRegister")
    .set_body(PackedFunc(std::function<void(Args, RetValue*)>([](Args args, RetValue* rv) {
        (void)args;
        (void)rv;
        DetectAndRegisterDevices();
    })));

}  // namespace kxc
