/*! \file src/base/device_api.cc
 * \brief 实现设备后端路由、设备发现及 PackedFunc 查询入口。
 */

#include "base/device_api.h"

#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "base/arena.h"
#include "base/packedfunc.h"
#include "base/registry.h"

#if KXC_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace kxc {

thread_local Arena* current_arena = nullptr;

namespace {

// 返回稳定的设备类型文本，供诊断和 JSON 使用。
const char* DeviceTypeName(DeviceTypeCode type) {
    switch (type) {
        case kCPU: return "cpu";
        case kCUDA: return "cuda";
        case kOpenCL: return "opencl";
        case kMetal: return "metal";
        case kUnknown: return "unknown";
    }
    return "unknown";
}

// 转义设备诊断字符串中的 JSON 特殊字符。
std::string EscapeJson(const std::string& value) {
    std::ostringstream os;
    for (char ch : value) {
        switch (ch) {
            case '\\': os << "\\\\"; break;
            case '"': os << "\\\""; break;
            case '\n': os << "\\n"; break;
            case '\r': os << "\\r"; break;
            case '\t': os << "\\t"; break;
            default: os << ch; break;
        }
    }
    return os.str();
}

// 将统一设备属性写入 JSON 对象，不暴露后端私有结构。
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
    os << "\"max_shared_memory_per_multiprocessor\":" << attrs.max_shared_memory_per_multiprocessor << ",";
    os << "\"max_registers_per_multiprocessor\":" << attrs.max_registers_per_multiprocessor << ",";
    os << "\"max_threads_per_multiprocessor\":" << attrs.max_threads_per_multiprocessor << ",";
    os << "\"compute_version_major\":" << attrs.compute_version_major << ",";
    os << "\"compute_version_minor\":" << attrs.compute_version_minor << ",";
    os << "\"arch\":\"" << EscapeJson(attrs.arch) << "\"}";
}

// 将一组设备诊断记录编码为稳定的 JSON 契约。
std::string DeviceInfosToJSON(const std::vector<DeviceInfo>& infos) {
    std::ostringstream os;
    os << "{\"devices\":[";
    for (size_t i = 0; i < infos.size(); ++i) {
        if (i) os << ',';
        const auto& info = infos[i];
        os << "{\"device_type\":" << static_cast<int>(info.device_type) << ',';
        os << "\"device_type_name\":\"" << EscapeJson(info.device_type_name) << "\",";
        os << "\"device_id\":" << info.device_id << ',';
        os << "\"target_kind\":\"" << EscapeJson(info.target_kind) << "\",";
        os << "\"available\":" << (info.available ? "true" : "false") << ',';
        os << "\"status\":\"" << EscapeJson(info.status) << "\",\"attrs\":";
        WriteAttrJson(os, info.attrs);
        os << '}';
    }
    os << "]}";
    return os.str();
}

// 探测单个设备，并把探测异常收敛为 unavailable 诊断记录。
DeviceInfo QueryDeviceInfo(DeviceTypeCode type, int id,
                           std::string unavailable_status = {}) {
    DeviceInfo info;
    info.device_type = type;
    info.device_id = id;
    info.device_type_name = DeviceTypeName(type);
    info.status = std::move(unavailable_status);
    // 单个后端探测失败只形成 unavailable 诊断项，不能中断其他设备枚举。
    try {
        Device device(type, id);
        DeviceAPI* api = GetDeviceAPI(type);
        info.target_kind = api->GetTargetKind(device);
        info.attrs = api->GetDeviceAttributes(device);
        info.available = info.attrs.exists != 0;
        if (info.status.empty()) info.status = info.available ? "ok" : "unavailable";
    } catch (const std::exception& error) {
        info.available = false;
        if (info.status.empty()) info.status = error.what();
        info.target_kind = DeviceTypeName(type);
        info.attrs.exists = 0;
        info.attrs.device_name = info.status;
    }
    return info;
}

// 根据两端设备选择唯一负责解释复制方向的后端。
DeviceAPI* CopyAPI(const Device& from, const Device& to) {
    if (from.device_type() == kCPU && to.device_type() == kCPU) {
        return GetCPUDeviceAPI();
    }
    // 任一端为 CUDA 时由 CUDA 后端选择 H2D、D2H 或 D2D 复制方向。
    if (from.device_type() == kCUDA || to.device_type() == kCUDA) {
        if (from.device_type() == kCUDA && to.device_type() == kCUDA &&
            from.device_id() != to.device_id()) {
            throw std::invalid_argument("cross-device CUDA copy is not supported");
        }
        return GetCUDADeviceAPI();
    }
    throw std::runtime_error("device copy route is not supported");
}

}  // namespace

// 定义 PackedFunc Value 到强类型 Device 的边界转换规则。
template <>
struct ArgConverter<Device> {
    // 从 PackedFunc 参数恢复经过类型检查的 Device 对象。
    static Device From(const Value& value, TypeCode type) {
        if (type != kObjectRef || value.v_object == nullptr) {
            throw std::runtime_error("Type mismatch: expected Device");
        }
        return Device(ObjectRef(value.v_object));
    }
};

// 返回进程级 DeviceAPI 注册与缓存管理器。
DeviceAPIManager* DeviceAPIManager::Global() {
    static DeviceAPIManager manager;
    return &manager;
}

// 按设备类型延迟解析并缓存后端单例。
DeviceAPI* DeviceAPIManager::GetAPI(DeviceTypeCode type) {
    std::lock_guard<std::mutex> lock(mutex_);
    const int key = static_cast<int>(type);
    auto found = apis_.find(key);
    if (found != apis_.end()) return found->second;
    DeviceAPI* api = nullptr;
    if (type == kCPU) api = GetCPUDeviceAPI();
    if (type == kCUDA) api = GetCUDADeviceAPI();
    if (!api) throw std::runtime_error("Device API not supported: " + std::to_string(key));
    apis_.emplace(key, api);
    return api;
}

// 提供统一的后端查找入口。
DeviceAPI* GetDeviceAPI(DeviceTypeCode type) {
    return DeviceAPIManager::Global()->GetAPI(type);
}

// 将当前可用设备列表转换为对象系统容器。
Array<Device> DeviceManager::ListAvailableDevices() const {
    Array<Device> result;
    for (const Device& device : ListDevices()) result.push_back(device);
    return result;
}

// 将包含不可用诊断项的完整设备信息转换为对象系统容器。
Array<DeviceInfo> DeviceManager::GetAllDeviceInfo() const {
    Array<DeviceInfo> result;
    for (const DeviceInfo& info : kxc::GetAllDeviceInfo()) result.push_back(info);
    return result;
}

// 统一处理零尺寸和对齐校验后，将分配路由到设备后端。
void* DeviceAlloc(const Device& device, size_t nbytes, size_t alignment) {
    // 零尺寸张量允许空 Storage，不把 nullptr 交给后端做指针运算。
    if (nbytes == 0) return nullptr;
    if (alignment != 0 && (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument("alignment must be zero or a power of two");
    }
    void* result = GetDeviceAPI(device.device_type())
                       ->AllocDataSpace(device, nbytes, alignment);
    if (!result) throw std::bad_alloc();
    return result;
}

// 将非空地址路由到其所有者设备后端释放。
void DeviceFree(const Device& device, void* ptr) {
    if (!ptr) return;
    GetDeviceAPI(device.device_type())->FreeDataSpace(device, ptr);
}

// 统一处理零字节契约后，将清零操作路由到设备后端。
void DeviceZero(const Device& device, void* ptr, size_t offset, size_t nbytes) {
    if (nbytes == 0) return;
    if (!ptr) throw std::invalid_argument("non-empty zero fill requires a pointer");
    GetDeviceAPI(device.device_type())->ZeroData(device, ptr, offset, nbytes);
}

// 校验统一指针契约，并按两端设备路由同步复制。
void DeviceCopySync(const Device& from_device, const void* from,
                    size_t from_offset, const Device& to_device, void* to,
                    size_t to_offset, size_t nbytes) {
    if (nbytes == 0) return;
    if (!from || !to) throw std::invalid_argument("non-empty copy requires pointers");
    CopyAPI(from_device, to_device)
        ->CopyDataSync(from_device, from, from_offset, to_device, to, to_offset,
                       nbytes);
}

// 将结构化设备属性投影为兼容 PackedFunc 的单项返回值。
void DeviceAPI::GetAttr(const Device& device, DeviceAttrKind kind, RetValue* rv) {
    const DeviceAttributes attrs = GetDeviceAttributes(device);
    switch (kind) {
        case DeviceAttrKind::kExist: *rv = attrs.exists; return;
        case DeviceAttrKind::kMaxThreadsPerBlock: *rv = attrs.max_threads_per_block; return;
        case DeviceAttrKind::kWarpSize: *rv = attrs.warp_size; return;
        case DeviceAttrKind::kMaxSharedMemoryPerBlock: *rv = attrs.max_shared_memory_per_block; return;
        case DeviceAttrKind::kComputeVersion: *rv = attrs.compute_version; return;
        case DeviceAttrKind::kDeviceName: *rv = attrs.device_name; return;
        case DeviceAttrKind::kMaxClockRate: *rv = attrs.max_clock_rate_khz; return;
        case DeviceAttrKind::kMultiProcessorCount: *rv = attrs.multi_processor_count; return;
        case DeviceAttrKind::kMaxRegistersPerBlock: *rv = attrs.max_registers_per_block; return;
        case DeviceAttrKind::kApiVersion: *rv = attrs.api_version; return;
        case DeviceAttrKind::kDriverVersion: *rv = attrs.driver_version; return;
        case DeviceAttrKind::kL2CacheSizeBytes: *rv = attrs.l2_cache_size_bytes; return;
        case DeviceAttrKind::kTotalGlobalMemory: *rv = attrs.total_global_memory; return;
        case DeviceAttrKind::kAvailableGlobalMemory: *rv = attrs.available_global_memory; return;
        case DeviceAttrKind::kMaxSharedMemoryPerMultiprocessor: *rv = attrs.max_shared_memory_per_multiprocessor; return;
        case DeviceAttrKind::kMaxRegistersPerMultiprocessor: *rv = attrs.max_registers_per_multiprocessor; return;
        case DeviceAttrKind::kMaxThreadsPerMultiprocessor: *rv = attrs.max_threads_per_multiprocessor; return;
        case DeviceAttrKind::kComputeVersionMajor: *rv = attrs.compute_version_major; return;
        case DeviceAttrKind::kComputeVersionMinor: *rv = attrs.compute_version_minor; return;
    }
    throw std::runtime_error("Unknown DeviceAttrKind");
}

// 通过对应后端收集完整结构化设备属性。
DeviceAttributes CollectDeviceAttributes(const Device& device) {
    return GetDeviceAPI(device.device_type())->GetDeviceAttributes(device);
}

// 通过对应后端查询单项设备属性。
void GetDeviceAttr(const Device& device, DeviceAttrKind kind, RetValue* rv) {
    GetDeviceAPI(device.device_type())->GetAttr(device, kind, rv);
}

// 为整数属性提供直接返回的便捷入口。
int64_t GetDeviceAttr(const Device& device, DeviceAttrKind kind) {
    RetValue result;
    GetDeviceAttr(device, kind, &result);
    return result.As<int64_t>();
}

// 枚举 CPU 与 CUDA 诊断信息，并保留不可用后端的原因。
std::vector<DeviceInfo> GetAllDeviceInfo() {
    std::vector<DeviceInfo> infos{QueryDeviceInfo(kCPU, 0)};
#if KXC_USE_CUDA
    int count = 0;
    cudaError_t error = cudaGetDeviceCount(&count);
    if (error == cudaSuccess && count > 0) {
        for (int id = 0; id < count; ++id) infos.push_back(QueryDeviceInfo(kCUDA, id));
    } else {
        std::string status = error == cudaSuccess
                                 ? "no CUDA device detected"
                                 : "CUDA device query failed: " + std::string(cudaGetErrorString(error));
        infos.push_back(QueryDeviceInfo(kCUDA, 0, status));
    }
#else
    // CPU-only 构建仍保留 CUDA 的不可用诊断项，便于调用方区分未编译和无设备。
    infos.push_back(QueryDeviceInfo(kCUDA, 0,
                                    "CUDA support is disabled (KXC_USE_CUDA=0)"));
#endif
    return infos;
}

// 从完整诊断列表中过滤当前真实可用的设备。
std::vector<Device> ListDevices() {
    std::vector<Device> result;
    for (const auto& info : GetAllDeviceInfo()) {
        if (info.available) result.emplace_back(info.device_type, info.device_id);
    }
    return result;
}

// 将当前可用设备编码为 JSON。
std::string ListDevicesJSON() {
    std::vector<DeviceInfo> result;
    for (const auto& info : GetAllDeviceInfo()) if (info.available) result.push_back(info);
    return DeviceInfosToJSON(result);
}

// 将包含不可用项的完整设备诊断编码为 JSON。
std::string GetAllDeviceInfoJSON() { return DeviceInfosToJSON(GetAllDeviceInfo()); }

// 注册单项属性查询的 PackedFunc 入口。
KXC_REGISTER_GLOBAL("device_api.GetAttr")
    .set_body(PackedFunc(std::function<void(Args, RetValue*)>([](Args args, RetValue* rv) {
        Device device = ArgConverter<Device>::From(args[0], args.type_code(0));
        int64_t kind = ArgConverter<int64_t>::From(args[1], args.type_code(1));
        GetDeviceAttr(device, static_cast<DeviceAttrKind>(kind), rv);
    })));

// 注册仅返回可用设备的兼容入口及显式 JSON 别名。
KXC_REGISTER_GLOBAL("device_api.ListDevices")
    .set_body(PackedFunc(std::function<void(Args, RetValue*)>([](Args, RetValue* rv) {
        *rv = ListDevicesJSON();
    })));
KXC_REGISTER_GLOBAL("device_api.ListDevicesJSON")
    .set_body(PackedFunc(std::function<void(Args, RetValue*)>([](Args, RetValue* rv) {
        *rv = ListDevicesJSON();
    })));
// 注册返回完整设备诊断的兼容入口及显式 JSON 别名。
KXC_REGISTER_GLOBAL("device_api.GetAllDeviceInfo")
    .set_body(PackedFunc(std::function<void(Args, RetValue*)>([](Args, RetValue* rv) {
        *rv = GetAllDeviceInfoJSON();
    })));
KXC_REGISTER_GLOBAL("device_api.GetAllDeviceInfoJSON")
    .set_body(PackedFunc(std::function<void(Args, RetValue*)>([](Args, RetValue* rv) {
        *rv = GetAllDeviceInfoJSON();
    })));
// 注册设备到代码生成 target kind 的查询入口。
KXC_REGISTER_GLOBAL("device_api.GetTargetKind")
    .set_body(PackedFunc(std::function<void(Args, RetValue*)>([](Args args, RetValue* rv) {
        Device device = ArgConverter<Device>::From(args[0], args.type_code(0));
        *rv = GetDeviceAPI(device.device_type())->GetTargetKind(device);
    })));
// 注册设备体系结构字符串的查询入口。
KXC_REGISTER_GLOBAL("device_api.GetTargetArch")
    .set_body(PackedFunc(std::function<void(Args, RetValue*)>([](Args args, RetValue* rv) {
        Device device = ArgConverter<Device>::From(args[0], args.type_code(0));
        *rv = CollectDeviceAttributes(device).arch;
    })));

}  // namespace kxc
