/*! \file src/base/device.cc
 * \brief 实现不可变物理设备身份、规范驻留和 DLPack 映射。
 */

#include "base/device.h"

#include <sstream>
#include <stdexcept>

namespace kxc {

namespace {

// 校验可以进入规范驻留表的物理设备身份。
void ValidateDeviceIdentity(DeviceTypeCode type, int id) {
    if (id < 0) {
        throw std::invalid_argument("Device id must be non-negative");
    }
    if (type == kCPU && id != 0) {
        throw std::invalid_argument("Only cpu:0 is supported");
    }
    if (type == kUnknown) {
        throw std::invalid_argument("Cannot construct an unknown Device");
    }
    if (type != kCPU && type != kCUDA && type != kOpenCL && type != kMetal) {
        throw std::invalid_argument("Unsupported DeviceTypeCode");
    }
}

// 返回稳定的设备类型文本，供显示和序列化使用。
const char* DeviceTypeName(DeviceTypeCode type) {
    switch (type) {
        case kCPU:
            return "cpu";
        case kCUDA:
            return "cuda";
        case kOpenCL:
            return "opencl";
        case kMetal:
            return "metal";
        case kUnknown:
            break;
    }
    return "unknown";
}

}  // namespace

// 所有公开构造都经 DeviceManager 复用规范对象。
Device::Device(DeviceTypeCode type, int id)
    : ObjectRef(DeviceManager::Global()->Get(type, id)) {}

// 从通用对象引用恢复 Device，并执行运行时类型检查。
Device::Device(const ObjectRef& ref) : ObjectRef(ref) {
    if (defined() && !ref.As<DeviceNode>()) {
        SetData(nullptr);
        throw std::invalid_argument("ObjectRef does not contain a Device");
    }
}

// 获取规范驻留的 CPU 设备。
Device Device::CPU(int id) {
    return DeviceManager::Global()->Get(kCPU, id);
}

// 获取规范驻留的 CUDA 设备；实际可用性由 DeviceAPI 查询。
Device Device::CUDA(int id) {
    return DeviceManager::Global()->Get(kCUDA, id);
}

// 返回设备类型，未定义对象不隐式解释为任何后端。
DeviceTypeCode Device::device_type() const {
    if (!defined()) throw std::runtime_error("Undefined Device has no type");
    return operator->()->device_type;
}

// 返回物理设备编号。
int Device::device_id() const {
    if (!defined()) throw std::runtime_error("Undefined Device has no id");
    return operator->()->device_id;
}

// 生成稳定的人类可读设备名。
std::string Device::ToString() const {
    if (!defined()) return "Device(undefined)";
    std::ostringstream os;
    os << DeviceTypeName(device_type()) << ':' << device_id();
    return os.str();
}

// 按物理身份比较设备，而非依赖对象地址。
bool Device::operator==(const Device& other) const {
    if (!defined() || !other.defined()) return defined() == other.defined();
    return device_type() == other.device_type() && device_id() == other.device_id();
}

// 复用物理身份相等性实现不等判断。
bool Device::operator!=(const Device& other) const {
    return !(*this == other);
}

// 返回经过 Device 类型约束的底层节点。
const DeviceNode* Device::operator->() const {
    return static_cast<const DeviceNode*>(object_);
}

// 返回进程级规范驻留管理器；静态局部变量保证线程安全初始化。
DeviceManager* DeviceManager::Global() {
    static DeviceManager manager;
    return &manager;
}

// 按 (type, id) 查找或创建强驻留的唯一 Device 对象。
Device DeviceManager::Get(DeviceTypeCode type, int id) {
    ValidateDeviceIdentity(type, id);
    const Key key{type, id};
    // 查找和创建必须位于同一临界区，确保同一物理设备只有一个规范节点。
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = devices_.find(key);
    if (it != devices_.end()) return it->second;

    auto* node = new DeviceNode();
    node->device_type = type;
    node->device_id = id;
    Device device{ObjectRef(node)};
    devices_.emplace(key, device);
    return device;
}

// 通过显式白名单映射内部设备类型，避免误认为两个枚举可以直接强转。
DLDeviceType ToDLDeviceType(DeviceTypeCode type) {
    switch (type) {
        case kCPU:
            return kDLCPU;
        case kCUDA:
            return kDLCUDA;
        case kOpenCL:
            return kDLOpenCL;
        case kMetal:
            return kDLMetal;
        case kUnknown:
            break;
    }
    throw std::invalid_argument("Device type has no DLPack mapping");
}

// 将受支持的 DLPack 类型恢复为内部设备类型。
DeviceTypeCode FromDLDeviceType(DLDeviceType type) {
    switch (type) {
        case kDLCPU:
            return kCPU;
        case kDLCUDA:
            return kCUDA;
        case kDLOpenCL:
            return kOpenCL;
        case kDLMetal:
            return kMetal;
        default:
            throw std::invalid_argument("Unsupported DLPack device type");
    }
}

// 将已定义的 Device 转为 DLPack 值类型描述。
DLDevice ToDLDevice(const Device& device) {
    if (!device.defined()) {
        throw std::invalid_argument("Cannot map an undefined Device to DLPack");
    }
    return DLDevice{ToDLDeviceType(device.device_type()), device.device_id()};
}

// 将 DLPack 描述重新纳入 DeviceManager 的规范驻留集合。
Device FromDLDevice(const DLDevice& device) {
    return DeviceManager::Global()->Get(FromDLDeviceType(device.device_type),
                                        device.device_id);
}

}  // namespace kxc
