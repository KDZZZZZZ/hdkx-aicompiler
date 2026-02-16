#include "base/device.h"

namespace kxc {

Device::Device(DeviceTypeCode type, int id) : device_type_(type), device_id_(id) {}

DeviceTypeCode Device::device_type() const { return device_type_; }

int Device::device_id() const { return device_id_; }

std::string Device::ToString() const {
    std::stringstream ss;
    ss << "Device(";
    switch (device_type_) {
        case kCPU:
            ss << "CPU";
            break;
        case kGPU:
            ss << "GPU";
            break;
        case kOpenCL:
            ss << "OpenCL";
            break;
        case kMetal:
            ss << "Metal";
            break;
        case kUnknown:
            ss << "Unknown";
            break;
    }
    ss << ", id=" << device_id_ << ")";
    return ss.str();
}

bool Device::operator==(const Device& other) const {
    return device_type_ == other.device_type_ && device_id_ == other.device_id_;
}

DeviceManager* DeviceManager::Global() {
    static DeviceManager manager;
    return &manager;
}

ObjectRef DeviceManager::GetOrCreate(DeviceTypeCode type, int id) {
    std::string key = std::to_string(type) + "_" + std::to_string(id);

    auto it = devices_.find(key);
    if (it != devices_.end()) {
        return it->second;
    }

    ObjectRef new_device_ref(new class Device(type, id));
    devices_[key] = new_device_ref;
    return new_device_ref;
}

ObjectRef Device(DeviceTypeCode type, int id) {
    return DeviceManager::Global()->GetOrCreate(type, id);
}

}  // namespace kxc

