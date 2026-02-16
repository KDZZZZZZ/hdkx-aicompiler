#include "base/virtual_device.h"

#include "base/registry.h"

#include <sstream>
#include <stdexcept>

namespace kxc {

namespace {

const class Device* AsDevice(const ObjectRef& obj) {
    if (!obj.defined()) {
        return nullptr;
    }
    const Object* raw = obj.get();
    if (!raw || raw->GetTypeId() != kKXC_DEVICE_TYPE) {
        throw std::runtime_error("VirtualDevice.device_obj is not a Device");
    }
    return static_cast<const class Device*>(raw);
}

ObjectRef CloneDevice(const class Device& device) {
    return ObjectRef(new class Device(device));
}

Target AsTarget(const ObjectRef& obj) {
    if (!obj.defined()) {
        return Target();
    }
    const Object* raw = obj.get();
    if (!raw || raw->GetTypeId() != TargetNode::_type_index) {
        throw std::runtime_error("Expected Target object");
    }
    return Target(obj);
}

void ValidateTargetDevicePair(const ObjectRef& device_obj, const Target& target) {
    if (!device_obj.defined() || !target.defined()) {
        return;
    }
    const class Device* dev = AsDevice(device_obj);
    if (!dev) {
        return;
    }
    if (target->device_type != kUnknown && target->device_type != dev->device_type()) {
        throw std::runtime_error("VirtualDevice device/target mismatch in device_type");
    }
    if (target->device_id >= 0 && target->device_id != dev->device_id()) {
        throw std::runtime_error("VirtualDevice device/target mismatch in device_id");
    }
}

}  // namespace

bool VirtualDeviceNode::has_device() const {
    return device_obj.defined();
}

bool VirtualDeviceNode::has_target() const {
    return target.defined();
}

bool VirtualDeviceNode::IsFullyUnconstrained() const {
    return !device_obj.defined() && !target.defined() && memory_scope.empty() &&
           virtual_device_id == kInvalidVirtualDeviceId;
}

bool VirtualDeviceNode::IsFullyConstrained() const {
    return device_obj.defined() && target.defined() &&
           virtual_device_id != kInvalidVirtualDeviceId;
}

class Device VirtualDeviceNode::device() const {
    const class Device* dev = AsDevice(device_obj);
    if (!dev) {
        throw std::runtime_error("VirtualDevice does not contain a Device");
    }
    return *dev;
}

VirtualDevice::VirtualDevice(const class Device& device, Target target,
                             std::string memory_scope, int virtual_device_id) {
    VirtualDeviceNode* node = new VirtualDeviceNode();
    node->device_obj = CloneDevice(device);
    node->target = std::move(target);
    node->memory_scope = std::move(memory_scope);
    node->virtual_device_id =
        (virtual_device_id == kInvalidVirtualDeviceId) ? device.device_id() : virtual_device_id;
    ValidateTargetDevicePair(node->device_obj, node->target);
    SetData(node);
}

VirtualDevice::VirtualDevice(Target target, std::string memory_scope, int virtual_device_id) {
    VirtualDeviceNode* node = new VirtualDeviceNode();
    node->target = std::move(target);
    node->memory_scope = std::move(memory_scope);
    node->virtual_device_id = virtual_device_id;
    if (node->virtual_device_id == kInvalidVirtualDeviceId && node->target.defined()) {
        node->virtual_device_id = node->target->device_id;
    }
    SetData(node);
}

const VirtualDeviceNode* VirtualDevice::operator->() const {
    return static_cast<const VirtualDeviceNode*>(object_);
}

bool VirtualDevice::IsFullyUnconstrained() const {
    return !defined() || operator->()->IsFullyUnconstrained();
}

bool VirtualDevice::IsFullyConstrained() const {
    return defined() && operator->()->IsFullyConstrained();
}

class Device VirtualDevice::device() const {
    if (!defined()) {
        throw std::runtime_error("Undefined VirtualDevice has no Device");
    }
    return operator->()->device();
}

std::string VirtualDevice::ToString() const {
    if (!defined()) {
        return "VirtualDevice(undefined)";
    }
    const VirtualDeviceNode* n = operator->();
    std::stringstream ss;
    ss << "VirtualDevice(";
    if (n->has_device()) {
        ss << n->device().ToString();
    } else {
        ss << "device=none";
    }
    ss << ", target=";
    if (n->has_target()) {
        ss << n->target.ToString();
    } else {
        ss << "none";
    }
    ss << ", memory_scope=" << (n->memory_scope.empty() ? "\"\"" : n->memory_scope)
       << ", virtual_device_id=" << n->virtual_device_id << ")";
    return ss.str();
}

VirtualDevice VirtualDevice::FullyUnconstrained() {
    VirtualDeviceNode* node = new VirtualDeviceNode();
    return VirtualDevice(ObjectRef(node));
}

VirtualDevice VirtualDevice::ForDevice(const class Device& device) {
    return VirtualDevice(device, Target(), "", device.device_id());
}

VirtualDevice VirtualDevice::ForTarget(const Target& target) {
    return VirtualDevice(target, "", target.defined() ? target->device_id : kInvalidVirtualDeviceId);
}

VirtualDevice VirtualDevice::ForDeviceAndTarget(const class Device& device, const Target& target) {
    return VirtualDevice(device, target, "", device.device_id());
}

KXC_REGISTER_GLOBAL("virtual_device.FullyUnconstrained")
    .set_body(ToPackedFunc([]() -> ObjectRef { return VirtualDevice::FullyUnconstrained(); }));

KXC_REGISTER_GLOBAL("virtual_device.ForDevice")
    .set_body(ToPackedFunc([](ObjectRef device_ref) -> ObjectRef {
        const class Device* device = AsDevice(device_ref);
        if (!device) {
            throw std::runtime_error("virtual_device.ForDevice expects a defined Device");
        }
        return VirtualDevice::ForDevice(*device);
    }));

KXC_REGISTER_GLOBAL("virtual_device.ForTarget")
    .set_body(ToPackedFunc([](ObjectRef target_ref) -> ObjectRef {
        Target target = AsTarget(target_ref);
        return VirtualDevice::ForTarget(target);
    }));

KXC_REGISTER_GLOBAL("virtual_device.ForDeviceAndTarget")
    .set_body(ToPackedFunc([](ObjectRef device_ref, ObjectRef target_ref) -> ObjectRef {
        const class Device* device = AsDevice(device_ref);
        if (!device) {
            throw std::runtime_error("virtual_device.ForDeviceAndTarget expects a defined Device");
        }
        Target target = AsTarget(target_ref);
        return VirtualDevice::ForDeviceAndTarget(*device, target);
    }));

}  // namespace kxc
