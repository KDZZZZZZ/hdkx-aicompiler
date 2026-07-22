/*! \file src/target/virtual_device.cc
 * \brief 实现基础对象、设备、NDArray、Target、执行计划、PassContext 和 profiling 支撑逻辑。
 */

#include "kxc/target/virtual_device.h"
#include "kxc/support/object_registration.h"

#include "kxc/ffi/registration.h"

#include <sstream>
#include <stdexcept>

namespace kxc {

KXC_OBJECT_DEFINE(VirtualDeviceNode)

namespace {

// 从可选 ObjectRef 恢复 Target，并拒绝其他对象类型。
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

// 当两侧约束同时存在时，验证逻辑放置中的物理身份一致。
void ValidateTargetDevicePair(const Device& device, const Target& target) {
    if (!device.defined() || !target.defined()) {
        return;
    }
    if (target->device_type != kUnknown && target->device_type != device.device_type()) {
        throw std::runtime_error("VirtualDevice device/target mismatch in device_type");
    }
    if (target->device_id >= 0 && target->device_id != device.device_id()) {
        throw std::runtime_error("VirtualDevice device/target mismatch in device_id");
    }
}

}  // namespace

// 判断逻辑放置是否绑定物理 Device。
bool VirtualDeviceNode::has_device() const {
    return device.defined();
}

// 判断逻辑放置是否携带编译 Target。
bool VirtualDeviceNode::has_target() const {
    return target.defined();
}

// 判断所有放置维度是否均未施加约束。
bool VirtualDeviceNode::IsFullyUnconstrained() const {
    return !device.defined() && !target.defined() && memory_scope.empty() &&
           virtual_device_id == kInvalidVirtualDeviceId;
}

// 判断执行设备与编译目标是否都已确定。
bool VirtualDeviceNode::IsFullyConstrained() const {
    // 逻辑放置是否完整由 Device 与 Target 决定，不要求复用物理 device_id 作为逻辑 ID。
    return device.defined() && target.defined();
}

// 构造可同时包含物理 Device、Target、内存域与逻辑编号的放置约束。
VirtualDevice::VirtualDevice(const Device& device, Target target,
                             std::string memory_scope, int virtual_device_id) {
    VirtualDeviceNode* node = new VirtualDeviceNode();
    node->device = device;
    node->target = std::move(target);
    node->memory_scope = std::move(memory_scope);
    // virtual_device_id 是编译期逻辑身份，与物理 Device::device_id 保持解耦。
    node->virtual_device_id = virtual_device_id;
    ValidateTargetDevicePair(node->device, node->target);
    SetData(node);
}

// 构造尚未绑定物理 Device 的 Target 侧放置约束。
VirtualDevice::VirtualDevice(Target target, std::string memory_scope, int virtual_device_id) {
    VirtualDeviceNode* node = new VirtualDeviceNode();
    node->target = std::move(target);
    node->memory_scope = std::move(memory_scope);
    node->virtual_device_id = virtual_device_id;
    SetData(node);
}

// 返回经过 VirtualDevice 类型约束的底层节点。
const VirtualDeviceNode* VirtualDevice::operator->() const {
    return static_cast<const VirtualDeviceNode*>(object_);
}

// 将未定义句柄也视为无约束放置。
bool VirtualDevice::IsFullyUnconstrained() const {
    return !defined() || operator->()->IsFullyUnconstrained();
}

// 判断已定义对象是否同时包含 Device 与 Target。
bool VirtualDevice::IsFullyConstrained() const {
    return defined() && operator->()->IsFullyConstrained();
}

// 返回已绑定的物理 Device，缺失约束时明确失败。
Device VirtualDevice::device() const {
    if (!defined()) {
        throw std::runtime_error("Undefined VirtualDevice has no Device");
    }
    if (!operator->()->device.defined()) {
        throw std::runtime_error("VirtualDevice does not contain a Device");
    }
    return operator->()->device;
}

// 输出逻辑放置的各维约束，便于 Pass 和计划诊断。
std::string VirtualDevice::ToString() const {
    if (!defined()) {
        return "VirtualDevice(undefined)";
    }
    const VirtualDeviceNode* n = operator->();
    std::stringstream ss;
    ss << "VirtualDevice(";
    if (n->has_device()) {
        ss << n->device.ToString();
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

// 创建显式的全无约束逻辑设备对象。
VirtualDevice VirtualDevice::FullyUnconstrained() {
    VirtualDeviceNode* node = new VirtualDeviceNode();
    return VirtualDevice(ObjectRef(node));
}

// 创建仅绑定物理设备的逻辑放置。
VirtualDevice VirtualDevice::ForDevice(const Device& device) {
    return VirtualDevice(device);
}

// 创建仅绑定编译目标的逻辑放置。
VirtualDevice VirtualDevice::ForTarget(const Target& target) {
    return VirtualDevice(target);
}

// 创建物理设备与编译目标均已确定的逻辑放置。
VirtualDevice VirtualDevice::ForDeviceAndTarget(const Device& device, const Target& target) {
    return VirtualDevice(device, target);
}

// 注册 VirtualDevice 工厂函数的 PackedFunc 入口。
KXC_REGISTER_GLOBAL("virtual_device.FullyUnconstrained")
    .set_body(ToPackedFunc([]() -> ObjectRef { return VirtualDevice::FullyUnconstrained(); }));

KXC_REGISTER_GLOBAL("virtual_device.ForDevice")
    .set_body(ToPackedFunc([](ObjectRef device_ref) -> ObjectRef {
        Device device(device_ref);
        if (!device.defined()) {
            throw std::runtime_error("virtual_device.ForDevice expects a defined Device");
        }
        return VirtualDevice::ForDevice(device);
    }));

KXC_REGISTER_GLOBAL("virtual_device.ForTarget")
    .set_body(ToPackedFunc([](ObjectRef target_ref) -> ObjectRef {
        Target target = AsTarget(target_ref);
        return VirtualDevice::ForTarget(target);
    }));

KXC_REGISTER_GLOBAL("virtual_device.ForDeviceAndTarget")
    .set_body(ToPackedFunc([](ObjectRef device_ref, ObjectRef target_ref) -> ObjectRef {
        Device device(device_ref);
        if (!device.defined()) {
            throw std::runtime_error("virtual_device.ForDeviceAndTarget expects a defined Device");
        }
        Target target = AsTarget(target_ref);
        return VirtualDevice::ForDeviceAndTarget(device, target);
    }));

}  // namespace kxc

namespace kxc::builtin_anchor {
void TargetVirtualDevice() {}
}  // namespace kxc::builtin_anchor
