/*! \file src/base/target.cc
 * \brief 实现基础对象、设备、NDArray、Target、执行计划、PassContext 和 profiling 支撑逻辑。
 */

#include "base/target.h"

#include <sstream>
#include <stdexcept>

namespace kxc {

const TargetNode* Target::operator->() const {
    return static_cast<const TargetNode*>(object_);
}

std::string Target::ToString() const {
    if (!defined()) {
        return "Target(undefined)";
    }
    const TargetNode* node = operator->();
    std::stringstream ss;
    ss << "Target(kind=" << node->kind
       << ", device_type=" << static_cast<int>(node->device_type)
       << ", device_id=" << node->device_id
       << ", arch=" << node->attrs.arch
       << ", compute_version=" << node->attrs.compute_version
       << ", max_threads=" << node->attrs.max_threads_per_block
       << ", warp_size=" << node->attrs.warp_size
       << ", shared_mem=" << node->attrs.max_shared_memory_per_block
       << ")";
    return ss.str();
}

Target BuildTarget(const class Device& device) {
    DeviceAPI* api = GetDeviceAPI(device.device_type());
    TargetNode* node = new TargetNode();
    node->kind = api->GetTargetKind(device);
    node->device_type = device.device_type();
    node->device_id = device.device_id();
    node->attrs = api->GetDeviceAttributes(device);
    return Target(ObjectRef(node));
}

Target BuildTarget(DeviceTypeCode type, int device_id) {
    ObjectRef dev_ref = DeviceManager::Global()->GetOrCreate(type, device_id);
    const Object* obj = dev_ref.get();
    if (!obj || obj->GetTypeId() != kKXC_DEVICE_TYPE) {
        throw std::runtime_error("BuildTarget expects a Device object");
    }
    return BuildTarget(*static_cast<const class Device*>(obj));
}

}  // namespace kxc

