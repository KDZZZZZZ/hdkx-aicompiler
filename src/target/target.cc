/*! \file src/target/target.cc
 * \brief 实现基础对象、设备、NDArray、Target、执行计划、PassContext 和 profiling 支撑逻辑。
 */

#include "kxc/target/target.h"
#include "kxc/support/object_registration.h"
#include "kxc/runtime/device_api.h"

#include <sstream>
#include <stdexcept>

namespace kxc {

KXC_OBJECT_DEFINE(TargetNode)

// 返回经过 Target 类型约束的底层节点。
const TargetNode* Target::operator->() const {
    return static_cast<const TargetNode*>(object_);
}

// 输出 target kind、设备身份及编译所需硬件能力快照。
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

// 从物理设备后端构造用于编译决策的 Target 能力快照。
Target BuildTarget(const Device& device) {
    // Target 的 kind 与硬件属性由对应 DeviceAPI 查询，避免编译端维护重复的设备表。
    DeviceAPI* api = GetDeviceAPI(device.device_type());
    TargetNode* node = new TargetNode();
    node->kind = api->GetTargetKind(device);
    node->device_type = device.device_type();
    node->device_id = device.device_id();
    node->attrs = api->GetDeviceAttributes(device);
    return Target(ObjectRef(node));
}

}  // namespace kxc
