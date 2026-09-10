/*! \file src/target/target.cc
 * \brief 实现基础对象、设备、NDArray、Target、执行计划、PassContext 和 profiling 支撑逻辑。
 */

#include "kxc/target/target.h"
#include "kxc/support/object_registration.h"
#include "kxc/runtime/device_api.h"
#include "support/canonical.h"

#include <sstream>
#include <stdexcept>
#include <utility>

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

namespace {
void AppendPipelineIdentityField(std::string* canonical,
                                 const std::string& name,
                                 const std::string& value) {
    support::CanonicalBytesEncoder field;
    field.Field(name, value);
    *canonical += std::move(field).Take();
}
}  // namespace

std::string Target::CanonicalBytes() const {
    const auto* node = As<TargetNode>();
    if (!node) {
        throw std::invalid_argument(
            "CanonicalTargetSnapshot requires a defined TargetNode");
    }
    std::string canonical;
    AppendPipelineIdentityField(&canonical, "kind", "target-snapshot-v1");
    AppendPipelineIdentityField(&canonical, "target_kind", node->kind);
    AppendPipelineIdentityField(
        &canonical, "device_type",
        std::to_string(static_cast<int>(node->device_type)));
    AppendPipelineIdentityField(&canonical, "device_id",
                                std::to_string(node->device_id));
    const DeviceAttributes& attrs = node->attrs;
    const auto append_integer = [&canonical](const char* name, int64_t value) {
        AppendPipelineIdentityField(&canonical, name, std::to_string(value));
    };
    append_integer("exists", attrs.exists);
    append_integer("max_threads_per_block", attrs.max_threads_per_block);
    append_integer("warp_size", attrs.warp_size);
    append_integer("max_shared_memory_per_block",
                   attrs.max_shared_memory_per_block);
    AppendPipelineIdentityField(&canonical, "compute_version",
                                attrs.compute_version);
    AppendPipelineIdentityField(&canonical, "device_name", attrs.device_name);
    append_integer("max_clock_rate_khz", attrs.max_clock_rate_khz);
    append_integer("max_registers_per_block", attrs.max_registers_per_block);
    append_integer("api_version", attrs.api_version);
    append_integer("driver_version", attrs.driver_version);
    append_integer("l2_cache_size_bytes", attrs.l2_cache_size_bytes);
    append_integer("total_global_memory", attrs.total_global_memory);
    // available_global_memory is a volatile observation, not a codegen
    // capability. It is deliberately excluded from reusable identity.
    append_integer("max_shared_memory_per_multiprocessor",
                   attrs.max_shared_memory_per_multiprocessor);
    append_integer("max_registers_per_multiprocessor",
                   attrs.max_registers_per_multiprocessor);
    append_integer("max_threads_per_multiprocessor",
                   attrs.max_threads_per_multiprocessor);
    append_integer("compute_version_major", attrs.compute_version_major);
    append_integer("compute_version_minor", attrs.compute_version_minor);
    append_integer("multi_processor_count", attrs.multi_processor_count);
    AppendPipelineIdentityField(&canonical, "arch", attrs.arch);
    return canonical;
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
