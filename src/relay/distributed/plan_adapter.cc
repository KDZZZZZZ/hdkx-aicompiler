/*! \file src/relay/distributed/plan_adapter.cc
 * \brief Converts Relay communication attrs to the distributed plan schema.
 */

#include "plan_adapter.h"

#include "kxc/relay/op.h"

namespace kxc::relay::distributed_internal {

CommExecAttrs AdaptCommExecAttrs(const std::string& op_name,
                                 const ObjectRef& attrs) {
    CommExecAttrs result;
    result.kind = op_name;
    if (const auto* copy = attrs.As<DeviceCopyAttrsNode>()) {
        result.src_virtual_device = copy->src_virtual_device;
        result.dst_virtual_device = copy->dst_virtual_device;
        result.async = copy->async;
        result.in_group = copy->in_group;
    } else if (const auto* collective = attrs.As<CollectiveAttrsNode>()) {
        result.kind = collective->kind;
        result.reduce_kind = collective->reduce_kind;
        result.in_group = collective->in_group;
        result.group_id = collective->group_id;
        result.root_worker = collective->root_worker;
    }
    return result;
}

}  // namespace kxc::relay::distributed_internal
