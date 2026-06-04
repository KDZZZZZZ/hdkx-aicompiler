/*! \file include/base/disco_placement.h
 * \brief 定义基础对象系统、容器、设备、NDArray、Target、PassContext 和 profiling 公共类型。
 */

#pragma once

#include <string>

#include "base/container.h"
#include "base/target.h"
#include "base/virtual_device.h"

namespace kxc {

class WorkerPlacementNode : public Object {
public:
    int worker_id{-1};
    int group_id{0};
    int local_rank{-1};
    ObjectRef device_obj;
    Target target;
    VirtualDevice virtual_device;

    KXC_OBJECT_DECLARE

    bool has_device() const;
    bool has_target() const;
};

KXC_OBJECT_DEFINE(WorkerPlacementNode)

class WorkerPlacement : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    WorkerPlacement(const ObjectRef& ref) : ObjectRef(ref) {}
    WorkerPlacement(int worker_id, int group_id, int local_rank, ObjectRef device_obj,
                    Target target, VirtualDevice virtual_device);

    const WorkerPlacementNode* operator->() const;
    std::string ToString() const;
};

class DiscoPlacementNode : public Object {
public:
    Array<WorkerPlacement> workers;
    Map<VirtualDevice, int> vd_to_worker;
    int num_groups{1};

    KXC_OBJECT_DECLARE

    bool empty() const;
    int FindWorker(const VirtualDevice& virtual_device) const;
};

KXC_OBJECT_DEFINE(DiscoPlacementNode)

class DiscoPlacement : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    DiscoPlacement(const ObjectRef& ref) : ObjectRef(ref) {}
    DiscoPlacement(Array<WorkerPlacement> workers, Map<VirtualDevice, int> vd_to_worker,
                   int num_groups = 1);

    const DiscoPlacementNode* operator->() const;
    bool empty() const;
    int FindWorker(const VirtualDevice& virtual_device) const;
    std::string ToString() const;
};

DiscoPlacement BuildDiscoPlacement(const Array<VirtualDevice>& virtual_devices,
                                   int num_groups = 1);
int FindWorkerForVirtualDevice(const DiscoPlacement& placement,
                               const VirtualDevice& virtual_device);

}  // namespace kxc

