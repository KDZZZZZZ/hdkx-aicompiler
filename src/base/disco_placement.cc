/*! \file src/base/disco_placement.cc
 * \brief 实现基础对象、设备、NDArray、Target、执行计划、PassContext 和 profiling 支撑逻辑。
 */

#include "base/disco_placement.h"

#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "base/device.h"

namespace kxc {

namespace {

std::string VirtualDeviceIdentity(const VirtualDevice& vd) {
    if (!vd.defined()) {
        return "virtual_device:undefined";
    }
    std::stringstream ss;
    ss << "virtual_device:";
    if (vd->device_obj.defined()) {
        const auto* dev = static_cast<const class Device*>(vd->device_obj.get());
        if (dev) {
            ss << "dev(" << static_cast<int>(dev->device_type()) << "," << dev->device_id() << ")";
        }
    } else {
        ss << "dev(none)";
    }
    if (vd->target.defined()) {
        ss << "target(" << static_cast<int>(vd->target->device_type) << ","
           << vd->target->device_id << "," << vd->target->kind << ")";
    } else {
        ss << "target(none)";
    }
    ss << "scope(" << vd->memory_scope << ")";
    ss << "vdid(" << vd->virtual_device_id << ")";
    return ss.str();
}

}  // namespace

bool WorkerPlacementNode::has_device() const {
    return device_obj.defined();
}

bool WorkerPlacementNode::has_target() const {
    return target.defined();
}

WorkerPlacement::WorkerPlacement(int worker_id, int group_id, int local_rank, ObjectRef device_obj,
                                 Target target, VirtualDevice virtual_device) {
    WorkerPlacementNode* node = new WorkerPlacementNode();
    node->worker_id = worker_id;
    node->group_id = group_id;
    node->local_rank = local_rank;
    node->device_obj = std::move(device_obj);
    node->target = std::move(target);
    node->virtual_device = std::move(virtual_device);
    SetData(node);
}

const WorkerPlacementNode* WorkerPlacement::operator->() const {
    return static_cast<const WorkerPlacementNode*>(object_);
}

std::string WorkerPlacement::ToString() const {
    if (!defined()) {
        return "WorkerPlacement(undefined)";
    }
    std::stringstream ss;
    ss << "WorkerPlacement(worker=" << operator->()->worker_id
       << ", group=" << operator->()->group_id
       << ", local_rank=" << operator->()->local_rank;
    if (operator->()->target.defined()) {
        ss << ", target=" << operator->()->target.ToString();
    }
    if (operator->()->virtual_device.defined()) {
        ss << ", virtual_device=" << operator->()->virtual_device.ToString();
    }
    ss << ")";
    return ss.str();
}

bool DiscoPlacementNode::empty() const {
    return workers.empty();
}

int DiscoPlacementNode::FindWorker(const VirtualDevice& virtual_device) const {
    if (!virtual_device.defined()) {
        return -1;
    }
    if (vd_to_worker.count(virtual_device)) {
        return vd_to_worker.at(virtual_device);
    }
    std::string key = VirtualDeviceIdentity(virtual_device);
    for (const auto& worker : workers) {
        if (!worker.defined() || !worker->virtual_device.defined()) {
            continue;
        }
        if (VirtualDeviceIdentity(worker->virtual_device) == key) {
            return worker->worker_id;
        }
    }
    return -1;
}

DiscoPlacement::DiscoPlacement(Array<WorkerPlacement> workers, Map<VirtualDevice, int> vd_to_worker,
                               int num_groups) {
    DiscoPlacementNode* node = new DiscoPlacementNode();
    node->workers = std::move(workers);
    node->vd_to_worker = std::move(vd_to_worker);
    node->num_groups = num_groups > 0 ? num_groups : 1;
    SetData(node);
}

const DiscoPlacementNode* DiscoPlacement::operator->() const {
    return static_cast<const DiscoPlacementNode*>(object_);
}

bool DiscoPlacement::empty() const {
    return !defined() || operator->()->empty();
}

int DiscoPlacement::FindWorker(const VirtualDevice& virtual_device) const {
    if (!defined()) {
        return -1;
    }
    return operator->()->FindWorker(virtual_device);
}

std::string DiscoPlacement::ToString() const {
    if (!defined()) {
        return "DiscoPlacement(undefined)";
    }
    std::stringstream ss;
    ss << "DiscoPlacement(num_workers=" << operator->()->workers.size()
       << ", num_groups=" << operator->()->num_groups << ")";
    return ss.str();
}

DiscoPlacement BuildDiscoPlacement(const Array<VirtualDevice>& virtual_devices, int num_groups) {
    Array<WorkerPlacement> workers;
    Map<VirtualDevice, int> vd_to_worker;
    std::unordered_map<std::string, int> key_to_worker;

    int next_worker_id = 0;
    for (const auto& vd : virtual_devices) {
        if (!vd.defined()) {
            continue;
        }
        std::string key = VirtualDeviceIdentity(vd);
        auto it = key_to_worker.find(key);
        if (it != key_to_worker.end()) {
            vd_to_worker.Set(vd, it->second);
            continue;
        }

        Target target = vd->target;
        if (!target.defined() && vd->device_obj.defined()) {
            const auto* dev = static_cast<const class Device*>(vd->device_obj.get());
            if (dev) {
                target = BuildTarget(*dev);
            }
        }

        int worker_id = next_worker_id++;
        int group_id = 0;
        int local_rank = worker_id;
        workers.push_back(WorkerPlacement(worker_id, group_id, local_rank, vd->device_obj, target,
                                          vd));
        vd_to_worker.Set(vd, worker_id);
        key_to_worker[key] = worker_id;
    }

    return DiscoPlacement(workers, vd_to_worker, num_groups);
}

int FindWorkerForVirtualDevice(const DiscoPlacement& placement,
                               const VirtualDevice& virtual_device) {
    if (!placement.defined()) {
        return -1;
    }
    return placement.FindWorker(virtual_device);
}

}  // namespace kxc
