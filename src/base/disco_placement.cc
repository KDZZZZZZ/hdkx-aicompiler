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

// 生成值语义身份键，使等价但不同 ObjectRef 的逻辑设备可映射到同一 worker。
std::string VirtualDeviceIdentity(const VirtualDevice& vd) {
    if (!vd.defined()) {
        return "virtual_device:undefined";
    }
    std::stringstream ss;
    ss << "virtual_device:";
    if (vd->device.defined()) {
        ss << "dev(" << static_cast<int>(vd->device.device_type()) << ","
           << vd->device.device_id() << ")";
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

// 判断 worker 是否已绑定物理设备。
bool WorkerPlacementNode::has_device() const {
    return device.defined();
}

// 判断 worker 是否已有代码生成目标。
bool WorkerPlacementNode::has_target() const {
    return target.defined();
}

// 构造单个 worker 的组身份、物理设备和逻辑放置记录。
WorkerPlacement::WorkerPlacement(int worker_id, int group_id, int local_rank, Device device,
                                 Target target, VirtualDevice virtual_device) {
    // 直接保存强类型 Device，防止任意 ObjectRef 绕过设备类型与身份校验。
    WorkerPlacementNode* node = new WorkerPlacementNode();
    node->worker_id = worker_id;
    node->group_id = group_id;
    node->local_rank = local_rank;
    node->device = std::move(device);
    node->target = std::move(target);
    node->virtual_device = std::move(virtual_device);
    SetData(node);
}

// 返回经过 WorkerPlacement 类型约束的底层节点。
const WorkerPlacementNode* WorkerPlacement::operator->() const {
    return static_cast<const WorkerPlacementNode*>(object_);
}

// 输出 worker 身份及其可用放置约束。
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

// 判断放置表是否包含任何 worker。
bool DiscoPlacementNode::empty() const {
    return workers.empty();
}

// 优先按 ObjectRef 映射查找，再按值语义身份匹配等价逻辑设备。
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

// 构造逻辑设备到 worker 的完整放置表，并规范化组数下限。
DiscoPlacement::DiscoPlacement(Array<WorkerPlacement> workers, Map<VirtualDevice, int> vd_to_worker,
                               int num_groups) {
    DiscoPlacementNode* node = new DiscoPlacementNode();
    node->workers = std::move(workers);
    node->vd_to_worker = std::move(vd_to_worker);
    node->num_groups = num_groups > 0 ? num_groups : 1;
    SetData(node);
}

// 返回经过 DiscoPlacement 类型约束的底层节点。
const DiscoPlacementNode* DiscoPlacement::operator->() const {
    return static_cast<const DiscoPlacementNode*>(object_);
}

// 未定义放置与无 worker 放置都按空表处理。
bool DiscoPlacement::empty() const {
    return !defined() || operator->()->empty();
}

// 在已定义放置表中查询逻辑设备对应 worker。
int DiscoPlacement::FindWorker(const VirtualDevice& virtual_device) const {
    if (!defined()) {
        return -1;
    }
    return operator->()->FindWorker(virtual_device);
}

// 输出放置表规模和组数摘要。
std::string DiscoPlacement::ToString() const {
    if (!defined()) {
        return "DiscoPlacement(undefined)";
    }
    std::stringstream ss;
    ss << "DiscoPlacement(num_workers=" << operator->()->workers.size()
       << ", num_groups=" << operator->()->num_groups << ")";
    return ss.str();
}

// 对等价逻辑设备去重，并为每个唯一放置分配稳定 worker 编号。
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
        if (!target.defined() && vd->device.defined()) {
            // Target 缺失时由物理 Device 的后端能力构造，而不是仅凭类型编号猜测。
            target = BuildTarget(vd->device);
        }

        int worker_id = next_worker_id++;
        int group_id = 0;
        int local_rank = worker_id;
        workers.push_back(WorkerPlacement(worker_id, group_id, local_rank, vd->device, target,
                                          vd));
        vd_to_worker.Set(vd, worker_id);
        key_to_worker[key] = worker_id;
    }

    return DiscoPlacement(workers, vd_to_worker, num_groups);
}

// 提供可接受未定义 placement 的安全查询入口。
int FindWorkerForVirtualDevice(const DiscoPlacement& placement,
                               const VirtualDevice& virtual_device) {
    if (!placement.defined()) {
        return -1;
    }
    return placement.FindWorker(virtual_device);
}

}  // namespace kxc
