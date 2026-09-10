/*! \file src/distributed/placement.cc
 * \brief 实现基础对象、设备、NDArray、Target、执行计划、PassContext 和 profiling 支撑逻辑。
 */

#include "kxc/distributed/placement.h"
#include "kxc/support/object_registration.h"

#include <algorithm>
#include <map>
#include <memory>
#include <utility>
#include <vector>
#include <sstream>

#include "support/canonical.h"
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "kxc/runtime/device.h"

namespace kxc {

KXC_OBJECT_DEFINE(WorkerPlacementNode)
KXC_OBJECT_DEFINE(DiscoPlacementNode)

namespace {

// 生成值语义身份键，使等价但不同 ObjectRef 的逻辑设备可映射到同一 worker。
std::string VirtualDeviceIdentity(const VirtualDevice& vd) {
    const auto* node = vd.As<VirtualDeviceNode>();
    if (!node || !node->device.defined()) throw std::invalid_argument("placement requires a physical VirtualDevice");
    const Target target = node->target.defined() ? node->target : BuildTarget(node->device);
    if (!target.As<TargetNode>()) throw std::invalid_argument("placement Target has the wrong node type");
    if (target->device_type != node->device.device_type() || target->device_id != node->device.device_id()) {
        throw std::invalid_argument("placement VirtualDevice target/device mismatch");
    }
    support::CanonicalBytesEncoder key;
    key.Field("kind", "disco-virtual-device-v2");
    key.Field("target", target.CanonicalBytes());
    key.Field("scope", node->memory_scope);
    key.Field("logical_id", std::to_string(node->virtual_device_id));
    return std::move(key).Take();
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
    Validate();
}

void WorkerPlacement::Validate() const {
    const auto* worker = As<WorkerPlacementNode>();
    if (!worker || worker->worker_id < 0 || worker->group_id < 0 || worker->local_rank < 0 ||
        !worker->device.defined() || !worker->target.As<TargetNode>() || !worker->virtual_device.As<VirtualDeviceNode>()) {
        throw std::invalid_argument("incomplete WorkerPlacement");
    }
    if (worker->target->device_type != worker->device.device_type() || worker->target->device_id != worker->device.device_id() ||
        worker->virtual_device->device != worker->device ||
        (worker->virtual_device->target.defined() ? worker->virtual_device->target : BuildTarget(worker->device)).CanonicalBytes() != worker->target.CanonicalBytes()) {
        throw std::invalid_argument("WorkerPlacement has conflicting device/target constraints");
    }
    (void)VirtualDeviceIdentity(worker->virtual_device);
}

// 返回经过 WorkerPlacement 类型约束的底层节点。
const WorkerPlacementNode* WorkerPlacement::operator->() const {
    const auto* node = As<WorkerPlacementNode>();
    if (!node) throw std::invalid_argument("expected WorkerPlacement");
    return node;
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
    if (!virtual_device.defined()) return -1;
    const auto key = VirtualDeviceIdentity(virtual_device);
    if (vd_to_worker.count(virtual_device)) {
        const int id = vd_to_worker.at(virtual_device);
        for (const auto& worker : workers) {
            if (worker->worker_id == id && VirtualDeviceIdentity(worker->virtual_device) == key) return id;
        }
        return -1;
    }
    for (const auto& worker : workers) {
        if (VirtualDeviceIdentity(worker->virtual_device) == key) return worker->worker_id;
    }
    return -1;
}

// 构造逻辑设备到 worker 的完整放置表，并规范化组数下限。
DiscoPlacement::DiscoPlacement(Array<WorkerPlacement> workers, Map<VirtualDevice, int> vd_to_worker,
                               int num_groups) {
    auto node = std::make_unique<DiscoPlacementNode>();
    std::vector<WorkerPlacement> ordered(workers.begin(), workers.end());
    for (const auto& worker : ordered) worker.Validate();
    std::sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) { return a->worker_id < b->worker_id; });
    for (const auto& worker : ordered) node->workers.push_back(worker);
    node->vd_to_worker = std::move(vd_to_worker);
    node->num_groups = num_groups;
    SetData(node.release());
    Validate();
}

void DiscoPlacement::Validate() const {
    const auto* placement = As<DiscoPlacementNode>();
    if (!placement || placement->workers.empty() || placement->num_groups <= 0 ||
        placement->workers.size() % static_cast<size_t>(placement->num_groups) != 0) {
        throw std::invalid_argument("placement requires positive evenly divided worker groups");
    }
    const auto width = placement->workers.size() / static_cast<size_t>(placement->num_groups);
    std::unordered_set<std::string> devices;
    for (size_t i = 0; i < placement->workers.size(); ++i) {
        const auto& worker = placement->workers[i]; worker.Validate();
        if (worker->worker_id != static_cast<int>(i) || worker->group_id != static_cast<int>(i / width) ||
            worker->local_rank != static_cast<int>(i % width) || !devices.insert(VirtualDeviceIdentity(worker->virtual_device)).second) {
            throw std::invalid_argument("placement requires dense worker ids, consistent groups/ranks and unique logical devices");
        }
    }
    for (const auto& entry : placement->vd_to_worker) {
        if (entry.second < 0 || static_cast<size_t>(entry.second) >= placement->workers.size() ||
            VirtualDeviceIdentity(entry.first) != VirtualDeviceIdentity(placement->workers[entry.second]->virtual_device)) {
            throw std::invalid_argument("placement index does not match its worker contracts");
        }
    }
}

// 返回经过 DiscoPlacement 类型约束的底层节点。
const DiscoPlacementNode* DiscoPlacement::operator->() const {
    const auto* node = As<DiscoPlacementNode>();
    if (!node) throw std::invalid_argument("expected DiscoPlacement");
    return node;
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
    std::map<std::string, VirtualDevice> unique;
    for (const auto& vd : virtual_devices) unique.emplace(VirtualDeviceIdentity(vd), vd);
    if (unique.empty() || num_groups <= 0 || unique.size() % static_cast<size_t>(num_groups) != 0) {
        throw std::invalid_argument("placement requires positive evenly divided worker groups");
    }
    const int width = static_cast<int>(unique.size() / static_cast<size_t>(num_groups));
    Array<WorkerPlacement> workers;
    Map<VirtualDevice, int> index;
    std::unordered_map<std::string, int> ids;
    for (const auto& item : unique) {
        const int id = static_cast<int>(workers.size());
        const auto& vd = item.second;
        const Target target = vd->target.defined() ? vd->target : BuildTarget(vd->device);
        workers.push_back(WorkerPlacement(id, id / width, id % width, vd->device, target, vd));
        ids.emplace(item.first, id);
    }
    for (const auto& vd : virtual_devices) index.Set(vd, ids.at(VirtualDeviceIdentity(vd)));
    return DiscoPlacement(workers, index, num_groups);
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
