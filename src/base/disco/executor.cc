/*! \file src/base/disco/executor.cc
 * \brief 实现 ExecutionPlan 解释器，并把通信节点分派给注入的 CCLBackend。
 */

#include "base/disco/executor.h"

#include <stdexcept>
#include <string>

#include "base/profiling.h"
#include "relay/op.h"

namespace kxc {
namespace disco {

namespace {

// 规范节点 worker 集合；未显式指定时退回放置结果或 worker0。
Array<int> EffectiveWorkerSet(const Array<int>& worker_set, int fallback_worker) {
    if (!worker_set.empty()) {
        return worker_set;
    }
    return {fallback_worker >= 0 ? fallback_worker : 0};
}

// 构造执行计划统一 profiling 事件描述。
profiling::EventSpec MakeExecutorSpec(const std::string& event_type) {
    profiling::EventSpec spec;
    spec.component = "execution_plan";
    spec.event_type = event_type;
    return spec;
}

}  // namespace

// 绑定一个 Disco 会话和通信后端，拒绝不可执行的空依赖。
ExecutionPlanExecutor::ExecutionPlanExecutor(DiscoSession session,
                                             std::shared_ptr<CCLBackend> ccl_backend)
    : session_(std::move(session)), ccl_backend_(std::move(ccl_backend)) {
    if (!session_.defined()) {
        throw std::runtime_error("ExecutionPlanExecutor requires a defined DiscoSession");
    }
    if (!ccl_backend_) {
        throw std::runtime_error("ExecutionPlanExecutor requires a CCL backend");
    }
}

// 按计划顺序解释 kernel、通信和 barrier 节点，并维护 value 到 DRef 的表。
Map<int, DRef> ExecutionPlanExecutor::Execute(const ExecutionPlan& plan,
                                              const Map<int, DRef>& initial_values) {
    if (!plan.defined()) {
        throw std::runtime_error("Execute requires a defined ExecutionPlan");
    }
    profiling::ScopedSpan execute_span(profiling::CurrentContext(),
                                       MakeExecutorSpec("execute_plan"));
    execute_span.AddMetric("node_count", static_cast<double>(plan->nodes.size()));
    execute_span.AddMetric("value_count", static_cast<double>(plan->num_values));
    values_ = initial_values;

    for (size_t node_index = 0; node_index < plan->nodes.size(); ++node_index) {
        const auto& node_ref = plan->nodes[node_index];
        if (!node_ref.defined()) {
            continue;
        }
        // 节点类型通过对象系统 TypeId 分派，未知类型必须失败而不能跳过。
        if (node_ref.get()->GetTypeId() == KernelExecNode::_type_index) {
            profiling::ScopedSpan node_span(profiling::CurrentContext(),
                                            MakeExecutorSpec("execute_kernel_node"));
            node_span.AddField("node_index", std::to_string(node_index));
            ExecuteKernel(plan, static_cast<const KernelExecNode*>(node_ref.get()));
            continue;
        }
        if (node_ref.get()->GetTypeId() == CommExecNode::_type_index) {
            profiling::ScopedSpan node_span(profiling::CurrentContext(),
                                            MakeExecutorSpec("execute_comm_node"));
            node_span.AddField("node_index", std::to_string(node_index));
            ExecuteComm(plan, static_cast<const CommExecNode*>(node_ref.get()));
            continue;
        }
        if (node_ref.get()->GetTypeId() == BarrierExecNode::_type_index) {
            const auto* barrier = static_cast<const BarrierExecNode*>(node_ref.get());
            profiling::ScopedSpan node_span(profiling::CurrentContext(),
                                            MakeExecutorSpec("execute_barrier_node"));
            node_span.AddField("node_index", std::to_string(node_index));
            node_span.AddField("tag", barrier->tag);
            for (int worker : EffectiveWorkerSet(barrier->worker_set, 0)) {
                ccl_backend_->SyncWorker(session_, worker);
            }
            continue;
        }
        throw std::runtime_error("ExecutionPlan contains unknown node type");
    }
    return values_;
}

// 执行完整计划并返回声明的唯一输出 DRef。
DRef ExecutionPlanExecutor::ExecuteForOutput(const ExecutionPlan& plan,
                                             const Map<int, DRef>& initial_values) {
    Execute(plan, initial_values);
    int output = plan->output_value;
    if (output < 0 || !values_.count(output)) {
        throw std::runtime_error("ExecutionPlan output value is missing after execution");
    }
    return values_.at(output);
}

// 确保 value 已有 DRef；给定原型时复制各 worker 本地值作为初始内容。
DRef ExecutionPlanExecutor::EnsureValue(const ExecutionPlan& plan, int value_id,
                                        const DRef& prototype) {
    if (values_.count(value_id)) {
        return values_.at(value_id);
    }
    DRef ref = session_.NewDRef();
    values_.Set(value_id, ref);
    if (!prototype.valid()) {
        return ref;
    }
    for (int worker = 0; worker < session_.num_workers(); ++worker) {
        runtime::NDArray src = session_.Get(worker, prototype);
        if (!src.defined()) continue;
        ccl_backend_->Copy(session_, prototype, ref, worker, worker);
    }
    return ref;
}

// 真实 module lookup/launch 尚未接入；当前边界必须失败，禁止伪造输出。
void ExecutionPlanExecutor::ExecuteKernel(const ExecutionPlan& plan, const KernelExecNode* kernel) {
    (void)plan;
    if (!kernel) throw std::runtime_error("ExecutionPlan contains an invalid kernel node");
    profiling::ScopedSpan kernel_span(profiling::CurrentContext(),
                                      MakeExecutorSpec("kernel_exec"));
    kernel_span.AddField("op_name", kernel->op_name);
    kernel_span.AddField("kernel_symbol", kernel->kernel_symbol);
    kernel_span.AddMetric("input_count", static_cast<double>(kernel->input_values.size()));
    kernel_span.AddMetric("output_count", static_cast<double>(kernel->output_values.size()));
    const std::string message =
        "ExecutionPlan CompiledModule launch is not implemented";
    kernel_span.SetStatus("error");
    kernel_span.SetMessage(message);
    throw std::runtime_error(message);
}

// 根据通信算子名和结构化 attrs 分派到 CCLBackend，并登记输出 DRef。
void ExecutionPlanExecutor::ExecuteComm(const ExecutionPlan& plan, const CommExecNode* comm) {
    if (!comm || comm->output_values.empty()) {
        return;
    }
    profiling::ScopedSpan comm_span(profiling::CurrentContext(),
                                    MakeExecutorSpec("comm_exec"));
    comm_span.AddField("op_name", comm->op_name);
    comm_span.AddMetric("input_count", static_cast<double>(comm->input_values.size()));
    comm_span.AddMetric("output_count", static_cast<double>(comm->output_values.size()));
    if (comm->input_values.empty()) {
        throw std::runtime_error("Communication node requires at least one input");
    }

    int input_id = comm->input_values[0];
    if (!values_.count(input_id)) {
        throw std::runtime_error("Communication input value is missing");
    }

    DRef src = values_.at(input_id);
    int output_id = comm->output_values[0];
    DRef dst = EnsureValue(plan, output_id, src);

    // device.copy 可由 attrs 中的显式 VirtualDevice 覆盖 value 默认放置。
    if (comm->op_name == "device.copy") {
        int src_worker = ResolveWorkerForValue(plan, input_id);
        int dst_worker = ResolveWorkerForValue(plan, output_id);
        if (comm->attrs.defined()) {
            if (auto* attrs = comm->attrs.As<relay::DeviceCopyAttrsNode>()) {
                if (attrs->src_virtual_device.defined()) {
                    src_worker =
                        ResolveWorkerForVirtualDevice(plan, attrs->src_virtual_device);
                }
                if (attrs->dst_virtual_device.defined()) {
                    dst_worker =
                        ResolveWorkerForVirtualDevice(plan, attrs->dst_virtual_device);
                }
            }
        }
        ccl_backend_->Copy(session_, src, dst, src_worker, dst_worker);
    } else if (comm->op_name == "device.allreduce") {
        std::string reduce_kind = "sum";
        bool in_group = true;
        if (comm->attrs.defined()) {
            if (auto* attrs = comm->attrs.As<relay::CollectiveAttrsNode>()) {
                reduce_kind = attrs->reduce_kind;
                in_group = attrs->in_group;
            }
        }
        ccl_backend_->AllReduce(session_, src, dst, reduce_kind, in_group);
    } else if (comm->op_name == "device.broadcast_from_worker0") {
        bool in_group = true;
        if (comm->attrs.defined()) {
            if (auto* attrs = comm->attrs.As<relay::CollectiveAttrsNode>()) {
                in_group = attrs->in_group;
            }
        }
        ccl_backend_->BroadcastFromWorker0(session_, src, dst, in_group);
    } else if (comm->op_name == "device.scatter_from_worker0") {
        bool in_group = true;
        if (comm->attrs.defined()) {
            if (auto* attrs = comm->attrs.As<relay::CollectiveAttrsNode>()) {
                in_group = attrs->in_group;
            }
        }
        ccl_backend_->ScatterFromWorker0(session_, src, dst, in_group);
    } else if (comm->op_name == "device.gather_to_worker0") {
        bool in_group = true;
        if (comm->attrs.defined()) {
            if (auto* attrs = comm->attrs.As<relay::CollectiveAttrsNode>()) {
                in_group = attrs->in_group;
            }
        }
        ccl_backend_->GatherToWorker0(session_, src, dst, in_group);
    } else if (comm->op_name == "device.send_to_worker") {
        int receiver_worker = 0;
        if (comm->attrs.defined()) {
            if (auto* attrs = comm->attrs.As<relay::CollectiveAttrsNode>()) {
                receiver_worker = attrs->root_worker;
            }
        }
        ccl_backend_->SendToWorker(session_, src, dst, receiver_worker);
    } else if (comm->op_name == "device.recv_from_worker") {
        int sender_worker = 0;
        if (comm->attrs.defined()) {
            if (auto* attrs = comm->attrs.As<relay::CollectiveAttrsNode>()) {
                sender_worker = attrs->root_worker;
            }
        }
        ccl_backend_->RecvFromWorker(session_, src, dst, sender_worker);
    } else {
        throw std::runtime_error("Unsupported communication op in executor: " + comm->op_name);
    }

    // 多输出通信节点当前共享同一 DRef，保持执行计划中的别名关系。
    values_.Set(output_id, dst);
    for (size_t i = 1; i < comm->output_values.size(); ++i) {
        values_.Set(comm->output_values[i], dst);
    }
}

// 通过 value 的 VirtualDevice 约束解析 worker，缺失映射时使用 worker0。
int ExecutionPlanExecutor::ResolveWorkerForValue(const ExecutionPlan& plan, int value_id) const {
    if (!plan.defined()) return 0;
    if (!plan->value_virtual_devices.count(value_id)) {
        return 0;
    }
    return ResolveWorkerForVirtualDevice(plan, plan->value_virtual_devices.at(value_id));
}

// 仅通过 DiscoPlacement 把逻辑设备映射为 worker，禁止复用物理 device_id。
int ExecutionPlanExecutor::ResolveWorkerForVirtualDevice(const ExecutionPlan& plan,
                                                         const VirtualDevice& vd) const {
    if (!plan.defined() || !vd.defined()) {
        return 0;
    }
    if (plan->pass_ctx.has_disco_placement()) {
        int worker = FindWorkerForVirtualDevice(plan->pass_ctx.disco_placement(), vd);
        if (worker >= 0) return worker;
    }
    // 物理 device id 与 worker id 属于不同命名空间；参与 Disco 时只能由 placement 映射。
    return 0;
}

}  // namespace disco
}  // namespace kxc
