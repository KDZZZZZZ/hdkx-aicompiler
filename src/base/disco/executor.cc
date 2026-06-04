/*! \file src/base/disco/executor.cc
 * \brief 实现 Disco 线程会话、执行计划解释器和 CPU/NCCL 通信后端。
 */

#include "base/disco/executor.h"

#include <stdexcept>
#include <string>

#include "base/device.h"
#include "base/profiling.h"
#include "base/packedfunc.h"
#include "base/registry.h"
#include "relay/op.h"

namespace kxc {
namespace disco {

namespace {

Array<int> EffectiveWorkerSet(const Array<int>& worker_set, int fallback_worker) {
    if (!worker_set.empty()) {
        return worker_set;
    }
    return {fallback_worker >= 0 ? fallback_worker : 0};
}

profiling::EventSpec MakeExecutorSpec(const std::string& event_type) {
    profiling::EventSpec spec;
    spec.component = "execution_plan";
    spec.event_type = event_type;
    return spec;
}

}  // namespace

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

DRef ExecutionPlanExecutor::ExecuteForOutput(const ExecutionPlan& plan,
                                             const Map<int, DRef>& initial_values) {
    Execute(plan, initial_values);
    int output = plan->output_value;
    if (output < 0 || !values_.count(output)) {
        throw std::runtime_error("ExecutionPlan output value is missing after execution");
    }
    return values_.at(output);
}

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

void ExecutionPlanExecutor::ExecuteKernel(const ExecutionPlan& plan, const KernelExecNode* kernel) {
    if (!kernel || kernel->output_values.empty()) {
        return;
    }
    profiling::ScopedSpan kernel_span(profiling::CurrentContext(),
                                      MakeExecutorSpec("kernel_exec"));
    kernel_span.AddField("op_name", kernel->op_name);
    kernel_span.AddField("kernel_symbol", kernel->kernel_symbol);
    kernel_span.AddMetric("input_count", static_cast<double>(kernel->input_values.size()));
    kernel_span.AddMetric("output_count", static_cast<double>(kernel->output_values.size()));

    if (kernel->input_values.empty()) {
        for (int out_id : kernel->output_values) {
            DRef out = EnsureValue(plan, out_id, DRef());
            Array<int64_t> scalar_shape = {1};
            for (int worker : EffectiveWorkerSet(kernel->worker_set, 0)) {
                session_.Set(worker, out, runtime::NDArray(scalar_shape, "float32"));
            }
        }
        return;
    }

    int input_id = kernel->input_values[0];
    if (!values_.count(input_id)) {
        throw std::runtime_error("Kernel input value is not available");
    }
    DRef input = values_.at(input_id);
    for (int out_id : kernel->output_values) {
        DRef output = EnsureValue(plan, out_id, input);
        for (int worker : EffectiveWorkerSet(kernel->worker_set,
                                             ResolveWorkerForValue(plan, input_id))) {
            ccl_backend_->Copy(session_, input, output, worker, worker);
        }
        values_.Set(out_id, output);
    }
}

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

    values_.Set(output_id, dst);
    for (size_t i = 1; i < comm->output_values.size(); ++i) {
        values_.Set(comm->output_values[i], dst);
    }
}

int ExecutionPlanExecutor::ResolveWorkerForValue(const ExecutionPlan& plan, int value_id) const {
    if (!plan.defined()) return 0;
    if (!plan->value_virtual_devices.count(value_id)) {
        return 0;
    }
    return ResolveWorkerForVirtualDevice(plan, plan->value_virtual_devices.at(value_id));
}

int ExecutionPlanExecutor::ResolveWorkerForVirtualDevice(const ExecutionPlan& plan,
                                                         const VirtualDevice& vd) const {
    if (!plan.defined() || !vd.defined()) {
        return 0;
    }
    if (plan->pass_ctx.has_disco_placement()) {
        int worker = FindWorkerForVirtualDevice(plan->pass_ctx.disco_placement(), vd);
        if (worker >= 0) return worker;
    }
    if (vd->device_obj.defined()) {
        const auto* dev = static_cast<const class Device*>(vd->device_obj.get());
        if (dev) return dev->device_id();
    }
    if (vd->target.defined()) {
        return vd->target->device_id;
    }
    return 0;
}

KXC_REGISTER_GLOBAL("kxc.disco.execute_plan")
    .set_body(ToPackedFunc([](DiscoSession session, ExecutionPlan plan) -> ObjectRef {
        ExecutionPlanExecutor executor(std::move(session), CreateCpuCCLBackend());
        return ObjectRef(executor.Execute(plan, Map<int, DRef>()));
    }));

KXC_REGISTER_GLOBAL("kxc.disco.execute_plan_output")
    .set_body(ToPackedFunc([](DiscoSession session, ExecutionPlan plan) -> ObjectRef {
        ExecutionPlanExecutor executor(std::move(session), CreateCpuCCLBackend());
        return ObjectRef(executor.ExecuteForOutput(plan, Map<int, DRef>()));
    }));

KXC_REGISTER_GLOBAL("kxc.disco.execute_plan_json")
    .set_body(ToPackedFunc([](DiscoSession session, std::string json_text) -> ObjectRef {
        ExecutionPlanExecutor executor(std::move(session), CreateCpuCCLBackend());
        ExecutionPlan plan = DeserializeExecutionPlanFromJson(json_text);
        return ObjectRef(executor.Execute(plan, Map<int, DRef>()));
    }));

KXC_REGISTER_GLOBAL("kxc.disco.execute_plan_json_output")
    .set_body(ToPackedFunc([](DiscoSession session, std::string json_text) -> ObjectRef {
        ExecutionPlanExecutor executor(std::move(session), CreateCpuCCLBackend());
        ExecutionPlan plan = DeserializeExecutionPlanFromJson(json_text);
        return ObjectRef(executor.ExecuteForOutput(plan, Map<int, DRef>()));
    }));

KXC_REGISTER_GLOBAL("kxc.disco.execute_plan_json_file")
    .set_body(ToPackedFunc([](DiscoSession session, std::string path) -> ObjectRef {
        ExecutionPlanExecutor executor(std::move(session), CreateCpuCCLBackend());
        ExecutionPlan plan = LoadExecutionPlanFromJsonFile(path);
        return ObjectRef(executor.Execute(plan, Map<int, DRef>()));
    }));

KXC_REGISTER_GLOBAL("kxc.disco.execute_plan_json_file_output")
    .set_body(ToPackedFunc([](DiscoSession session, std::string path) -> ObjectRef {
        ExecutionPlanExecutor executor(std::move(session), CreateCpuCCLBackend());
        ExecutionPlan plan = LoadExecutionPlanFromJsonFile(path);
        return ObjectRef(executor.ExecuteForOutput(plan, Map<int, DRef>()));
    }));

}  // namespace disco
}  // namespace kxc
