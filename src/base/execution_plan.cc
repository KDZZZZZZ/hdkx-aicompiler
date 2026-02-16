#include "base/execution_plan.h"

#include <sstream>

namespace kxc {

KernelExec::KernelExec(std::string op_name, tir::PrimFunc primfunc, Array<int> input_values,
                       Array<int> output_values, Array<int> worker_set) {
    KernelExecNode* node = new KernelExecNode();
    node->kind = ExecNodeKind::kKernel;
    node->op_name = std::move(op_name);
    node->primfunc = std::move(primfunc);
    node->input_values = std::move(input_values);
    node->output_values = std::move(output_values);
    node->worker_set = std::move(worker_set);
    SetData(node);
}

CommExec::CommExec(std::string op_name, ObjectRef attrs, Array<int> input_values,
                   Array<int> output_values, Array<int> worker_set) {
    CommExecNode* node = new CommExecNode();
    node->kind = ExecNodeKind::kComm;
    node->op_name = std::move(op_name);
    node->attrs = std::move(attrs);
    node->input_values = std::move(input_values);
    node->output_values = std::move(output_values);
    node->worker_set = std::move(worker_set);
    SetData(node);
}

BarrierExec::BarrierExec(std::string tag, Array<int> worker_set) {
    BarrierExecNode* node = new BarrierExecNode();
    node->kind = ExecNodeKind::kBarrier;
    node->tag = std::move(tag);
    node->worker_set = std::move(worker_set);
    SetData(node);
}

ExecutionPlan::ExecutionPlan(Array<ObjectRef> nodes, Map<int, VirtualDevice> value_virtual_devices,
                             int num_values, PassContext pass_ctx, int output_value) {
    ExecutionPlanNode* node = new ExecutionPlanNode();
    node->nodes = std::move(nodes);
    node->value_virtual_devices = std::move(value_virtual_devices);
    node->num_values = num_values;
    node->pass_ctx = std::move(pass_ctx);
    node->output_value = output_value;
    SetData(node);
}

std::string ExecutionPlan::ToString() const {
    if (!defined()) {
        return "ExecutionPlan(undefined)";
    }
    std::stringstream ss;
    ss << "ExecutionPlan(nodes=" << operator->()->nodes.size()
       << ", values=" << operator->()->num_values
       << ", output=" << operator->()->output_value << ")";
    return ss.str();
}

bool IsCommunicationOpName(const std::string& op_name) {
    return op_name.rfind("device.", 0) == 0;
}

}  // namespace kxc

