/*! \file src/distributed/execution_plan.cc
 * \brief Implements the execution-plan object model independently of persistence.
 */

#include "kxc/distributed/execution_plan.h"
#include "kxc/support/object_registration.h"

#include <sstream>
#include <utility>

namespace kxc {

KXC_OBJECT_DEFINE(ExecNodeBaseNode)
KXC_OBJECT_DEFINE(KernelExecNode)
KXC_OBJECT_DEFINE(CommExecNode)
KXC_OBJECT_DEFINE(BarrierExecNode)
KXC_OBJECT_DEFINE(ExecutionPlanNode)

KernelExec::KernelExec(std::string op_name,
                       Array<int> input_values, Array<int> output_values,
                       Array<int> worker_set, std::string kernel_symbol) {
  auto* node = new KernelExecNode();
  node->kind = ExecNodeKind::kKernel;
  node->op_name = std::move(op_name);
  node->kernel_symbol = std::move(kernel_symbol);
  node->input_values = std::move(input_values);
  node->output_values = std::move(output_values);
  node->worker_set = std::move(worker_set);
  SetData(node);
}

CommExec::CommExec(std::string op_name, CommExecAttrs attrs,
                   Array<int> input_values, Array<int> output_values,
                   Array<int> worker_set) {
  auto* node = new CommExecNode();
  node->kind = ExecNodeKind::kComm;
  node->op_name = std::move(op_name);
  node->attrs = std::move(attrs);
  node->input_values = std::move(input_values);
  node->output_values = std::move(output_values);
  node->worker_set = std::move(worker_set);
  SetData(node);
}

BarrierExec::BarrierExec(std::string tag, Array<int> worker_set) {
  auto* node = new BarrierExecNode();
  node->kind = ExecNodeKind::kBarrier;
  node->tag = std::move(tag);
  node->worker_set = std::move(worker_set);
  SetData(node);
}

ExecutionPlan::ExecutionPlan(Array<ObjectRef> nodes,
                             Map<int, VirtualDevice> value_virtual_devices,
                             Array<int> input_value_ids,
                             Array<int> constant_value_ids,
                             Map<int, Array<int64_t>> value_shapes,
                             Map<int, std::string> value_dtypes,
                             int num_values, PassContext pass_ctx,
                             DiscoPlacement placement, int output_value) {
  auto* node = new ExecutionPlanNode();
  node->nodes = std::move(nodes);
  node->value_virtual_devices = std::move(value_virtual_devices);
  node->input_value_ids = std::move(input_value_ids);
  node->constant_value_ids = std::move(constant_value_ids);
  node->value_shapes = std::move(value_shapes);
  node->value_dtypes = std::move(value_dtypes);
  node->num_values = num_values;
  node->pass_ctx = std::move(pass_ctx);
  node->placement = std::move(placement);
  node->output_value = output_value;
  SetData(node);
}

std::string ExecutionPlan::ToString() const {
  if (!defined()) return "ExecutionPlan(undefined)";
  std::stringstream ss;
  ss << "ExecutionPlan(nodes=" << operator->()->nodes.size()
     << ", values=" << operator->()->num_values
     << ", inputs=" << operator->()->input_value_ids.size()
     << ", constants=" << operator->()->constant_value_ids.size()
     << ", output=" << operator->()->output_value << ")";
  return ss.str();
}

}  // namespace kxc
