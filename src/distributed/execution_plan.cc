/*! \file src/distributed/execution_plan.cc
 * \brief Implements the execution-plan object model independently of persistence.
 */

#include "kxc/distributed/execution_plan.h"
#include "kxc/support/object_registration.h"

#include <sstream>
#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_set>
#include <vector>
#include <utility>

#include "kxc/runtime/ndarray.h"
#include "kxc/runtime/session.h"
#include "runtime/internal/compiled_module_node.h"

namespace kxc {

KXC_OBJECT_DEFINE(ExecNodeBaseNode)
KXC_OBJECT_DEFINE(KernelExecNode)
KXC_OBJECT_DEFINE(CommExecNode)
KXC_OBJECT_DEFINE(BarrierExecNode)
KXC_OBJECT_DEFINE(ExecutionPlanNode)

namespace {
Array<int> OrderedWorkers(const Array<int>& workers) {
  std::vector<int> ordered(workers.begin(), workers.end());
  std::sort(ordered.begin(), ordered.end());
  Array<int> result;
  for (int worker : ordered) result.push_back(worker);
  return result;
}
}  // namespace

KernelExec::KernelExec(std::string op_name,
                       Array<int> input_values, Array<int> output_values,
                       Array<int> worker_set, std::string kernel_symbol,
                       std::string kernel_abi) {
  auto* node = new KernelExecNode();
  SetData(node);
  node->kind = ExecNodeKind::kKernel;
  node->op_name = std::move(op_name);
  node->kernel_symbol = std::move(kernel_symbol);
  node->kernel_abi = std::move(kernel_abi);
  node->input_values = std::move(input_values);
  node->output_values = std::move(output_values);
  node->worker_set = OrderedWorkers(worker_set);
}

CommExec::CommExec(std::string op_name, CommExecAttrs attrs,
                   Array<int> input_values, Array<int> output_values,
                   Array<int> worker_set) {
  auto* node = new CommExecNode();
  SetData(node);
  node->kind = ExecNodeKind::kComm;
  node->op_name = std::move(op_name);
  node->attrs = std::move(attrs);
  node->input_values = std::move(input_values);
  node->output_values = std::move(output_values);
  node->worker_set = OrderedWorkers(worker_set);
}

BarrierExec::BarrierExec(std::string tag, Array<int> worker_set) {
  auto* node = new BarrierExecNode();
  SetData(node);
  node->kind = ExecNodeKind::kBarrier;
  node->tag = std::move(tag);
  node->worker_set = OrderedWorkers(worker_set);
}

ExecutionPlan::ExecutionPlan(Array<ObjectRef> nodes,
                             Map<int, VirtualDevice> value_virtual_devices,
                             Array<int> input_value_ids,
                             Array<int> constant_value_ids,
                             Map<int, Array<int64_t>> value_shapes,
                             Map<int, std::string> value_dtypes,
                             int num_values, PassContext pass_ctx,
                             DiscoPlacement placement, int output_value)
    : ExecutionPlan(std::move(nodes), std::move(value_virtual_devices),
                    std::move(input_value_ids), std::move(constant_value_ids),
                    std::move(value_shapes), std::move(value_dtypes), num_values,
                    std::move(pass_ctx), std::move(placement), Array<int>{output_value}) {}

ExecutionPlan::ExecutionPlan(Array<ObjectRef> nodes,
                             Map<int, VirtualDevice> value_virtual_devices,
                             Array<int> input_value_ids, Array<int> constant_value_ids,
                             Map<int, Array<int64_t>> value_shapes,
                             Map<int, std::string> value_dtypes,
                             int num_values, PassContext pass_ctx,
                             DiscoPlacement placement, Array<int> output_value_ids) {
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
  node->output_value = output_value_ids.empty() ? -1 : output_value_ids[0];
  node->output_value_ids = std::move(output_value_ids);
  SetData(node);
}

ExecutionPlan ExecutionPlan::FromExecutablePlan(const api::CompiledModule& module,
    const runtime::ExecutablePlan& source, const DiscoPlacement& placement,
    const Array<int>& input_workers, const Array<int>& call_workers, int output_worker) {
  source.Validate();
  placement.Validate();
  if (source.mode() != runtime::ExecutablePlanMode::kStatic ||
      !source.state_value_ids().empty() || source.request_batching()) {
    throw std::invalid_argument("distributed graph binding requires static fresh-output state-free plans");
  }
  runtime::RuntimeSession::Validate(module, source);
  const auto calls = source.calls();
  const auto inputs = source.input_value_ids();
  if (calls.empty() || source.output_value_ids().empty() ||
      input_workers.size() != inputs.size() || call_workers.size() != calls.size()) {
    throw std::invalid_argument("distributed graph binding requires explicit input/call workers and graph outputs");
  }
  const auto check_worker = [&](int worker) {
    if (worker < 0 || static_cast<size_t>(worker) >= placement->workers.size()) {
      throw std::invalid_argument("distributed graph binding worker is out of range");
    }
  };
  check_worker(output_worker);
  for (int worker : input_workers) check_worker(worker);
  for (int worker : call_workers) check_worker(worker);
  const auto target = module.As<api::CompiledModuleNode>()->target_.CanonicalBytes();
  for (const auto& worker : placement->workers) {
    if (worker->device != Device::CPU() || worker->target.CanonicalBytes() != target) {
      throw std::invalid_argument("distributed graph binding requires matching CPU:0 module/worker Targets");
    }
  }
  std::map<int64_t, runtime::ValueSpec> specs;
  std::unordered_set<int64_t> storages;
  for (const auto& value : source.values()) {
    if (value->is_alias || value->is_state || value->valid_bytes != -1 ||
        value->write_mode != runtime::ValueWriteMode::kAllocate ||
        !storages.insert(value->storage_id).second) {
      throw std::invalid_argument("distributed graph binding rejects alias, state, partial values and storage reuse");
    }
    specs.emplace(value->value_id, value);
  }
  for (int64_t value : source.output_value_ids()) {
    if (specs.at(value)->is_constant) {
      throw std::invalid_argument("distributed graph binding does not expose module constants as graph outputs");
    }
  }
  Array<ObjectRef> nodes;
  Map<int, VirtualDevice> devices;
  Map<int, Array<int64_t>> shapes;
  Map<int, std::string> dtypes;
  Array<int> input_ids, output_ids;
  std::map<int64_t, int> homes;
  std::map<std::pair<int64_t, int>, int> locations;
  int count = 0;
  const auto add_value = [&](int64_t value, int worker) {
    if (count == std::numeric_limits<int>::max()) {
      throw std::invalid_argument("distributed graph value count exceeds int range");
    }
    const int id = count++;
    const auto& spec = specs.at(value);
    devices.Set(id, placement->workers[worker]->virtual_device);
    shapes.Set(id, spec.shape());
    dtypes.Set(id, runtime::DataTypeToString(spec->dtype));
    locations.emplace(std::make_pair(value, worker), id);
    return id;
  };
  const auto on_worker = [&](int64_t value, int worker) {
    const auto found = locations.find({value, worker});
    if (found != locations.end()) return found->second;
    const int home = homes.at(value);
    const int original = locations.at({value, home});
    const int copy = add_value(value, worker);
    CommExecAttrs attrs;
    attrs.src_virtual_device = placement->workers[home]->virtual_device;
    attrs.dst_virtual_device = placement->workers[worker]->virtual_device;
    attrs.in_group = false;
    nodes.push_back(CommExec("device.copy", attrs, {original}, {copy}, {home, worker}));
    return copy;
  };
  for (size_t i = 0; i < inputs.size(); ++i) {
    homes.emplace(inputs[i], input_workers[i]);
    input_ids.push_back(add_value(inputs[i], input_workers[i]));
  }
  for (size_t index = 0; index < calls.size(); ++index) {
    const auto& call = calls[index];
    const int worker = call_workers[index];
    const auto launch = module.launch_metadata(call->symbol);
    if (launch->device != Device::CPU() || launch->backend != codegen::CodeGenBackend::kLLVM) {
      throw std::invalid_argument("distributed graph binding requires CPU:0 LLVM kernels");
    }
    Array<int> in, out;
    for (int64_t value : call.input_value_ids()) {
      if (!specs.at(value)->is_constant) in.push_back(on_worker(value, worker));
    }
    for (int64_t value : call.output_value_ids()) {
      homes.emplace(value, worker);
      out.push_back(add_value(value, worker));
    }
    nodes.push_back(KernelExec("compiled_graph", in, out, {worker}, call->symbol,
                              module.signature(call->symbol).CanonicalBytes()));
  }
  for (int64_t value : source.output_value_ids()) output_ids.push_back(on_worker(value, output_worker));
  Array<VirtualDevice> virtual_devices;
  for (const auto& worker : placement->workers) virtual_devices.push_back(worker->virtual_device);
  ExecutionPlan result(nodes, devices, input_ids, {}, shapes, dtypes, count,
      PassContext::FromVirtualDevices(virtual_devices), placement, output_ids);
  result.Validate();
  return result;
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

bool IsCommunicationOpName(const std::string& op_name) {
  return op_name == "device.copy" || op_name == "device.allreduce" ||
         op_name == "device.broadcast_from_worker0" || op_name == "device.scatter_from_worker0" ||
         op_name == "device.gather_to_worker0" || op_name == "device.send_to_worker" ||
         op_name == "device.recv_from_worker";
}

void ExecutionPlan::Validate() const {
  const auto* plan = As<ExecutionPlanNode>();
  if (!plan) throw std::invalid_argument("ExecutionPlan is undefined or has the wrong node type");
  plan->placement.Validate();
  if (plan->num_values <= 0 || plan->value_shapes.size() != static_cast<size_t>(plan->num_values) ||
      plan->value_dtypes.size() != static_cast<size_t>(plan->num_values) ||
      plan->value_virtual_devices.size() != static_cast<size_t>(plan->num_values)) {
    throw std::invalid_argument("ExecutionPlan requires complete, dense value metadata");
  }
  const auto check_id = [&](int id) {
    if (id < 0 || id >= plan->num_values) throw std::invalid_argument("ExecutionPlan value id out of range");
  };
  for (int id = 0; id < plan->num_values; ++id) {
    if (!plan->value_shapes.count(id) || !plan->value_dtypes.count(id) || !plan->value_virtual_devices.count(id) ||
        plan->placement.FindWorker(plan->value_virtual_devices.at(id)) < 0) {
      throw std::invalid_argument("ExecutionPlan value lacks explicit metadata or placement: " + std::to_string(id));
    }
    const auto dtype = runtime::DataTypeFromString(plan->value_dtypes.at(id));
    if (!dtype.bits || dtype.bits % 8 || !dtype.lanes) throw std::invalid_argument("ExecutionPlan dtype must be byte-addressable");
    const auto& shape = plan->value_shapes.at(id);
    bool empty = false;
    for (int64_t dim : shape) {
      if (dim < 0) throw std::invalid_argument("ExecutionPlan requires static nonnegative shapes");
      empty = empty || dim == 0;
    }
    size_t bytes = dtype.bits / 8 * dtype.lanes;
    if (!empty) for (int64_t dim : shape) {
      if (static_cast<uint64_t>(dim) > std::numeric_limits<size_t>::max() / bytes) {
        throw std::invalid_argument("ExecutionPlan tensor byte size overflow");
      }
      bytes *= static_cast<size_t>(dim);
    }
  }
  std::unordered_set<int> defined;
  const auto source = [&](int id) {
    check_id(id);
    if (!defined.insert(id).second) throw std::invalid_argument("ExecutionPlan duplicate source value");
  };
  for (int id : plan->input_value_ids) source(id);
  for (int id : plan->constant_value_ids) source(id);
  for (size_t index = 0; index < plan->nodes.size(); ++index) {
    try {
      const auto& ref = plan->nodes[index];
      const auto* node = ref.As<ExecNodeBaseNode>();
      if (!node) throw std::invalid_argument("undefined or unknown node type");
      if (ref.As<KernelExecNode>()) {
        if (node->kind != ExecNodeKind::kKernel || node->output_values.empty()) throw std::invalid_argument("invalid kernel node");
      } else if (ref.As<CommExecNode>()) {
        if (node->kind != ExecNodeKind::kComm || node->input_values.size() != 1 || node->output_values.size() != 1) {
          throw std::invalid_argument("communication requires exactly one input and one fresh output");
        }
        if (!IsCommunicationOpName(ref.As<CommExecNode>()->op_name)) throw std::invalid_argument("unknown communication op");
      } else if (ref.As<BarrierExecNode>()) {
        if (node->kind != ExecNodeKind::kBarrier || !node->input_values.empty() || !node->output_values.empty()) {
          throw std::invalid_argument("barrier cannot define or consume values");
        }
      } else {
        throw std::invalid_argument("unknown node type");
      }
      if (node->worker_set.empty()) throw std::invalid_argument("node requires an explicit worker_set");
      int previous = -1;
      for (int worker : node->worker_set) {
        if (worker <= previous || worker >= static_cast<int>(plan->placement->workers.size())) {
          throw std::invalid_argument("worker_set must contain sorted unique valid workers");
        }
        previous = worker;
      }
      for (int id : node->input_values) {
        check_id(id);
        if (!defined.count(id)) throw std::invalid_argument("use before definition (or cycle), value " + std::to_string(id));
      }
      for (int id : node->output_values) {
        check_id(id);
        if (!defined.insert(id).second) throw std::invalid_argument("duplicate definition or undeclared alias, value " + std::to_string(id));
      }
    } catch (const std::exception& error) {
      throw std::invalid_argument("ExecutionPlan node[" + std::to_string(index) + "]: " + error.what());
    }
  }
  if (plan->output_value_ids.empty() || plan->output_value != plan->output_value_ids[0]) {
    throw std::invalid_argument("ExecutionPlan requires ordered outputs and a matching first output");
  }
  for (int value : plan->output_value_ids) {
    check_id(value);
    if (!defined.count(value)) throw std::invalid_argument("ExecutionPlan output is undefined");
  }
  if (defined.size() != static_cast<size_t>(plan->num_values)) {
    throw std::invalid_argument("ExecutionPlan contains values without a definition");
  }
}

}  // namespace kxc
