/*! \file src/distributed/executor.cc
 * \brief Executes admitted CPU plans using externally compiled modules and CCL.
 */

#include "kxc/distributed/executor.h"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <future>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "kxc/distributed/worker.h"
#include "kxc/profiling/profiling.h"
#include "runtime/internal/compiled_module_node.h"
#include "runtime/internal/kernel_argument_validation.h"

namespace kxc::disco {
namespace {

using runtime::NDArray;
using codegen::KernelArgRole;

[[noreturn]] void Fail(const std::string& message) {
    throw std::invalid_argument("ExecutionPlan " + message);
}

std::string FailureText(std::exception_ptr failure) {
    try { std::rethrow_exception(failure); }
    catch (const std::exception& error) { return error.what(); }
    catch (...) { return "non-standard exception"; }
}

profiling::EventSpec MakeSpec(const std::string& type) {
    profiling::EventSpec spec;
    spec.component = "execution_plan";
    spec.event_type = type;
    spec.fields["contract_version"] = std::to_string(kDistributedExecutionContractVersion);
    return spec;
}

int Home(const ExecutionPlan& plan, int value) {
    return plan->placement.FindWorker(plan->value_virtual_devices.at(value));
}

std::pair<int, int> CopyWorkers(const ExecutionPlan& plan, const CommExecNode* comm) {
    if (comm->op_name == "device.send_to_worker") return {0, comm->attrs.root_worker};
    if (comm->op_name == "device.recv_from_worker") return {comm->attrs.root_worker, 0};
    return {comm->attrs.src_virtual_device.defined()
                ? plan->placement.FindWorker(comm->attrs.src_virtual_device) : Home(plan, comm->input_values[0]),
            comm->attrs.dst_virtual_device.defined()
                ? plan->placement.FindWorker(comm->attrs.dst_virtual_device) : Home(plan, comm->output_values[0])};
}

bool IsPointToPoint(const CommExecNode* comm) {
    return comm->op_name == "device.copy" || comm->op_name == "device.send_to_worker" ||
           comm->op_name == "device.recv_from_worker";
}

bool SameShape(const Array<int64_t>& a, const Array<int64_t>& b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

codegen::KernelArgSpec ValueSpec(const ExecutionPlan& plan, int value, int worker) {
    return codegen::KernelArgSpec("value_" + std::to_string(value), KernelArgRole::kInput,
        runtime::DataTypeFromString(plan->value_dtypes.at(value)), plan->value_shapes.at(value),
        plan->placement->workers[worker]->device);
}

void ValidateArray(const ExecutionPlan& plan, int value, int worker, const NDArray& array) {
    const auto spec = ValueSpec(plan, value, worker);
    api::ValidateTensorArgument("distributed_value", 0, spec, array);
}

// Only invocation-local proof metadata. RuntimeSession remains the owner of persistent state.
struct Availability {
    std::vector<std::vector<bool>> present;
    std::vector<std::vector<uint64_t>> alignment;
};

Availability Preflight(const ExecutionPlan& plan, const DiscoSession& session,
                       const api::CompiledModule& module, const Map<int, DRef>& initial) {
    plan.Validate();
    const int workers = session.num_workers();
    if (plan->placement->workers.size() != static_cast<size_t>(workers) ||
        plan->placement->num_groups != session.num_groups()) Fail("placement/session group count mismatch");
    for (int worker = 0; worker < workers; ++worker) {
        if (plan->placement->workers[worker]->device != Device::CPU()) Fail("execution requires CPU:0 workers");
        session.SyncWorker(worker);  // Includes closed-session admission, without invoking CCL.
    }
    std::set<int> sources(plan->input_value_ids.begin(), plan->input_value_ids.end());
    sources.insert(plan->constant_value_ids.begin(), plan->constant_value_ids.end());
    if (sources.size() != initial.size()) Fail("initial values must exactly match inputs and constants");
    Availability proof{std::vector<std::vector<bool>>(plan->num_values, std::vector<bool>(workers)),
                       std::vector<std::vector<uint64_t>>(plan->num_values, std::vector<uint64_t>(workers))};
    for (int value : sources) {
        if (!initial.count(value)) Fail("missing initial value " + std::to_string(value));
        for (int worker = 0; worker < workers; ++worker) {
            const NDArray array = session.Get(worker, initial.at(value)); // Also verifies DRef ownership.
            if (array.defined()) {
                try { ValidateArray(plan, value, worker, array); }
                catch (...) { Fail("initial value " + std::to_string(value) + " worker " + std::to_string(worker) + ": " + FailureText(std::current_exception())); }
                proof.present[value][worker] = true;
            }
        }
    }
    const auto require = [&](int value, int worker) {
        if (worker < 0 || worker >= workers || !proof.present[value][worker]) {
            Fail("value " + std::to_string(value) + " unavailable on worker " + std::to_string(worker));
        }
    };
    const auto produce = [&](int value, int worker, uint64_t alignment) {
        proof.present[value][worker] = true;
        proof.alignment[value][worker] = alignment;
    };
    for (int value : sources) require(value, Home(plan, value));
    for (size_t index = 0; index < plan->nodes.size(); ++index) {
        try {
            const auto& ref = plan->nodes[index];
            if (const auto* kernel = ref.As<KernelExecNode>()) {
                if (!module.defined() || !module.IsReady()) Fail("requires a ready bound CompiledModule");
                if (kernel->kernel_symbol.empty() || !module.HasFunction(kernel->kernel_symbol)) Fail("module entry is missing: " + kernel->kernel_symbol);
                const auto signature = module.signature(kernel->kernel_symbol);
                if (kernel->kernel_abi.empty() || kernel->kernel_abi != signature.CanonicalBytes()) Fail("kernel ABI mismatch: " + kernel->kernel_symbol);
                const auto metadata = module.launch_metadata(kernel->kernel_symbol);
                metadata.Validate();
                if (metadata->device != Device::CPU() || metadata->backend != codegen::CodeGenBackend::kLLVM) Fail("kernel requires CPU:0 LLVM metadata");
                const auto& contract = api::internal::BorrowCompiledModuleInvocationContract(module, kernel->kernel_symbol);
                if (!contract.IsConstantShape(signature) || !contract.runtime_extent_scalars().empty()) Fail("only static stateless module invocation is supported");
                const auto& constants = api::internal::BorrowCompiledModuleConstants(module);
                const auto target_bytes = module.As<api::CompiledModuleNode>()->target_.CanonicalBytes();
                for (int worker : kernel->worker_set) {
                    if (target_bytes != plan->placement->workers[worker]->target.CanonicalBytes()) Fail("module Target mismatch on worker " + std::to_string(worker));
                }
                size_t input = 0, output = 0, output_bytes = 0;
                const auto arguments = signature.arguments();
                for (size_t slot = 0; slot < arguments.size(); ++slot) {
                    const auto& arg = arguments[slot];
                    if (arg->role == KernelArgRole::kConstant) {
                        if (!constants.count(arg->constant_key)) Fail("module constant is missing");
                        api::ValidateKernelArgument(signature, slot, arg, constants.at(arg->constant_key), constants);
                        continue;
                    }
                    const bool is_input = arg->role == KernelArgRole::kInput;
                    if ((!is_input && arg->role != KernelArgRole::kOutput) || (is_input && arg->mutable_data)) Fail("state, extents and mutable inputs are unsupported");
                    const auto& ids = is_input ? kernel->input_values : kernel->output_values;
                    size_t& position = is_input ? input : output;
                    if (position >= ids.size()) Fail("kernel value arity mismatch");
                    const int value = ids[position];
                    const auto expected = ValueSpec(plan, value, kernel->worker_set[0]);
                    if (!SameShape(arg.shape(), expected.shape()) || !api::SameDType(arg->dtype, expected->dtype) || arg->device != expected->device) Fail("kernel value contract mismatch for value " + std::to_string(value));
                    if (!is_input) {
                        const auto& tensor = contract.outputs()[position];
                        const auto shape = arg.shape();
                        size_t bytes = arg->dtype.bits / 8 * arg->dtype.lanes;
                        if (std::find(shape.begin(), shape.end(), 0) != shape.end()) bytes = 0;
                        else for (auto dim : shape) bytes *= static_cast<size_t>(dim); // Plan checked overflow.
                        if (bytes > tensor.max_bytes || bytes > std::numeric_limits<size_t>::max() - output_bytes) Fail("module output byte budget exceeded");
                        output_bytes += bytes;
                        for (size_t axis = 0; axis < shape.size(); ++axis) {
                            if (tensor.logical[axis].Evaluate({}) != static_cast<uint64_t>(shape[axis]) ||
                                tensor.valid[axis].Evaluate({}) != static_cast<uint64_t>(shape[axis])) Fail("partial logical/valid extents are unsupported");
                        }
                    }
                    for (int worker : kernel->worker_set) {
                        if (is_input) {
                            require(value, worker);
                            if (initial.count(value)) {
                                api::ValidateKernelArgument(signature, slot, arg, session.Get(worker, initial.at(value)));
                            } else if (proof.alignment[value][worker] % arg->alignment != 0) {
                                Fail("producer cannot guarantee kernel input alignment on worker " + std::to_string(worker));
                            }
                        } else {
                            if (arg->alignment > std::numeric_limits<size_t>::max()) Fail("kernel alignment exceeds host size type");
                            produce(value, worker, std::max<uint64_t>(arg->alignment, alignof(void*)));
                        }
                    }
                    ++position;
                }
                if (input != kernel->input_values.size() || output != kernel->output_values.size()) Fail("kernel value arity mismatch");
                if (contract.run_byte_budget() && output_bytes > contract.run_byte_budget()) Fail("module run byte budget exceeded");
            } else if (const auto* comm = ref.As<CommExecNode>()) {
                const auto& attrs = comm->attrs;
                if (attrs.async || attrs.group_id != 0 || (!attrs.kind.empty() && attrs.kind != comm->op_name)) Fail("unsupported communication attrs");
                if (attrs.reduce_kind != "sum") Fail("only sum reduction is supported");
                const bool transfer = comm->op_name == "device.send_to_worker" || comm->op_name == "device.recv_from_worker";
                if ((!transfer && attrs.root_worker != 0) || attrs.root_worker < 0 || attrs.root_worker >= workers) Fail("invalid communication root_worker");
                if (comm->op_name != "device.copy" && (attrs.src_virtual_device.defined() || attrs.dst_virtual_device.defined())) Fail("explicit src/dst placement is only supported for device.copy");
                const int in = comm->input_values[0], out = comm->output_values[0];
                const auto in_shape = plan->value_shapes.at(in), out_shape = plan->value_shapes.at(out);
                const auto dtype = runtime::DataTypeFromString(plan->value_dtypes.at(in));
                if (!api::SameDType(dtype, runtime::DataTypeFromString(plan->value_dtypes.at(out)))) Fail("communication dtype mismatch");
                if (IsPointToPoint(comm)) {
                    const auto route = CopyWorkers(plan, comm);
                    require(in, route.first);
                    if (route.second < 0 || route.second >= workers) Fail("missing destination placement");
                    const int group_width = workers / session.num_groups();
                    if (attrs.in_group && route.first / group_width != route.second / group_width) Fail("point-to-point route crosses groups while in_group is true");
                    const std::set<int> participants{route.first, route.second};
                    if (participants != std::set<int>(comm->worker_set.begin(), comm->worker_set.end())) Fail("point-to-point worker_set mismatch");
                    if (!SameShape(in_shape, out_shape)) Fail("point-to-point shape mismatch");
                    produce(out, route.second, alignof(std::max_align_t));
                } else {
                    if (comm->worker_set.size() != static_cast<size_t>(workers)) Fail("collective worker_set must include every session worker");
                    const int width = attrs.in_group ? workers / session.num_groups() : workers;
                    const bool scatter = comm->op_name == "device.scatter_from_worker0";
                    const bool gather = comm->op_name == "device.gather_to_worker0";
                    if (scatter || gather) {
                        if (in_shape.empty() || out_shape.size() != in_shape.size() ||
                            !std::equal(in_shape.begin() + 1, in_shape.end(), out_shape.begin() + 1)) Fail("scatter/gather rank or trailing shape mismatch");
                        if (scatter && (in_shape[0] % width != 0 || out_shape[0] != in_shape[0] / width)) Fail("scatter count mismatch");
                        if (gather && (in_shape[0] > std::numeric_limits<int64_t>::max() / width || out_shape[0] != in_shape[0] * width)) Fail("gather count mismatch");
                    } else if (!SameShape(in_shape, out_shape)) Fail("collective shape mismatch");
                    if (comm->op_name == "device.allreduce" &&
                        (dtype.lanes != 1 || (dtype.code != kDLFloat && dtype.code != kDLInt) || (dtype.bits != 32 && dtype.bits != 64))) Fail("unsupported sum allreduce dtype");
                    for (int root = 0; root < workers; root += width) {
                        for (int worker = root; worker < root + width; ++worker) {
                            require(in, scatter || comm->op_name == "device.broadcast_from_worker0" ? root : worker);
                            if (!gather || worker == root) produce(out, worker, alignof(std::max_align_t));
                        }
                    }
                }
            }
        } catch (...) {
            Fail("node[" + std::to_string(index) + "]: " + FailureText(std::current_exception()));
        }
    }
    for (int value : plan->output_value_ids) require(value, Home(plan, value));
    return proof;
}

class WorkerScope {
public:
    explicit WorkerScope(int worker) : previous_(CurrentWorkerId()) { SetCurrentWorkerId(worker); }
    ~WorkerScope() { SetCurrentWorkerId(previous_); }
private:
    int previous_;
};

}  // namespace

ExecutionPlanExecutor::ExecutionPlanExecutor(DiscoSession session, std::shared_ptr<CCLBackend> backend)
    : session_(std::move(session)), ccl_backend_(std::move(backend)) {
    if (!session_.As<DiscoSessionNode>() || !ccl_backend_) Fail("executor requires a session and CCL backend");
}

ExecutionPlanExecutor::ExecutionPlanExecutor(DiscoSession session, api::CompiledModule module,
                                           std::shared_ptr<CCLBackend> backend)
    : ExecutionPlanExecutor(std::move(session), std::move(backend)) {
    if (!module.IsReady()) Fail("executor requires a ready bound CompiledModule");
    module_ = std::move(module);
}

Map<int, DRef> ExecutionPlanExecutor::Execute(const ExecutionPlan& plan, const Map<int, DRef>& initial) {
    profiling::ScopedSpan execute_span(profiling::CurrentContext(), MakeSpec("execute_plan"));
    try {
        const auto proof = Preflight(plan, session_, module_, initial);
        execute_span.AddMetric("node_count", static_cast<double>(plan->nodes.size()));
        execute_span.AddMetric("value_count", plan->num_values);
        Map<int, DRef> values;
        for (const auto& entry : initial) values.Set(entry.first, entry.second);
        for (size_t index = 0; index < plan->nodes.size(); ++index) {
            const auto& ref = plan->nodes[index];
            const auto* node = ref.As<ExecNodeBaseNode>();
            auto spec = MakeSpec(ref.As<KernelExecNode>() ? "execute_kernel_node" : ref.As<CommExecNode>() ? "execute_comm_node" : "execute_barrier_node");
            spec.fields["node_index"] = std::to_string(index);
            profiling::ScopedSpan node_span(profiling::CurrentContext(), std::move(spec));
            try {
                if (const auto* kernel = ref.As<KernelExecNode>()) ExecuteKernel(kernel, index, values);
                else if (const auto* comm = ref.As<CommExecNode>()) ExecuteComm(plan, comm, values);
                else for (int worker : node->worker_set) {
                    try { ccl_backend_->SyncWorker(session_, worker); }
                    catch (...) { Fail("barrier worker " + std::to_string(worker) + ": " + FailureText(std::current_exception())); }
                }
                for (int value : node->output_values) for (int worker = 0; worker < session_.num_workers(); ++worker) {
                    if (proof.present[value][worker]) ValidateArray(plan, value, worker, session_.Get(worker, values.at(value)));
                }
            } catch (...) {
                const std::string message = "node[" + std::to_string(index) + "]: " + FailureText(std::current_exception());
                node_span.SetStatus("error"); node_span.SetMessage(message);
                Fail(message);
            }
        }
        return values;
    } catch (...) {
        execute_span.SetStatus("error"); execute_span.SetMessage(FailureText(std::current_exception()));
        throw;
    }
}

DRef ExecutionPlanExecutor::ExecuteForOutput(const ExecutionPlan& plan, const Map<int, DRef>& initial) {
    const auto values = Execute(plan, initial);
    return values.at(plan->output_value);
}

Array<DRef> ExecutionPlanExecutor::ExecuteForOutputs(const ExecutionPlan& plan, const Map<int, DRef>& initial) {
    const auto values = Execute(plan, initial);
    Array<DRef> outputs;
    for (int value : plan->output_value_ids) outputs.push_back(values.at(value));
    return outputs;
}

void ExecutionPlanExecutor::ExecuteKernel(const KernelExecNode* kernel, size_t index, Map<int, DRef>& values) {
    const auto context = profiling::CurrentContext();
    const auto run = profiling::CurrentRunId(), parent = profiling::CurrentSpanId();
    std::vector<std::future<api::ModuleInvocationResult>> jobs;
    // ponytail: one joined task per participating worker per node; use a persistent
    // worker pool only when measured scheduling overhead justifies it.
    for (int worker : kernel->worker_set) {
        Array<NDArray> inputs;
        for (int value : kernel->input_values) inputs.push_back(session_.Get(worker, values.at(value)));
        jobs.push_back(std::async(std::launch::async, [module = module_, symbol = kernel->kernel_symbol,
                              inputs = std::move(inputs), context, run, parent, worker, index] {
            WorkerScope worker_scope(worker);
            profiling::ActivationScope activation(context, run);
            auto spec = MakeSpec("kernel_exec");
            spec.worker_id = worker; spec.kernel_symbol = symbol; spec.device = "cpu:0";
            spec.fields["node_index"] = std::to_string(index);
            profiling::ScopedSpan span(context, std::move(spec), run, parent);
            try {
                auto result = module.Invoke(symbol, inputs, DeviceStream::Default(Device::CPU()));
                result.operation.Wait();
                return result;
            } catch (...) {
                const auto message = "worker " + std::to_string(worker) + " kernel '" + symbol + "': " + FailureText(std::current_exception());
                span.SetStatus("error"); span.SetMessage(message);
                Fail(message);
            }
        }));
    }
    std::vector<api::ModuleInvocationResult> results;
    std::exception_ptr failure;
    for (auto& job : jobs) {
        try { results.push_back(job.get()); }
        catch (...) { if (!failure) failure = std::current_exception(); }
    }
    if (failure) std::rethrow_exception(failure);
    for (size_t output = 0; output < kernel->output_values.size(); ++output) {
        DRef ref = session_.NewDRef();
        for (size_t worker = 0; worker < kernel->worker_set.size(); ++worker) session_.Set(kernel->worker_set[worker], ref, results[worker].outputs.at(output).storage);
        values.Set(kernel->output_values[output], ref);
    }
}

void ExecutionPlanExecutor::ExecuteComm(const ExecutionPlan& plan, const CommExecNode* comm, Map<int, DRef>& values) {
    const DRef src = values.at(comm->input_values[0]), dst = session_.NewDRef();
    auto spec = MakeSpec("comm_exec"); spec.op_name = comm->op_name;
    spec.fields["in_group"] = comm->attrs.in_group ? "true" : "false";
    const auto route = CopyWorkers(plan, comm);
    if (IsPointToPoint(comm)) {
        spec.fields["src_worker"] = std::to_string(route.first);
        spec.fields["dst_worker"] = std::to_string(route.second);
    } else spec.fields["worker_count"] = std::to_string(session_.num_workers());
    profiling::ScopedSpan span(profiling::CurrentContext(), std::move(spec));
    try {
        if (comm->op_name == "device.copy") ccl_backend_->Copy(session_, src, dst, route.first, route.second);
        else if (comm->op_name == "device.send_to_worker") ccl_backend_->SendToWorker(session_, src, dst, route.second);
        else if (comm->op_name == "device.recv_from_worker") ccl_backend_->RecvFromWorker(session_, src, dst, route.first);
        else if (comm->op_name == "device.allreduce") ccl_backend_->AllReduce(session_, src, dst, comm->attrs.reduce_kind, comm->attrs.in_group);
        else if (comm->op_name == "device.broadcast_from_worker0") ccl_backend_->BroadcastFromWorker0(session_, src, dst, comm->attrs.in_group);
        else if (comm->op_name == "device.scatter_from_worker0") ccl_backend_->ScatterFromWorker0(session_, src, dst, comm->attrs.in_group);
        else if (comm->op_name == "device.gather_to_worker0") ccl_backend_->GatherToWorker0(session_, src, dst, comm->attrs.in_group);
        values.Set(comm->output_values[0], dst);
    } catch (...) {
        const auto message = "communication " + comm->op_name +
            (IsPointToPoint(comm) ? " workers " + std::to_string(route.first) + " -> " + std::to_string(route.second) : " across " + std::to_string(session_.num_workers()) + " workers") + ": " + FailureText(std::current_exception());
        span.SetStatus("error"); span.SetMessage(message); Fail(message);
    }
}

}  // namespace kxc::disco
