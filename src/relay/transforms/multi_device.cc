#include "relay/transforms/multi_device.h"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "base/packedfunc.h"
#include "base/registry.h"
#include "relay/op.h"
#include "relay/transforms/lower.h"

namespace kxc {
namespace relay {

namespace {

VirtualDevice GetVirtualDeviceFromExpr(const Expr& expr) {
    if (!expr.defined()) {
        return VirtualDevice();
    }
    const RelayNode* relay_node = dynamic_cast<const RelayNode*>(expr.get());
    if (!relay_node) {
        return VirtualDevice();
    }
    return relay_node->virtual_device_;
}

bool SameVirtualDevice(const VirtualDevice& a, const VirtualDevice& b) {
    if (!a.defined() || !b.defined()) {
        return !a.defined() && !b.defined();
    }
    if (a.get() == b.get()) {
        return true;
    }
    auto extract_dev = [](const VirtualDevice& vd) -> std::pair<int, int> {
        if (vd->device_obj.defined()) {
            const auto* dev = static_cast<const class Device*>(vd->device_obj.get());
            return {static_cast<int>(dev->device_type()), dev->device_id()};
        }
        if (vd->target.defined()) {
            return {static_cast<int>(vd->target->device_type), vd->target->device_id};
        }
        return {static_cast<int>(kUnknown), kInvalidVirtualDeviceId};
    };
    auto da = extract_dev(a);
    auto db = extract_dev(b);
    return da.first == db.first && da.second == db.second &&
           a->memory_scope == b->memory_scope &&
           a->virtual_device_id == b->virtual_device_id;
}

void PushUniqueWorker(std::vector<int>* workers, int worker_id) {
    if (worker_id < 0) {
        return;
    }
    for (int w : *workers) {
        if (w == worker_id) {
            return;
        }
    }
    workers->push_back(worker_id);
}

Array<int> ToArray(const std::vector<int>& values) {
    Array<int> out;
    for (int v : values) {
        out.push_back(v);
    }
    return out;
}

int ResolveWorkerForVirtualDevice(const PassContext& pass_ctx, const VirtualDevice& vd) {
    if (!vd.defined()) {
        return -1;
    }
    if (pass_ctx.has_disco_placement()) {
        return FindWorkerForVirtualDevice(pass_ctx.disco_placement(), vd);
    }
    if (pass_ctx.default_device_obj().defined()) {
        const auto* dev = static_cast<const class Device*>(pass_ctx.default_device_obj().get());
        if (dev) {
            return dev->device_id();
        }
    }
    return 0;
}

Array<int> WorkerSetFromVirtualDevice(const PassContext& pass_ctx, const VirtualDevice& vd) {
    int worker = ResolveWorkerForVirtualDevice(pass_ctx, vd);
    if (worker < 0) {
        return {0};
    }
    return {worker};
}

std::string DTypeToString(const DLDataType& dtype) {
    std::string prefix = "unknown";
    if (dtype.code == kDLFloat) {
        prefix = "float";
    } else if (dtype.code == kDLInt) {
        prefix = "int";
    } else if (dtype.code == kDLUint) {
        prefix = dtype.bits == 1 ? "bool" : "uint";
    }
    if (prefix == "bool") {
        return "bool";
    }
    if (dtype.lanes > 1) {
        return prefix + std::to_string(dtype.bits) + "x" + std::to_string(dtype.lanes);
    }
    return prefix + std::to_string(dtype.bits);
}

std::string GetCallOpName(const CallNode* call) {
    if (!call) {
        return "";
    }
    const OpNode* op_node = call->op.As<OpNode>();
    if (!op_node) {
        return "";
    }
    return op_node->name;
}

class DeviceCommunicationInserter : public RelayPass {
public:
    Expr VisitCall(const CallNode* op, const Expr& ref) override {
        Expr rewritten = RelayPass::VisitCall(op, ref);
        const CallNode* call = rewritten.As<CallNode>();
        if (!call) {
            return rewritten;
        }

        const std::string op_name = GetCallOpName(call);
        if (op_name.empty()) {
            return rewritten;
        }

        if (IsCommunicationOpName(op_name)) {
            return NormalizeCommunicationCall(call, rewritten, op_name);
        }

        const VirtualDevice consumer_vd = GetVirtualDeviceFromExpr(rewritten);
        if (!consumer_vd.defined()) {
            return rewritten;
        }

        Array<Expr> new_args;
        bool changed = false;
        for (const auto& arg : call->args) {
            Expr new_arg = arg;
            const VirtualDevice producer_vd = GetVirtualDeviceFromExpr(arg);
            if (producer_vd.defined() && !SameVirtualDevice(producer_vd, consumer_vd)) {
                DeviceCopyAttrs attrs =
                    DeviceCopyAttrs::Create(producer_vd, consumer_vd, false, true);
                Call copy_call(Op::Get("device.copy"), {arg}, attrs);
                copy_call.set_virtual_device(consumer_vd);
                new_arg = copy_call;
                changed = true;
            }
            new_args.push_back(new_arg);
        }

        if (!changed) {
            return rewritten;
        }

        Call updated(call->op, new_args, call->attrs);
        updated.set_virtual_device(consumer_vd);
        return updated;
    }

private:
    Expr NormalizeCommunicationCall(const CallNode* call, const Expr& ref,
                                    const std::string& op_name) {
        if (op_name == "device.copy") {
            VirtualDevice src_vd =
                call->args.empty() ? VirtualDevice() : GetVirtualDeviceFromExpr(call->args[0]);
            VirtualDevice dst_vd = GetVirtualDeviceFromExpr(ref);
            bool async = false;
            bool in_group = true;
            if (call->attrs.defined()) {
                if (auto* attrs = call->attrs.As<DeviceCopyAttrsNode>()) {
                    if (attrs->src_virtual_device.defined()) {
                        src_vd = attrs->src_virtual_device;
                    }
                    if (attrs->dst_virtual_device.defined()) {
                        dst_vd = attrs->dst_virtual_device;
                    }
                    async = attrs->async;
                    in_group = attrs->in_group;
                }
            }
            Call normalized(call->op, call->args,
                            DeviceCopyAttrs::Create(src_vd, dst_vd, async, in_group));
            if (dst_vd.defined()) {
                normalized.set_virtual_device(dst_vd);
            } else {
                VirtualDevice vd = GetVirtualDeviceFromExpr(ref);
                if (vd.defined()) {
                    normalized.set_virtual_device(vd);
                }
            }
            return normalized;
        }

        std::string kind = op_name;
        std::string reduce_kind = "sum";
        bool in_group = true;
        int group_id = 0;
        int root_worker = 0;
        if (call->attrs.defined()) {
            if (auto* attrs = call->attrs.As<CollectiveAttrsNode>()) {
                if (!attrs->kind.empty()) {
                    kind = attrs->kind;
                }
                reduce_kind = attrs->reduce_kind;
                in_group = attrs->in_group;
                group_id = attrs->group_id;
                root_worker = attrs->root_worker;
            }
        }
        Call normalized(call->op, call->args,
                        CollectiveAttrs::Create(kind, reduce_kind, in_group, group_id,
                                                root_worker));
        VirtualDevice vd = GetVirtualDeviceFromExpr(ref);
        if (vd.defined()) {
            normalized.set_virtual_device(vd);
        }
        return normalized;
    }
};

class RelayToExecPlanBuilder : public RelayPassFunctor<int> {
public:
    explicit RelayToExecPlanBuilder(PassContext pass_ctx) : pass_ctx_(std::move(pass_ctx)) {}

    ExecutionPlan Build(const Function& func) {
        for (const auto& param : func->params) {
            int value_id = AllocateValue(param);
            var_bindings_[param.get()] = value_id;
            input_value_ids_.push_back(value_id);
            CaptureValueInfoFromType(value_id, param->type_annotation);
        }
        int out = Visit(func->body);
        for (const auto& param : func->params) {
            var_bindings_.erase(param.get());
        }
        return ExecutionPlan(nodes_, value_virtual_devices_, input_value_ids_,
                             constant_value_ids_, value_shapes_, value_dtypes_, next_value_id_,
                             pass_ctx_, out);
    }

protected:
    int Visit(const Expr& expr) override {
        if (!expr.defined()) {
            return -1;
        }
        auto it = memo_.find(expr.get());
        if (it != memo_.end()) {
            return it->second;
        }
        int value = RelayPassFunctor::Visit(expr);
        memo_[expr.get()] = value;
        return value;
    }

    int VisitVar(const VarNode* op, const Expr& ref) override {
        (void)op;
        auto it = var_bindings_.find(ref.get());
        if (it != var_bindings_.end()) {
            return it->second;
        }
        int value_id = AllocateValue(ref);
        return value_id;
    }

    int VisitConstant(const ConstantNode* op, const Expr& ref) override {
        int value_id = AllocateValue(ref);
        constant_value_ids_.push_back(value_id);
        Array<int64_t> shape;
        if (op && op->data.defined()) {
            for (const auto& dim : op->data->shape) {
                shape.push_back(dim);
            }
            value_shapes_.Set(value_id, shape);
            value_dtypes_.Set(value_id, DTypeToString(op->data->dl_tensor.dtype));
        }
        return value_id;
    }

    int VisitCall(const CallNode* op, const Expr& ref) override {
        std::string op_name = GetCallOpName(op);
        if (op_name.empty()) {
            throw std::runtime_error("LowerRelayToExecPlanPass only supports Call.op=Op");
        }

        Array<int> inputs;
        std::vector<int> input_vec;
        for (const auto& arg : op->args) {
            int arg_value = Visit(arg);
            input_vec.push_back(arg_value);
            inputs.push_back(arg_value);
        }

        int output = AllocateValue(ref);
        Array<int> outputs = {output};

        if (IsCommunicationOpName(op_name)) {
            Array<int> worker_set = ResolveCommunicationWorkerSet(op_name, op, ref, input_vec);
            nodes_.push_back(ObjectRef(CommExec(op_name, op->attrs, inputs, outputs, worker_set)));
            return output;
        }

        Array<int> worker_set = ResolveComputeWorkerSet(ref, input_vec);
        nodes_.push_back(
            ObjectRef(KernelExec(op_name, tir::PrimFunc(), inputs, outputs, worker_set)));
        return output;
    }

    int VisitFunction(const FunctionNode* op, const Expr& ref) override {
        (void)ref;
        return Visit(op->body);
    }

    int VisitLet(const LetNode* op, const Expr& ref) override {
        int value_id = Visit(op->value);
        var_bindings_[op->var.get()] = value_id;
        int body_id = Visit(op->body);
        var_bindings_.erase(op->var.get());
        memo_[ref.get()] = body_id;
        return body_id;
    }

    int VisitTuple(const TupleNode* op, const Expr& ref) override {
        Array<int> inputs;
        std::vector<int> input_vec;
        for (const auto& field : op->fields) {
            int field_id = Visit(field);
            input_vec.push_back(field_id);
            inputs.push_back(field_id);
        }
        int output = AllocateValue(ref);
        Array<int> outputs = {output};
        Array<int> worker_set = ResolveComputeWorkerSet(ref, input_vec);
        nodes_.push_back(ObjectRef(
            KernelExec("__make_tuple__", tir::PrimFunc(), inputs, outputs, worker_set)));
        return output;
    }

    int VisitTupleGetItem(const TupleGetItemNode* op, const Expr& ref) override {
        int tuple_id = Visit(op->tuple);
        int output = AllocateValue(ref);
        Array<int> inputs = {tuple_id};
        Array<int> outputs = {output};
        Array<int> worker_set = ResolveComputeWorkerSet(ref, {tuple_id});
        nodes_.push_back(ObjectRef(KernelExec("__tuple_getitem_" + std::to_string(op->index),
                                              tir::PrimFunc(), inputs, outputs, worker_set)));
        return output;
    }

    int VisitIf(const IfNode* op, const Expr& ref) override {
        (void)op;
        (void)ref;
        throw std::runtime_error(
            "LowerRelayToExecPlanPass does not support IfNode in phase-1");
    }

    int VisitDefault(const Expr& expr) override {
        (void)expr;
        throw std::runtime_error(
            "LowerRelayToExecPlanPass encountered an unsupported Relay node");
    }

private:
    void CaptureValueInfoFromType(int value_id, const Type& type) {
        const auto* tensor_type = type.As<TensorTypeNode>();
        if (!tensor_type) {
            return;
        }
        Array<int64_t> shape;
        for (const auto& dim : tensor_type->shape) {
            shape.push_back(dim);
        }
        value_shapes_.Set(value_id, shape);
        value_dtypes_.Set(value_id, tensor_type->dtype);
    }

    int AllocateValue(const Expr& expr) {
        int value_id = next_value_id_++;
        const VirtualDevice vd = GetVirtualDeviceFromExpr(expr);
        if (vd.defined()) {
            value_virtual_devices_.Set(value_id, vd);
        }
        return value_id;
    }

    Array<int> ResolveComputeWorkerSet(const Expr& ref, const std::vector<int>& input_values) {
        VirtualDevice vd = GetVirtualDeviceFromExpr(ref);
        if (!vd.defined()) {
            for (int input_id : input_values) {
                if (value_virtual_devices_.count(input_id)) {
                    vd = value_virtual_devices_.at(input_id);
                    break;
                }
            }
        }
        return WorkerSetFromVirtualDevice(pass_ctx_, vd);
    }

    Array<int> ResolveCommunicationWorkerSet(const std::string& op_name, const CallNode* call,
                                             const Expr& ref,
                                             const std::vector<int>& input_values) {
        std::vector<int> workers;
        if (op_name == "device.copy") {
            VirtualDevice src_vd =
                call->args.empty() ? VirtualDevice() : GetVirtualDeviceFromExpr(call->args[0]);
            VirtualDevice dst_vd = GetVirtualDeviceFromExpr(ref);
            if (call->attrs.defined()) {
                if (auto* attrs = call->attrs.As<DeviceCopyAttrsNode>()) {
                    if (attrs->src_virtual_device.defined()) {
                        src_vd = attrs->src_virtual_device;
                    }
                    if (attrs->dst_virtual_device.defined()) {
                        dst_vd = attrs->dst_virtual_device;
                    }
                }
            }
            PushUniqueWorker(&workers, ResolveWorkerForVirtualDevice(pass_ctx_, src_vd));
            PushUniqueWorker(&workers, ResolveWorkerForVirtualDevice(pass_ctx_, dst_vd));
        } else if (op_name == "device.allreduce" || op_name == "device.broadcast_from_worker0" ||
                   op_name == "device.scatter_from_worker0" ||
                   op_name == "device.gather_to_worker0") {
            if (pass_ctx_.has_disco_placement()) {
                for (const auto& worker : pass_ctx_.disco_placement()->workers) {
                    if (worker.defined()) {
                        PushUniqueWorker(&workers, worker->worker_id);
                    }
                }
            }
        } else if (op_name == "device.send_to_worker" || op_name == "device.recv_from_worker") {
            VirtualDevice vd = GetVirtualDeviceFromExpr(ref);
            PushUniqueWorker(&workers, ResolveWorkerForVirtualDevice(pass_ctx_, vd));
        }

        if (workers.empty()) {
            for (int input : input_values) {
                if (value_virtual_devices_.count(input)) {
                    PushUniqueWorker(
                        &workers,
                        ResolveWorkerForVirtualDevice(pass_ctx_,
                                                      value_virtual_devices_.at(input)));
                }
            }
        }
        if (workers.empty()) {
            workers.push_back(0);
        }
        return ToArray(workers);
    }

    PassContext pass_ctx_;
    Array<ObjectRef> nodes_;
    Map<int, VirtualDevice> value_virtual_devices_;
    Array<int> input_value_ids_;
    Array<int> constant_value_ids_;
    Map<int, Array<int64_t>> value_shapes_;
    Map<int, std::string> value_dtypes_;
    std::unordered_map<const Object*, int> memo_;
    std::unordered_map<const Object*, int> var_bindings_;
    int next_value_id_{0};
};

}  // namespace

PassContext BuildDiscoPlacementPass(const Function& func) {
    return ::kxc::BuildDiscoPlacementPass(func);
}

Function InsertDeviceCommunicationPass(const Function& func) {
    DeviceCommunicationInserter pass;
    return pass.Mutate(func);
}

tir::PrimFunc LowerRelayComputeToTIRPass(const Function& func) {
    return LowerToTIR(func);
}

ExecutionPlan LowerRelayToExecPlanPass(const Function& func) {
    if (!func.defined()) {
        throw std::runtime_error("LowerRelayToExecPlanPass expects a defined function");
    }
    PassContext pass_ctx = PassContext::Current();
    if (!pass_ctx.defined()) {
        pass_ctx = ::kxc::relay::BuildDiscoPlacementPass(func);
    }
    PassContext::Scope scope(pass_ctx);
    Function with_comm = InsertDeviceCommunicationPass(func);
    RelayToExecPlanBuilder builder(PassContext::Current());
    return builder.Build(with_comm);
}

KXC_REGISTER_GLOBAL("kxc.relay.transform.build_disco_placement")
    .set_body(ToPackedFunc([](Function func) -> std::string {
        return ::kxc::relay::BuildDiscoPlacementPass(func).ToString();
    }));

KXC_REGISTER_GLOBAL("kxc.relay.transform.insert_device_communication")
    .set_body(ToPackedFunc([](Function func) -> ObjectRef {
        return ObjectRef(InsertDeviceCommunicationPass(func));
    }));

KXC_REGISTER_GLOBAL("kxc.relay.transform.lower_compute_to_tir")
    .set_body(ToPackedFunc([](Function func) -> ObjectRef {
        return ObjectRef(LowerRelayComputeToTIRPass(func));
    }));

KXC_REGISTER_GLOBAL("kxc.relay.transform.lower_to_exec_plan")
    .set_body(ToPackedFunc([](Function func) -> ObjectRef {
        return ObjectRef(LowerRelayToExecPlanPass(func));
    }));

KXC_REGISTER_GLOBAL("kxc.relay.transform.lower_to_exec_plan_json")
    .set_body(ToPackedFunc([](Function func) -> std::string {
        return SerializeExecutionPlanToJson(LowerRelayToExecPlanPass(func));
    }));

KXC_REGISTER_GLOBAL("kxc.relay.transform.lower_to_exec_plan_json_file")
    .set_body(ToPackedFunc([](Function func, std::string path) -> std::string {
        ExecutionPlan plan = LowerRelayToExecPlanPass(func);
        SaveExecutionPlanToJsonFile(plan, path);
        return path;
    }));

}  // namespace relay
}  // namespace kxc
