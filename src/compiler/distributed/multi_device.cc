/*! \file src/compiler/distributed/multi_device.cc
 * \brief 实现 Relay 优化 pass 及其 pipeline 集成。
 */

#include "kxc/compiler/distributed/multi_device.h"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "kxc/ffi/packed_func.h"
#include "kxc/ffi/registration.h"
#include "kxc/relay/op.h"
#include "../../relay/distributed/plan_adapter.h"

namespace kxc {
namespace relay {

namespace {

OperatorSpec DeviceCopyOperatorSpec() {
    OperatorSpec spec;
    spec.name = "device.copy";
    spec.category = "device";
    spec.input_arity.num_inputs = 1;
    spec.output_arity = 1;
    spec.type_relation_key = "device_copy";
    spec.effect = OperatorEffectKind::kDeviceCommunication;
    spec.deterministic = true;
    spec.alias_contract = "no_alias";
    spec.lowering_kind = OperatorLoweringKind::kExecPlan;
    spec.lowering_key = "exec_plan";
    return spec;
}

const Op kDeviceCopyOpRegistration = Op::Register(DeviceCopyOperatorSpec());

// 从 Relay 节点读取可选的编译期 VirtualDevice 放置信息。
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

// 按物理设备、Target、内存域和逻辑编号比较两个放置约束。
bool SameVirtualDevice(const VirtualDevice& a, const VirtualDevice& b) {
    if (!a.defined() || !b.defined()) {
        return !a.defined() && !b.defined();
    }
    if (a.get() == b.get()) {
        return true;
    }
    auto extract_dev = [](const VirtualDevice& vd) -> std::pair<int, int> {
        if (vd->device.defined()) {
            return {static_cast<int>(vd->device.device_type()), vd->device.device_id()};
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

// 向 worker 列表追加非负且尚未出现的编号。
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

// 将临时 worker vector 转为执行计划可持有的对象系统 Array。
Array<int> ToArray(const std::vector<int>& values) {
    Array<int> out;
    for (int v : values) {
        out.push_back(v);
    }
    return out;
}

// 通过 DiscoPlacement 把 VirtualDevice 映射到 worker。
int ResolveWorkerForVirtualDevice(const DiscoPlacement& placement,
                                  const VirtualDevice& vd) {
    if (!vd.defined()) {
        return -1;
    }
    if (placement.defined()) {
        return FindWorkerForVirtualDevice(placement, vd);
    }
    // 没有 placement 时不能用物理 device_id 推导 worker；0 仅表示单 worker 默认值。
    return 0;
}

// 把单个 VirtualDevice 解析为执行节点 worker 集合。
Array<int> WorkerSetFromVirtualDevice(const DiscoPlacement& placement,
                                      const VirtualDevice& vd) {
    int worker = ResolveWorkerForVirtualDevice(placement, vd);
    if (worker < 0) {
        return {0};
    }
    return {worker};
}

// 将常量 DLPack dtype 转为执行计划中的文本 dtype。
std::string DTypeToString(const DLDataType& dtype) {
    std::string prefix = "unknown";
    if (dtype.code == kDLFloat) {
        prefix = "float";
    } else if (dtype.code == kDLInt) {
        prefix = "int";
    } else if (dtype.code == kDLUInt) {
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

// 取得直接 Op 调用的算子名称。
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

// 在生产者与消费者放置不同处插入 device.copy，并规范化已有通信属性。
class DeviceCommunicationInserter : public RelayPass {
public:
    // 重写调用实参，必要时插入跨设备复制节点。
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
    // 补齐 device.copy 或 collective 的默认属性和结果放置。
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

// 把 Relay 数据流线性化为带值编号、worker 集合和张量元数据的 ExecutionPlan。
class RelayToExecPlanBuilder : public RelayPassFunctor<int> {
public:
    // 保存用于设备到 worker 映射的 PassContext。
    RelayToExecPlanBuilder(PassContext pass_ctx, DiscoPlacement placement)
        : pass_ctx_(std::move(pass_ctx)), placement_(std::move(placement)) {}

    // 为参数、常量和函数体分配值编号并组装最终执行计划。
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
                              pass_ctx_, placement_, out);
    }

protected:
    // 按表达式对象身份记忆化其执行计划值编号。
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

    // 解析已绑定变量，未绑定自由变量分配新值编号。
    int VisitVar(const VarNode* op, const Expr& ref) override {
        (void)op;
        auto it = var_bindings_.find(ref.get());
        if (it != var_bindings_.end()) {
            return it->second;
        }
        int value_id = AllocateValue(ref);
        return value_id;
    }

    // 登记常量值及其 Storage-backed NDArray shape、dtype 元数据。
    int VisitConstant(const ConstantNode* op, const Expr& ref) override {
        int value_id = AllocateValue(ref);
        constant_value_ids_.push_back(value_id);
        Array<int64_t> shape;
        if (op && op->data.defined()) {
            shape = op->data.shape();
            value_shapes_.Set(value_id, shape);
            value_dtypes_.Set(value_id, DTypeToString(op->data.dtype()));
        }
        return value_id;
    }

    // 将普通调用生成 KernelExec，将通信调用生成 CommExec。
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
            nodes_.push_back(ObjectRef(CommExec(
                op_name,
                distributed_internal::AdaptCommExecAttrs(op_name, op->attrs),
                inputs, outputs, worker_set)));
            return output;
        }

        Array<int> worker_set = ResolveComputeWorkerSet(ref, input_vec);
        nodes_.push_back(
            ObjectRef(KernelExec(op_name, tir::PrimFunc(), inputs, outputs, worker_set)));
        return output;
    }

    // 函数节点以其 body 的计划值作为结果。
    int VisitFunction(const FunctionNode* op, const Expr& ref) override {
        (void)ref;
        return Visit(op->body);
    }

    // 建立 let 变量到计划值的词法绑定。
    int VisitLet(const LetNode* op, const Expr& ref) override {
        int value_id = Visit(op->value);
        var_bindings_[op->var.get()] = value_id;
        int body_id = Visit(op->body);
        var_bindings_.erase(op->var.get());
        memo_[ref.get()] = body_id;
        return body_id;
    }

    // 把 tuple 构造表示为内部合成 KernelExec。
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

    // 把 tuple 字段读取表示为内部合成 KernelExec。
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

    // 第一阶段执行计划暂不支持控制流，明确拒绝 If/While。
    int VisitWhile(const WhileNode* op, const Expr& ref) override {
        (void)op;
        (void)ref;
        throw std::runtime_error(
            "LowerRelayToExecPlanPass does not support WhileNode in phase-1");
    }

    int VisitIf(const IfNode* op, const Expr& ref) override {
        (void)op;
        (void)ref;
        throw std::runtime_error(
            "LowerRelayToExecPlanPass does not support IfNode in phase-1");
    }

    // 对未建模的 Relay 节点给出明确 lowering 错误。
    int VisitDefault(const Expr& expr) override {
        (void)expr;
        throw std::runtime_error(
            "LowerRelayToExecPlanPass encountered an unsupported Relay node");
    }

private:
    // 从 TensorType 捕获值的静态 shape 与 dtype。
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

    // 分配计划值编号并记录表达式的 VirtualDevice。
    int AllocateValue(const Expr& expr) {
        int value_id = next_value_id_++;
        const VirtualDevice vd = GetVirtualDeviceFromExpr(expr);
        if (vd.defined()) {
            value_virtual_devices_.Set(value_id, vd);
        }
        return value_id;
    }

    // 优先使用结果放置，否则继承首个有放置输入的 worker 集合。
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
        return WorkerSetFromVirtualDevice(placement_, vd);
    }

    // 根据复制端点或 collective 分组属性解析通信参与 worker。
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
            PushUniqueWorker(&workers, ResolveWorkerForVirtualDevice(placement_, src_vd));
            PushUniqueWorker(&workers, ResolveWorkerForVirtualDevice(placement_, dst_vd));
        } else if (op_name == "device.allreduce" || op_name == "device.broadcast_from_worker0" ||
                   op_name == "device.scatter_from_worker0" ||
                   op_name == "device.gather_to_worker0") {
            if (placement_.defined()) {
                for (const auto& worker : placement_->workers) {
                    if (worker.defined()) {
                        PushUniqueWorker(&workers, worker->worker_id);
                    }
                }
            }
        } else if (op_name == "device.send_to_worker" || op_name == "device.recv_from_worker") {
            VirtualDevice vd = GetVirtualDeviceFromExpr(ref);
            PushUniqueWorker(&workers, ResolveWorkerForVirtualDevice(placement_, vd));
        }

        if (workers.empty()) {
            for (int input : input_values) {
                if (value_virtual_devices_.count(input)) {
                    PushUniqueWorker(
                        &workers,
                        ResolveWorkerForVirtualDevice(placement_,
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
    DiscoPlacement placement_;
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

// 插入并规范化跨设备复制及 collective 调用。
Function InsertDeviceCommunicationPass(const Function& func) {
    DeviceCommunicationInserter pass;
    return pass.Mutate(func);
}

DiscoPlacement BuildDiscoPlacementPass(const Function& func) {
    const PassContext pass_ctx = PassContextFromRelay(func);
    return BuildDiscoPlacement(pass_ctx.virtual_devices(), /*num_groups=*/1);
}

// 生成包含放置、通信和值元数据的 ExecutionPlan。
ExecutionPlan LowerRelayToExecPlanPass(const Function& func) {
    if (!func.defined()) {
        throw std::runtime_error("LowerRelayToExecPlanPass expects a defined function");
    }
    PassContext pass_ctx = PassContext::Current();
    if (!pass_ctx.defined()) {
        pass_ctx = PassContextFromRelay(func);
    }
    DiscoPlacement placement = BuildDiscoPlacementPass(func);
    PassContext::Scope scope(pass_ctx);
    Function with_comm = InsertDeviceCommunicationPass(func);
    RelayToExecPlanBuilder builder(PassContext::Current(), placement);
    return builder.Build(with_comm);
}

KXC_REGISTER_GLOBAL("kxc.relay.transform.build_disco_placement")
    .set_body(ToPackedFunc([](Function func) -> std::string {
        return BuildDiscoPlacementPass(func).ToString();
    }));

KXC_REGISTER_GLOBAL("kxc.relay.transform.insert_device_communication")
    .set_body(ToPackedFunc([](Function func) -> ObjectRef {
        return ObjectRef(InsertDeviceCommunicationPass(func));
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

namespace kxc::builtin_anchor {
void CompilerDistributed() {}
}  // namespace kxc::builtin_anchor
