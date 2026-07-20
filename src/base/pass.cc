/*! \file src/base/pass.cc
 * \brief 实现基础对象、设备、NDArray、Target、执行计划、PassContext 和 profiling 支撑逻辑。
 */

#include "base/pass.h"

#include "base/target.h"

#include <sstream>
#include <string>
#include <stdexcept>
#include <unordered_set>

namespace kxc {

namespace {

constexpr const char* kPassCtxPrimaryVirtualDeviceAttr = "kxc.pass_ctx.primary_virtual_device";
constexpr const char* kPassCtxDefaultTargetAttr = "kxc.pass_ctx.default_target";
constexpr const char* kPassCtxDefaultDeviceAttr = "kxc.pass_ctx.default_device";
constexpr const char* kPassCtxMultiDeviceAttr = "kxc.pass_ctx.is_multi_device";
constexpr const char* kPassCtxDiscoPlacementAttr = "kxc.pass_ctx.disco_placement";

thread_local PassContext current_pass_ctx;
thread_local bool has_current_pass_ctx = false;

// 判断属性对象是否为 Target，避免跨类型 ObjectRef 强转。
bool IsTargetObjectRef(const ObjectRef& obj) {
    return obj.defined() && obj.get()->GetTypeId() == TargetNode::_type_index;
}

// 判断属性对象是否为 VirtualDevice。
bool IsVirtualDeviceObjectRef(const ObjectRef& obj) {
    return obj.defined() && obj.get()->GetTypeId() == VirtualDeviceNode::_type_index;
}

// 判断属性对象是否为 DiscoPlacement。
bool IsDiscoPlacementObjectRef(const ObjectRef& obj) {
    return obj.defined() && obj.get()->GetTypeId() == DiscoPlacementNode::_type_index;
}

// 生成用于判断多设备上下文的物理身份键，优先采用显式 Device。
std::string DeviceIdentityKey(const Device& device, const Target& target) {
    if (device.defined()) {
        return std::to_string(static_cast<int>(device.device_type())) + ":" +
               std::to_string(device.device_id());
    }
    if (target.defined()) {
        return std::to_string(static_cast<int>(target->device_type)) + ":" +
               std::to_string(target->device_id);
    }
    return "unconstrained";
}

// 遍历 Relay 图并按对象身份去重收集 VirtualDevice，visited 集合同时处理 DAG 共享。
void CollectRelayVirtualDevices(const Expr& expr, std::unordered_set<const Object*>* visited_exprs,
                                std::unordered_set<const Object*>* visited_virtual_devices,
                                Array<VirtualDevice>* out_virtual_devices) {
    if (!expr.defined()) {
        return;
    }
    const Object* obj = expr.get();
    if (!obj || visited_exprs->count(obj)) {
        return;
    }
    visited_exprs->insert(obj);

    const RelayNode* relay_node = dynamic_cast<const RelayNode*>(obj);
    if (relay_node && relay_node->virtual_device_.defined()) {
        const Object* vd_obj = relay_node->virtual_device_.get();
        if (vd_obj && !visited_virtual_devices->count(vd_obj)) {
            visited_virtual_devices->insert(vd_obj);
            out_virtual_devices->push_back(relay_node->virtual_device_);
        }
    }

    if (auto* call = expr.As<CallNode>()) {
        CollectRelayVirtualDevices(call->op, visited_exprs, visited_virtual_devices, out_virtual_devices);
        for (const auto& arg : call->args) {
            CollectRelayVirtualDevices(arg, visited_exprs, visited_virtual_devices, out_virtual_devices);
        }
        return;
    }
    if (auto* fn = expr.As<FunctionNode>()) {
        for (const auto& param : fn->params) {
            CollectRelayVirtualDevices(Expr(ObjectRef(param)), visited_exprs, visited_virtual_devices,
                                       out_virtual_devices);
        }
        CollectRelayVirtualDevices(fn->body, visited_exprs, visited_virtual_devices, out_virtual_devices);
        return;
    }
    if (auto* if_node = expr.As<IfNode>()) {
        CollectRelayVirtualDevices(if_node->cond, visited_exprs, visited_virtual_devices, out_virtual_devices);
        CollectRelayVirtualDevices(if_node->true_branch, visited_exprs, visited_virtual_devices,
                                   out_virtual_devices);
        CollectRelayVirtualDevices(if_node->false_branch, visited_exprs, visited_virtual_devices,
                                   out_virtual_devices);
        return;
    }
    if (auto* let_node = expr.As<LetNode>()) {
        CollectRelayVirtualDevices(Expr(ObjectRef(let_node->var)), visited_exprs, visited_virtual_devices,
                                   out_virtual_devices);
        CollectRelayVirtualDevices(let_node->value, visited_exprs, visited_virtual_devices,
                                   out_virtual_devices);
        CollectRelayVirtualDevices(let_node->body, visited_exprs, visited_virtual_devices,
                                   out_virtual_devices);
        return;
    }
    if (auto* tuple = expr.As<TupleNode>()) {
        for (const auto& field : tuple->fields) {
            CollectRelayVirtualDevices(field, visited_exprs, visited_virtual_devices, out_virtual_devices);
        }
        return;
    }
    if (auto* tuple_get = expr.As<TupleGetItemNode>()) {
        CollectRelayVirtualDevices(tuple_get->tuple, visited_exprs, visited_virtual_devices,
                                   out_virtual_devices);
        return;
    }
}

// 将源表达式的放置与已推导类型复制到重建节点。
Expr CopyRelayVirtualDevice(const Expr& source, const Expr& dest) {
    if (!source.defined() || !dest.defined()) {
        return dest;
    }
    const RelayNode* src_node = dynamic_cast<const RelayNode*>(source.get());
    RelayNode* dst_node = const_cast<RelayNode*>(dynamic_cast<const RelayNode*>(dest.get()));
    if (!src_node || !dst_node) {
        return dest;
    }
    dst_node->virtual_device_ = src_node->virtual_device_;
    dst_node->checked_type_ = src_node->checked_type_;
    return dest;
}

}  // namespace

// 输出 PassContext 的设备、目标和 Disco 放置摘要。
std::string PassContext::ToString() const {
    if (!defined_) {
        return "PassContext(undefined)";
    }
    std::stringstream ss;
    ss << "PassContext(multi_device=" << (is_multi_device_ ? "true" : "false")
       << ", virtual_devices=" << virtual_devices_.size();
    if (primary_virtual_device_.defined()) {
        ss << ", primary=" << primary_virtual_device_.ToString();
    } else {
        ss << ", primary=none";
    }
    if (default_target_.defined()) {
        ss << ", target=" << default_target_.ToString();
    } else {
        ss << ", target=none";
    }
    if (disco_placement_.defined()) {
        ss << ", disco_workers=" << disco_placement_->workers.size()
           << ", disco_groups=" << disco_placement_->num_groups;
    } else {
        ss << ", disco=none";
    }
    ss << ")";
    return ss.str();
}

// 返回当前线程的 PassContext；未设置时返回未定义上下文。
PassContext PassContext::Current() {
    if (!has_current_pass_ctx) {
        return PassContext();
    }
    return current_pass_ctx;
}

// 从逻辑设备集合推导默认 Device、Target、多设备标志及 DiscoPlacement。
PassContext PassContext::BuildFromVirtualDevices(const Array<VirtualDevice>& virtual_devices) {
    PassContext ctx;
    if (virtual_devices.empty()) {
        return ctx;
    }

    ctx.defined_ = true;
    ctx.virtual_devices_ = virtual_devices;
    ctx.primary_virtual_device_ = virtual_devices[0];
    // PassContext 保留强类型 Device，使后续 Target 推导不再依赖 ObjectRef 强制转换。
    ctx.default_device_ = ctx.primary_virtual_device_->device;

    if (ctx.primary_virtual_device_->target.defined()) {
        ctx.default_target_ = ctx.primary_virtual_device_->target;
    } else if (ctx.default_device_.defined()) {
        ctx.default_target_ = BuildTarget(ctx.default_device_);
    }

    std::unordered_set<std::string> identities;
    for (const auto& vd : virtual_devices) {
        if (!vd.defined()) continue;
        identities.insert(DeviceIdentityKey(vd->device, vd->target));
    }
    ctx.is_multi_device_ = identities.size() > 1;
    ctx.disco_placement_ = BuildDiscoPlacement(virtual_devices, /*num_groups=*/1);
    return ctx;
}

// 扫描 Relay 表达式中的逻辑放置并构造 PassContext。
PassContext PassContext::FromRelay(const Expr& expr) {
    Array<VirtualDevice> virtual_devices;
    std::unordered_set<const Object*> visited_exprs;
    std::unordered_set<const Object*> visited_virtual_devices;
    CollectRelayVirtualDevices(expr, &visited_exprs, &visited_virtual_devices, &virtual_devices);
    return BuildFromVirtualDevices(virtual_devices);
}

// 将 Relay Function 作为表达式扫描放置上下文。
PassContext PassContext::FromRelay(const Function& func) {
    return FromRelay(Expr(ObjectRef(func)));
}

// 从 PrimFunc 的保留属性恢复强类型 PassContext。
PassContext PassContext::FromTIR(const tir::PrimFunc& func) {
    PassContext ctx;
    if (!func.defined()) {
        return ctx;
    }

    if (func->attrs.count(String(kPassCtxPrimaryVirtualDeviceAttr))) {
        ObjectRef vd_ref = func->attrs.at(String(kPassCtxPrimaryVirtualDeviceAttr));
        if (IsVirtualDeviceObjectRef(vd_ref)) {
            ctx = BuildFromVirtualDevices({VirtualDevice(vd_ref)});
        }
    }

    if (func->attrs.count(String(kPassCtxDefaultTargetAttr))) {
        ObjectRef target_ref = func->attrs.at(String(kPassCtxDefaultTargetAttr));
        if (IsTargetObjectRef(target_ref)) {
            if (!ctx.defined_) ctx.defined_ = true;
            ctx.default_target_ = Target(target_ref);
        }
    }

    if (func->attrs.count(String(kPassCtxDefaultDeviceAttr))) {
        ObjectRef device_ref = func->attrs.at(String(kPassCtxDefaultDeviceAttr));
        // Device(ObjectRef) 在属性边界完成类型检查，非法对象不会静默进入 PassContext。
        Device device(device_ref);
        if (!ctx.defined_) ctx.defined_ = true;
        ctx.default_device_ = std::move(device);
    }

    if (func->attrs.count(String(kPassCtxMultiDeviceAttr))) {
        ObjectRef multi_ref = func->attrs.at(String(kPassCtxMultiDeviceAttr));
        if (auto* imm = multi_ref.As<tir::IntImmNode>()) {
            if (!ctx.defined_) ctx.defined_ = true;
            ctx.is_multi_device_ = imm->value != 0;
        }
    }

    if (func->attrs.count(String(kPassCtxDiscoPlacementAttr))) {
        ObjectRef disco_ref = func->attrs.at(String(kPassCtxDiscoPlacementAttr));
        if (IsDiscoPlacementObjectRef(disco_ref)) {
            if (!ctx.defined_) ctx.defined_ = true;
            ctx.disco_placement_ = DiscoPlacement(disco_ref);
        }
    }

    return ctx;
}

// 从 Compiler 的唯一 Target 建立完整单设备上下文，供 Relay/TIR pass 共享。
PassContext PassContext::FromTarget(const Target& target) {
    if (!IsTargetObjectRef(target) || target->device_type == kUnknown ||
        target->device_id < 0 || target->kind.empty()) {
        throw std::invalid_argument("PassContext requires a complete Target");
    }
    const Device device(target->device_type, target->device_id);
    return BuildFromVirtualDevices(
        {VirtualDevice::ForDeviceAndTarget(device, target)});
}

// 合并时保留 Relay 已建立的多设备/Disco 信息，但不允许覆盖其物理身份。
PassContext PassContext::MergeTarget(const PassContext& base_ctx,
                                     const Target& target) {
    PassContext target_ctx = FromTarget(target);
    if (!base_ctx.defined()) return target_ctx;

    const Device expected(target->device_type, target->device_id);
    if (base_ctx.default_device_.defined() &&
        base_ctx.default_device_ != expected) {
        throw std::invalid_argument(
            "Relay placement device conflicts with CompileConfig target");
    }
    if (base_ctx.default_target_.defined()) {
        const Target& placed = base_ctx.default_target_;
        if (!IsTargetObjectRef(placed) || placed->kind != target->kind ||
            placed->device_type != target->device_type ||
            placed->device_id != target->device_id) {
            throw std::invalid_argument(
                "Relay placement target conflicts with CompileConfig target");
        }
    }

    PassContext merged = base_ctx;
    merged.defined_ = true;
    merged.default_target_ = target;
    merged.default_device_ = expected;
    if (!merged.primary_virtual_device_.defined()) {
        merged.primary_virtual_device_ =
            VirtualDevice::ForDeviceAndTarget(expected, target);
        merged.virtual_devices_ = {merged.primary_virtual_device_};
    }
    return merged;
}

// 基于现有上下文替换 DiscoPlacement，同时保留其他推导结果。
PassContext PassContext::WithDiscoPlacement(const PassContext& base_ctx,
                                            const DiscoPlacement& disco_placement) {
    PassContext ctx = base_ctx;
    if (!ctx.defined_ && disco_placement.defined()) {
        ctx.defined_ = true;
    }
    ctx.disco_placement_ = disco_placement;
    return ctx;
}

// 设置当前线程 PassContext，避免跨线程共享可变编译状态。
void PassContext::SetCurrent(const PassContext& pass_ctx) {
    current_pass_ctx = pass_ctx;
    has_current_pass_ctx = pass_ctx.defined_;
}

// 清除当前线程 PassContext。
void PassContext::ClearCurrent() {
    current_pass_ctx = PassContext();
    has_current_pass_ctx = false;
}

// 进入作用域时保存旧上下文并安装新的已定义上下文。
PassContext::Scope::Scope(const PassContext& pass_ctx) {
    previous_ = std::make_unique<PassContext>(PassContext::Current());
    if (!pass_ctx.defined()) {
        active_ = false;
        return;
    }
    PassContext::SetCurrent(pass_ctx);
    active_ = true;
}

// 离开作用域时恢复旧上下文，保证嵌套 Pass 不泄漏状态。
PassContext::Scope::~Scope() {
    if (!active_) {
        return;
    }
    if (previous_ && previous_->defined()) {
        PassContext::SetCurrent(*previous_);
    } else {
        PassContext::ClearCurrent();
    }
}

// 复制 PrimFunc 属性并附加可跨 Pass 传递的上下文字段。
Map<String, ObjectRef> AttachPassContextAttrs(const Map<String, ObjectRef>& attrs,
                                              const PassContext& pass_ctx) {
    Map<String, ObjectRef> new_attrs;
    for (const auto& kv : attrs) {
        new_attrs.Set(kv.first, kv.second);
    }

    if (!pass_ctx.defined()) {
        return new_attrs;
    }
    if (pass_ctx.primary_virtual_device().defined()) {
        new_attrs.Set(String(kPassCtxPrimaryVirtualDeviceAttr),
                      ObjectRef(pass_ctx.primary_virtual_device()));
    }
    if (pass_ctx.default_target().defined()) {
        new_attrs.Set(String(kPassCtxDefaultTargetAttr), ObjectRef(pass_ctx.default_target()));
    }
    if (pass_ctx.default_device().defined()) {
        new_attrs.Set(String(kPassCtxDefaultDeviceAttr), ObjectRef(pass_ctx.default_device()));
    }
    new_attrs.Set(String(kPassCtxMultiDeviceAttr),
                  tir::IntImm(pass_ctx.is_multi_device() ? 1 : 0, tir::DataType::Bool()));
    if (pass_ctx.has_disco_placement()) {
        new_attrs.Set(String(kPassCtxDiscoPlacementAttr), ObjectRef(pass_ctx.disco_placement()));
    }
    return new_attrs;
}

// 从 Relay 放置约束建立 DiscoPlacement，缺失时生成默认单组映射。
PassContext BuildDiscoPlacementPass(const Expr& expr) {
    PassContext pass_ctx = PassContext::FromRelay(expr);
    if (!pass_ctx.has_disco_placement()) {
        pass_ctx = PassContext::WithDiscoPlacement(
            pass_ctx, BuildDiscoPlacement(pass_ctx.virtual_devices(), /*num_groups=*/1));
    }
    return pass_ctx;
}

// 为 Relay Function 提供 BuildDiscoPlacementPass 重载。
PassContext BuildDiscoPlacementPass(const Function& func) {
    return BuildDiscoPlacementPass(Expr(ObjectRef(func)));
}

// 在当前或推导出的 PassContext 作用域内递归重写 Relay 表达式。
Expr RelayPass::Mutate(const Expr& expr) {
    PassContext pass_ctx = PassContext::Current();
    if (!pass_ctx.defined()) {
        pass_ctx = PassContext::FromRelay(expr);
    }
    PassContext::Scope scope(pass_ctx);
    return VisitExpr(expr);
}

// 重写 Relay Function 并验证结果仍为 Function。
Function RelayPass::Mutate(const Function& func) {
    if (!func.defined()) return func;
    Expr out = Mutate(Expr(ObjectRef(func)));
    if (!out.defined()) return Function();
    if (!out.As<FunctionNode>()) {
        throw std::runtime_error("RelayPass expected Function result when mutating Function");
    }
    return Function(out);
}

// 重写 Relay Var 并保持强类型返回。
Var RelayPass::Mutate(const Var& var) { return MutateToVar(var); }

// 常量没有递归子节点，默认保持原对象。
Expr RelayPass::VisitConstant(const ConstantNode* op, const Expr& ref) {
    (void)op;
    return ref;
}

// 将 Var 分派给可覆写的强类型变量重写入口。
Expr RelayPass::VisitVar(const VarNode* op, const Expr& ref) {
    (void)op;
    return ref;
}

// 算子描述节点不可变，默认保持原对象。
Expr RelayPass::VisitOp(const relay::OpNode* op, const Expr& ref) {
    (void)op;
    return ref;
}

// 递归重写调用目标与参数，并把原放置和类型元数据复制到新节点。
Expr RelayPass::VisitCall(const CallNode* op, const Expr& ref) {
    auto new_op = Mutate(op->op);
    Array<Expr> new_args;
    bool changed = (new_op.get() != op->op.get());

    for (const auto& arg : op->args) {
        auto new_arg = Mutate(arg);
        if (new_arg.get() != arg.get()) changed = true;
        new_args.push_back(new_arg);
    }

    if (!changed) return ref;
    return CopyRelayVirtualDevice(ref, Call(new_op, new_args, op->attrs));
}

// 递归重写函数参数和函数体，同时保留函数属性与放置元数据。
Expr RelayPass::VisitFunction(const FunctionNode* op, const Expr& ref) {
    Array<Var> new_params;
    bool changed = false;
    for (const auto& param : op->params) {
        Var new_param = MutateToVar(param);
        if (new_param.get() != param.get()) changed = true;
        new_params.push_back(new_param);
    }

    auto new_body = Mutate(op->body);
    if (new_body.get() != op->body.get()) changed = true;

    if (!changed) return ref;
    return CopyRelayVirtualDevice(ref, Function(changed ? new_params : op->params, new_body));
}

// 递归重写条件与两个分支，并保留结果放置元数据。
Expr RelayPass::VisitIf(const IfNode* op, const Expr& ref) {
    auto new_cond = Mutate(op->cond);
    auto new_true = Mutate(op->true_branch);
    auto new_false = Mutate(op->false_branch);

    if (new_cond.get() == op->cond.get() && new_true.get() == op->true_branch.get() &&
        new_false.get() == op->false_branch.get()) {
        return ref;
    }
    return CopyRelayVirtualDevice(ref, If(new_cond, new_true, new_false));
}

// 递归重写 Let 绑定变量、值与作用域体。
Expr RelayPass::VisitLet(const LetNode* op, const Expr& ref) {
    auto new_var = MutateToVar(op->var);
    auto new_value = Mutate(op->value);
    auto new_body = Mutate(op->body);

    if (new_var.get() == op->var.get() && new_value.get() == op->value.get() &&
        new_body.get() == op->body.get()) {
        return ref;
    }
    return CopyRelayVirtualDevice(ref, Let(new_var, new_value, new_body));
}

// 递归重写 Tuple 各字段并保留元数据。
Expr RelayPass::VisitTuple(const TupleNode* op, const Expr& ref) {
    Array<Expr> new_fields;
    bool changed = false;
    for (const auto& field : op->fields) {
        auto new_field = Mutate(field);
        if (new_field.get() != field.get()) changed = true;
        new_fields.push_back(new_field);
    }

    if (!changed) return ref;
    return CopyRelayVirtualDevice(ref, Tuple(new_fields));
}

// 递归重写 TupleGetItem 的源 Tuple 并保留索引与元数据。
Expr RelayPass::VisitTupleGetItem(const TupleGetItemNode* op, const Expr& ref) {
    auto new_tuple = Mutate(op->tuple);
    if (new_tuple.get() == op->tuple.get()) return ref;
    return CopyRelayVirtualDevice(ref, TupleGetItem(new_tuple, op->index));
}

// 默认保持 Var；派生 Pass 可覆写以替换绑定身份。
Var RelayPass::MutateToVar(const Var& var) {
    if (!var.defined()) return var;
    Expr new_var = Mutate(Expr(ObjectRef(var)));
    if (!new_var.defined()) {
        throw std::runtime_error("RelayPass mutated Var into undefined expression");
    }
    if (!new_var.As<VarNode>()) {
        throw std::runtime_error("RelayPass expects variable position to remain Var");
    }
    return Var(new_var);
}

// 递归重写 TIR 表达式入口。
tir::PrimExpr TIRPass::Mutate(const tir::PrimExpr& expr) { return VisitExpr(expr); }

// 递归重写 TIR 语句入口。
tir::Stmt TIRPass::Mutate(const tir::Stmt& stmt) { return VisitStmt(stmt); }

// 在当前或从属性恢复的 PassContext 中重写 PrimFunc。
tir::PrimFunc TIRPass::Mutate(const tir::PrimFunc& func) {
    PassContext pass_ctx = PassContext::Current();
    if (!pass_ctx.defined()) {
        pass_ctx = PassContext::FromTIR(func);
    }
    PassContext::Scope scope(pass_ctx);
    return VisitPrimFunc(func);
}

// 重写参数、buffer map、函数体并重新附加 PassContext 属性。
tir::PrimFunc TIRPass::VisitPrimFunc(const tir::PrimFunc& func) {
    if (!func.defined()) return func;

    Array<tir::Var> new_params;
    bool params_changed = false;
    for (const auto& p : func->params) {
        tir::Var np = MutateToVar(p);
        if (np.get() != p.get()) params_changed = true;
        new_params.push_back(np);
    }

    tir::Stmt new_body = Mutate(func->body);
    bool body_changed = (new_body.get() != func->body.get());

    Map<tir::Var, tir::Buffer> new_buffer_map;
    bool buffer_map_changed = false;
    for (const auto& kv : func->buffer_map) {
        tir::Var new_key = MutateToVar(kv.first);
        tir::Buffer new_val = MutateBuffer(kv.second);
        if (new_key.get() != kv.first.get() || new_val.get() != kv.second.get()) {
            buffer_map_changed = true;
        }
        new_buffer_map.Set(new_key, new_val);
    }

    if (!params_changed && !body_changed && !buffer_map_changed) {
        return func;
    }

    return tir::PrimFunc(params_changed ? new_params : func->params, new_body,
                         buffer_map_changed ? new_buffer_map : func->buffer_map, func->attrs);
}

// 整数立即数没有子节点，默认保持原对象。
tir::PrimExpr TIRPass::VisitIntImm(const tir::IntImmNode* op, const tir::PrimExpr& ref) {
    (void)op;
    return ref;
}

// 浮点立即数没有子节点，默认保持原对象。
tir::PrimExpr TIRPass::VisitFloatImm(const tir::FloatImmNode* op, const tir::PrimExpr& ref) {
    (void)op;
    return ref;
}

// 将 TIR Var 分派给可覆写的强类型变量重写入口。
tir::PrimExpr TIRPass::VisitVar(const tir::VarNode* op, const tir::PrimExpr& ref) {
    (void)op;
    return ref;
}

// 递归重写加法两侧并重建表达式。
tir::PrimExpr TIRPass::VisitAdd(const tir::AddNode* op, const tir::PrimExpr& ref) {
    tir::PrimExpr new_a = Mutate(op->a);
    tir::PrimExpr new_b = Mutate(op->b);
    if (new_a.get() == op->a.get() && new_b.get() == op->b.get()) return ref;
    return tir::Add(new_a, new_b);
}

// 递归重写减法两侧并重建表达式。
tir::PrimExpr TIRPass::VisitSub(const tir::SubNode* op, const tir::PrimExpr& ref) {
    tir::PrimExpr new_a = Mutate(op->a);
    tir::PrimExpr new_b = Mutate(op->b);
    if (new_a.get() == op->a.get() && new_b.get() == op->b.get()) return ref;
    return tir::Sub(new_a, new_b);
}

// 递归重写乘法两侧并重建表达式。
tir::PrimExpr TIRPass::VisitMul(const tir::MulNode* op, const tir::PrimExpr& ref) {
    tir::PrimExpr new_a = Mutate(op->a);
    tir::PrimExpr new_b = Mutate(op->b);
    if (new_a.get() == op->a.get() && new_b.get() == op->b.get()) return ref;
    return tir::Mul(new_a, new_b);
}

// 递归重写除法两侧并重建表达式。
tir::PrimExpr TIRPass::VisitDiv(const tir::DivNode* op, const tir::PrimExpr& ref) {
    tir::PrimExpr new_a = Mutate(op->a);
    tir::PrimExpr new_b = Mutate(op->b);
    if (new_a.get() == op->a.get() && new_b.get() == op->b.get()) return ref;
    return tir::Div(new_a, new_b);
}

// 递归重写取模两侧并重建表达式。
tir::PrimExpr TIRPass::VisitMod(const tir::ModNode* op, const tir::PrimExpr& ref) {
    tir::PrimExpr new_a = Mutate(op->a);
    tir::PrimExpr new_b = Mutate(op->b);
    if (new_a.get() == op->a.get() && new_b.get() == op->b.get()) return ref;
    return tir::Mod(new_a, new_b);
}

// 递归重写 Min 两侧并重建表达式。
tir::PrimExpr TIRPass::VisitMin(const tir::MinNode* op, const tir::PrimExpr& ref) {
    tir::PrimExpr new_a = Mutate(op->a);
    tir::PrimExpr new_b = Mutate(op->b);
    if (new_a.get() == op->a.get() && new_b.get() == op->b.get()) return ref;
    return tir::Min(new_a, new_b);
}

// 递归重写 Max 两侧并重建表达式。
tir::PrimExpr TIRPass::VisitMax(const tir::MaxNode* op, const tir::PrimExpr& ref) {
    tir::PrimExpr new_a = Mutate(op->a);
    tir::PrimExpr new_b = Mutate(op->b);
    if (new_a.get() == op->a.get() && new_b.get() == op->b.get()) return ref;
    return tir::Max(new_a, new_b);
}

// 递归重写相等比较两侧并重建表达式。
tir::PrimExpr TIRPass::VisitEQ(const tir::EQNode* op, const tir::PrimExpr& ref) {
    tir::PrimExpr new_a = Mutate(op->a);
    tir::PrimExpr new_b = Mutate(op->b);
    if (new_a.get() == op->a.get() && new_b.get() == op->b.get()) return ref;
    return tir::EQ(new_a, new_b);
}

// 递归重写小于比较两侧并重建表达式。
tir::PrimExpr TIRPass::VisitLT(const tir::LTNode* op, const tir::PrimExpr& ref) {
    tir::PrimExpr new_a = Mutate(op->a);
    tir::PrimExpr new_b = Mutate(op->b);
    if (new_a.get() == op->a.get() && new_b.get() == op->b.get()) return ref;
    return tir::LT(new_a, new_b);
}

// 递归重写逻辑与两侧并重建表达式。
tir::PrimExpr TIRPass::VisitAnd(const tir::AndNode* op, const tir::PrimExpr& ref) {
    tir::PrimExpr new_a = Mutate(op->a);
    tir::PrimExpr new_b = Mutate(op->b);
    if (new_a.get() == op->a.get() && new_b.get() == op->b.get()) return ref;
    return tir::And(new_a, new_b);
}

// 递归重写逻辑或两侧并重建表达式。
tir::PrimExpr TIRPass::VisitOr(const tir::OrNode* op, const tir::PrimExpr& ref) {
    tir::PrimExpr new_a = Mutate(op->a);
    tir::PrimExpr new_b = Mutate(op->b);
    if (new_a.get() == op->a.get() && new_b.get() == op->b.get()) return ref;
    return tir::Or(new_a, new_b);
}

// 递归重写逻辑非操作数并重建表达式。
tir::PrimExpr TIRPass::VisitNot(const tir::NotNode* op, const tir::PrimExpr& ref) {
    tir::PrimExpr new_value = Mutate(op->value);
    if (new_value.get() == op->value.get()) return ref;
    return tir::Not(new_value);
}

// 重写 Load 的 buffer 变量、索引和谓词并保留 dtype。
tir::PrimExpr TIRPass::VisitLoad(const tir::LoadNode* op, const tir::PrimExpr& ref) {
    tir::Var new_buffer_var = MutateToVar(op->buffer_var);
    tir::PrimExpr new_index = Mutate(op->index);
    tir::PrimExpr new_pred = Mutate(op->predicate);
    if (new_buffer_var.get() == op->buffer_var.get() && new_index.get() == op->index.get() &&
        new_pred.get() == op->predicate.get()) {
        return ref;
    }
    return tir::Load(new_buffer_var, new_index, new_pred);
}

// 递归重写 TIR Call 参数并保留调用目标与 dtype。
tir::PrimExpr TIRPass::VisitCall(const tir::CallNode* op, const tir::PrimExpr& ref) {
    Array<tir::PrimExpr> new_args;
    bool changed = false;
    for (const auto& arg : op->args) {
        tir::PrimExpr new_arg = Mutate(arg);
        if (new_arg.get() != arg.get()) changed = true;
        new_args.push_back(new_arg);
    }
    if (!changed) return ref;
    return tir::Call(ref.dtype(), op->name, new_args);
}

// 递归重写 Select 条件和两个值分支。
tir::PrimExpr TIRPass::VisitSelect(const tir::SelectNode* op, const tir::PrimExpr& ref) {
    tir::PrimExpr new_cond = Mutate(op->condition);
    tir::PrimExpr new_true = Mutate(op->true_value);
    tir::PrimExpr new_false = Mutate(op->false_value);
    if (new_cond.get() == op->condition.get() && new_true.get() == op->true_value.get() &&
        new_false.get() == op->false_value.get()) {
        return ref;
    }
    return tir::Select(new_cond, new_true, new_false);
}

// 重写 LetStmt 的绑定变量、值和作用域体。
tir::Stmt TIRPass::VisitLetStmt(const tir::LetStmtNode* op, const tir::Stmt& ref) {
    tir::Var new_var = MutateToVar(op->var);
    tir::PrimExpr new_value = Mutate(op->value);
    tir::Stmt new_body = Mutate(op->body);
    if (new_var.get() == op->var.get() && new_value.get() == op->value.get() &&
        new_body.get() == op->body.get()) {
        return ref;
    }
    return tir::LetStmt(new_var, new_value, new_body);
}

// 重写 Store 的 buffer 变量、值、索引和谓词。
tir::Stmt TIRPass::VisitStore(const tir::StoreNode* op, const tir::Stmt& ref) {
    tir::Var new_buffer_var = MutateToVar(op->buffer_var);
    tir::PrimExpr new_value = Mutate(op->value);
    tir::PrimExpr new_index = Mutate(op->index);
    tir::PrimExpr new_pred = Mutate(op->predicate);
    if (new_buffer_var.get() == op->buffer_var.get() && new_value.get() == op->value.get() &&
        new_index.get() == op->index.get() && new_pred.get() == op->predicate.get()) {
        return ref;
    }
    return tir::Store(new_buffer_var, new_value, new_index, new_pred);
}

// 重写循环变量、范围和循环体，同时保留循环种类与注解。
tir::Stmt TIRPass::VisitFor(const tir::ForNode* op, const tir::Stmt& ref) {
    tir::Var new_loop_var = MutateToVar(op->loop_var);
    tir::PrimExpr new_min = Mutate(op->min);
    tir::PrimExpr new_extent = Mutate(op->extent);
    tir::Stmt new_body = Mutate(op->body);
    if (new_loop_var.get() == op->loop_var.get() && new_min.get() == op->min.get() &&
        new_extent.get() == op->extent.get() && new_body.get() == op->body.get()) {
        return ref;
    }
    return tir::For(new_loop_var, new_min, new_extent, op->for_type, new_body);
}

// 重写条件语句的条件、真分支和可选假分支。
tir::Stmt TIRPass::VisitIfThenElse(const tir::IfThenElseNode* op, const tir::Stmt& ref) {
    tir::PrimExpr new_cond = Mutate(op->condition);
    tir::Stmt new_then = Mutate(op->then_case);
    tir::Stmt new_else = Mutate(op->else_case);
    if (new_cond.get() == op->condition.get() && new_then.get() == op->then_case.get() &&
        new_else.get() == op->else_case.get()) {
        return ref;
    }
    return tir::IfThenElse(new_cond, new_then, new_else);
}

// 重写 Allocate 的变量、维度、条件和作用域体，并保留存储注解。
tir::Stmt TIRPass::VisitAllocate(const tir::AllocateNode* op, const tir::Stmt& ref) {
    tir::Var new_buffer_var = MutateToVar(op->buffer_var);
    Array<tir::PrimExpr> new_extents;
    bool extents_changed = false;
    for (const auto& extent : op->extents) {
        tir::PrimExpr new_extent = Mutate(extent);
        if (new_extent.get() != extent.get()) extents_changed = true;
        new_extents.push_back(new_extent);
    }
    tir::PrimExpr new_cond = Mutate(op->condition);
    tir::Stmt new_body = Mutate(op->body);
    if (new_buffer_var.get() == op->buffer_var.get() && !extents_changed &&
        new_cond.get() == op->condition.get() && new_body.get() == op->body.get()) {
        return ref;
    }
    return tir::Allocate(new_buffer_var, op->dtype, extents_changed ? new_extents : op->extents,
                         new_cond, new_body);
}

// 重写 AttrStmt 的节点、值与作用域体并保留属性键。
tir::Stmt TIRPass::VisitAttrStmt(const tir::AttrStmtNode* op, const tir::Stmt& ref) {
    tir::PrimExpr new_value = Mutate(op->value);
    tir::Stmt new_body = Mutate(op->body);
    if (new_value.get() == op->value.get() && new_body.get() == op->body.get()) {
        return ref;
    }
    return tir::AttrStmt(op->node, op->attr_key, new_value, new_body);
}

// 递归重建 Block 的迭代变量、读写区域、初始化和主体。
tir::Stmt TIRPass::VisitBlock(const tir::BlockNode* op, const tir::Stmt& ref) {
    Array<tir::IterVar> new_iter_vars;
    bool iter_vars_changed = false;
    for (const auto& iv : op->iter_vars) {
        tir::IterVar new_iv = MutateIterVar(iv);
        if (new_iv.get() != iv.get()) iter_vars_changed = true;
        new_iter_vars.push_back(new_iv);
    }

    Array<tir::BufferRegion> new_reads;
    bool reads_changed = false;
    for (const auto& r : op->reads) {
        tir::BufferRegion nr = MutateBufferRegion(r);
        if (nr.get() != r.get()) reads_changed = true;
        new_reads.push_back(nr);
    }

    Array<tir::BufferRegion> new_writes;
    bool writes_changed = false;
    for (const auto& w : op->writes) {
        tir::BufferRegion nw = MutateBufferRegion(w);
        if (nw.get() != w.get()) writes_changed = true;
        new_writes.push_back(nw);
    }

    tir::Stmt new_body = Mutate(op->body);
    tir::Stmt new_init = Mutate(op->init);

    if (!iter_vars_changed && !reads_changed && !writes_changed &&
        new_body.get() == op->body.get() && new_init.get() == op->init.get()) {
        return ref;
    }

    return tir::Block(iter_vars_changed ? new_iter_vars : op->iter_vars,
                      reads_changed ? new_reads : op->reads,
                      writes_changed ? new_writes : op->writes, op->name_hint, new_body,
                      new_init);
}

// 按原顺序重写语句序列。
tir::Stmt TIRPass::VisitSeqStmt(const tir::SeqStmtNode* op, const tir::Stmt& ref) {
    Array<tir::Stmt> new_seq;
    bool changed = false;
    for (const auto& s : op->seq) {
        tir::Stmt ns = Mutate(s);
        if (ns.get() != s.get()) changed = true;
        new_seq.push_back(ns);
    }
    if (!changed) return ref;
    return tir::SeqStmt(new_seq);
}

// 重写 Evaluate 持有的表达式。
tir::Stmt TIRPass::VisitEvaluate(const tir::EvaluateNode* op, const tir::Stmt& ref) {
    tir::PrimExpr new_value = Mutate(op->value);
    if (new_value.get() == op->value.get()) return ref;
    return tir::Evaluate(new_value);
}

// 默认保持 TIR Var；派生 Pass 可覆写以替换绑定身份。
tir::Var TIRPass::MutateToVar(const tir::Var& var) {
    if (!var.defined()) return var;
    const tir::PrimExpr& var_expr = static_cast<const tir::PrimExpr&>(var);
    tir::PrimExpr new_var_expr = Mutate(var_expr);
    if (!new_var_expr.defined()) {
        throw std::runtime_error("TIRPass mutated Var into undefined expression");
    }
    auto* var_node = new_var_expr.As<tir::VarNode>();
    if (!var_node) {
        throw std::runtime_error("TIRPass expects variable position to remain tir::Var");
    }
    return tir::Var(new_var_expr);
}

// 重写 Range 的起点和跨度。
tir::Range TIRPass::MutateRange(const tir::Range& range) {
    if (!range.defined()) return range;
    tir::PrimExpr new_min = Mutate(range->min);
    tir::PrimExpr new_extent = Mutate(range->extent);
    if (new_min.get() == range->min.get() && new_extent.get() == range->extent.get()) {
        return range;
    }
    return tir::Range(new_min, new_extent);
}

// 重写 IterVar 的范围和绑定变量并保留迭代类型。
tir::IterVar TIRPass::MutateIterVar(const tir::IterVar& iv) {
    if (!iv.defined()) return iv;
    tir::Range new_dom = MutateRange(iv->dom);
    tir::Var new_var = MutateToVar(iv->var);
    if (new_dom.get() == iv->dom.get() && new_var.get() == iv->var.get()) {
        return iv;
    }
    return tir::IterVar(new_dom, new_var, iv->iter_type, iv->thread_tag);
}

// 重写 Buffer 的数据变量、shape、strides 和偏移元数据。
tir::Buffer TIRPass::MutateBuffer(const tir::Buffer& buffer) {
    if (!buffer.defined()) return buffer;

    tir::Var new_data = MutateToVar(buffer->data);

    Array<tir::PrimExpr> new_shape;
    bool shape_changed = false;
    for (const auto& s : buffer->shape) {
        tir::PrimExpr ns = Mutate(s);
        if (ns.get() != s.get()) shape_changed = true;
        new_shape.push_back(ns);
    }

    Array<tir::PrimExpr> new_strides;
    bool strides_changed = false;
    for (const auto& s : buffer->strides) {
        tir::PrimExpr ns = Mutate(s);
        if (ns.get() != s.get()) strides_changed = true;
        new_strides.push_back(ns);
    }

    tir::PrimExpr new_elem_offset = Mutate(buffer->elem_offset);

    if (new_data.get() == buffer->data.get() && !shape_changed && !strides_changed &&
        new_elem_offset.get() == buffer->elem_offset.get()) {
        return buffer;
    }

    return tir::Buffer(new_data, buffer->dtype, shape_changed ? new_shape : buffer->shape,
                       strides_changed ? new_strides : buffer->strides, new_elem_offset,
                       buffer->name, buffer->data_alignment, buffer->offset_factor);
}

// 重写 BufferRegion 的 Buffer 与各维 Range。
tir::BufferRegion TIRPass::MutateBufferRegion(const tir::BufferRegion& region) {
    if (!region.defined()) return region;
    tir::Buffer new_buffer = MutateBuffer(region->buffer);
    Array<tir::Range> new_region;
    bool region_changed = false;
    for (const auto& r : region->region) {
        tir::Range nr = MutateRange(r);
        if (nr.get() != r.get()) region_changed = true;
        new_region.push_back(nr);
    }
    if (new_buffer.get() == region->buffer.get() && !region_changed) {
        return region;
    }
    return tir::BufferRegion(new_buffer, region_changed ? new_region : region->region);
}

}  // namespace kxc
