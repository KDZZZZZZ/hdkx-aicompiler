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

bool IsDeviceObjectRef(const ObjectRef& obj) {
    return obj.defined() && obj.get()->GetTypeId() == kKXC_DEVICE_TYPE;
}

bool IsTargetObjectRef(const ObjectRef& obj) {
    return obj.defined() && obj.get()->GetTypeId() == TargetNode::_type_index;
}

bool IsVirtualDeviceObjectRef(const ObjectRef& obj) {
    return obj.defined() && obj.get()->GetTypeId() == VirtualDeviceNode::_type_index;
}

bool IsDiscoPlacementObjectRef(const ObjectRef& obj) {
    return obj.defined() && obj.get()->GetTypeId() == DiscoPlacementNode::_type_index;
}

std::string DeviceIdentityKey(const ObjectRef& device_obj, const Target& target) {
    if (IsDeviceObjectRef(device_obj)) {
        const auto* dev = static_cast<const class Device*>(device_obj.get());
        return std::to_string(static_cast<int>(dev->device_type())) + ":" +
               std::to_string(dev->device_id());
    }
    if (target.defined()) {
        return std::to_string(static_cast<int>(target->device_type)) + ":" +
               std::to_string(target->device_id);
    }
    return "unconstrained";
}

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
            CollectRelayVirtualDevices(Expr(param), visited_exprs, visited_virtual_devices,
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
        CollectRelayVirtualDevices(Expr(let_node->var), visited_exprs, visited_virtual_devices,
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
    return dest;
}

}  // namespace

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

PassContext PassContext::Current() {
    if (!has_current_pass_ctx) {
        return PassContext();
    }
    return current_pass_ctx;
}

PassContext PassContext::BuildFromVirtualDevices(const Array<VirtualDevice>& virtual_devices) {
    PassContext ctx;
    if (virtual_devices.empty()) {
        return ctx;
    }

    ctx.defined_ = true;
    ctx.virtual_devices_ = virtual_devices;
    ctx.primary_virtual_device_ = virtual_devices[0];
    ctx.default_device_obj_ = ctx.primary_virtual_device_->device_obj;

    if (ctx.primary_virtual_device_->target.defined()) {
        ctx.default_target_ = ctx.primary_virtual_device_->target;
    } else if (IsDeviceObjectRef(ctx.default_device_obj_)) {
        const auto* dev = static_cast<const class Device*>(ctx.default_device_obj_.get());
        ctx.default_target_ = BuildTarget(*dev);
    }

    std::unordered_set<std::string> identities;
    for (const auto& vd : virtual_devices) {
        if (!vd.defined()) continue;
        identities.insert(DeviceIdentityKey(vd->device_obj, vd->target));
    }
    ctx.is_multi_device_ = identities.size() > 1;
    ctx.disco_placement_ = BuildDiscoPlacement(virtual_devices, /*num_groups=*/1);
    return ctx;
}

PassContext PassContext::FromRelay(const Expr& expr) {
    Array<VirtualDevice> virtual_devices;
    std::unordered_set<const Object*> visited_exprs;
    std::unordered_set<const Object*> visited_virtual_devices;
    CollectRelayVirtualDevices(expr, &visited_exprs, &visited_virtual_devices, &virtual_devices);
    return BuildFromVirtualDevices(virtual_devices);
}

PassContext PassContext::FromRelay(const Function& func) {
    return FromRelay(Expr(func));
}

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
        if (IsDeviceObjectRef(device_ref)) {
            if (!ctx.defined_) ctx.defined_ = true;
            ctx.default_device_obj_ = device_ref;
        }
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

PassContext PassContext::WithDiscoPlacement(const PassContext& base_ctx,
                                            const DiscoPlacement& disco_placement) {
    PassContext ctx = base_ctx;
    if (!ctx.defined_ && disco_placement.defined()) {
        ctx.defined_ = true;
    }
    ctx.disco_placement_ = disco_placement;
    return ctx;
}

void PassContext::SetCurrent(const PassContext& pass_ctx) {
    current_pass_ctx = pass_ctx;
    has_current_pass_ctx = pass_ctx.defined_;
}

void PassContext::ClearCurrent() {
    current_pass_ctx = PassContext();
    has_current_pass_ctx = false;
}

PassContext::Scope::Scope(const PassContext& pass_ctx) {
    previous_ = std::make_unique<PassContext>(PassContext::Current());
    if (!pass_ctx.defined()) {
        active_ = false;
        return;
    }
    PassContext::SetCurrent(pass_ctx);
    active_ = true;
}

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
    if (pass_ctx.default_device_obj().defined()) {
        new_attrs.Set(String(kPassCtxDefaultDeviceAttr), pass_ctx.default_device_obj());
    }
    new_attrs.Set(String(kPassCtxMultiDeviceAttr),
                  tir::IntImm(pass_ctx.is_multi_device() ? 1 : 0, tir::DataType::Bool()));
    if (pass_ctx.has_disco_placement()) {
        new_attrs.Set(String(kPassCtxDiscoPlacementAttr), ObjectRef(pass_ctx.disco_placement()));
    }
    return new_attrs;
}

PassContext BuildDiscoPlacementPass(const Expr& expr) {
    PassContext pass_ctx = PassContext::FromRelay(expr);
    if (!pass_ctx.has_disco_placement()) {
        pass_ctx = PassContext::WithDiscoPlacement(
            pass_ctx, BuildDiscoPlacement(pass_ctx.virtual_devices(), /*num_groups=*/1));
    }
    return pass_ctx;
}

PassContext BuildDiscoPlacementPass(const Function& func) {
    return BuildDiscoPlacementPass(Expr(func));
}

Expr RelayPass::Mutate(const Expr& expr) {
    PassContext pass_ctx = PassContext::Current();
    if (!pass_ctx.defined()) {
        pass_ctx = PassContext::FromRelay(expr);
    }
    PassContext::Scope scope(pass_ctx);
    return VisitExpr(expr);
}

Function RelayPass::Mutate(const Function& func) {
    if (!func.defined()) return func;
    Expr out = Mutate(Expr(func));
    if (!out.defined()) return Function();
    if (!out.As<FunctionNode>()) {
        throw std::runtime_error("RelayPass expected Function result when mutating Function");
    }
    return Function(out);
}

Var RelayPass::Mutate(const Var& var) { return MutateToVar(var); }

Expr RelayPass::VisitConstant(const ConstantNode* op, const Expr& ref) {
    (void)op;
    return ref;
}

Expr RelayPass::VisitVar(const VarNode* op, const Expr& ref) {
    (void)op;
    return ref;
}

Expr RelayPass::VisitOp(const relay::OpNode* op, const Expr& ref) {
    (void)op;
    return ref;
}

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

Expr RelayPass::VisitTupleGetItem(const TupleGetItemNode* op, const Expr& ref) {
    auto new_tuple = Mutate(op->tuple);
    if (new_tuple.get() == op->tuple.get()) return ref;
    return CopyRelayVirtualDevice(ref, TupleGetItem(new_tuple, op->index));
}

Var RelayPass::MutateToVar(const Var& var) {
    if (!var.defined()) return var;
    Expr new_var = Mutate(Expr(var));
    if (!new_var.defined()) {
        throw std::runtime_error("RelayPass mutated Var into undefined expression");
    }
    if (!new_var.As<VarNode>()) {
        throw std::runtime_error("RelayPass expects variable position to remain Var");
    }
    return Var(new_var);
}

tir::PrimExpr TIRPass::Mutate(const tir::PrimExpr& expr) { return VisitExpr(expr); }

tir::Stmt TIRPass::Mutate(const tir::Stmt& stmt) { return VisitStmt(stmt); }

tir::PrimFunc TIRPass::Mutate(const tir::PrimFunc& func) {
    PassContext pass_ctx = PassContext::Current();
    if (!pass_ctx.defined()) {
        pass_ctx = PassContext::FromTIR(func);
    }
    PassContext::Scope scope(pass_ctx);
    return VisitPrimFunc(func);
}

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

tir::PrimExpr TIRPass::VisitIntImm(const tir::IntImmNode* op, const tir::PrimExpr& ref) {
    (void)op;
    return ref;
}

tir::PrimExpr TIRPass::VisitFloatImm(const tir::FloatImmNode* op, const tir::PrimExpr& ref) {
    (void)op;
    return ref;
}

tir::PrimExpr TIRPass::VisitVar(const tir::VarNode* op, const tir::PrimExpr& ref) {
    (void)op;
    return ref;
}

tir::PrimExpr TIRPass::VisitAdd(const tir::AddNode* op, const tir::PrimExpr& ref) {
    tir::PrimExpr new_a = Mutate(op->a);
    tir::PrimExpr new_b = Mutate(op->b);
    if (new_a.get() == op->a.get() && new_b.get() == op->b.get()) return ref;
    return tir::Add(new_a, new_b);
}

tir::PrimExpr TIRPass::VisitSub(const tir::SubNode* op, const tir::PrimExpr& ref) {
    tir::PrimExpr new_a = Mutate(op->a);
    tir::PrimExpr new_b = Mutate(op->b);
    if (new_a.get() == op->a.get() && new_b.get() == op->b.get()) return ref;
    return tir::Sub(new_a, new_b);
}

tir::PrimExpr TIRPass::VisitMul(const tir::MulNode* op, const tir::PrimExpr& ref) {
    tir::PrimExpr new_a = Mutate(op->a);
    tir::PrimExpr new_b = Mutate(op->b);
    if (new_a.get() == op->a.get() && new_b.get() == op->b.get()) return ref;
    return tir::Mul(new_a, new_b);
}

tir::PrimExpr TIRPass::VisitDiv(const tir::DivNode* op, const tir::PrimExpr& ref) {
    tir::PrimExpr new_a = Mutate(op->a);
    tir::PrimExpr new_b = Mutate(op->b);
    if (new_a.get() == op->a.get() && new_b.get() == op->b.get()) return ref;
    return tir::Div(new_a, new_b);
}

tir::PrimExpr TIRPass::VisitMod(const tir::ModNode* op, const tir::PrimExpr& ref) {
    tir::PrimExpr new_a = Mutate(op->a);
    tir::PrimExpr new_b = Mutate(op->b);
    if (new_a.get() == op->a.get() && new_b.get() == op->b.get()) return ref;
    return tir::Mod(new_a, new_b);
}

tir::PrimExpr TIRPass::VisitMin(const tir::MinNode* op, const tir::PrimExpr& ref) {
    tir::PrimExpr new_a = Mutate(op->a);
    tir::PrimExpr new_b = Mutate(op->b);
    if (new_a.get() == op->a.get() && new_b.get() == op->b.get()) return ref;
    return tir::Min(new_a, new_b);
}

tir::PrimExpr TIRPass::VisitMax(const tir::MaxNode* op, const tir::PrimExpr& ref) {
    tir::PrimExpr new_a = Mutate(op->a);
    tir::PrimExpr new_b = Mutate(op->b);
    if (new_a.get() == op->a.get() && new_b.get() == op->b.get()) return ref;
    return tir::Max(new_a, new_b);
}

tir::PrimExpr TIRPass::VisitEQ(const tir::EQNode* op, const tir::PrimExpr& ref) {
    tir::PrimExpr new_a = Mutate(op->a);
    tir::PrimExpr new_b = Mutate(op->b);
    if (new_a.get() == op->a.get() && new_b.get() == op->b.get()) return ref;
    return tir::EQ(new_a, new_b);
}

tir::PrimExpr TIRPass::VisitLT(const tir::LTNode* op, const tir::PrimExpr& ref) {
    tir::PrimExpr new_a = Mutate(op->a);
    tir::PrimExpr new_b = Mutate(op->b);
    if (new_a.get() == op->a.get() && new_b.get() == op->b.get()) return ref;
    return tir::LT(new_a, new_b);
}

tir::PrimExpr TIRPass::VisitAnd(const tir::AndNode* op, const tir::PrimExpr& ref) {
    tir::PrimExpr new_a = Mutate(op->a);
    tir::PrimExpr new_b = Mutate(op->b);
    if (new_a.get() == op->a.get() && new_b.get() == op->b.get()) return ref;
    return tir::And(new_a, new_b);
}

tir::PrimExpr TIRPass::VisitOr(const tir::OrNode* op, const tir::PrimExpr& ref) {
    tir::PrimExpr new_a = Mutate(op->a);
    tir::PrimExpr new_b = Mutate(op->b);
    if (new_a.get() == op->a.get() && new_b.get() == op->b.get()) return ref;
    return tir::Or(new_a, new_b);
}

tir::PrimExpr TIRPass::VisitNot(const tir::NotNode* op, const tir::PrimExpr& ref) {
    tir::PrimExpr new_value = Mutate(op->value);
    if (new_value.get() == op->value.get()) return ref;
    return tir::Not(new_value);
}

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

tir::Stmt TIRPass::VisitAttrStmt(const tir::AttrStmtNode* op, const tir::Stmt& ref) {
    tir::PrimExpr new_value = Mutate(op->value);
    tir::Stmt new_body = Mutate(op->body);
    if (new_value.get() == op->value.get() && new_body.get() == op->body.get()) {
        return ref;
    }
    return tir::AttrStmt(op->node, op->attr_key, new_value, new_body);
}

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

tir::Stmt TIRPass::VisitEvaluate(const tir::EvaluateNode* op, const tir::Stmt& ref) {
    tir::PrimExpr new_value = Mutate(op->value);
    if (new_value.get() == op->value.get()) return ref;
    return tir::Evaluate(new_value);
}

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

tir::Range TIRPass::MutateRange(const tir::Range& range) {
    if (!range.defined()) return range;
    tir::PrimExpr new_min = Mutate(range->min);
    tir::PrimExpr new_extent = Mutate(range->extent);
    if (new_min.get() == range->min.get() && new_extent.get() == range->extent.get()) {
        return range;
    }
    return tir::Range(new_min, new_extent);
}

tir::IterVar TIRPass::MutateIterVar(const tir::IterVar& iv) {
    if (!iv.defined()) return iv;
    tir::Range new_dom = MutateRange(iv->dom);
    tir::Var new_var = MutateToVar(iv->var);
    if (new_dom.get() == iv->dom.get() && new_var.get() == iv->var.get()) {
        return iv;
    }
    return tir::IterVar(new_dom, new_var, iv->iter_type, iv->thread_tag);
}

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
