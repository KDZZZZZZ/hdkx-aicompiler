/*! \file src/relay/visitor.cc
 * \brief Implements Relay visitors and PassContext inference.
 */

#include "kxc/relay/visitor.h"

#include <stdexcept>
#include <unordered_set>

namespace kxc {
namespace {

void CollectRelayVirtualDevices(
    const Expr& expr, std::unordered_set<const Object*>* visited_exprs,
    std::unordered_set<const Object*>* visited_virtual_devices,
    Array<VirtualDevice>* out_virtual_devices) {
    if (!expr.defined()) return;
    const Object* obj = expr.get();
    if (!obj || visited_exprs->count(obj)) return;
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
        CollectRelayVirtualDevices(call->op, visited_exprs,
                                   visited_virtual_devices, out_virtual_devices);
        for (const auto& arg : call->args) {
            CollectRelayVirtualDevices(arg, visited_exprs,
                                       visited_virtual_devices,
                                       out_virtual_devices);
        }
    } else if (auto* fn = expr.As<FunctionNode>()) {
        for (const auto& param : fn->params) {
            CollectRelayVirtualDevices(Expr(ObjectRef(param)), visited_exprs,
                                       visited_virtual_devices,
                                       out_virtual_devices);
        }
        CollectRelayVirtualDevices(fn->body, visited_exprs,
                                   visited_virtual_devices, out_virtual_devices);
    } else if (auto* if_node = expr.As<IfNode>()) {
        CollectRelayVirtualDevices(if_node->cond, visited_exprs,
                                   visited_virtual_devices, out_virtual_devices);
        CollectRelayVirtualDevices(if_node->true_branch, visited_exprs,
                                   visited_virtual_devices, out_virtual_devices);
        CollectRelayVirtualDevices(if_node->false_branch, visited_exprs,
                                   visited_virtual_devices, out_virtual_devices);
    } else if (auto* let_node = expr.As<LetNode>()) {
        CollectRelayVirtualDevices(Expr(ObjectRef(let_node->var)), visited_exprs,
                                   visited_virtual_devices, out_virtual_devices);
        CollectRelayVirtualDevices(let_node->value, visited_exprs,
                                   visited_virtual_devices, out_virtual_devices);
        CollectRelayVirtualDevices(let_node->body, visited_exprs,
                                   visited_virtual_devices, out_virtual_devices);
    } else if (auto* tuple = expr.As<TupleNode>()) {
        for (const auto& field : tuple->fields) {
            CollectRelayVirtualDevices(field, visited_exprs,
                                       visited_virtual_devices,
                                       out_virtual_devices);
        }
    } else if (auto* tuple_get = expr.As<TupleGetItemNode>()) {
        CollectRelayVirtualDevices(tuple_get->tuple, visited_exprs,
                                   visited_virtual_devices, out_virtual_devices);
    }
}

Expr CopyRelayVirtualDevice(const Expr& source, const Expr& dest) {
    if (!source.defined() || !dest.defined()) return dest;
    const RelayNode* src_node = dynamic_cast<const RelayNode*>(source.get());
    RelayNode* dst_node =
        const_cast<RelayNode*>(dynamic_cast<const RelayNode*>(dest.get()));
    if (!src_node || !dst_node) return dest;
    dst_node->virtual_device_ = src_node->virtual_device_;
    dst_node->checked_type_ = src_node->checked_type_;
    return dest;
}

}  // namespace

namespace relay {

PassContext PassContextFromRelay(const Expr& expr) {
    Array<VirtualDevice> virtual_devices;
    std::unordered_set<const Object*> visited_exprs;
    std::unordered_set<const Object*> visited_virtual_devices;
    CollectRelayVirtualDevices(expr, &visited_exprs, &visited_virtual_devices,
                               &virtual_devices);
    return PassContext::FromVirtualDevices(virtual_devices);
}

PassContext PassContextFromRelay(const Function& func) {
    return PassContextFromRelay(Expr(ObjectRef(func)));
}

}  // namespace relay

Expr RelayPass::Mutate(const Expr& expr) {
    PassContext pass_ctx = PassContext::Current();
    if (!pass_ctx.defined()) {
        pass_ctx = relay::PassContextFromRelay(expr);
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

}  // namespace kxc
