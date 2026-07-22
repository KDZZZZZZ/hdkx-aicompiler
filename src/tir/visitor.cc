/*! \file src/tir/visitor.cc
 * \brief Implements TIR visitors and PassContext attribute adapters.
 */

#include "kxc/tir/visitor.h"

#include <utility>

namespace kxc {
namespace {

constexpr const char* kPassCtxPrimaryVirtualDeviceAttr =
    "kxc.pass_ctx.primary_virtual_device";
constexpr const char* kPassCtxDefaultTargetAttr =
    "kxc.pass_ctx.default_target";
constexpr const char* kPassCtxDefaultDeviceAttr =
    "kxc.pass_ctx.default_device";
constexpr const char* kPassCtxMultiDeviceAttr =
    "kxc.pass_ctx.is_multi_device";

bool IsTargetObjectRef(const ObjectRef& obj) {
    return obj.defined() && obj.get()->GetTypeId() == TargetNode::_type_index;
}

bool IsVirtualDeviceObjectRef(const ObjectRef& obj) {
    return obj.defined() &&
           obj.get()->GetTypeId() == VirtualDeviceNode::_type_index;
}

}  // namespace

namespace tir {

PassContext PassContextFromTIR(const PrimFunc& func) {
    if (!func.defined()) return PassContext();

    Array<VirtualDevice> virtual_devices;
    Target default_target;
    Device default_device;
    bool is_multi_device = false;
    bool defined = false;

    if (func->attrs.count(String(kPassCtxPrimaryVirtualDeviceAttr))) {
        ObjectRef vd_ref =
            func->attrs.at(String(kPassCtxPrimaryVirtualDeviceAttr));
        if (IsVirtualDeviceObjectRef(vd_ref)) {
            virtual_devices.push_back(VirtualDevice(vd_ref));
            defined = true;
        }
    }
    if (func->attrs.count(String(kPassCtxDefaultTargetAttr))) {
        ObjectRef target_ref = func->attrs.at(String(kPassCtxDefaultTargetAttr));
        if (IsTargetObjectRef(target_ref)) {
            default_target = Target(target_ref);
            defined = true;
        }
    }
    if (func->attrs.count(String(kPassCtxDefaultDeviceAttr))) {
        default_device =
            Device(func->attrs.at(String(kPassCtxDefaultDeviceAttr)));
        defined = true;
    }
    if (func->attrs.count(String(kPassCtxMultiDeviceAttr))) {
        ObjectRef multi_ref =
            func->attrs.at(String(kPassCtxMultiDeviceAttr));
        if (auto* imm = multi_ref.As<IntImmNode>()) {
            is_multi_device = imm->value != 0;
            defined = true;
        }
    }

    PassContext inferred =
        PassContext::FromVirtualDevices(virtual_devices);
    if (!default_target.defined()) default_target = inferred.default_target();
    if (!default_device.defined()) default_device = inferred.default_device();
    return PassContext::FromComponents(
        virtual_devices, std::move(default_target), std::move(default_device),
        is_multi_device || inferred.is_multi_device(), defined);
}

Map<String, ObjectRef> AttachPassContextAttrs(
    const Map<String, ObjectRef>& attrs, const PassContext& pass_ctx) {
    Map<String, ObjectRef> new_attrs;
    for (const auto& kv : attrs) new_attrs.Set(kv.first, kv.second);
    if (!pass_ctx.defined()) return new_attrs;

    if (pass_ctx.primary_virtual_device().defined()) {
        new_attrs.Set(String(kPassCtxPrimaryVirtualDeviceAttr),
                      ObjectRef(pass_ctx.primary_virtual_device()));
    }
    if (pass_ctx.default_target().defined()) {
        new_attrs.Set(String(kPassCtxDefaultTargetAttr),
                      ObjectRef(pass_ctx.default_target()));
    }
    if (pass_ctx.default_device().defined()) {
        new_attrs.Set(String(kPassCtxDefaultDeviceAttr),
                      ObjectRef(pass_ctx.default_device()));
    }
    new_attrs.Set(String(kPassCtxMultiDeviceAttr),
                  IntImm(pass_ctx.is_multi_device() ? 1 : 0,
                         DataType::Bool()));
    return new_attrs;
}

}  // namespace tir

tir::PrimExpr TIRPass::Mutate(const tir::PrimExpr& expr) { return VisitExpr(expr); }

// 递归重写 TIR 语句入口。
tir::Stmt TIRPass::Mutate(const tir::Stmt& stmt) { return VisitStmt(stmt); }

// 在当前或从属性恢复的 PassContext 中重写 PrimFunc。
tir::PrimFunc TIRPass::Mutate(const tir::PrimFunc& func) {
    PassContext pass_ctx = PassContext::Current();
    if (!pass_ctx.defined()) {
        pass_ctx = tir::PassContextFromTIR(func);
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

// 重写线程索引变量、启动范围与作用域体，同时保留结构化 CUDA 索引类别。
tir::Stmt TIRPass::VisitThreadBinding(const tir::ThreadBindingNode* op,
                                      const tir::Stmt& ref) {
    tir::Var new_thread_var = MutateToVar(op->thread_var);
    tir::PrimExpr new_extent = Mutate(op->extent);
    tir::Stmt new_body = Mutate(op->body);
    if (new_thread_var.get() == op->thread_var.get() &&
        new_extent.get() == op->extent.get() &&
        new_body.get() == op->body.get()) {
        return ref;
    }
    return tir::ThreadBinding(new_thread_var, op->thread_index, new_extent, new_body);
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
