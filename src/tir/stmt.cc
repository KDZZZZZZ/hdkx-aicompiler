/*! \file src/tir/stmt.cc
 * \brief 实现 TIR 节点构造、工具函数和 pass 基础能力。
 */

#include "kxc/tir/stmt.h"
#include "kxc/support/object_registration.h"

#include <stdexcept>

namespace kxc {
namespace tir {

KXC_OBJECT_DEFINE(StmtNode)
KXC_OBJECT_DEFINE(LetStmtNode)
KXC_OBJECT_DEFINE(StoreNode)
KXC_OBJECT_DEFINE(ForNode)
KXC_OBJECT_DEFINE_WITH_KEY(ThreadBindingNode, "kxc.tir.ThreadBindingNode")
KXC_OBJECT_DEFINE(IfThenElseNode)
KXC_OBJECT_DEFINE(AllocateNode)
KXC_OBJECT_DEFINE(AttrStmtNode)
KXC_OBJECT_DEFINE(RangeNode)
KXC_OBJECT_DEFINE_WITH_KEY(IterVarNode, "kxc.tir.IterVarNode")
KXC_OBJECT_DEFINE(BufferNode)
KXC_OBJECT_DEFINE(BufferRegionNode)
KXC_OBJECT_DEFINE(BlockNode)
KXC_OBJECT_DEFINE(SeqStmtNode)
KXC_OBJECT_DEFINE(EvaluateNode)
KXC_OBJECT_DEFINE(PrimFuncNode)

LetStmt::LetStmt(Var var, PrimExpr value, Stmt body) {
    auto* node = new LetStmtNode();
    node->var = std::move(var);
    node->value = std::move(value);
    node->body = std::move(body);
    SetData(node);
}

Store::Store(Var buffer_var, PrimExpr value, PrimExpr index, PrimExpr predicate) {
    auto* node = new StoreNode();
    node->buffer_var = std::move(buffer_var);
    node->value = std::move(value);
    node->index = std::move(index);
    node->predicate = std::move(predicate);
    SetData(node);
}

For::For(Var loop_var, PrimExpr min, PrimExpr extent, ForType for_type, Stmt body) {
    auto* node = new ForNode();
    node->loop_var = std::move(loop_var);
    node->min = std::move(min);
    node->extent = std::move(extent);
    node->for_type = for_type;
    node->body = std::move(body);
    SetData(node);
}

// 构造结构化线程绑定时拒绝空字段和未知索引，避免 CUDA 后端猜测语义。
ThreadBinding::ThreadBinding(Var thread_var, ThreadIndexKind thread_index,
                             PrimExpr extent, Stmt body) {
    if (!thread_var.defined() || !thread_var.As<VarNode>()) {
        throw std::invalid_argument("ThreadBinding requires a defined thread Var");
    }
    if (!extent.defined() || !body.defined()) {
        throw std::invalid_argument("ThreadBinding requires extent and body");
    }
    const int kind = static_cast<int>(thread_index);
    if (kind < static_cast<int>(ThreadIndexKind::kBlockIdxX) ||
        kind > static_cast<int>(ThreadIndexKind::kThreadIdxZ)) {
        throw std::invalid_argument("ThreadBinding uses an unknown thread index");
    }
    auto* node = new ThreadBindingNode();
    node->thread_var = std::move(thread_var);
    node->thread_index = thread_index;
    node->extent = std::move(extent);
    node->body = std::move(body);
    SetData(node);
}

// undefined 或其他 Stmt 类型不能被静态解释为线程绑定。
const ThreadBindingNode* ThreadBinding::operator->() const {
    const auto* node = As<ThreadBindingNode>();
    if (!node) throw std::runtime_error("undefined or invalid ThreadBinding");
    return node;
}

IfThenElse::IfThenElse(PrimExpr condition, Stmt then_case, Stmt else_case) {
    auto* node = new IfThenElseNode();
    node->condition = std::move(condition);
    node->then_case = std::move(then_case);
    node->else_case = std::move(else_case);
    SetData(node);
}

Allocate::Allocate(Var buffer_var, DataType dtype, Array<PrimExpr> extents, PrimExpr condition,
                   Stmt body) {
    auto* node = new AllocateNode();
    node->buffer_var = std::move(buffer_var);
    node->dtype = dtype;
    node->extents = std::move(extents);
    node->condition = std::move(condition);
    node->body = std::move(body);
    SetData(node);
}

AttrStmt::AttrStmt(ObjectRef node_ref, std::string attr_key, PrimExpr value, Stmt body) {
    auto* node = new AttrStmtNode();
    node->node = std::move(node_ref);
    node->attr_key = std::move(attr_key);
    node->value = std::move(value);
    node->body = std::move(body);
    SetData(node);
}

Range::Range(PrimExpr min, PrimExpr extent) {
    auto* node = new RangeNode();
    node->min = std::move(min);
    node->extent = std::move(extent);
    SetData(node);
}

IterVar::IterVar(Range dom, Var var, IterVarType iter_type, std::string thread_tag) {
    auto* node = new IterVarNode();
    node->dom = std::move(dom);
    node->var = std::move(var);
    node->iter_type = iter_type;
    node->thread_tag = std::move(thread_tag);
    SetData(node);
}

Buffer::Buffer(Var data, DataType dtype, Array<PrimExpr> shape, Array<PrimExpr> strides,
               PrimExpr elem_offset, std::string name, int data_alignment, int offset_factor) {
    auto* node = new BufferNode();
    node->data = std::move(data);
    node->dtype = dtype;
    node->shape = std::move(shape);
    node->strides = std::move(strides);
    node->elem_offset = std::move(elem_offset);
    node->name = std::move(name);
    node->data_alignment = data_alignment;
    node->offset_factor = offset_factor;
    SetData(node);
}

BufferRegion::BufferRegion(Buffer buffer, Array<Range> region) {
    auto* node = new BufferRegionNode();
    node->buffer = std::move(buffer);
    node->region = std::move(region);
    SetData(node);
}

Block::Block(Array<IterVar> iter_vars, Array<BufferRegion> reads, Array<BufferRegion> writes,
             std::string name_hint, Stmt body, Stmt init) {
    auto* node = new BlockNode();
    node->iter_vars = std::move(iter_vars);
    node->reads = std::move(reads);
    node->writes = std::move(writes);
    node->name_hint = std::move(name_hint);
    node->body = std::move(body);
    node->init = std::move(init);
    SetData(node);
}

SeqStmt::SeqStmt(Array<Stmt> seq) {
    auto* node = new SeqStmtNode();
    node->seq = std::move(seq);
    SetData(node);
}

Evaluate::Evaluate(PrimExpr value) {
    auto* node = new EvaluateNode();
    node->value = std::move(value);
    SetData(node);
}

PrimFunc::PrimFunc(Array<Var> params, Stmt body, Map<Var, Buffer> buffer_map,
                   Map<String, ObjectRef> attrs) {
    auto* node = new PrimFuncNode();
    node->params = std::move(params);
    node->body = std::move(body);
    node->buffer_map = std::move(buffer_map);
    node->attrs = std::move(attrs);
    SetData(node);
}

}  // namespace tir
}  // namespace kxc
