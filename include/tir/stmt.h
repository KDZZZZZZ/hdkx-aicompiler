#pragma once
#include "tir/expr.h"
#include "base/expr.h"
#include <vector>
#include <string>

namespace kxc {
namespace tir {

class StmtNode : public Object {
public:
    KXC_OBJECT_DECLARE
    virtual void VisitAttrs(AttrVisitor& visitor) {};
};
KXC_OBJECT_DEFINE(StmtNode)

class Stmt : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    const StmtNode* operator->() const {return static_cast<const StmtNode*>(object_);}
};

// 1. LetStmt: let var = value; body;
class LetStmtNode : public StmtNode {
public:
    Var var;
    PrimExpr value;
    Stmt body;

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(LetStmtNode)

class LetStmt : public Stmt {
public:
    using Stmt::Stmt;
    LetStmt(Var var, PrimExpr value, Stmt body) {
        auto* node = new LetStmtNode();
        node->var = var;
        node->value = value;
        node->body = body;
        SetData(node);
    }
};

// 2. Store: buffer_var[index] = value;
class StoreNode : public StmtNode {
public:
    Var buffer_var;
    PrimExpr value;
    PrimExpr index;
    PrimExpr predicate; // Optional mask

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(StoreNode)

class Store : public Stmt {
public:
    using Stmt::Stmt;
    Store(Var buffer_var, PrimExpr value, PrimExpr index, PrimExpr predicate = PrimExpr()) {
        auto* node = new StoreNode();
        node->buffer_var = buffer_var;
        node->value = value;
        node->index = index;
        node->predicate = predicate;
        SetData(node);
    }
};

// 3. For Loop
enum class ForType {
    Serial = 0,
    Parallel = 1,
    Vectorized = 2,
    Unrolled = 3
};

class ForNode : public StmtNode {
public:
    Var loop_var;
    PrimExpr min;
    PrimExpr extent;
    ForType for_type;
    // DeviceAPI device_api; // Optional device context
    Stmt body;

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(ForNode)

class For : public Stmt {
public:
    using Stmt::Stmt;
    For(Var loop_var, PrimExpr min, PrimExpr extent, ForType for_type, Stmt body) {
        auto* node = new ForNode();
        node->loop_var = loop_var;
        node->min = min;
        node->extent = extent;
        node->for_type = for_type;
        node->body = body;
        SetData(node);
    }
};

// 4. IfThenElse
class IfThenElseNode : public StmtNode {
public:
    PrimExpr condition;
    Stmt then_case;
    Stmt else_case; // Optional

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(IfThenElseNode)

class IfThenElse : public Stmt {
public:
    using Stmt::Stmt;
    IfThenElse(PrimExpr condition, Stmt then_case, Stmt else_case = Stmt()) {
        auto* node = new IfThenElseNode();
        node->condition = condition;
        node->then_case = then_case;
        node->else_case = else_case;
        SetData(node);
    }
};

// 5. Allocate: float buffer[size]; body;
class AllocateNode : public StmtNode {
public:
    Var buffer_var;
    DataType dtype;
    std::vector<PrimExpr> extents;
    PrimExpr condition; // Optional condition
    Stmt body;

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(AllocateNode)

class Allocate : public Stmt {
public:
    using Stmt::Stmt;
    Allocate(Var buffer_var, DataType dtype, std::vector<PrimExpr> extents, PrimExpr condition, Stmt body) {
        auto* node = new AllocateNode();
        node->buffer_var = buffer_var;
        node->dtype = dtype;
        node->extents = std::move(extents);
        node->condition = condition;
        node->body = body;
        SetData(node);
    }
};

// 6. AttrStmt: Annotate scope (e.g. thread binding)
class AttrStmtNode : public StmtNode {
public:
    ObjectRef node; // The object being annotated (e.g., iter_var)
    std::string attr_key;
    PrimExpr value;
    Stmt body;

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(AttrStmtNode)

class AttrStmt : public Stmt {
public:
    using Stmt::Stmt;
    AttrStmt(ObjectRef node, std::string attr_key, PrimExpr value, Stmt body) {
        auto* n = new AttrStmtNode();
        n->node = node;
        n->attr_key = std::move(attr_key);
        n->value = value;
        n->body = body;
        SetData(n);
    }
};

// Forward declarations
class Range;
class IterVar;
class Buffer;
class BufferRegion;

// Range
class RangeNode : public Object {
public:
    PrimExpr min;
    PrimExpr extent;
    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(RangeNode)

class Range : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    Range(PrimExpr min, PrimExpr extent) {
        auto* node = new RangeNode();
        node->min = min;
        node->extent = extent;
        SetData(node);
    }
    const RangeNode* operator->() const { return static_cast<const RangeNode*>(object_); }
};

// IterVar (TIR)
enum class IterVarType : int {
    kDataPar = 0,
    kThreadIndex = 1,
    kCommReduce = 2,
    kOrdered = 3,
    kOpaque = 4,
    kVectorized = 5,
    kParallel = 6,
    kUnrolled = 7
};

class IterVarNode : public Object {
public:
    Range dom;
    Var var;
    IterVarType iter_type;
    std::string thread_tag;
    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(IterVarNode)

class IterVar : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    IterVar(Range dom, Var var, IterVarType iter_type, std::string thread_tag = "") {
        auto* node = new IterVarNode();
        node->dom = dom;
        node->var = var;
        node->iter_type = iter_type;
        node->thread_tag = thread_tag;
        SetData(node);
    }
    const IterVarNode* operator->() const { return static_cast<const IterVarNode*>(object_); }
};

// Buffer
class BufferNode : public Object {
public:
    Var data;
    DataType dtype;
    std::vector<PrimExpr> shape;
    std::vector<PrimExpr> strides;
    PrimExpr elem_offset;
    std::string name;
    int data_alignment;
    int offset_factor;
    
    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(BufferNode)

class Buffer : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    Buffer(Var data, DataType dtype, std::vector<PrimExpr> shape, std::vector<PrimExpr> strides, PrimExpr elem_offset, std::string name, int data_alignment, int offset_factor) {
        auto* node = new BufferNode();
        node->data = data;
        node->dtype = dtype;
        node->shape = std::move(shape);
        node->strides = std::move(strides);
        node->elem_offset = elem_offset;
        node->name = std::move(name);
        node->data_alignment = data_alignment;
        node->offset_factor = offset_factor;
        SetData(node);
    }
    const BufferNode* operator->() const { return static_cast<const BufferNode*>(object_); }
};

// BufferRegion
class BufferRegionNode : public Object {
public:
    Buffer buffer;
    std::vector<Range> region;
    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(BufferRegionNode)

class BufferRegion : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    BufferRegion(Buffer buffer, std::vector<Range> region) {
        auto* node = new BufferRegionNode();
        node->buffer = buffer;
        node->region = std::move(region);
        SetData(node);
    }
    const BufferRegionNode* operator->() const { return static_cast<const BufferRegionNode*>(object_); }
};

// 7. Block (TensorIR)
class BlockNode : public StmtNode {
public:
    std::vector<IterVar> iter_vars;
    std::vector<BufferRegion> reads;
    std::vector<BufferRegion> writes;
    std::string name_hint;
    Stmt body;
    Stmt init; // Optional

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(BlockNode)

class Block : public Stmt {
public:
    using Stmt::Stmt;
    Block(std::vector<IterVar> iter_vars, std::vector<BufferRegion> reads, std::vector<BufferRegion> writes, std::string name_hint, Stmt body, Stmt init = Stmt()) {
        auto* node = new BlockNode();
        node->iter_vars = std::move(iter_vars);
        node->reads = std::move(reads);
        node->writes = std::move(writes);
        node->name_hint = std::move(name_hint);
        node->body = body;
        node->init = init;
        SetData(node);
    }
};

// SeqStmt in TVM is a sequence of statements.
// Block in recent TVM refers to a scoped computation block (TensorIR).
// Here we implement a simple sequence block (SeqStmt style).
class SeqStmtNode : public StmtNode {
public:
    std::vector<Stmt> seq;
    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(SeqStmtNode)

class SeqStmt : public Stmt {
public:
    using Stmt::Stmt;
    explicit SeqStmt(std::vector<Stmt> seq) {
        auto* node = new SeqStmtNode();
        node->seq = std::move(seq);
        SetData(node);
    }
};

// 8. Evaluate: Execute expression for side effects
class EvaluateNode : public StmtNode {
public:
    PrimExpr value;
    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(EvaluateNode)

class Evaluate : public Stmt {
public:
    using Stmt::Stmt;
    explicit Evaluate(PrimExpr value) {
        auto* node = new EvaluateNode();
        node->value = value;
        SetData(node);
    }
};

} // namespace tir
} // namespace kxc
