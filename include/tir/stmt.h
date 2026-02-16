#pragma once

#include <string>
#include <vector>

#include "base/container.h"
#include "base/expr.h"
#include "tir/expr.h"

namespace kxc {
namespace tir {

class StmtNode : public Object {
public:
    KXC_OBJECT_DECLARE
    virtual void VisitAttrs(AttrVisitor& visitor) { (void)visitor; }
};
KXC_OBJECT_DEFINE(StmtNode)

class Stmt : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    const StmtNode* operator->() const { return static_cast<const StmtNode*>(object_); }
};

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
    LetStmt(Var var, PrimExpr value, Stmt body);
};

class StoreNode : public StmtNode {
public:
    Var buffer_var;
    PrimExpr value;
    PrimExpr index;
    PrimExpr predicate;

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(StoreNode)

class Store : public Stmt {
public:
    using Stmt::Stmt;
    Store(Var buffer_var, PrimExpr value, PrimExpr index, PrimExpr predicate = PrimExpr());
};

enum class ForType {
    Serial = 0,
    Parallel = 1,
    Vectorized = 2,
    Unrolled = 3,
};

class ForNode : public StmtNode {
public:
    Var loop_var;
    PrimExpr min;
    PrimExpr extent;
    ForType for_type;
    Stmt body;

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(ForNode)

class For : public Stmt {
public:
    using Stmt::Stmt;
    For(Var loop_var, PrimExpr min, PrimExpr extent, ForType for_type, Stmt body);
};

class IfThenElseNode : public StmtNode {
public:
    PrimExpr condition;
    Stmt then_case;
    Stmt else_case;

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(IfThenElseNode)

class IfThenElse : public Stmt {
public:
    using Stmt::Stmt;
    IfThenElse(PrimExpr condition, Stmt then_case, Stmt else_case = Stmt());
};

class AllocateNode : public StmtNode {
public:
    Var buffer_var;
    DataType dtype;
    Array<PrimExpr> extents;
    PrimExpr condition;
    Stmt body;

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(AllocateNode)

class Allocate : public Stmt {
public:
    using Stmt::Stmt;
    Allocate(Var buffer_var, DataType dtype, Array<PrimExpr> extents, PrimExpr condition,
             Stmt body);
};

class AttrStmtNode : public StmtNode {
public:
    ObjectRef node;
    std::string attr_key;
    PrimExpr value;
    Stmt body;

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(AttrStmtNode)

class AttrStmt : public Stmt {
public:
    using Stmt::Stmt;
    AttrStmt(ObjectRef node, std::string attr_key, PrimExpr value, Stmt body);
};

class Range;
class IterVar;
class Buffer;
class BufferRegion;

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
    Range(PrimExpr min, PrimExpr extent);
    const RangeNode* operator->() const { return static_cast<const RangeNode*>(object_); }
};

enum class IterVarType : int {
    kDataPar = 0,
    kThreadIndex = 1,
    kCommReduce = 2,
    kOrdered = 3,
    kOpaque = 4,
    kVectorized = 5,
    kParallel = 6,
    kUnrolled = 7,
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
    IterVar(Range dom, Var var, IterVarType iter_type, std::string thread_tag = "");
    const IterVarNode* operator->() const { return static_cast<const IterVarNode*>(object_); }
};

class BufferNode : public Object {
public:
    Var data;
    DataType dtype;
    Array<PrimExpr> shape;
    Array<PrimExpr> strides;
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
    Buffer(Var data, DataType dtype, Array<PrimExpr> shape, Array<PrimExpr> strides,
           PrimExpr elem_offset, std::string name, int data_alignment, int offset_factor);
    const BufferNode* operator->() const { return static_cast<const BufferNode*>(object_); }
};

class BufferRegionNode : public Object {
public:
    Buffer buffer;
    Array<Range> region;
    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(BufferRegionNode)

class BufferRegion : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    BufferRegion(Buffer buffer, Array<Range> region);
    const BufferRegionNode* operator->() const { return static_cast<const BufferRegionNode*>(object_); }
};

class BlockNode : public StmtNode {
public:
    Array<IterVar> iter_vars;
    Array<BufferRegion> reads;
    Array<BufferRegion> writes;
    std::string name_hint;
    Stmt body;
    Stmt init;

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(BlockNode)

class Block : public Stmt {
public:
    using Stmt::Stmt;
    Block(Array<IterVar> iter_vars, Array<BufferRegion> reads, Array<BufferRegion> writes,
          std::string name_hint, Stmt body, Stmt init = Stmt());
};

class SeqStmtNode : public StmtNode {
public:
    Array<Stmt> seq;
    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(SeqStmtNode)

class SeqStmt : public Stmt {
public:
    using Stmt::Stmt;
    explicit SeqStmt(Array<Stmt> seq);
};

class EvaluateNode : public StmtNode {
public:
    PrimExpr value;
    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(EvaluateNode)

class Evaluate : public Stmt {
public:
    using Stmt::Stmt;
    explicit Evaluate(PrimExpr value);
};

class PrimFuncNode : public Object {
public:
    Array<Var> params;
    Stmt body;
    Map<Var, Buffer> buffer_map;
    Map<String, ObjectRef> attrs;

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(PrimFuncNode)

class PrimFunc : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    PrimFunc(Array<Var> params, Stmt body, Map<Var, Buffer> buffer_map = {},
             Map<String, ObjectRef> attrs = {});
    const PrimFuncNode* operator->() const { return static_cast<const PrimFuncNode*>(object_); }
};

}  // namespace tir
}  // namespace kxc