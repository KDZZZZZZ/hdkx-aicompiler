#pragma once
#include "tir/expr.h"
#include "base/expr.h"
#include <vector>
#include <string>

namespace kxc {
namespace tir {

class StmtNode : public Object {
public:
    const TypeIndex GetTypeId() const override { return kKXC_OBJECT_TYPE + 200; }
    virtual void VisitAttrs(AttrVisitor& visitor) {};
};

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

    const TypeIndex GetTypeId() const override { return kKXC_OBJECT_TYPE + 201; }
};

class LetStmt : public Stmt {
public:
    using Stmt::Stmt;
    LetStmt(Var var, PrimExpr value, Stmt body) {
        auto* node = new LetStmtNode();
        node->var = var;
        node->value = value;
        node->body = body;
        object_ = node;
        if (object_) object_->IncRef();
    }
};

// 2. Store: buffer_var[index] = value;
class StoreNode : public StmtNode {
public:
    Var buffer_var;
    PrimExpr value;
    PrimExpr index;
    PrimExpr predicate; // Optional mask

    const TypeIndex GetTypeId() const override { return kKXC_OBJECT_TYPE + 202; }
};

class Store : public Stmt {
public:
    using Stmt::Stmt;
    Store(Var buffer_var, PrimExpr value, PrimExpr index, PrimExpr predicate = PrimExpr()) {
        auto* node = new StoreNode();
        node->buffer_var = buffer_var;
        node->value = value;
        node->index = index;
        node->predicate = predicate;
        object_ = node;
        if (object_) object_->IncRef();
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

    const TypeIndex GetTypeId() const override { return kKXC_OBJECT_TYPE + 203; }
};

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
        object_ = node;
        if (object_) object_->IncRef();
    }
};

// 4. IfThenElse
class IfThenElseNode : public StmtNode {
public:
    PrimExpr condition;
    Stmt then_case;
    Stmt else_case; // Optional

    const TypeIndex GetTypeId() const override { return kKXC_OBJECT_TYPE + 204; }
};

class IfThenElse : public Stmt {
public:
    using Stmt::Stmt;
    IfThenElse(PrimExpr condition, Stmt then_case, Stmt else_case = Stmt()) {
        auto* node = new IfThenElseNode();
        node->condition = condition;
        node->then_case = then_case;
        node->else_case = else_case;
        object_ = node;
        if (object_) object_->IncRef();
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

    const TypeIndex GetTypeId() const override { return kKXC_OBJECT_TYPE + 205; }
};

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
        object_ = node;
        if (object_) object_->IncRef();
    }
};

// 6. AttrStmt: Annotate scope (e.g. thread binding)
class AttrStmtNode : public StmtNode {
public:
    ObjectRef node; // The object being annotated (e.g., iter_var)
    std::string attr_key;
    PrimExpr value;
    Stmt body;

    const TypeIndex GetTypeId() const override { return kKXC_OBJECT_TYPE + 206; }
};

class AttrStmt : public Stmt {
public:
    using Stmt::Stmt;
    AttrStmt(ObjectRef node, std::string attr_key, PrimExpr value, Stmt body) {
        auto* n = new AttrStmtNode();
        n->node = node;
        n->attr_key = std::move(attr_key);
        n->value = value;
        n->body = body;
        object_ = n;
        if (object_) object_->IncRef();
    }
};

// 7. Block: { stmt1; stmt2; ... }
class BlockNode : public StmtNode {
public:
    Stmt first;
    Stmt rest;

    const TypeIndex GetTypeId() const override { return kKXC_OBJECT_TYPE + 207; }
};

// SeqStmt in TVM is a sequence of statements.
// Block in recent TVM refers to a scoped computation block (TensorIR).
// Here we implement a simple sequence block (SeqStmt style).
class SeqStmtNode : public StmtNode {
public:
    std::vector<Stmt> seq;
    const TypeIndex GetTypeId() const override { return kKXC_OBJECT_TYPE + 207; }
};

class SeqStmt : public Stmt {
public:
    using Stmt::Stmt;
    explicit SeqStmt(std::vector<Stmt> seq) {
        auto* node = new SeqStmtNode();
        node->seq = std::move(seq);
        object_ = node;
        if (object_) object_->IncRef();
    }
};

// 8. Evaluate: Execute expression for side effects
class EvaluateNode : public StmtNode {
public:
    PrimExpr value;
    const TypeIndex GetTypeId() const override { return kKXC_OBJECT_TYPE + 208; }
};

class Evaluate : public Stmt {
public:
    using Stmt::Stmt;
    explicit Evaluate(PrimExpr value) {
        auto* node = new EvaluateNode();
        node->value = value;
        object_ = node;
        if (object_) object_->IncRef();
    }
};

} // namespace tir
} // namespace kxc
