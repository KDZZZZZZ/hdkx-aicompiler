#pragma once
#include "base/expr.h"
#include "base/tensor.h"
#include "base/ndarray.h"
#include "base/container.h"
#include <vector>
#include <string>
#include <cstdint>

namespace kxc {

class RelayNode : public ExprNode {
public:
    // Relay specific metadata can go here
};

class Relay : public Expr {
public:
    using Expr::Expr;
};

// --- Type ---
class TensorTypeNode : public TypeNode {
public:
    Array<int64_t> shape;
    std::string dtype;

    KXC_OBJECT_DECLARE
};

KXC_OBJECT_DEFINE(TensorTypeNode)

class TensorType : public Type {
public:
    using Type::Type;

    TensorType(Array<int64_t> shape, std::string dtype) {
        auto* node = new TensorTypeNode();
        node->shape = std::move(shape);
        node->dtype = std::move(dtype);
        SetData(node);
    }

    const TensorTypeNode* operator->() const {
        return static_cast<const TensorTypeNode*>(object_);
    }
};

// --- Var ---
class IdNode : public Object {
public:
    std::string name_hint;
    KXC_OBJECT_DECLARE
};

KXC_OBJECT_DEFINE(IdNode)

class Id : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    explicit Id(std::string name) {
        auto* node = new IdNode();
        node->name_hint = std::move(name);
        SetData(node);
    }
    const IdNode* operator->() const { return static_cast<const IdNode*>(object_); }
};

class VarNode : public RelayNode {
public:
    Id vid;
    Type type_annotation;

    KXC_OBJECT_DECLARE

    void VisitAttrs(AttrVisitor& visitor) override {
        // visitor("vid", &vid);
    }
};

KXC_OBJECT_DEFINE(VarNode)

class Var : public Relay {
public:
    using Relay::Relay;
    explicit Var(std::string name) { 
        VarNode* node = new VarNode();
        node->vid = Id(std::move(name));
        SetData(node);
    }
    Var(std::string name, Type type_annotation) {
        VarNode* node = new VarNode();
        node->vid = Id(std::move(name));
        node->type_annotation = std::move(type_annotation);
        SetData(node);
    }
    const VarNode* operator->() const {
        return static_cast<const VarNode*>(object_);
    }
};

// --- Constant ---
class ConstantNode : public RelayNode {
public:
    runtime::NDArray data;

    KXC_OBJECT_DECLARE
};

KXC_OBJECT_DEFINE(ConstantNode)

class Constant : public Relay {
public:
    using Relay::Relay;
    explicit Constant(runtime::NDArray data) {
        ConstantNode* node = new ConstantNode();
        node->data = data;
        SetData(node);
    }
    const ConstantNode* operator->() const {
        return static_cast<const ConstantNode*>(object_);
    }
};

// --- Call ---
class CallNode : public RelayNode {
public:
    Expr op;
    Array<Expr> args;
    ObjectRef attrs; // Changed from Attrs to ObjectRef to allow any Attrs type

    KXC_OBJECT_DECLARE
};

KXC_OBJECT_DEFINE(CallNode)

class Call : public Relay {
public:
    using Relay::Relay;
    Call(Expr op, Array<Expr> args, ObjectRef attrs = ObjectRef()) {
        CallNode* node = new CallNode();
        node->op = op;
        node->args = std::move(args);
        node->attrs = attrs;
        SetData(node);
    }
    const CallNode* operator->() const {
        return static_cast<const CallNode*>(object_);
    }
};

// --- Function ---
class FunctionNode : public RelayNode {
public:
    Array<Var> params;
    Expr body;
    // Type ret_type;

    KXC_OBJECT_DECLARE
};

KXC_OBJECT_DEFINE(FunctionNode)

class Function : public Relay {
public:
    using Relay::Relay;
    Function(Array<Var> params, Expr body) {
        FunctionNode* node = new FunctionNode();
        node->params = std::move(params);
        node->body = body;
        SetData(node);
    }
    const FunctionNode* operator->() const {
        return static_cast<const FunctionNode*>(object_);
    }
};

// --- Tuple ---
class TupleNode : public RelayNode {
public:
    Array<Expr> fields;

    KXC_OBJECT_DECLARE
};

KXC_OBJECT_DEFINE(TupleNode)

class Tuple : public Relay {
public:
    using Relay::Relay;
    explicit Tuple(Array<Expr> fields) {
        TupleNode* node = new TupleNode();
        node->fields = std::move(fields);
        SetData(node);
    }
    const TupleNode* operator->() const {
        return static_cast<const TupleNode*>(object_);
    }
};

// --- TupleGetItem ---
class TupleGetItemNode : public RelayNode {
public:
    Expr tuple;
    int index;

    KXC_OBJECT_DECLARE
};

KXC_OBJECT_DEFINE(TupleGetItemNode)

class TupleGetItem : public Relay {
public:
    using Relay::Relay;
    TupleGetItem(Expr tuple, int index) {
        TupleGetItemNode* node = new TupleGetItemNode();
        node->tuple = tuple;
        node->index = index;
        SetData(node);
    }
    const TupleGetItemNode* operator->() const {
        return static_cast<const TupleGetItemNode*>(object_);
    }
};

// --- If ---
class IfNode : public RelayNode {
public:
    Expr cond;
    Expr true_branch;
    Expr false_branch;

    KXC_OBJECT_DECLARE
};

KXC_OBJECT_DEFINE(IfNode)

class If : public Relay {
public:
    using Relay::Relay;
    If(Expr cond, Expr true_branch, Expr false_branch) {
        IfNode* node = new IfNode();
        node->cond = cond;
        node->true_branch = true_branch;
        node->false_branch = false_branch;
        SetData(node);
    }
    const IfNode* operator->() const {
        return static_cast<const IfNode*>(object_);
    }
};

// --- Let ---
class LetNode : public RelayNode {
public:
    Var var;
    Expr value;
    Expr body;

    KXC_OBJECT_DECLARE
};

KXC_OBJECT_DEFINE(LetNode)

class Let : public Relay {
public:
    using Relay::Relay;
    Let(Var var, Expr value, Expr body) {
        LetNode* node = new LetNode();
        node->var = var;
        node->value = value;
        node->body = body;
        SetData(node);
    }
    const LetNode* operator->() const {
        return static_cast<const LetNode*>(object_);
    }
};

}
