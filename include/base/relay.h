#pragma once
#include "expr.h"
#include "tensor.h"
#include <vector>
#include <string>

namespace kxc {

class RelayNode : public ExprNode {
public:
    // Relay specific metadata can go here
};

class Relay : public Expr {
public:
    using Expr::Expr;
};

// --- Var ---
class IdNode : public Object {
public:
    std::string name_hint;
    const TypeIndex GetTypeId() const override { return kKXC_OBJECT_TYPE + 4; }
};

class Id : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    explicit Id(std::string name) {
        auto* node = new IdNode();
        node->name_hint = std::move(name);
        object_ = node;
        if (object_) object_->IncRef();
    }
    const IdNode* operator->() const { return static_cast<const IdNode*>(object_); }
};

class VarNode : public RelayNode {
public:
    Id vid;
    // Type type_annotation;

    const TypeIndex GetTypeId() const override {
        return kKXC_OBJECT_TYPE + 5;
    }
    void VisitAttrs(AttrVisitor& visitor) override {
        // visitor("vid", &vid);
    }
};

class Var : public Relay {
public:
    using Relay::Relay;
    explicit Var(std::string name) { 
        VarNode* node = new VarNode();
        node->vid = Id(std::move(name));
        object_ = node;
        if (object_) object_->IncRef();
    }
    const VarNode* operator->() const {
        return static_cast<const VarNode*>(object_);
    }
};

// --- Constant ---
class ConstantNode : public RelayNode {
public:
    Tensor data;

    const TypeIndex GetTypeId() const override {
        return kKXC_OBJECT_TYPE + 6;
    }
};

class Constant : public Relay {
public:
    using Relay::Relay;
    explicit Constant(Tensor data) {
        ConstantNode* node = new ConstantNode();
        node->data = data;
        object_ = node;
        if (object_) object_->IncRef();
    }
    const ConstantNode* operator->() const {
        return static_cast<const ConstantNode*>(object_);
    }
};

// --- Call ---
class CallNode : public RelayNode {
public:
    Expr op;
    std::vector<Expr> args;
    ObjectRef attrs; // Changed from Attrs to ObjectRef to allow any Attrs type

    const TypeIndex GetTypeId() const override {
        return kKXC_OBJECT_TYPE + 8;
    }
};

class Call : public Relay {
public:
    using Relay::Relay;
    Call(Expr op, std::vector<Expr> args, ObjectRef attrs = ObjectRef()) {
        CallNode* node = new CallNode();
        node->op = op;
        node->args = std::move(args);
        node->attrs = attrs;
        object_ = node;
        if (object_) object_->IncRef();
    }
    const CallNode* operator->() const {
        return static_cast<const CallNode*>(object_);
    }
};

// --- Function ---
class FunctionNode : public RelayNode {
public:
    std::vector<Var> params;
    Expr body;
    // Type ret_type;

    const TypeIndex GetTypeId() const override {
        return kKXC_OBJECT_TYPE + 9;
    }
};

class Function : public Relay {
public:
    using Relay::Relay;
    Function(std::vector<Var> params, Expr body) {
        FunctionNode* node = new FunctionNode();
        node->params = std::move(params);
        node->body = body;
        object_ = node;
        if (object_) object_->IncRef();
    }
    const FunctionNode* operator->() const {
        return static_cast<const FunctionNode*>(object_);
    }
};

// --- Tuple ---
class TupleNode : public RelayNode {
public:
    std::vector<Expr> fields;

    const TypeIndex GetTypeId() const override {
        return kKXC_OBJECT_TYPE + 14;
    }
};

class Tuple : public Relay {
public:
    using Relay::Relay;
    explicit Tuple(std::vector<Expr> fields) {
        TupleNode* node = new TupleNode();
        node->fields = std::move(fields);
        object_ = node;
        if (object_) object_->IncRef();
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

    const TypeIndex GetTypeId() const override {
        return kKXC_OBJECT_TYPE + 15;
    }
};

class TupleGetItem : public Relay {
public:
    using Relay::Relay;
    TupleGetItem(Expr tuple, int index) {
        TupleGetItemNode* node = new TupleGetItemNode();
        node->tuple = tuple;
        node->index = index;
        object_ = node;
        if (object_) object_->IncRef();
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

    const TypeIndex GetTypeId() const override {
        return kKXC_OBJECT_TYPE + 16;
    }
};

class If : public Relay {
public:
    using Relay::Relay;
    If(Expr cond, Expr true_branch, Expr false_branch) {
        IfNode* node = new IfNode();
        node->cond = cond;
        node->true_branch = true_branch;
        node->false_branch = false_branch;
        object_ = node;
        if (object_) object_->IncRef();
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

    const TypeIndex GetTypeId() const override {
        return kKXC_OBJECT_TYPE + 17;
    }
};

class Let : public Relay {
public:
    using Relay::Relay;
    Let(Var var, Expr value, Expr body) {
        LetNode* node = new LetNode();
        node->var = var;
        node->value = value;
        node->body = body;
        object_ = node;
        if (object_) object_->IncRef();
    }
    const LetNode* operator->() const {
        return static_cast<const LetNode*>(object_);
    }
};

}
