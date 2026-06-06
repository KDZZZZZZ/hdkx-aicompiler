/*! \file include/relay/relay.h
 * \brief Defines Relay IR nodes, type nodes, and user-facing handles.
 */

#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "base/container.h"
#include "base/expr.h"
#include "base/ndarray.h"
#include "base/tensor.h"
#include "base/virtual_device.h"

namespace kxc {

class RelayNode : public ExprNode {
public:
    VirtualDevice virtual_device_;
};

class Relay : public Expr {
public:
    using Expr::Expr;

    VirtualDevice virtual_device() const {
        return static_cast<const RelayNode*>(object_)->virtual_device_;
    }

    void set_virtual_device(VirtualDevice virtual_device) const {
        const_cast<RelayNode*>(static_cast<const RelayNode*>(object_))->virtual_device_ =
            std::move(virtual_device);
    }
};

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

    TensorType(Array<int64_t> shape, std::string dtype);
    const TensorTypeNode* operator->() const;
};

class TupleTypeNode : public TypeNode {
public:
    Array<Type> fields;

    KXC_OBJECT_DECLARE
};

KXC_OBJECT_DEFINE(TupleTypeNode)

class TupleType : public Type {
public:
    using Type::Type;

    explicit TupleType(Array<Type> fields);
    const TupleTypeNode* operator->() const;
};

std::string TensorTypeToString(const TensorTypeNode* type);
std::string TypeToString(const Type& type);
bool TypeEqual(const Type& lhs, const Type& rhs);

class IdNode : public Object {
public:
    std::string name_hint;
    KXC_OBJECT_DECLARE
};

KXC_OBJECT_DEFINE(IdNode)

class Id : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    explicit Id(std::string name);
    const IdNode* operator->() const;
};

class VarNode : public RelayNode {
public:
    Id vid;
    Type type_annotation;

    KXC_OBJECT_DECLARE

    void VisitAttrs(AttrVisitor& visitor) override {
        (void)visitor;
    }
};

KXC_OBJECT_DEFINE(VarNode)

class Var : public Relay {
public:
    using Relay::Relay;
    explicit Var(std::string name);
    Var(std::string name, Type type_annotation);
    const VarNode* operator->() const;
};

class ConstantNode : public RelayNode {
public:
    runtime::NDArray data;

    KXC_OBJECT_DECLARE
};

KXC_OBJECT_DEFINE(ConstantNode)

class Constant : public Relay {
public:
    using Relay::Relay;
    explicit Constant(runtime::NDArray data);
    const ConstantNode* operator->() const;
};

class CallNode : public RelayNode {
public:
    Expr op;
    Array<Expr> args;
    ObjectRef attrs;

    KXC_OBJECT_DECLARE
};

KXC_OBJECT_DEFINE(CallNode)

class Call : public Relay {
public:
    using Relay::Relay;
    Call(Expr op, Array<Expr> args, ObjectRef attrs = ObjectRef());
    const CallNode* operator->() const;
};

class FunctionNode : public RelayNode {
public:
    Array<Var> params;
    Expr body;

    KXC_OBJECT_DECLARE
};

KXC_OBJECT_DEFINE(FunctionNode)

class Function : public Relay {
public:
    using Relay::Relay;
    Function(Array<Var> params, Expr body);
    const FunctionNode* operator->() const;
};

class TupleNode : public RelayNode {
public:
    Array<Expr> fields;

    KXC_OBJECT_DECLARE
};

KXC_OBJECT_DEFINE(TupleNode)

class Tuple : public Relay {
public:
    using Relay::Relay;
    explicit Tuple(Array<Expr> fields);
    const TupleNode* operator->() const;
};

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
    TupleGetItem(Expr tuple, int index);
    const TupleGetItemNode* operator->() const;
};

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
    If(Expr cond, Expr true_branch, Expr false_branch);
    const IfNode* operator->() const;
};

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
    Let(Var var, Expr value, Expr body);
    const LetNode* operator->() const;
};

}  // namespace kxc
