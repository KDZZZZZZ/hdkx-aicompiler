/*! \file include/kxc/relay/relay.h
 * \brief 定义 Relay IR 节点、类型节点和用户可见句柄。
 */

#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "kxc/support/container.h"
#include "kxc/ir/expr.h"
#include "kxc/runtime/ndarray.h"
#include "kxc/target/virtual_device.h"

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


class TensorType : public Type {
public:
    using Type::Type;

    TensorType(Array<int64_t> shape, std::string dtype);
    const TensorTypeNode* operator->() const;
};

/*! \brief Tuple 类型节点，保存 Tuple 表达式每个字段的静态类型。 */
class TupleTypeNode : public TypeNode {
public:
    Array<Type> fields;

    KXC_OBJECT_DECLARE
};


/*! \brief Tuple 类型句柄，用于在类型推导中表达多值或结构化结果。 */
class TupleType : public Type {
public:
    using Type::Type;

    explicit TupleType(Array<Type> fields);
    const TupleTypeNode* operator->() const;
};

/*! \brief 将 TensorType 格式化为可读字符串，主要用于错误信息和 IR 调试输出。 */
std::string TensorTypeToString(const TensorTypeNode* type);
/*! \brief 统一格式化当前支持的 Type，未定义类型会输出为 unknown。 */
std::string TypeToString(const Type& type);
/*! \brief 递归比较两个类型的结构、shape 和 dtype 是否完全一致。 */
bool TypeEqual(const Type& lhs, const Type& rhs);

class IdNode : public Object {
public:
    std::string name_hint;
    KXC_OBJECT_DECLARE
};


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


class Let : public Relay {
public:
    using Relay::Relay;
    Let(Var var, Expr value, Expr body);
    const LetNode* operator->() const;
};

}  // namespace kxc
