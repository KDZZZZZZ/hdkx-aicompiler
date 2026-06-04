/*! \file include/relay/relay.h
 * \brief 定义 Relay IR 节点、算子注册、attrs 和 Relay 到 TE lowering 属性。
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

/*!
 * \brief Relay 表达式节点基类，额外携带设备规划信息。
 */
class RelayNode : public ExprNode {
public:
    // Device planning metadata, similar to TVM's virtual_device_ on relay::ExprNode.
    VirtualDevice virtual_device_;
};

/*!
 * \brief Relay 表达式句柄基类。
 *
 * 具体表达式如 Var、Call、Function 都继承该句柄，提供统一的 virtual device
 * 读写接口。
 */
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

/*! \brief Tensor 类型节点，描述 shape 和 dtype。 */
class TensorTypeNode : public TypeNode {
public:
    Array<int64_t> shape;
    std::string dtype;

    KXC_OBJECT_DECLARE
};

KXC_OBJECT_DEFINE(TensorTypeNode)

/*! \brief Tensor 类型句柄。 */
class TensorType : public Type {
public:
    using Type::Type;

    TensorType(Array<int64_t> shape, std::string dtype);
    const TensorTypeNode* operator->() const;
};

/*! \brief Relay 变量名节点。 */
class IdNode : public Object {
public:
    std::string name_hint;
    KXC_OBJECT_DECLARE
};

KXC_OBJECT_DEFINE(IdNode)

/*! \brief Relay 变量名句柄。 */
class Id : public ObjectRef {
public:
    using ObjectRef::ObjectRef;
    explicit Id(std::string name);
    const IdNode* operator->() const;
};

/*! \brief Relay 变量节点，表示函数参数或 let-bound 局部变量。 */
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

/*! \brief Relay 变量句柄。 */
class Var : public Relay {
public:
    using Relay::Relay;
    explicit Var(std::string name);
    Var(std::string name, Type type_annotation);
    const VarNode* operator->() const;
};

/*! \brief Relay 常量节点，持有 runtime::NDArray 数据。 */
class ConstantNode : public RelayNode {
public:
    runtime::NDArray data;

    KXC_OBJECT_DECLARE
};

KXC_OBJECT_DEFINE(ConstantNode)

/*! \brief Relay 常量句柄。 */
class Constant : public Relay {
public:
    using Relay::Relay;
    explicit Constant(runtime::NDArray data);
    const ConstantNode* operator->() const;
};

/*! \brief Relay 调用节点，op 可以是算子或其他可调用表达式。 */
class CallNode : public RelayNode {
public:
    Expr op;
    Array<Expr> args;
    ObjectRef attrs;

    KXC_OBJECT_DECLARE
};

KXC_OBJECT_DEFINE(CallNode)

/*! \brief Relay 调用句柄。 */
class Call : public Relay {
public:
    using Relay::Relay;
    Call(Expr op, Array<Expr> args, ObjectRef attrs = ObjectRef());
    const CallNode* operator->() const;
};

/*! \brief Relay 函数节点，包含参数列表和函数体表达式。 */
class FunctionNode : public RelayNode {
public:
    Array<Var> params;
    Expr body;

    KXC_OBJECT_DECLARE
};

KXC_OBJECT_DEFINE(FunctionNode)

/*! \brief Relay 函数句柄，是 Compiler::Compile 的主要输入。 */
class Function : public Relay {
public:
    using Relay::Relay;
    Function(Array<Var> params, Expr body);
    const FunctionNode* operator->() const;
};

/*! \brief Relay 元组节点。 */
class TupleNode : public RelayNode {
public:
    Array<Expr> fields;

    KXC_OBJECT_DECLARE
};

KXC_OBJECT_DEFINE(TupleNode)

/*! \brief Relay 元组句柄。 */
class Tuple : public Relay {
public:
    using Relay::Relay;
    explicit Tuple(Array<Expr> fields);
    const TupleNode* operator->() const;
};

/*! \brief Relay 元组取项节点。 */
class TupleGetItemNode : public RelayNode {
public:
    Expr tuple;
    int index;

    KXC_OBJECT_DECLARE
};

KXC_OBJECT_DEFINE(TupleGetItemNode)

/*! \brief Relay 元组取项句柄。 */
class TupleGetItem : public Relay {
public:
    using Relay::Relay;
    TupleGetItem(Expr tuple, int index);
    const TupleGetItemNode* operator->() const;
};

/*! \brief Relay 条件表达式节点。 */
class IfNode : public RelayNode {
public:
    Expr cond;
    Expr true_branch;
    Expr false_branch;

    KXC_OBJECT_DECLARE
};

KXC_OBJECT_DEFINE(IfNode)

/*! \brief Relay 条件表达式句柄。 */
class If : public Relay {
public:
    using Relay::Relay;
    If(Expr cond, Expr true_branch, Expr false_branch);
    const IfNode* operator->() const;
};

/*! \brief Relay let binding 节点。 */
class LetNode : public RelayNode {
public:
    Var var;
    Expr value;
    Expr body;

    KXC_OBJECT_DECLARE
};

KXC_OBJECT_DEFINE(LetNode)

/*! \brief Relay let binding 句柄。 */
class Let : public Relay {
public:
    using Relay::Relay;
    Let(Var var, Expr value, Expr body);
    const LetNode* operator->() const;
};

}  // namespace kxc
