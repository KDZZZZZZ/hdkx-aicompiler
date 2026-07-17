/*! \file include/tir/expr.h
 * \brief 定义 TIR PrimExpr、Stmt、PrimFunc 和 pass 工具。
 */

#pragma once

#include <stdint.h>

#include <functional>
#include <string>
#include <vector>

#include "base/container.h"
#include "base/expr.h"

namespace kxc {
namespace tir {

/*! \brief TIR 标量/向量数据类型描述，包含类型 code、位宽和 lanes。 */
struct DataType {
    uint8_t code;
    uint8_t bits;
    uint16_t lanes;
    bool operator==(const DataType& other) const {
        return code == other.code && bits == other.bits && lanes == other.lanes;
    }
    bool operator!=(const DataType& other) const { return !(*this == other); }
    static DataType Int(int bits, int lanes = 1) { return {0, static_cast<uint8_t>(bits), static_cast<uint16_t>(lanes)}; }
    static DataType UInt(int bits, int lanes = 1) { return {1, static_cast<uint8_t>(bits), static_cast<uint16_t>(lanes)}; }
    static DataType Float(int bits, int lanes = 1) { return {2, static_cast<uint8_t>(bits), static_cast<uint16_t>(lanes)}; }
    static DataType Handle(int bits = 64, int lanes = 1) { return {3, static_cast<uint8_t>(bits), static_cast<uint16_t>(lanes)}; }
    static DataType Bool(int lanes = 1) { return UInt(1, lanes); }
    static DataType Void() { return {4, 0, 0}; }
};

/*! \brief 所有 TIR primitive expression 节点的基类。 */
class PrimExprNode : public ExprNode {
public:
    DataType dtype;
    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(PrimExprNode)

/*! \brief TIR primitive expression 的引用类型。 */
class PrimExpr : public Expr {
public:
    using Expr::Expr;
    PrimExpr(int32_t value);
    PrimExpr(int64_t value);
    PrimExpr(float value);
    PrimExpr(double value);
    PrimExpr(bool value);

    const PrimExprNode* operator->() const { return static_cast<const PrimExprNode*>(object_); }
    DataType dtype() const { return operator->()->dtype; }
};

/*! \brief 整数字面量表达式节点。 */
class IntImmNode : public PrimExprNode {
public:
    int64_t value;
    KXC_OBJECT_DECLARE
    void VisitAttrs(AttrVisitor& visitor) override { (void)visitor; }
};
KXC_OBJECT_DEFINE(IntImmNode)

/*! \brief 整数字面量表达式引用类型。 */
class IntImm : public PrimExpr {
public:
    using PrimExpr::PrimExpr;
    explicit IntImm(int64_t value, DataType dtype = DataType::Int(32));
    const IntImmNode* operator->() const { return static_cast<const IntImmNode*>(object_); }
};

/*! \brief 浮点字面量表达式节点。 */
class FloatImmNode : public PrimExprNode {
public:
    double value;
    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(FloatImmNode)

/*! \brief 浮点字面量表达式引用类型。 */
class FloatImm : public PrimExpr {
public:
    using PrimExpr::PrimExpr;
    explicit FloatImm(double value, DataType dtype = DataType::Float(32));
};

/*! \brief TIR 符号变量节点。 */
class VarNode : public PrimExprNode {
public:
    std::string name_hint;
    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE_WITH_KEY(VarNode, "kxc.tir.VarNode")

/*! \brief TIR 符号变量引用类型。 */
class Var : public PrimExpr {
public:
    Var() = default;
    Var(const ObjectRef& n) : PrimExpr(n) {}
    explicit Var(std::string name_hint, DataType dtype = DataType::Int(32));
    const VarNode* operator->() const { return static_cast<const VarNode*>(object_); }
    bool operator==(const Var& other) const { return object_ == other.object_; }
    bool operator!=(const Var& other) const { return !(*this == other); }
};

/*! \brief 二元表达式节点基类，保存左右操作数。 */
class BinaryOpNode : public PrimExprNode {
public:
    PrimExpr a;
    PrimExpr b;
    void VisitAttrs(AttrVisitor& visitor) override { (void)visitor; }
};

#define DEFINE_BINARY_OP(OpName)                                \
    class OpName##Node : public BinaryOpNode {                  \
    public:                                                     \
        KXC_OBJECT_DECLARE                                      \
    };                                                          \
    KXC_OBJECT_DEFINE(OpName##Node)                             \
    class OpName : public PrimExpr {                            \
    public:                                                     \
        using PrimExpr::PrimExpr;                               \
        OpName(PrimExpr a, PrimExpr b);                         \
    };

DEFINE_BINARY_OP(Add)
DEFINE_BINARY_OP(Sub)
DEFINE_BINARY_OP(Mul)
DEFINE_BINARY_OP(Div)
DEFINE_BINARY_OP(Mod)
DEFINE_BINARY_OP(Min)
DEFINE_BINARY_OP(Max)

#define DEFINE_LOGIC_OP(OpName)                                 \
    class OpName##Node : public BinaryOpNode {                  \
    public:                                                     \
        KXC_OBJECT_DECLARE                                      \
    };                                                          \
    KXC_OBJECT_DEFINE(OpName##Node)                             \
    class OpName : public PrimExpr {                            \
    public:                                                     \
        using PrimExpr::PrimExpr;                               \
        OpName(PrimExpr a, PrimExpr b);                         \
    };

DEFINE_LOGIC_OP(EQ)
DEFINE_LOGIC_OP(LT)
DEFINE_LOGIC_OP(And)
DEFINE_LOGIC_OP(Or)

/*! \brief 逻辑非表达式节点。 */
class NotNode : public PrimExprNode {
public:
    PrimExpr value;
    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(NotNode)

/*! \brief 逻辑非表达式引用类型。 */
class Not : public PrimExpr {
public:
    using PrimExpr::PrimExpr;
    explicit Not(PrimExpr value);
};

/*! \brief 从 buffer 指针按下标读取的表达式节点。 */
class LoadNode : public PrimExprNode {
public:
    Var buffer_var;
    PrimExpr index;
    PrimExpr predicate;

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(LoadNode)

/*! \brief Load 表达式引用类型。 */
class Load : public PrimExpr {
public:
    using PrimExpr::PrimExpr;
    Load(Var buffer_var, PrimExpr index, PrimExpr predicate = PrimExpr());
};

/*! \brief 对外部函数、intrinsic 或 runtime helper 的调用表达式节点。 */
class CallNode : public PrimExprNode {
public:
    std::string name;
    Array<PrimExpr> args;

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE_WITH_KEY(CallNode, "kxc.tir.CallNode")

/*! \brief Call 表达式引用类型。 */
class Call : public PrimExpr {
public:
    using PrimExpr::PrimExpr;
    Call(DataType dtype, std::string name, Array<PrimExpr> args);
};

/*! \brief 条件选择表达式节点，语义等价于 condition ? true_value : false_value。 */
class SelectNode : public PrimExprNode {
public:
    PrimExpr condition;
    PrimExpr true_value;
    PrimExpr false_value;

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(SelectNode)

/*! \brief Select 表达式引用类型。 */
class Select : public PrimExpr {
public:
    using PrimExpr::PrimExpr;
    Select(PrimExpr condition, PrimExpr true_value, PrimExpr false_value);
};

inline PrimExpr operator+(PrimExpr a, PrimExpr b) { return Add(a, b); }
inline PrimExpr operator-(PrimExpr a, PrimExpr b) { return Sub(a, b); }
inline PrimExpr operator*(PrimExpr a, PrimExpr b) { return Mul(a, b); }
inline PrimExpr operator/(PrimExpr a, PrimExpr b) { return Div(a, b); }
inline PrimExpr operator%(PrimExpr a, PrimExpr b) { return Mod(a, b); }
inline PrimExpr operator==(PrimExpr a, PrimExpr b) { return EQ(a, b); }
inline PrimExpr operator<(PrimExpr a, PrimExpr b) { return LT(a, b); }
inline PrimExpr operator&&(PrimExpr a, PrimExpr b) { return And(a, b); }
inline PrimExpr operator||(PrimExpr a, PrimExpr b) { return Or(a, b); }
inline PrimExpr operator!(PrimExpr a) { return Not(a); }

}  // namespace tir
}  // namespace kxc

namespace std {
template <>
struct hash<kxc::tir::Var> {
    size_t operator()(const kxc::tir::Var& k) const {
        return std::hash<const kxc::Object*>()(k.get());
    }
};
}
