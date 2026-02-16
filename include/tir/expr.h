#pragma once

#include <stdint.h>

#include <functional>
#include <string>
#include <vector>

#include "base/container.h"
#include "base/expr.h"

namespace kxc {
namespace tir {

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

class PrimExprNode : public ExprNode {
public:
    DataType dtype;
    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(PrimExprNode)

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

class IntImmNode : public PrimExprNode {
public:
    int64_t value;
    KXC_OBJECT_DECLARE
    void VisitAttrs(AttrVisitor& visitor) override { (void)visitor; }
};
KXC_OBJECT_DEFINE(IntImmNode)

class IntImm : public PrimExpr {
public:
    using PrimExpr::PrimExpr;
    explicit IntImm(int64_t value, DataType dtype = DataType::Int(32));
    const IntImmNode* operator->() const { return static_cast<const IntImmNode*>(object_); }
};

class FloatImmNode : public PrimExprNode {
public:
    double value;
    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(FloatImmNode)

class FloatImm : public PrimExpr {
public:
    using PrimExpr::PrimExpr;
    explicit FloatImm(double value, DataType dtype = DataType::Float(32));
};

class VarNode : public PrimExprNode {
public:
    std::string name_hint;
    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(VarNode)

class Var : public PrimExpr {
public:
    using PrimExpr::PrimExpr;
    explicit Var(std::string name_hint, DataType dtype = DataType::Int(32));
    const VarNode* operator->() const { return static_cast<const VarNode*>(object_); }
    bool operator==(const Var& other) const { return object_ == other.object_; }
    bool operator!=(const Var& other) const { return !(*this == other); }
};

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

class NotNode : public PrimExprNode {
public:
    PrimExpr value;
    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(NotNode)

class Not : public PrimExpr {
public:
    using PrimExpr::PrimExpr;
    explicit Not(PrimExpr value);
};

class LoadNode : public PrimExprNode {
public:
    Var buffer_var;
    PrimExpr index;
    PrimExpr predicate;

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(LoadNode)

class Load : public PrimExpr {
public:
    using PrimExpr::PrimExpr;
    Load(Var buffer_var, PrimExpr index, PrimExpr predicate = PrimExpr());
};

class CallNode : public PrimExprNode {
public:
    std::string name;
    Array<PrimExpr> args;

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(CallNode)

class Call : public PrimExpr {
public:
    using PrimExpr::PrimExpr;
    Call(DataType dtype, std::string name, Array<PrimExpr> args);
};

class SelectNode : public PrimExprNode {
public:
    PrimExpr condition;
    PrimExpr true_value;
    PrimExpr false_value;

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(SelectNode)

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