#pragma once
#include <stdint.h>
#include <string>
#include <vector>
#include "base/expr.h"

namespace kxc {
namespace tir {

// --- Type System ---
struct DataType {
    uint8_t code;
    uint8_t bits;
    uint16_t lanes;
    bool operator==(const DataType& other) const {
        return code == other.code && bits == other.bits && lanes == other.lanes;
    }
    bool operator!=(const DataType& other) const {
        return !(*this == other);
    }
    static DataType Int(int bits, int lanes = 1) { return {0, (uint8_t)bits, (uint16_t)lanes}; }
    static DataType UInt(int bits, int lanes = 1) { return {1, (uint8_t)bits, (uint16_t)lanes}; }
    static DataType Float(int bits, int lanes = 1) { return {2, (uint8_t)bits, (uint16_t)lanes}; }
    static DataType Handle(int bits = 64, int lanes = 1) { return {3, (uint8_t)bits, (uint16_t)lanes}; }
    static DataType Bool(int lanes = 1) { return UInt(1, lanes); }
    static DataType Void() { return {4, 0, 0}; }
};

// --- Base Expression ---
class PrimExprNode : public ExprNode {
public:
    DataType dtype;
    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(PrimExprNode)

class PrimExpr : public Expr {
public:
    using Expr::Expr;
    // Allow implicit construction from int, float, bool for convenience
    PrimExpr(int32_t value);
    PrimExpr(int64_t value);
    PrimExpr(float value);
    PrimExpr(double value);
    PrimExpr(bool value);

    const PrimExprNode* operator->() const { return static_cast<const PrimExprNode*>(object_); }
    DataType dtype() const { return operator->()->dtype; }
};

// --- 1. Constants ---
class IntImmNode : public PrimExprNode {
public:
    int64_t value;
    KXC_OBJECT_DECLARE
    void VisitAttrs(AttrVisitor& visitor) override {}
};
KXC_OBJECT_DEFINE(IntImmNode)

class IntImm : public PrimExpr {
public:
    using PrimExpr::PrimExpr;
    explicit IntImm(int64_t value, DataType dtype = DataType::Int(32)) {
        auto* node = new IntImmNode();
        node->value = value;
        node->dtype = dtype;
        SetData(node);
    }
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
    explicit FloatImm(double value, DataType dtype = DataType::Float(32)) {
        auto* node = new FloatImmNode();
        node->value = value;
        node->dtype = dtype;
        SetData(node);
    }
};

// Implement implicit constructors after IntImm/FloatImm definition
inline PrimExpr::PrimExpr(int32_t value) : PrimExpr(IntImm(value, DataType::Int(32))) {}
inline PrimExpr::PrimExpr(int64_t value) : PrimExpr(IntImm(value, DataType::Int(64))) {}
inline PrimExpr::PrimExpr(float value) : PrimExpr(FloatImm(value, DataType::Float(32))) {}
inline PrimExpr::PrimExpr(double value) : PrimExpr(FloatImm(value, DataType::Float(64))) {}
inline PrimExpr::PrimExpr(bool value) : PrimExpr(IntImm(value, DataType::Bool())) {}

// --- 2. Variables ---
class VarNode : public PrimExprNode {
public:
    std::string name_hint;
    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(VarNode)

class Var : public PrimExpr {
public:
    using PrimExpr::PrimExpr;
    explicit Var(std::string name_hint, DataType dtype = DataType::Int(32)) {
        auto* node = new VarNode();
        node->name_hint = std::move(name_hint);
        node->dtype = dtype;
        SetData(node);
    }
    const VarNode* operator->() const { return static_cast<const VarNode*>(object_); }
};

// --- 3. Arithmetic & Logic Operations ---
// Abstract Base for Binary Ops
class BinaryOpNode : public PrimExprNode {
public:
    PrimExpr a;
    PrimExpr b;
    void VisitAttrs(AttrVisitor& visitor) override {}
};

// Helper macro for defining binary ops
#define DEFINE_BINARY_OP(OpName) \
    class OpName##Node : public BinaryOpNode { \
    public: \
        KXC_OBJECT_DECLARE \
    }; \
    KXC_OBJECT_DEFINE(OpName##Node) \
    class OpName : public PrimExpr { \
    public: \
        using PrimExpr::PrimExpr; \
        OpName(PrimExpr a, PrimExpr b) { \
            auto* node = new OpName##Node(); \
            node->a = a; \
            node->b = b; \
            node->dtype = a.dtype(); /* Simplified type inference */ \
            SetData(node); \
        } \
    };

// Arithmetic
DEFINE_BINARY_OP(Add)
DEFINE_BINARY_OP(Sub)
DEFINE_BINARY_OP(Mul)
DEFINE_BINARY_OP(Div)
DEFINE_BINARY_OP(Mod)
DEFINE_BINARY_OP(Min)
DEFINE_BINARY_OP(Max)

// Logic (Return Bool)
// Need specialized constructor for Bool return type
#define DEFINE_LOGIC_OP(OpName) \
    class OpName##Node : public BinaryOpNode { \
    public: \
        KXC_OBJECT_DECLARE \
    }; \
    KXC_OBJECT_DEFINE(OpName##Node) \
    class OpName : public PrimExpr { \
    public: \
        using PrimExpr::PrimExpr; \
        OpName(PrimExpr a, PrimExpr b) { \
            auto* node = new OpName##Node(); \
            node->a = a; \
            node->b = b; \
            node->dtype = DataType::Bool(); \
            object_ = node; \
        } \
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
    explicit Not(PrimExpr value) {
        auto* node = new NotNode();
        node->value = value;
        node->dtype = DataType::Bool();
        SetData(node);
    }
};

// --- 4. Memory Operations ---
class LoadNode : public PrimExprNode {
public:
    Var buffer_var;
    PrimExpr index;
    PrimExpr predicate; // For predicated load (optional)

    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(LoadNode)

class Load : public PrimExpr {
public:
    using PrimExpr::PrimExpr;
    Load(Var buffer_var, PrimExpr index, PrimExpr predicate = PrimExpr()) {
        auto* node = new LoadNode();
        node->buffer_var = buffer_var;
        node->index = index;
        node->predicate = predicate;
        node->dtype = buffer_var->dtype; // Load type is buffer element type
        object_ = node;
    }
};

// --- 5. Special Operations ---
class CallNode : public PrimExprNode {
public:
    // Function name or Op
    std::string name;
    std::vector<PrimExpr> args;
    
    KXC_OBJECT_DECLARE
};
KXC_OBJECT_DEFINE(CallNode)

class Call : public PrimExpr {
public:
    using PrimExpr::PrimExpr;
    Call(DataType dtype, std::string name, std::vector<PrimExpr> args) {
        auto* node = new CallNode();
        node->dtype = dtype;
        node->name = std::move(name);
        node->args = std::move(args);
        object_ = node;
    }
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
    Select(PrimExpr condition, PrimExpr true_value, PrimExpr false_value) {
        auto* node = new SelectNode();
        node->condition = condition;
        node->true_value = true_value;
        node->false_value = false_value;
        node->dtype = true_value.dtype();
        SetData(node);
    }
};

// --- Operator Overloads ---
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

} // namespace tir
} // namespace kxc
