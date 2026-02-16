#include "tir/expr.h"

namespace kxc {
namespace tir {

IntImm::IntImm(int64_t value, DataType dtype) {
    auto* node = new IntImmNode();
    node->value = value;
    node->dtype = dtype;
    SetData(node);
}

FloatImm::FloatImm(double value, DataType dtype) {
    auto* node = new FloatImmNode();
    node->value = value;
    node->dtype = dtype;
    SetData(node);
}

PrimExpr::PrimExpr(int32_t value) : PrimExpr(IntImm(value, DataType::Int(32))) {}
PrimExpr::PrimExpr(int64_t value) : PrimExpr(IntImm(value, DataType::Int(64))) {}
PrimExpr::PrimExpr(float value) : PrimExpr(FloatImm(value, DataType::Float(32))) {}
PrimExpr::PrimExpr(double value) : PrimExpr(FloatImm(value, DataType::Float(64))) {}
PrimExpr::PrimExpr(bool value) : PrimExpr(IntImm(value, DataType::Bool())) {}

Var::Var(std::string name_hint, DataType dtype) {
    auto* node = new VarNode();
    node->name_hint = std::move(name_hint);
    node->dtype = dtype;
    SetData(node);
}

#define KXC_DEFINE_BINARY_OP_CTOR(OpName) \
    OpName::OpName(PrimExpr a, PrimExpr b) { \
        auto* node = new OpName##Node(); \
        node->a = std::move(a); \
        node->b = std::move(b); \
        node->dtype = node->a.dtype(); \
        SetData(node); \
    }

KXC_DEFINE_BINARY_OP_CTOR(Add)
KXC_DEFINE_BINARY_OP_CTOR(Sub)
KXC_DEFINE_BINARY_OP_CTOR(Mul)
KXC_DEFINE_BINARY_OP_CTOR(Div)
KXC_DEFINE_BINARY_OP_CTOR(Mod)
KXC_DEFINE_BINARY_OP_CTOR(Min)
KXC_DEFINE_BINARY_OP_CTOR(Max)

#undef KXC_DEFINE_BINARY_OP_CTOR

#define KXC_DEFINE_LOGIC_OP_CTOR(OpName) \
    OpName::OpName(PrimExpr a, PrimExpr b) { \
        auto* node = new OpName##Node(); \
        node->a = std::move(a); \
        node->b = std::move(b); \
        node->dtype = DataType::Bool(); \
        SetData(node); \
    }

KXC_DEFINE_LOGIC_OP_CTOR(EQ)
KXC_DEFINE_LOGIC_OP_CTOR(LT)
KXC_DEFINE_LOGIC_OP_CTOR(And)
KXC_DEFINE_LOGIC_OP_CTOR(Or)

#undef KXC_DEFINE_LOGIC_OP_CTOR

Not::Not(PrimExpr value) {
    auto* node = new NotNode();
    node->value = std::move(value);
    node->dtype = DataType::Bool();
    SetData(node);
}

Load::Load(Var buffer_var, PrimExpr index, PrimExpr predicate) {
    auto* node = new LoadNode();
    node->buffer_var = std::move(buffer_var);
    node->index = std::move(index);
    node->predicate = std::move(predicate);
    node->dtype = node->buffer_var->dtype;
    SetData(node);
}

Call::Call(DataType dtype, std::string name, Array<PrimExpr> args) {
    auto* node = new CallNode();
    node->dtype = dtype;
    node->name = std::move(name);
    node->args = std::move(args);
    SetData(node);
}

Select::Select(PrimExpr condition, PrimExpr true_value, PrimExpr false_value) {
    auto* node = new SelectNode();
    node->condition = std::move(condition);
    node->true_value = std::move(true_value);
    node->false_value = std::move(false_value);
    node->dtype = node->true_value.dtype();
    SetData(node);
}

}  // namespace tir
}  // namespace kxc

