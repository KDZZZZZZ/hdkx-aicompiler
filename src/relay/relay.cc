/*! \file src/relay/relay.cc
 * \brief 实现 Relay IR 句柄和类型辅助函数。
 */

#include "relay/relay.h"

#include <sstream>
#include <utility>

namespace kxc {

TensorType::TensorType(Array<int64_t> shape, std::string dtype) {
    auto* node = new TensorTypeNode();
    node->shape = std::move(shape);
    node->dtype = std::move(dtype);
    SetData(node);
}

const TensorTypeNode* TensorType::operator->() const {
    return static_cast<const TensorTypeNode*>(object_);
}

TupleType::TupleType(Array<Type> fields) {
    auto* node = new TupleTypeNode();
    node->fields = std::move(fields);
    SetData(node);
}

const TupleTypeNode* TupleType::operator->() const {
    return static_cast<const TupleTypeNode*>(object_);
}

std::string TensorTypeToString(const TensorTypeNode* type) {
    if (!type) {
        return "<non-tensor>";
    }
    std::ostringstream os;
    os << "Tensor[";
    for (size_t i = 0; i < type->shape.size(); ++i) {
        if (i) os << ", ";
        os << type->shape[i];
    }
    os << "; " << type->dtype << "]";
    return os.str();
}

std::string TypeToString(const Type& type) {
    if (!type.defined()) {
        return "<unknown>";
    }
    if (const auto* tensor = type.As<TensorTypeNode>()) {
        return TensorTypeToString(tensor);
    }
    if (const auto* tuple = type.As<TupleTypeNode>()) {
        std::ostringstream os;
        os << "Tuple(";
        for (size_t i = 0; i < tuple->fields.size(); ++i) {
            if (i) os << ", ";
            os << TypeToString(tuple->fields[i]);
        }
        os << ")";
        return os.str();
    }
    return "<type>";
}

bool TypeEqual(const Type& lhs, const Type& rhs) {
    if (!lhs.defined() || !rhs.defined()) {
        return !lhs.defined() && !rhs.defined();
    }
    if (lhs.get() == rhs.get()) {
        return true;
    }
    if (const auto* left_tensor = lhs.As<TensorTypeNode>()) {
        const auto* right_tensor = rhs.As<TensorTypeNode>();
        if (!right_tensor || left_tensor->dtype != right_tensor->dtype ||
            left_tensor->shape.size() != right_tensor->shape.size()) {
            return false;
        }
        for (size_t i = 0; i < left_tensor->shape.size(); ++i) {
            if (left_tensor->shape[i] != right_tensor->shape[i]) {
                return false;
            }
        }
        return true;
    }
    if (const auto* left_tuple = lhs.As<TupleTypeNode>()) {
        const auto* right_tuple = rhs.As<TupleTypeNode>();
        if (!right_tuple || left_tuple->fields.size() != right_tuple->fields.size()) {
            return false;
        }
        for (size_t i = 0; i < left_tuple->fields.size(); ++i) {
            if (!TypeEqual(left_tuple->fields[i], right_tuple->fields[i])) {
                return false;
            }
        }
        return true;
    }
    return false;
}

Id::Id(std::string name) {
    auto* node = new IdNode();
    node->name_hint = std::move(name);
    SetData(node);
}

const IdNode* Id::operator->() const { return static_cast<const IdNode*>(object_); }

Var::Var(std::string name) {
    auto* node = new VarNode();
    node->vid = Id(std::move(name));
    SetData(node);
}

Var::Var(std::string name, Type type_annotation) {
    auto* node = new VarNode();
    node->vid = Id(std::move(name));
    node->type_annotation = std::move(type_annotation);
    SetData(node);
}

const VarNode* Var::operator->() const { return static_cast<const VarNode*>(object_); }

Constant::Constant(runtime::NDArray data) {
    auto* node = new ConstantNode();
    node->data = std::move(data);
    SetData(node);
}

const ConstantNode* Constant::operator->() const {
    return static_cast<const ConstantNode*>(object_);
}

Call::Call(Expr op, Array<Expr> args, ObjectRef attrs) {
    auto* node = new CallNode();
    node->op = std::move(op);
    node->args = std::move(args);
    node->attrs = std::move(attrs);
    SetData(node);
}

const CallNode* Call::operator->() const { return static_cast<const CallNode*>(object_); }

Function::Function(Array<Var> params, Expr body) {
    auto* node = new FunctionNode();
    node->params = std::move(params);
    node->body = std::move(body);
    SetData(node);
}

const FunctionNode* Function::operator->() const {
    return static_cast<const FunctionNode*>(object_);
}

Tuple::Tuple(Array<Expr> fields) {
    auto* node = new TupleNode();
    node->fields = std::move(fields);
    SetData(node);
}

const TupleNode* Tuple::operator->() const { return static_cast<const TupleNode*>(object_); }

TupleGetItem::TupleGetItem(Expr tuple, int index) {
    auto* node = new TupleGetItemNode();
    node->tuple = std::move(tuple);
    node->index = index;
    SetData(node);
}

const TupleGetItemNode* TupleGetItem::operator->() const {
    return static_cast<const TupleGetItemNode*>(object_);
}

If::If(Expr cond, Expr true_branch, Expr false_branch) {
    auto* node = new IfNode();
    node->cond = std::move(cond);
    node->true_branch = std::move(true_branch);
    node->false_branch = std::move(false_branch);
    SetData(node);
}

const IfNode* If::operator->() const { return static_cast<const IfNode*>(object_); }

Let::Let(Var var, Expr value, Expr body) {
    auto* node = new LetNode();
    node->var = std::move(var);
    node->value = std::move(value);
    node->body = std::move(body);
    SetData(node);
}

const LetNode* Let::operator->() const { return static_cast<const LetNode*>(object_); }

}  // namespace kxc
