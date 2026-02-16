#include "relay/relay.h"

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

