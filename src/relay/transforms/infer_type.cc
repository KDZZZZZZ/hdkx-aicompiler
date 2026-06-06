/*! \file src/relay/transforms/infer_type.cc
 * \brief Implements Relay static type and shape inference.
 */

#include "relay/transforms/infer_type.h"

#include <any>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "base/pass.h"
#include "relay/op_attr_types.h"
#include "relay/op.h"

namespace kxc {
namespace relay {

namespace {

std::string DTypeToString(const DLDataType& dtype) {
    if (dtype.lanes != 1) {
        throw std::runtime_error("InferType only supports scalar-lane DLDataType");
    }
    if (dtype.code == kDLFloat) {
        if (dtype.bits == 32) return "float32";
        if (dtype.bits == 64) return "float64";
    }
    if (dtype.code == kDLInt) {
        if (dtype.bits == 8) return "int8";
        if (dtype.bits == 32) return "int32";
        if (dtype.bits == 64) return "int64";
    }
    if (dtype.code == kDLUint) {
        if (dtype.bits == 1) return "bool";
        if (dtype.bits == 8) return "uint8";
    }
    throw std::runtime_error("Unsupported constant dtype in InferType");
}

class TypeInferencer : public RelayPassFunctor<Type> {
public:
    Type Infer(const Function& func) {
        if (!func.defined()) {
            throw std::runtime_error("InferTypePass expects a defined Function");
        }
        return Visit(Expr(ObjectRef(func)));
    }

protected:
    std::unordered_map<const Object*, Type> memo_;
    std::unordered_map<const Object*, Type> var_env_;

    Type Visit(const Expr& expr) override {
        if (!expr.defined()) {
            return Type();
        }
        auto it = memo_.find(expr.get());
        if (it != memo_.end()) {
            return it->second;
        }
        Type result = RelayPassFunctor<Type>::Visit(expr);
        if (result.defined()) {
            SetCheckedType(expr, result);
        }
        memo_[expr.get()] = result;
        return result;
    }

    Type VisitConstant(const ConstantNode* op, const Expr& ref) override {
        if (!op->data.defined()) {
            throw std::runtime_error("InferType found undefined Constant data");
        }
        Array<int64_t> shape;
        for (int64_t dim : op->data->shape) {
            shape.push_back(dim);
        }
        return TensorType(shape, DTypeToString(op->data->dl_tensor.dtype));
    }

    Type VisitVar(const VarNode* op, const Expr& ref) override {
        auto env_it = var_env_.find(ref.get());
        if (env_it != var_env_.end()) {
            return env_it->second;
        }
        if (op->type_annotation.defined()) {
            return op->type_annotation;
        }
        if (ref.checked_type().defined()) {
            return ref.checked_type();
        }
        throw std::runtime_error("InferType requires a type for variable: " +
                                 op->vid->name_hint);
    }

    Type VisitCall(const CallNode* op, const Expr& ref) override {
        (void)ref;
        const auto* op_node = op->op.As<OpNode>();
        if (!op_node) {
            throw std::runtime_error("InferType currently supports calls to registered Ops only");
        }

        Array<Type> input_types;
        for (const auto& arg : op->args) {
            Type arg_type = Visit(arg);
            if (!arg_type.defined()) {
                throw std::runtime_error("InferType failed to infer argument type for op: " +
                                         op_node->name);
            }
            input_types.push_back(arg_type);
        }

        auto attr_it = op_node->attrs.find("FInferType");
        if (attr_it == op_node->attrs.end()) {
            throw std::runtime_error("No FInferType registered for op: " + op_node->name);
        }
        auto* infer_rule = std::any_cast<FInferType>(&attr_it->second);
        if (!infer_rule) {
            throw std::runtime_error("Bad FInferType type for op: " + op_node->name);
        }
        Attrs attrs = op->attrs.defined() ? Attrs(op->attrs) : Attrs();
        Type out_type = (*infer_rule)(attrs, input_types);
        if (!out_type.defined()) {
            throw std::runtime_error("FInferType returned an undefined type for op: " +
                                     op_node->name);
        }
        return out_type;
    }

    Type VisitFunction(const FunctionNode* op, const Expr& ref) override {
        for (const auto& param : op->params) {
            if (!param->type_annotation.defined()) {
                throw std::runtime_error("InferType requires TensorType annotation on parameter: " +
                                         param->vid->name_hint);
            }
            if (!param->type_annotation.As<TensorTypeNode>()) {
                throw std::runtime_error("InferType currently supports TensorType parameters only: " +
                                         param->vid->name_hint);
            }
            var_env_[param.get()] = param->type_annotation;
            SetCheckedType(Expr(ObjectRef(param)), param->type_annotation);
        }

        Type body_type = Visit(op->body);
        SetCheckedType(ref, body_type);
        return body_type;
    }

    Type VisitIf(const IfNode* op, const Expr& ref) override {
        (void)ref;
        Type cond_type = Visit(op->cond);
        const auto* cond_tensor = cond_type.As<TensorTypeNode>();
        if (!cond_tensor || cond_tensor->dtype != "bool" || !cond_tensor->shape.empty()) {
            throw std::runtime_error("If condition must be a scalar bool TensorType");
        }
        Type true_type = Visit(op->true_branch);
        Type false_type = Visit(op->false_branch);
        if (!TypeEqual(true_type, false_type)) {
            throw std::runtime_error("If branch type mismatch: " + TypeToString(true_type) +
                                     " vs " + TypeToString(false_type));
        }
        return true_type;
    }

    Type VisitLet(const LetNode* op, const Expr& ref) override {
        (void)ref;
        Type value_type = Visit(op->value);
        if (op->var->type_annotation.defined() &&
            !TypeEqual(op->var->type_annotation, value_type)) {
            throw std::runtime_error("Let variable annotation mismatch for " +
                                     op->var->vid->name_hint + ": " +
                                     TypeToString(op->var->type_annotation) + " vs " +
                                     TypeToString(value_type));
        }
        var_env_[op->var.get()] = value_type;
        SetCheckedType(Expr(ObjectRef(op->var)), value_type);
        return Visit(op->body);
    }

    Type VisitTuple(const TupleNode* op, const Expr& ref) override {
        (void)ref;
        Array<Type> fields;
        for (const auto& field : op->fields) {
            fields.push_back(Visit(field));
        }
        return TupleType(fields);
    }

    Type VisitTupleGetItem(const TupleGetItemNode* op, const Expr& ref) override {
        (void)ref;
        Type tuple_type = Visit(op->tuple);
        const auto* tuple = tuple_type.As<TupleTypeNode>();
        if (!tuple) {
            throw std::runtime_error("TupleGetItem expects TupleType, got " +
                                     TypeToString(tuple_type));
        }
        if (op->index < 0 || static_cast<size_t>(op->index) >= tuple->fields.size()) {
            throw std::runtime_error("TupleGetItem index out of range: " +
                                     std::to_string(op->index));
        }
        return tuple->fields[static_cast<size_t>(op->index)];
    }

    Type VisitOp(const OpNode* op, const Expr& ref) override {
        (void)op;
        (void)ref;
        return Type();
    }

    Type VisitDefault(const Expr& expr) override {
        (void)expr;
        throw std::runtime_error("InferType encountered unsupported Relay node");
    }
};

}  // namespace

Function InferTypePass(const Function& func) {
    TypeInferencer inferencer;
    inferencer.Infer(func);
    return func;
}

}  // namespace relay
}  // namespace kxc
