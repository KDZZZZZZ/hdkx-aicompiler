/*! \file src/relay/transforms/infer_type.cc
 * \brief 实现 Relay 静态类型和 shape 推导。
 */

#include "kxc/relay/transforms/infer_type.h"

#include <any>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "kxc/relay/visitor.h"
#include "kxc/relay/op_attr_types.h"
#include "kxc/relay/op.h"

namespace kxc {
namespace relay {

namespace {

// 将 Constant 的 DLPack dtype 转为 Relay TensorType dtype 名称。
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
    if (dtype.code == kDLUInt) {
        if (dtype.bits == 8) return "uint8";
    }
    if (dtype.code == kDLBool && dtype.bits == 8) return "bool";
    throw std::runtime_error("Unsupported constant dtype in InferType");
}

// 递归推导 Relay 表达式类型，并把结果写回 checked_type 缓存。
class TypeInferencer : public RelayPassFunctor<Type> {
public:
    // 推导完整函数并返回函数表达式的结果类型。
    Type Infer(const Function& func) {
        if (!func.defined()) {
            throw std::runtime_error("InferTypePass expects a defined Function");
        }
        return Visit(Expr(ObjectRef(func)));
    }

protected:
    std::unordered_map<const Object*, Type> memo_;
    std::unordered_map<const Object*, Type> var_env_;

    // 提供按对象身份记忆化的统一访问入口。
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

    // 从 Storage-backed NDArray 的 shape 与 dtype 构造常量 TensorType。
    Type VisitConstant(const ConstantNode* op, const Expr& ref) override {
        if (!op->data.defined()) {
            throw std::runtime_error("InferType found undefined Constant data");
        }
        Array<int64_t> shape;
        for (int64_t dim : op->data->shape_storage) {
            shape.push_back(dim);
        }
        return TensorType(shape, DTypeToString(op->data->dl_tensor.dtype));
    }

    // 从局部环境、声明或已有 checked_type 解析变量类型。
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

    // 推导实参类型并调用算子注册的 FInferType 规则。
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

    // 建立参数类型环境并推导函数体类型。
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

    // 校验标量布尔条件和两个分支的一致类型。
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

    // 推导 let 绑定值、校验注解并扩展变量环境。
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

    // 按字段顺序构造 TupleType。
    Type VisitTuple(const TupleNode* op, const Expr& ref) override {
        (void)ref;
        Array<Type> fields;
        for (const auto& field : op->fields) {
            fields.push_back(Visit(field));
        }
        return TupleType(fields);
    }

    // 校验 tuple 和索引后返回目标字段类型。
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

    // 独立 Op 节点没有可推导的值类型。
    Type VisitOp(const OpNode* op, const Expr& ref) override {
        (void)op;
        (void)ref;
        return Type();
    }

    // 对尚未支持的 Relay 节点给出明确错误。
    Type VisitDefault(const Expr& expr) override {
        (void)expr;
        throw std::runtime_error("InferType encountered unsupported Relay node");
    }
};

}  // namespace

// 运行类型推导并返回已填充 checked_type 的原函数对象。
Function InferTypePass(const Function& func) {
    TypeInferencer inferencer;
    inferencer.Infer(func);
    return func;
}

}  // namespace relay
}  // namespace kxc
