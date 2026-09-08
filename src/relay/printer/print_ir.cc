/*! \file src/relay/printer/print_ir.cc
 * \brief 实现 Relay IR 文本打印和调试工具。
 */

#include "kxc/relay/printer/print_ir.h"

#include <sstream>
#include <string>
#include <unordered_map>

#include "kxc/relay/op.h"

namespace kxc {
namespace relay {
namespace printer {

namespace {

// 生成指定宽度的缩进字符串。
std::string Indent(int n) { return std::string(n, ' '); }

// 在表达式已有 checked_type 时追加类型文本。
std::string CheckedTypeSuffix(const Expr& expr) {
    Type checked_type = expr.checked_type();
    return checked_type.defined() ? " : " + TypeToString(checked_type) : "";
}

// 将 DLPack dtype 转换为调试输出使用的稳定名称。
std::string DTypeToString(const DLDataType& dtype) {
    if (dtype.code == kDLFloat) {
        return "float" + std::to_string(dtype.bits);
    }
    if (dtype.code == kDLInt) {
        return "int" + std::to_string(dtype.bits);
    }
    if (dtype.code == kDLUInt) return "uint" + std::to_string(dtype.bits);
    if (dtype.code == kDLBool) return "bool";
    return "dtype(code=" + std::to_string(dtype.code) + ",bits=" +
           std::to_string(dtype.bits) + ")";
}

// 递归打印 Relay 表达式及其类型、参数和张量元数据。
//
// Relay 的表达式是 DAG 而不是树：Transformer 的每个残差都让同一个子表达式被
// 多个消费者引用。按树遍历会把共享子图重复展开，文本随共享点数量指数增长——
// 3 层 MiniMind（330 节点）就能产出 GB 级文本并耗尽内存。
//
// 因此内部节点（Call/Function/If/While/Let/Tuple/TupleGetItem）先统计引用数：
// 只被引用一次的节点输出与从前完全一致；真正被共享的节点首次出现时带 `#id`
// 标记，其后只打印 `<shared #id>` 并停止递归。叶子（Var/Constant）重复打印是
// O(1)，不参与记忆，输出保持原样。
class RelayIRPrinter {
public:
    // 绑定输出流和每层缩进宽度。
    RelayIRPrinter(std::ostream& os, int indent_spaces)
        : os_(os), indent_spaces_(indent_spaces) {}

    // 统计内部节点引用数；自身带 visited 集合，走一遍 DAG 而不是树。
    void CountReferences(const Expr& expr) {
        if (!expr.defined() || !IsInteriorNode(expr)) return;
        const Object* node = expr.get();
        if (++references_[node] > 1) return;  // 已展开过，子节点不再重复统计
        ForEachChild(expr, [this](const Expr& child) { CountReferences(child); });
    }

    // 按节点类别打印当前表达式并递归处理子节点。
    void Print(const Expr& expr, int indent) {
        if (!expr.defined()) {
            os_ << Indent(indent) << "<undef-expr>\n";
            return;
        }
        if (IsInteriorNode(expr)) {
            const Object* node = expr.get();
            const auto reference = references_.find(node);
            if (reference != references_.end() && reference->second > 1) {
                const auto emitted = shared_ids_.find(node);
                if (emitted != shared_ids_.end()) {
                    os_ << Indent(indent) << "<shared #" << emitted->second << ">\n";
                    return;
                }
                shared_ids_.emplace(node, next_shared_id_);
                pending_shared_id_ = next_shared_id_++;
            }
        }

        if (const auto* var = expr.As<VarNode>()) {
            os_ << Indent(indent) << "Var(" << var->vid->name_hint << ")"
                << CheckedTypeSuffix(expr) << "\n";
            return;
        }
        if (const auto* constant = expr.As<ConstantNode>()) {
            os_ << Indent(indent) << "Constant(shape=[";
            for (size_t i = 0; i < constant->data->shape_storage.size(); ++i) {
                if (i) os_ << ", ";
                os_ << constant->data->shape_storage[i];
            }
            os_ << "], dtype=" << DTypeToString(constant->data->dl_tensor.dtype) << ")"
                << CheckedTypeSuffix(expr) << "\n";
            return;
        }
        if (const auto* call = expr.As<CallNode>()) {
            std::string op_name = "<expr-op>";
            if (const auto* op_node = call->op.As<OpNode>()) {
                op_name = op_node->name;
            }
            os_ << Indent(indent) << "Call(op=" << op_name << ", args=" << call->args.size()
                << ")" << CheckedTypeSuffix(expr) << SharedSuffix() << "\n";
            for (size_t i = 0; i < call->args.size(); ++i) {
                os_ << Indent(indent + indent_spaces_) << "arg[" << i << "]:\n";
                Print(call->args[i], indent + indent_spaces_ * 2);
            }
            return;
        }
        if (const auto* func = expr.As<FunctionNode>()) {
            os_ << Indent(indent) << "Function(params=" << func->params.size() << ")"
                << CheckedTypeSuffix(expr) << SharedSuffix() << "\n";
            for (size_t i = 0; i < func->params.size(); ++i) {
                os_ << Indent(indent + indent_spaces_) << "param[" << i
                    << "]=" << func->params[i]->vid->name_hint << "\n";
            }
            os_ << Indent(indent + indent_spaces_) << "body:\n";
            Print(func->body, indent + indent_spaces_ * 2);
            return;
        }
        if (const auto* if_node = expr.As<IfNode>()) {
            os_ << Indent(indent) << "If" << CheckedTypeSuffix(expr) << SharedSuffix() << "\n";
            os_ << Indent(indent + indent_spaces_) << "cond:\n";
            Print(if_node->cond, indent + indent_spaces_ * 2);
            os_ << Indent(indent + indent_spaces_) << "true:\n";
            Print(if_node->true_branch, indent + indent_spaces_ * 2);
            os_ << Indent(indent + indent_spaces_) << "false:\n";
            Print(if_node->false_branch, indent + indent_spaces_ * 2);
            return;
        }
        if (const auto* while_node = expr.As<WhileNode>()) {
            os_ << Indent(indent) << "While(max_trip_count="
                << while_node->max_trip_count << ", var="
                << while_node->loop_var->vid->name_hint << ")"
                << CheckedTypeSuffix(expr) << SharedSuffix() << "\n";
            os_ << Indent(indent + indent_spaces_) << "initial_state:\n";
            Print(while_node->initial_state, indent + indent_spaces_ * 2);
            os_ << Indent(indent + indent_spaces_) << "condition:\n";
            Print(while_node->condition, indent + indent_spaces_ * 2);
            os_ << Indent(indent + indent_spaces_) << "body:\n";
            Print(while_node->body, indent + indent_spaces_ * 2);
            return;
        }
        if (const auto* let_node = expr.As<LetNode>()) {
            os_ << Indent(indent) << "Let(var=" << let_node->var->vid->name_hint << ")"
                << CheckedTypeSuffix(expr) << SharedSuffix() << "\n";
            os_ << Indent(indent + indent_spaces_) << "value:\n";
            Print(let_node->value, indent + indent_spaces_ * 2);
            os_ << Indent(indent + indent_spaces_) << "body:\n";
            Print(let_node->body, indent + indent_spaces_ * 2);
            return;
        }
        if (const auto* tuple = expr.As<TupleNode>()) {
            os_ << Indent(indent) << "Tuple(fields=" << tuple->fields.size() << ")"
                << CheckedTypeSuffix(expr) << SharedSuffix() << "\n";
            for (size_t i = 0; i < tuple->fields.size(); ++i) {
                os_ << Indent(indent + indent_spaces_) << "field[" << i << "]:\n";
                Print(tuple->fields[i], indent + indent_spaces_ * 2);
            }
            return;
        }
        if (const auto* tuple_get = expr.As<TupleGetItemNode>()) {
            os_ << Indent(indent) << "TupleGetItem(index=" << tuple_get->index << ")"
                << CheckedTypeSuffix(expr) << SharedSuffix() << "\n";
            os_ << Indent(indent + indent_spaces_) << "tuple:\n";
            Print(tuple_get->tuple, indent + indent_spaces_ * 2);
            return;
        }

        os_ << Indent(indent) << "<expr>\n";
    }

private:
    // 只有内部节点参与共享记忆；叶子重复打印代价是 O(1)。
    static bool IsInteriorNode(const Expr& expr) {
        return expr.As<CallNode>() || expr.As<FunctionNode>() || expr.As<IfNode>() ||
               expr.As<WhileNode>() || expr.As<LetNode>() || expr.As<TupleNode>() ||
               expr.As<TupleGetItemNode>();
    }

    template <typename Visitor>
    static void ForEachChild(const Expr& expr, const Visitor& visit) {
        if (const auto* call = expr.As<CallNode>()) {
            for (const Expr& argument : call->args) visit(argument);
        } else if (const auto* func = expr.As<FunctionNode>()) {
            visit(func->body);
        } else if (const auto* if_node = expr.As<IfNode>()) {
            visit(if_node->cond);
            visit(if_node->true_branch);
            visit(if_node->false_branch);
        } else if (const auto* while_node = expr.As<WhileNode>()) {
            visit(while_node->initial_state);
            visit(while_node->condition);
            visit(while_node->body);
        } else if (const auto* let_node = expr.As<LetNode>()) {
            visit(let_node->value);
            visit(let_node->body);
        } else if (const auto* tuple = expr.As<TupleNode>()) {
            for (const Expr& field : tuple->fields) visit(field);
        } else if (const auto* tuple_get = expr.As<TupleGetItemNode>()) {
            visit(tuple_get->tuple);
        }
    }

    // 取出当前节点的共享编号后缀；未被共享的节点返回空串，输出与从前一致。
    std::string SharedSuffix() {
        if (pending_shared_id_ < 0) return "";
        const std::string suffix = " #" + std::to_string(pending_shared_id_);
        pending_shared_id_ = -1;
        return suffix;
    }

    std::ostream& os_;
    int indent_spaces_{2};
    std::unordered_map<const Object*, int> references_;
    std::unordered_map<const Object*, int> shared_ids_;
    int next_shared_id_{0};
    int pending_shared_id_{-1};
};

}  // namespace

// 配置公开 IR 打印器的缩进宽度。
IRPrinter::IRPrinter(int indent_spaces) : indent_spaces_(indent_spaces) {}

// 打印任意 Relay 表达式。
void IRPrinter::Run(const Expr& expr, std::ostream& os) const {
    RelayIRPrinter printer(os, indent_spaces_);
    // 先统计引用，再打印：只有真正被共享的内部节点才带 `#id` 标记。
    printer.CountReferences(expr);
    printer.Print(expr, 0);
}

// 将 Function 适配为 Expr 后打印。
void IRPrinter::Run(const Function& func, std::ostream& os) const {
    Run(Expr(ObjectRef(func)), os);
}

// 直接把表达式转储到调用方流。
void DumpExpr(const Expr& expr, std::ostream& os, int indent_spaces) {
    IRPrinter printer(indent_spaces);
    printer.Run(expr, os);
}

// 直接把函数转储到调用方流。
void DumpFunction(const Function& func, std::ostream& os, int indent_spaces) {
    DumpExpr(Expr(ObjectRef(func)), os, indent_spaces);
}

// 返回表达式的完整文本表示。
std::string ToText(const Expr& expr, int indent_spaces) {
    std::ostringstream os;
    DumpExpr(expr, os, indent_spaces);
    return os.str();
}

// 返回函数的完整文本表示。
std::string ToText(const Function& func, int indent_spaces) {
    std::ostringstream os;
    DumpFunction(func, os, indent_spaces);
    return os.str();
}

}  // namespace printer
}  // namespace relay
}  // namespace kxc
