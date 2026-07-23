/*! \file src/relay/pass/print_ir.cc
 * \brief 实现 Relay IR 文本打印和调试工具。
 */

#include "kxc/relay/pass/print_ir.h"

#include <sstream>

#include "kxc/relay/op.h"

namespace kxc {
namespace relay {
namespace pass {

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

// 递归打印 Relay 表达式树及其类型、参数和张量元数据。
class RelayIRPrinter {
public:
    // 绑定输出流和每层缩进宽度。
    RelayIRPrinter(std::ostream& os, int indent_spaces)
        : os_(os), indent_spaces_(indent_spaces) {}

    // 按节点类别打印当前表达式并递归处理子节点。
    void Print(const Expr& expr, int indent) {
        if (!expr.defined()) {
            os_ << Indent(indent) << "<undef-expr>\n";
            return;
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
                << ")" << CheckedTypeSuffix(expr) << "\n";
            for (size_t i = 0; i < call->args.size(); ++i) {
                os_ << Indent(indent + indent_spaces_) << "arg[" << i << "]:\n";
                Print(call->args[i], indent + indent_spaces_ * 2);
            }
            return;
        }
        if (const auto* func = expr.As<FunctionNode>()) {
            os_ << Indent(indent) << "Function(params=" << func->params.size() << ")"
                << CheckedTypeSuffix(expr) << "\n";
            for (size_t i = 0; i < func->params.size(); ++i) {
                os_ << Indent(indent + indent_spaces_) << "param[" << i
                    << "]=" << func->params[i]->vid->name_hint << "\n";
            }
            os_ << Indent(indent + indent_spaces_) << "body:\n";
            Print(func->body, indent + indent_spaces_ * 2);
            return;
        }
        if (const auto* if_node = expr.As<IfNode>()) {
            os_ << Indent(indent) << "If" << CheckedTypeSuffix(expr) << "\n";
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
                << CheckedTypeSuffix(expr) << "\n";
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
                << CheckedTypeSuffix(expr) << "\n";
            os_ << Indent(indent + indent_spaces_) << "value:\n";
            Print(let_node->value, indent + indent_spaces_ * 2);
            os_ << Indent(indent + indent_spaces_) << "body:\n";
            Print(let_node->body, indent + indent_spaces_ * 2);
            return;
        }
        if (const auto* tuple = expr.As<TupleNode>()) {
            os_ << Indent(indent) << "Tuple(fields=" << tuple->fields.size() << ")"
                << CheckedTypeSuffix(expr) << "\n";
            for (size_t i = 0; i < tuple->fields.size(); ++i) {
                os_ << Indent(indent + indent_spaces_) << "field[" << i << "]:\n";
                Print(tuple->fields[i], indent + indent_spaces_ * 2);
            }
            return;
        }
        if (const auto* tuple_get = expr.As<TupleGetItemNode>()) {
            os_ << Indent(indent) << "TupleGetItem(index=" << tuple_get->index << ")"
                << CheckedTypeSuffix(expr) << "\n";
            os_ << Indent(indent + indent_spaces_) << "tuple:\n";
            Print(tuple_get->tuple, indent + indent_spaces_ * 2);
            return;
        }

        os_ << Indent(indent) << "<expr>\n";
    }

private:
    std::ostream& os_;
    int indent_spaces_{2};
};

}  // namespace

// 配置公开 IR 打印 pass 的缩进宽度。
IRPrinterPass::IRPrinterPass(int indent_spaces) : indent_spaces_(indent_spaces) {}

// 打印任意 Relay 表达式。
void IRPrinterPass::Run(const Expr& expr, std::ostream& os) const {
    RelayIRPrinter printer(os, indent_spaces_);
    printer.Print(expr, 0);
}

// 将 Function 适配为 Expr 后打印。
void IRPrinterPass::Run(const Function& func, std::ostream& os) const {
    Run(Expr(ObjectRef(func)), os);
}

// 直接把表达式转储到调用方流。
void DumpExpr(const Expr& expr, std::ostream& os, int indent_spaces) {
    IRPrinterPass pass(indent_spaces);
    pass.Run(expr, os);
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

}  // namespace pass
}  // namespace relay
}  // namespace kxc
