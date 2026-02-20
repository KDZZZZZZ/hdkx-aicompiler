#include "relay/pass/print_ir.h"

#include <sstream>

#include "relay/op.h"

namespace kxc {
namespace relay {
namespace pass {

namespace {

std::string Indent(int n) { return std::string(n, ' '); }

std::string DTypeToString(const DLDataType& dtype) {
    if (dtype.code == kDLFloat) {
        return "float" + std::to_string(dtype.bits);
    }
    if (dtype.code == kDLInt) {
        return "int" + std::to_string(dtype.bits);
    }
    if (dtype.code == kDLUint) {
        return dtype.bits == 1 ? "bool" : ("uint" + std::to_string(dtype.bits));
    }
    return "dtype(code=" + std::to_string(dtype.code) + ",bits=" +
           std::to_string(dtype.bits) + ")";
}

class RelayIRPrinter {
public:
    RelayIRPrinter(std::ostream& os, int indent_spaces)
        : os_(os), indent_spaces_(indent_spaces) {}

    void Print(const Expr& expr, int indent) {
        if (!expr.defined()) {
            os_ << Indent(indent) << "<undef-expr>\n";
            return;
        }

        if (const auto* var = expr.As<VarNode>()) {
            os_ << Indent(indent) << "Var(" << var->vid->name_hint << ")\n";
            return;
        }
        if (const auto* constant = expr.As<ConstantNode>()) {
            os_ << Indent(indent) << "Constant(shape=[";
            for (size_t i = 0; i < constant->data->shape.size(); ++i) {
                if (i) os_ << ", ";
                os_ << constant->data->shape[i];
            }
            os_ << "], dtype=" << DTypeToString(constant->data->dl_tensor.dtype) << ")\n";
            return;
        }
        if (const auto* call = expr.As<CallNode>()) {
            std::string op_name = "<expr-op>";
            if (const auto* op_node = call->op.As<OpNode>()) {
                op_name = op_node->name;
            }
            os_ << Indent(indent) << "Call(op=" << op_name << ", args=" << call->args.size()
                << ")\n";
            for (size_t i = 0; i < call->args.size(); ++i) {
                os_ << Indent(indent + indent_spaces_) << "arg[" << i << "]:\n";
                Print(call->args[i], indent + indent_spaces_ * 2);
            }
            return;
        }
        if (const auto* func = expr.As<FunctionNode>()) {
            os_ << Indent(indent) << "Function(params=" << func->params.size() << ")\n";
            for (size_t i = 0; i < func->params.size(); ++i) {
                os_ << Indent(indent + indent_spaces_) << "param[" << i
                    << "]=" << func->params[i]->vid->name_hint << "\n";
            }
            os_ << Indent(indent + indent_spaces_) << "body:\n";
            Print(func->body, indent + indent_spaces_ * 2);
            return;
        }
        if (const auto* if_node = expr.As<IfNode>()) {
            os_ << Indent(indent) << "If\n";
            os_ << Indent(indent + indent_spaces_) << "cond:\n";
            Print(if_node->cond, indent + indent_spaces_ * 2);
            os_ << Indent(indent + indent_spaces_) << "true:\n";
            Print(if_node->true_branch, indent + indent_spaces_ * 2);
            os_ << Indent(indent + indent_spaces_) << "false:\n";
            Print(if_node->false_branch, indent + indent_spaces_ * 2);
            return;
        }
        if (const auto* let_node = expr.As<LetNode>()) {
            os_ << Indent(indent) << "Let(var=" << let_node->var->vid->name_hint << ")\n";
            os_ << Indent(indent + indent_spaces_) << "value:\n";
            Print(let_node->value, indent + indent_spaces_ * 2);
            os_ << Indent(indent + indent_spaces_) << "body:\n";
            Print(let_node->body, indent + indent_spaces_ * 2);
            return;
        }
        if (const auto* tuple = expr.As<TupleNode>()) {
            os_ << Indent(indent) << "Tuple(fields=" << tuple->fields.size() << ")\n";
            for (size_t i = 0; i < tuple->fields.size(); ++i) {
                os_ << Indent(indent + indent_spaces_) << "field[" << i << "]:\n";
                Print(tuple->fields[i], indent + indent_spaces_ * 2);
            }
            return;
        }
        if (const auto* tuple_get = expr.As<TupleGetItemNode>()) {
            os_ << Indent(indent) << "TupleGetItem(index=" << tuple_get->index << ")\n";
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

IRPrinterPass::IRPrinterPass(int indent_spaces) : indent_spaces_(indent_spaces) {}

void IRPrinterPass::Run(const Expr& expr, std::ostream& os) const {
    RelayIRPrinter printer(os, indent_spaces_);
    printer.Print(expr, 0);
}

void IRPrinterPass::Run(const Function& func, std::ostream& os) const {
    Run(Expr(func), os);
}

void DumpExpr(const Expr& expr, std::ostream& os, int indent_spaces) {
    IRPrinterPass pass(indent_spaces);
    pass.Run(expr, os);
}

void DumpFunction(const Function& func, std::ostream& os, int indent_spaces) {
    DumpExpr(Expr(func), os, indent_spaces);
}

std::string ToText(const Expr& expr, int indent_spaces) {
    std::ostringstream os;
    DumpExpr(expr, os, indent_spaces);
    return os.str();
}

std::string ToText(const Function& func, int indent_spaces) {
    std::ostringstream os;
    DumpFunction(func, os, indent_spaces);
    return os.str();
}

}  // namespace pass
}  // namespace relay
}  // namespace kxc
