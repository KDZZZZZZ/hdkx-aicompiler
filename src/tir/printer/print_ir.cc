/*! \file src/tir/printer/print_ir.cc
 * \brief 实现 TIR IR 文本打印工具。
 */

#include "kxc/tir/printer/print_ir.h"

#include "kxc/tir/visitor.h"

#include <sstream>
#include <string>

namespace kxc {
namespace tir {
namespace printer {

namespace {

std::string Indent(int n) { return std::string(n, ' '); }

std::string DTypeToString(const DataType& dt) {
    if (dt.code == 2) return "float" + std::to_string(dt.bits);
    if (dt.code == 0) return "int" + std::to_string(dt.bits);
    if (dt.code == 1) return dt.bits == 1 ? "bool" : ("uint" + std::to_string(dt.bits));
    if (dt.code == 3) return "handle";
    return "dtype(code=" + std::to_string(dt.code) + ",bits=" + std::to_string(dt.bits) + ")";
}

// 结构化枚举只在这里映射为 CUDA 拼写，其他 pass 不处理裸 thread tag。
const char* ThreadIndexName(ThreadIndexKind kind) {
    switch (kind) {
        case ThreadIndexKind::kBlockIdxX: return "blockIdx.x";
        case ThreadIndexKind::kBlockIdxY: return "blockIdx.y";
        case ThreadIndexKind::kBlockIdxZ: return "blockIdx.z";
        case ThreadIndexKind::kThreadIdxX: return "threadIdx.x";
        case ThreadIndexKind::kThreadIdxY: return "threadIdx.y";
        case ThreadIndexKind::kThreadIdxZ: return "threadIdx.z";
    }
    return "unknown_thread_index";
}

const char* ForTypeName(ForType type) {
    switch (type) {
        case ForType::Serial: return "serial";
        case ForType::Parallel: return "parallel";
        case ForType::Vectorized: return "vectorized";
        case ForType::Unrolled: return "unrolled";
    }
    return "unknown";
}

class IRPrinterImpl : private TIRExprFunctor<std::string>, private TIRStmtFunctor<void> {
public:
    IRPrinterImpl(std::ostream& os, int indent_spaces) : os_(os), indent_spaces_(indent_spaces) {}

    void PrintPrimFunc(const PrimFunc& f) {
        if (!f.defined()) {
            os_ << "<undef-primfunc>\n";
            return;
        }

        os_ << "\n================ TIR PrimFunc ================\n";
        os_ << "params(" << f->params.size() << "):\n";
        for (size_t i = 0; i < f->params.size(); ++i) {
            const auto& p = f->params[i];
            os_ << "  [" << i << "] " << p->name_hint << " : " << DTypeToString(p->dtype) << "\n";
        }
        os_ << "buffer_map(" << f->buffer_map.size() << "):\n";
        for (const auto& kv : f->buffer_map) {
            const auto& v = kv.first;
            const auto& b = kv.second;
            os_ << "  " << v->name_hint << " -> " << b->name << " shape=[";
            for (size_t i = 0; i < b->shape.size(); ++i) {
                if (i) os_ << ", ";
                os_ << PrintExpr(b->shape[i]);
            }
            os_ << "] dtype=" << DTypeToString(b->dtype) << "\n";
        }
        os_ << "attrs(" << f->attrs.size() << "):\n";
        for (const auto& kv : f->attrs) {
            os_ << "  " << std::string(kv.first) << "\n";
        }
        os_ << "body:\n";
        PrintStmt(f->body, indent_spaces_);
    }

private:
    std::string PrintExpr(const PrimExpr& expr) { return VisitExpr(expr); }

    void PrintStmt(const Stmt& stmt, int indent) {
        indent_ = indent;
        VisitStmt(stmt);
    }

    std::string VisitIntImm(const IntImmNode* op, const PrimExpr& ref) override {
        (void)ref;
        return std::to_string(op->value);
    }

    std::string VisitFloatImm(const FloatImmNode* op, const PrimExpr& ref) override {
        (void)ref;
        std::ostringstream os;
        os << op->value;
        return os.str();
    }

    std::string VisitVar(const VarNode* op, const PrimExpr& ref) override {
        (void)ref;
        return op->name_hint;
    }

    std::string VisitLoad(const LoadNode* op, const PrimExpr& ref) override {
        (void)ref;
        if (op->predicate.defined()) {
            return op->buffer_var->name_hint + "[" + PrintExpr(op->index) + "]"
                   + " if " + PrintExpr(op->predicate);
        }
        return op->buffer_var->name_hint + "[" + PrintExpr(op->index) + "]";
    }

#define KXC_PRINT_BINARY_EXPR(NodeType, Symbol) \
    std::string Visit##NodeType(const NodeType##Node* op, const PrimExpr& ref) override { \
        (void)ref; \
        return "(" + PrintExpr(op->a) + " " Symbol " " + PrintExpr(op->b) + ")"; \
    }

    KXC_PRINT_BINARY_EXPR(Add, "+")
    KXC_PRINT_BINARY_EXPR(Sub, "-")
    KXC_PRINT_BINARY_EXPR(Mul, "*")
    KXC_PRINT_BINARY_EXPR(Div, "/")
    KXC_PRINT_BINARY_EXPR(Mod, "%")
    KXC_PRINT_BINARY_EXPR(EQ, "==")
    KXC_PRINT_BINARY_EXPR(LT, "<")
    KXC_PRINT_BINARY_EXPR(And, "&&")
    KXC_PRINT_BINARY_EXPR(Or, "||")

#undef KXC_PRINT_BINARY_EXPR

    std::string VisitMin(const MinNode* op, const PrimExpr& ref) override {
        (void)ref;
        return "min(" + PrintExpr(op->a) + ", " + PrintExpr(op->b) + ")";
    }

    std::string VisitMax(const MaxNode* op, const PrimExpr& ref) override {
        (void)ref;
        return "max(" + PrintExpr(op->a) + ", " + PrintExpr(op->b) + ")";
    }

    std::string VisitNot(const NotNode* op, const PrimExpr& ref) override {
        (void)ref;
        return "(!" + PrintExpr(op->value) + ")";
    }

    std::string VisitSelect(const SelectNode* op, const PrimExpr& ref) override {
        (void)ref;
        return "select(" + PrintExpr(op->condition) + ", " + PrintExpr(op->true_value) + ", "
               + PrintExpr(op->false_value) + ")";
    }

    std::string VisitCall(const CallNode* op, const PrimExpr& ref) override {
        (void)ref;
        std::ostringstream os;
        os << op->name << "(";
        for (size_t i = 0; i < op->args.size(); ++i) {
            if (i) os << ", ";
            os << PrintExpr(op->args[i]);
        }
        os << ")";
        return os.str();
    }

    std::string VisitExprDefault(const PrimExpr& expr) override {
        (void)expr;
        return "<expr>";
    }

    void VisitLetStmt(const LetStmtNode* op, const Stmt& ref) override {
        (void)ref;
        os_ << Indent(indent_) << "let " << op->var->name_hint << " = " << PrintExpr(op->value) << " in\n";
        PrintStmt(op->body, indent_ + indent_spaces_);
    }

    void VisitStore(const StoreNode* op, const Stmt& ref) override {
        (void)ref;
        os_ << Indent(indent_) << op->buffer_var->name_hint << "[" << PrintExpr(op->index)
            << "] = " << PrintExpr(op->value);
        if (op->predicate.defined()) {
            os_ << " if " << PrintExpr(op->predicate);
        }
        os_ << ";\n";
    }

    void VisitFor(const ForNode* op, const Stmt& ref) override {
        (void)ref;
        os_ << Indent(indent_) << "for[" << ForTypeName(op->for_type) << "] ("
            << op->loop_var->name_hint << " = " << PrintExpr(op->min)
            << "; " << op->loop_var->name_hint << " < (" << PrintExpr(op->min) << " + "
            << PrintExpr(op->extent) << "); " << op->loop_var->name_hint << "++) {\n";
        PrintStmt(op->body, indent_ + indent_spaces_);
        os_ << Indent(indent_) << "}\n";
    }

    // 打印显式绑定范围和变量，便于测试确认 pass 没有退化为 AttrStmt 字符串约定。
    void VisitThreadBinding(const ThreadBindingNode* op, const Stmt& ref) override {
        (void)ref;
        os_ << Indent(indent_) << "thread_binding(" << op->thread_var->name_hint
            << " = " << ThreadIndexName(op->thread_index)
            << ", extent=" << PrintExpr(op->extent) << ") {\n";
        PrintStmt(op->body, indent_ + indent_spaces_);
        os_ << Indent(indent_) << "}\n";
    }

    void VisitIfThenElse(const IfThenElseNode* op, const Stmt& ref) override {
        (void)ref;
        os_ << Indent(indent_) << "if (" << PrintExpr(op->condition) << ") {\n";
        PrintStmt(op->then_case, indent_ + indent_spaces_);
        if (op->else_case.defined()) {
            os_ << Indent(indent_) << "} else {\n";
            PrintStmt(op->else_case, indent_ + indent_spaces_);
        }
        os_ << Indent(indent_) << "}\n";
    }

    void VisitAllocate(const AllocateNode* op, const Stmt& ref) override {
        (void)ref;
        os_ << Indent(indent_) << "allocate " << op->buffer_var->name_hint << " : "
            << DTypeToString(op->dtype) << " [";
        for (size_t i = 0; i < op->extents.size(); ++i) {
            if (i) os_ << ", ";
            os_ << PrintExpr(op->extents[i]);
        }
        os_ << "]";
        if (op->condition.defined()) {
            os_ << " if (" << PrintExpr(op->condition) << ")";
        }
        os_ << " {\n";
        PrintStmt(op->body, indent_ + indent_spaces_);
        os_ << Indent(indent_) << "}\n";
    }

    void VisitAttrStmt(const AttrStmtNode* op, const Stmt& ref) override {
        (void)ref;
        os_ << Indent(indent_) << "attr[" << op->attr_key << "] = " << PrintExpr(op->value) << " {\n";
        PrintStmt(op->body, indent_ + indent_spaces_);
        os_ << Indent(indent_) << "}\n";
    }

    void VisitBlock(const BlockNode* op, const Stmt& ref) override {
        (void)ref;
        os_ << Indent(indent_) << "block(" << op->name_hint << ") {\n";
        if (!op->iter_vars.empty()) {
            os_ << Indent(indent_ + indent_spaces_) << "iter_vars:\n";
            for (const auto& iv : op->iter_vars) {
                os_ << Indent(indent_ + indent_spaces_ * 2) << iv->var->name_hint << " in ["
                    << PrintExpr(iv->dom->min) << ", " << PrintExpr(iv->dom->extent) << ")\n";
            }
        }
        if (op->init.defined()) {
            os_ << Indent(indent_ + indent_spaces_) << "init:\n";
            PrintStmt(op->init, indent_ + indent_spaces_ * 2);
        }
        os_ << Indent(indent_ + indent_spaces_) << "body:\n";
        PrintStmt(op->body, indent_ + indent_spaces_ * 2);
        os_ << Indent(indent_) << "}\n";
    }

    void VisitSeqStmt(const SeqStmtNode* op, const Stmt& ref) override {
        (void)ref;
        for (const auto& s : op->seq) {
            PrintStmt(s, indent_);
        }
    }

    void VisitEvaluate(const EvaluateNode* op, const Stmt& ref) override {
        (void)ref;
        os_ << Indent(indent_) << "evaluate(" << PrintExpr(op->value) << ");\n";
    }

    void VisitStmtDefault(const Stmt& stmt) override {
        (void)stmt;
        os_ << Indent(indent_) << "<stmt>\n";
    }

    std::ostream& os_;
    int indent_spaces_ = 2;
    int indent_ = 0;
};

}  // namespace

IRPrinter::IRPrinter(int indent_spaces) : indent_spaces_(indent_spaces) {}

void IRPrinter::Run(const PrimFunc& func, std::ostream& os) const {
    IRPrinterImpl printer(os, indent_spaces_);
    printer.PrintPrimFunc(func);
}

void DumpPrimFunc(const PrimFunc& func, std::ostream& os, int indent_spaces) {
    IRPrinter printer(indent_spaces);
    printer.Run(func, os);
}

}  // namespace printer
}  // namespace tir
}  // namespace kxc
