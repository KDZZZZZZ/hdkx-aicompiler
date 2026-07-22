/*! \file src/codegen/c/codegen_c.cc
 * \brief 实现仅用于诊断和导出的 TIR 到 C 源码发射器。
 */

#include "internal/codegen_c.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace kxc {
namespace codegen {

// 将 TIR 标量类型转换为 C 类型；不支持的类型返回 void 以便诊断源码暴露缺口。
std::string CSourceEmitter::DTypeToCType(tir::DataType dtype) {
    if (dtype.code == 2) {  // Float
        if (dtype.bits == 32) return "float";
        if (dtype.bits == 64) return "double";
        if (dtype.bits == 16) return "_Float16";
    }
    if (dtype.code == 0) {  // Int
        if (dtype.bits == 8) return "int8_t";
        if (dtype.bits == 16) return "int16_t";
        if (dtype.bits == 32) return "int32_t";
        if (dtype.bits == 64) return "int64_t";
    }
    if (dtype.code == 1) {  // UInt
        if (dtype.bits == 1) return "bool";
        if (dtype.bits == 8) return "uint8_t";
        if (dtype.bits == 16) return "uint16_t";
        if (dtype.bits == 32) return "uint32_t";
        if (dtype.bits == 64) return "uint64_t";
    }
    return "void";
}

// 优先保留 IR 名称；匿名变量使用单次 Generate 内稳定递增的名称。
std::string CSourceEmitter::GetVarName(const tir::Var& var) {
    auto it = var_name_map_.find(var.get());
    if (it != var_name_map_.end()) return it->second;
    std::string name = var->name_hint.empty()
                           ? ("v" + std::to_string(var_counter_++))
                           : var->name_hint;
    var_name_map_[var.get()] = name;
    return name;
}

// 每一缩进层使用两个空格，使生成源码保持稳定且便于测试比较。
void CSourceEmitter::PrintIndent() {
    for (int i = 0; i < indent_; ++i) stream_ << ' ';
}

// 重置全部可变状态后发射函数，保证同一 emitter 可安全地顺序复用。
std::string CSourceEmitter::Generate(const tir::PrimFunc& func, const std::string& name) {
    stream_.str("");
    var_name_map_.clear();
    var_counter_ = 0;
    indent_ = 0;

    // 诊断源码显式包含当前发射内容依赖的标准类型、分配和数学声明。
    stream_ << "#include <stdint.h>\n";
    stream_ << "#include <stdlib.h>\n";
    stream_ << "#include <math.h>\n\n";

    // C 源码仍展示现有内部 packed ABI，但它不是公共可执行 backend 契约。
    stream_ << "int32_t " << name << "(void** packed_args) {\n";
    indent_ += 2;

    // 参数顺序严格沿用 PrimFunc.params，并从 buffer_map 恢复元素 dtype。
    for (size_t i = 0; i < func->params.size(); ++i) {
        const auto& param = func->params[i];
        std::string vname = GetVarName(param);

        tir::DataType elem_dtype = param->dtype;
        if (func->buffer_map.count(param)) {
            elem_dtype = func->buffer_map.at(param)->dtype;
        }

        PrintIndent();
        stream_ << DTypeToCType(elem_dtype) << "* __restrict__ " << vname
                << " = (" << DTypeToCType(elem_dtype) << "*)packed_args["
                << i << "];\n";
    }
    stream_ << "\n";

    // 函数体由语句发射器递归展开。
    GenStmt(func->body);

    // 诊断函数使用零返回值表示源码中展示的正常完成路径。
    PrintIndent();
    stream_ << "return 0;\n";
    indent_ -= 2;
    stream_ << "}\n";

    return stream_.str();
}

// 按节点类型递归生成表达式；未知节点保留显式占位，避免伪装为可执行支持。
std::string CSourceEmitter::GenExpr(const tir::PrimExpr& expr) {
    if (!expr.defined()) return "0";

    if (auto* n = expr.As<tir::IntImmNode>()) {
        if (n->dtype.bits == 64) return std::to_string(n->value) + "LL";
        return std::to_string(n->value);
    }
    if (auto* n = expr.As<tir::FloatImmNode>()) {
        // Handle special values
        if (std::isinf(n->value)) {
            return n->value > 0 ? "INFINITY" : "(-INFINITY)";
        }
        char buf[64];
        snprintf(buf, sizeof(buf), n->dtype.bits == 32 ? "%.8ef" : "%.16e", n->value);
        return buf;
    }
    if (auto* n = expr.As<tir::VarNode>()) {
        auto it = var_name_map_.find(n);
        if (it != var_name_map_.end()) return it->second;
        return n->name_hint.empty() ? "???" : n->name_hint;
    }

    // Binary ops
    if (auto* n = expr.As<tir::AddNode>()) return "(" + GenExpr(n->a) + " + " + GenExpr(n->b) + ")";
    if (auto* n = expr.As<tir::SubNode>()) return "(" + GenExpr(n->a) + " - " + GenExpr(n->b) + ")";
    if (auto* n = expr.As<tir::MulNode>()) return "(" + GenExpr(n->a) + " * " + GenExpr(n->b) + ")";
    if (auto* n = expr.As<tir::DivNode>()) return "(" + GenExpr(n->a) + " / " + GenExpr(n->b) + ")";
    if (auto* n = expr.As<tir::ModNode>()) return "(" + GenExpr(n->a) + " % " + GenExpr(n->b) + ")";
    if (auto* n = expr.As<tir::MinNode>()) {
        std::string a = GenExpr(n->a), b = GenExpr(n->b);
        return "((" + a + ") < (" + b + ") ? (" + a + ") : (" + b + "))";
    }
    if (auto* n = expr.As<tir::MaxNode>()) {
        std::string a = GenExpr(n->a), b = GenExpr(n->b);
        return "((" + a + ") > (" + b + ") ? (" + a + ") : (" + b + "))";
    }

    // Logic/comparison ops
    if (auto* n = expr.As<tir::EQNode>()) return "(" + GenExpr(n->a) + " == " + GenExpr(n->b) + ")";
    if (auto* n = expr.As<tir::LTNode>()) return "(" + GenExpr(n->a) + " < " + GenExpr(n->b) + ")";
    if (auto* n = expr.As<tir::AndNode>()) return "(" + GenExpr(n->a) + " && " + GenExpr(n->b) + ")";
    if (auto* n = expr.As<tir::OrNode>()) return "(" + GenExpr(n->a) + " || " + GenExpr(n->b) + ")";
    if (auto* n = expr.As<tir::NotNode>()) return "(!" + GenExpr(n->value) + ")";

    // Load
    if (auto* n = expr.As<tir::LoadNode>()) {
        std::string buf = GenExpr(tir::PrimExpr(ObjectRef(n->buffer_var)));
        std::string idx = GenExpr(n->index);
        if (n->predicate.defined()) {
            return "(" + GenExpr(n->predicate) + " ? " + buf + "[" + idx + "] : 0)";
        }
        return buf + "[" + idx + "]";
    }

    // Call
    if (auto* n = expr.As<tir::CallNode>()) {
        std::string name = n->name;
        if (name == "tir.exp") name = "expf";
        else if (name == "tir.log") name = "logf";
        else if (name == "tir.sqrt") name = "sqrtf";
        else if (name == "tir.tanh") name = "tanhf";
        else if (name == "tir.fabs") name = "fabsf";

        std::string result = name + "(";
        for (size_t i = 0; i < n->args.size(); ++i) {
            if (i > 0) result += ", ";
            result += GenExpr(n->args[i]);
        }
        return result + ")";
    }

    // Select
    if (auto* n = expr.As<tir::SelectNode>()) {
        return "(" + GenExpr(n->condition) + " ? " +
               GenExpr(n->true_value) + " : " + GenExpr(n->false_value) + ")";
    }

    return "/* unsupported expr */";
}

// ==================== 语句生成 ====================

// 按语句类型递归发射控制流、存储和局部分配。
void CSourceEmitter::GenStmt(const tir::Stmt& stmt) {
    if (!stmt.defined()) return;

    if (auto* n = stmt.As<tir::ForNode>()) {
        std::string var = GetVarName(n->loop_var);
        std::string min_str = GenExpr(n->min);
        std::string ext_str = GenExpr(n->extent);
        PrintIndent();
        stream_ << "for (int32_t " << var << " = " << min_str << "; "
                << var << " < (" << min_str << " + " << ext_str << "); ++" << var << ") {\n";
        indent_ += 2;
        GenStmt(n->body);
        indent_ -= 2;
        PrintIndent();
        stream_ << "}\n";
        return;
    }

    if (auto* n = stmt.As<tir::StoreNode>()) {
        std::string buf = GenExpr(tir::PrimExpr(ObjectRef(n->buffer_var)));
        std::string idx = GenExpr(n->index);
        std::string val = GenExpr(n->value);
        if (n->predicate.defined()) {
            PrintIndent();
            stream_ << "if (" << GenExpr(n->predicate) << ") {\n";
            indent_ += 2;
            PrintIndent();
            stream_ << buf << "[" << idx << "] = " << val << ";\n";
            indent_ -= 2;
            PrintIndent();
            stream_ << "}\n";
        } else {
            PrintIndent();
            stream_ << buf << "[" << idx << "] = " << val << ";\n";
        }
        return;
    }

    if (auto* n = stmt.As<tir::AllocateNode>()) {
        std::string var = GetVarName(n->buffer_var);
        std::string ctype = DTypeToCType(n->dtype);
        // Compute total elements
        std::string size = "1";
        for (const auto& ext : n->extents) {
            size = "(" + size + " * " + GenExpr(ext) + ")";
        }
        PrintIndent();
        stream_ << ctype << " " << var << "[" << size << "];\n";
        GenStmt(n->body);
        return;
    }

    if (auto* n = stmt.As<tir::IfThenElseNode>()) {
        PrintIndent();
        stream_ << "if (" << GenExpr(n->condition) << ") {\n";
        indent_ += 2;
        GenStmt(n->then_case);
        indent_ -= 2;
        if (n->else_case.defined()) {
            PrintIndent();
            stream_ << "} else {\n";
            indent_ += 2;
            GenStmt(n->else_case);
            indent_ -= 2;
        }
        PrintIndent();
        stream_ << "}\n";
        return;
    }

    if (auto* n = stmt.As<tir::LetStmtNode>()) {
        std::string var = GetVarName(n->var);
        std::string ctype = DTypeToCType(n->var->dtype);
        PrintIndent();
        stream_ << ctype << " " << var << " = " << GenExpr(n->value) << ";\n";
        GenStmt(n->body);
        return;
    }

    if (auto* n = stmt.As<tir::SeqStmtNode>()) {
        for (const auto& s : n->seq) {
            GenStmt(s);
        }
        return;
    }

    if (auto* n = stmt.As<tir::EvaluateNode>()) {
        PrintIndent();
        stream_ << GenExpr(n->value) << ";\n";
        return;
    }

    PrintIndent();
    stream_ << "/* unsupported stmt */\n";
}

}  // namespace codegen
}  // namespace kxc
