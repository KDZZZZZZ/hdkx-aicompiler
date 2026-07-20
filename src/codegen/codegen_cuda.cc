/*! \file src/codegen/codegen_cuda.cc
 * \brief 实现线程绑定 TIR 到可由 NVRTC 编译的 CUDA C++ 源码发射。
 */

#include "codegen/codegen_cuda.h"

#include <cmath>
#include <cctype>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

namespace kxc::codegen {
namespace {

/*! \brief 判断名称是否可以直接作为 C/C++ 标识符，防止符号注入源码。 */
bool IsIdentifier(const std::string& text) {
    if (text.empty() ||
        !(std::isalpha(static_cast<unsigned char>(text[0])) || text[0] == '_')) {
        return false;
    }
    for (char ch : text) {
        const unsigned char value = static_cast<unsigned char>(ch);
        if (!(std::isalnum(value) || ch == '_')) return false;
    }
    return true;
}

/*! \brief 把任意提示名清洗为变量标识符；最终唯一性由 CodeGenCUDA 维护。 */
std::string SanitizeIdentifier(const std::string& hint) {
    std::string result;
    result.reserve(hint.size() + 1);
    for (char ch : hint) {
        const unsigned char value = static_cast<unsigned char>(ch);
        result.push_back(std::isalnum(value) || ch == '_' ? ch : '_');
    }
    if (result.empty()) result = "v";
    if (!(std::isalpha(static_cast<unsigned char>(result[0])) || result[0] == '_')) {
        result.insert(result.begin(), '_');
    }
    return result;
}

/*! \brief 返回浮点 intrinsic 在当前精度下对应的 CUDA C 函数名。 */
std::string MathFunction(const tir::CallNode* call) {
    const bool single = call->dtype.code == 2 && call->dtype.bits == 32;
    if (call->name == "tir.exp") return single ? "expf" : "exp";
    if (call->name == "tir.log") return single ? "logf" : "log";
    if (call->name == "tir.sqrt") return single ? "sqrtf" : "sqrt";
    if (call->name == "tir.tanh") return single ? "tanhf" : "tanh";
    if (call->name == "tir.fabs") return single ? "fabsf" : "fabs";
    throw std::runtime_error("CodeGenCUDA: unsupported call '" + call->name + "'");
}

}  // namespace

std::string CodeGenCUDA::Generate(const tir::PrimFunc& function,
                                  const std::string& symbol) {
    if (!function.defined()) throw std::invalid_argument("CodeGenCUDA requires PrimFunc");
    if (!IsIdentifier(symbol)) {
        throw std::invalid_argument("CUDA kernel symbol is not a valid identifier");
    }

    output_.str("");
    output_.clear();
    indent_ = 0;
    next_variable_id_ = 0;
    saw_thread_binding_ = false;
    variable_names_.clear();
    used_names_.clear();

    output_ << "#include <stdint.h>\n";
    output_ << "#include <math.h>\n";
    output_ << "#include <cuda_fp16.h>\n\n";
    output_ << "#include <cuda_fp16.h>\n\n";
    output_ << "extern \"C\" __global__ void " << symbol << "(";
    for (size_t i = 0; i < function->params.size(); ++i) {
        if (i != 0) output_ << ", ";
        const tir::Var& parameter = function->params[i];
        if (!function->buffer_map.count(parameter)) {
            throw std::runtime_error(
                "CodeGenCUDA: scalar PrimFunc parameters are unsupported");
        }
        const tir::DataType dtype = function->buffer_map.at(parameter)->dtype;
        output_ << DTypeName(dtype) << "* __restrict__ " << VarName(parameter);
    }
    output_ << ") {\n";
    indent_ = 2;
    GenStmt(function->body);
    if (!saw_thread_binding_) {
        throw std::invalid_argument(
            "CodeGenCUDA requires a thread-bound PrimFunc");
    }
    output_ << "}\n";
    return output_.str();
}

std::string CodeGenCUDA::DTypeName(tir::DataType dtype) const {
    if (dtype.lanes != 1) {
        throw std::runtime_error("CodeGenCUDA: vector dtype is unsupported");
    }
    if (dtype.code == 2) {
        if (dtype.bits == 16) return "__half";
        if (dtype.bits == 32) return "float";
        if (dtype.bits == 64) return "double";
    }
    if (dtype.code == 0) {
        if (dtype.bits == 8) return "int8_t";
        if (dtype.bits == 16) return "int16_t";
        if (dtype.bits == 32) return "int32_t";
        if (dtype.bits == 64) return "int64_t";
    }
    if (dtype.code == 1) {
        if (dtype.bits == 1) return "bool";
        if (dtype.bits == 8) return "uint8_t";
        if (dtype.bits == 16) return "uint16_t";
        if (dtype.bits == 32) return "uint32_t";
        if (dtype.bits == 64) return "uint64_t";
    }
    throw std::runtime_error("CodeGenCUDA: unsupported scalar dtype");
}

std::string CodeGenCUDA::VarName(const tir::Var& variable) {
    if (!variable.defined()) throw std::runtime_error("CodeGenCUDA: undefined Var");
    auto found = variable_names_.find(variable.get());
    if (found != variable_names_.end()) return found->second;

    const std::string base = SanitizeIdentifier(variable->name_hint);
    std::string name = base;
    while (!used_names_.insert(name).second) {
        name = base + "_" + std::to_string(next_variable_id_++);
    }
    variable_names_.emplace(variable.get(), name);
    return name;
}

const char* CodeGenCUDA::ThreadBuiltin(tir::ThreadIndexKind kind) {
    switch (kind) {
        case tir::ThreadIndexKind::kBlockIdxX: return "blockIdx.x";
        case tir::ThreadIndexKind::kBlockIdxY: return "blockIdx.y";
        case tir::ThreadIndexKind::kBlockIdxZ: return "blockIdx.z";
        case tir::ThreadIndexKind::kThreadIdxX: return "threadIdx.x";
        case tir::ThreadIndexKind::kThreadIdxY: return "threadIdx.y";
        case tir::ThreadIndexKind::kThreadIdxZ: return "threadIdx.z";
    }
    throw std::runtime_error("CodeGenCUDA: unknown thread index kind");
}

void CodeGenCUDA::Indent() {
    for (int i = 0; i < indent_; ++i) output_ << ' ';
}

std::string CodeGenCUDA::GenExpr(const tir::PrimExpr& expression) {
    if (!expression.defined()) {
        throw std::runtime_error("CodeGenCUDA: undefined expression");
    }
    if (const auto* node = expression.As<tir::IntImmNode>()) {
        if (node->dtype.code == 1 && node->dtype.bits != 1) {
            return std::to_string(static_cast<uint64_t>(node->value)) + "ULL";
        }
        return std::to_string(node->value) +
               (node->dtype.bits == 64 ? "LL" : "");
    }
    if (const auto* node = expression.As<tir::FloatImmNode>()) {
        if (std::isnan(node->value)) return "NAN";
        if (std::isinf(node->value)) return node->value > 0 ? "INFINITY" : "(-INFINITY)";
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer),
                      node->dtype.bits == 32 ? "%.9ef" : "%.17e", node->value);
        return buffer;
    }
    if (const auto* node = expression.As<tir::VarNode>()) {
        const auto found = variable_names_.find(node);
        if (found == variable_names_.end()) {
            throw std::runtime_error("CodeGenCUDA: undefined variable '" +
                                     node->name_hint + "'");
        }
        return found->second;
    }

    const auto binary = [&](const tir::BinaryOpNode* node, const char* op) {
        return "(" + GenExpr(node->a) + " " + op + " " + GenExpr(node->b) + ")";
    };
    if (const auto* node = expression.As<tir::AddNode>()) return binary(node, "+");
    if (const auto* node = expression.As<tir::SubNode>()) return binary(node, "-");
    if (const auto* node = expression.As<tir::MulNode>()) return binary(node, "*");
    if (const auto* node = expression.As<tir::DivNode>()) return binary(node, "/");
    if (const auto* node = expression.As<tir::ModNode>()) {
        if (node->dtype.code == 2) {
            const char* function = node->dtype.bits == 32 ? "fmodf" : "fmod";
            return std::string(function) + "(" + GenExpr(node->a) + ", " +
                   GenExpr(node->b) + ")";
        }
        return binary(node, "%");
    }
    if (const auto* node = expression.As<tir::EQNode>()) return binary(node, "==");
    if (const auto* node = expression.As<tir::LTNode>()) return binary(node, "<");
    if (const auto* node = expression.As<tir::AndNode>()) return binary(node, "&&");
    if (const auto* node = expression.As<tir::OrNode>()) return binary(node, "||");
    if (const auto* node = expression.As<tir::MinNode>()) {
        const std::string lhs = GenExpr(node->a);
        const std::string rhs = GenExpr(node->b);
        return "((" + lhs + ") < (" + rhs + ") ? (" + lhs + ") : (" + rhs + "))";
    }
    if (const auto* node = expression.As<tir::MaxNode>()) {
        const std::string lhs = GenExpr(node->a);
        const std::string rhs = GenExpr(node->b);
        return "((" + lhs + ") > (" + rhs + ") ? (" + lhs + ") : (" + rhs + "))";
    }
    if (const auto* node = expression.As<tir::NotNode>()) {
        return "(!" + GenExpr(node->value) + ")";
    }
    if (const auto* node = expression.As<tir::LoadNode>()) {
        const std::string buffer = GenExpr(tir::PrimExpr(ObjectRef(node->buffer_var)));
        const std::string load = buffer + "[" + GenExpr(node->index) + "]";
        return node->predicate.defined()
                   ? "(" + GenExpr(node->predicate) + " ? " + load + " : 0)"
                   : load;
    }
    if (const auto* node = expression.As<tir::SelectNode>()) {
        return "(" + GenExpr(node->condition) + " ? " +
               GenExpr(node->true_value) + " : " + GenExpr(node->false_value) + ")";
    }
    if (const auto* node = expression.As<tir::CallNode>()) {
        if (node->name == "cast") {
            if (node->args.size() != 1) {
                throw std::runtime_error("CodeGenCUDA: cast expects one argument");
            }
            return "static_cast<" + DTypeName(node->dtype) + ">(" +
                   GenExpr(node->args[0]) + ")";
        }
        const std::string function = MathFunction(node);
        std::string result = function + "(";
        for (size_t i = 0; i < node->args.size(); ++i) {
            if (i != 0) result += ", ";
            result += GenExpr(node->args[i]);
        }
        return result + ")";
    }
    throw std::runtime_error("CodeGenCUDA: unsupported expression node");
}

void CodeGenCUDA::GenStmt(const tir::Stmt& statement) {
    if (!statement.defined()) throw std::runtime_error("CodeGenCUDA: undefined statement");
    if (const auto* node = statement.As<tir::ThreadBindingNode>()) {
        saw_thread_binding_ = true;
        const std::string variable = VarName(node->thread_var);
        Indent();
        output_ << "const " << DTypeName(node->thread_var->dtype) << ' ' << variable
                << " = static_cast<" << DTypeName(node->thread_var->dtype) << ">(" 
                << ThreadBuiltin(node->thread_index) << ");\n";
        // 即使 launch metadata 大于逻辑 extent，越界线程也不会进入 body；这使发射器
        // 对向上取整的 block 数保持正确，而不依赖调度器必须恰好整除。
        Indent();
        output_ << "if (" << variable << " < " << GenExpr(node->extent) << ") {\n";
        indent_ += 2;
        GenStmt(node->body);
        indent_ -= 2;
        Indent();
        output_ << "}\n";
        return;
    }
    if (const auto* node = statement.As<tir::ForNode>()) {
        if (node->for_type != tir::ForType::Serial &&
            node->for_type != tir::ForType::Unrolled) {
            throw std::runtime_error("CodeGenCUDA: unsupported ForType");
        }
        const std::string variable = VarName(node->loop_var);
        const std::string minimum = GenExpr(node->min);
        Indent();
        if (node->for_type == tir::ForType::Unrolled) output_ << "#pragma unroll\n";
        Indent();
        output_ << "for (" << DTypeName(node->loop_var->dtype) << ' ' << variable
                << " = " << minimum << "; " << variable << " < (" << minimum
                << " + " << GenExpr(node->extent) << "); ++" << variable << ") {\n";
        indent_ += 2;
        GenStmt(node->body);
        indent_ -= 2;
        Indent();
        output_ << "}\n";
        return;
    }
    if (const auto* node = statement.As<tir::StoreNode>()) {
        const std::string buffer = GenExpr(tir::PrimExpr(ObjectRef(node->buffer_var)));
        if (node->predicate.defined()) {
            Indent();
            output_ << "if (" << GenExpr(node->predicate) << ") {\n";
            indent_ += 2;
        }
        Indent();
        output_ << buffer << '[' << GenExpr(node->index) << "] = "
                << GenExpr(node->value) << ";\n";
        if (node->predicate.defined()) {
            indent_ -= 2;
            Indent();
            output_ << "}\n";
        }
        return;
    }
    if (const auto* node = statement.As<tir::LetStmtNode>()) {
        Indent();
        output_ << DTypeName(node->var->dtype) << ' ' << VarName(node->var)
                << " = " << GenExpr(node->value) << ";\n";
        GenStmt(node->body);
        return;
    }
    if (const auto* node = statement.As<tir::IfThenElseNode>()) {
        Indent();
        output_ << "if (" << GenExpr(node->condition) << ") {\n";
        indent_ += 2;
        GenStmt(node->then_case);
        indent_ -= 2;
        if (node->else_case.defined()) {
            Indent();
            output_ << "} else {\n";
            indent_ += 2;
            GenStmt(node->else_case);
            indent_ -= 2;
        }
        Indent();
        output_ << "}\n";
        return;
    }
    if (const auto* node = statement.As<tir::AllocateNode>()) {
        Indent();
        output_ << "{\n";
        indent_ += 2;
        std::string count = "1";
        for (const auto& extent : node->extents) {
            if (!extent.As<tir::IntImmNode>()) {
                throw std::runtime_error("CodeGenCUDA: Allocate requires constant extents");
            }
            count = "(" + count + " * " + GenExpr(extent) + ")";
        }
        Indent();
        output_ << DTypeName(node->dtype) << ' ' << VarName(node->buffer_var)
                << '[' << count << "];\n";
        if (node->condition.defined()) {
            Indent();
            output_ << "if (" << GenExpr(node->condition) << ") {\n";
            indent_ += 2;
        }
        GenStmt(node->body);
        if (node->condition.defined()) {
            indent_ -= 2;
            Indent();
            output_ << "}\n";
        }
        indent_ -= 2;
        Indent();
        output_ << "}\n";
        return;
    }
    if (const auto* node = statement.As<tir::SeqStmtNode>()) {
        for (const auto& child : node->seq) GenStmt(child);
        return;
    }
    if (const auto* node = statement.As<tir::EvaluateNode>()) {
        Indent();
        output_ << GenExpr(node->value) << ";\n";
        return;
    }
    throw std::runtime_error("CodeGenCUDA: unsupported statement node");
}

}  // namespace kxc::codegen
