#pragma once

#include <sstream>
#include <string>
#include <unordered_map>

#include "codegen/codegen.h"
#include "tir/stmt.h"

namespace kxc {
namespace codegen {

class CodeGenC {
public:
    // 将PrimFunc生成为C源代码字符串
    std::string Generate(const tir::PrimFunc& func, const std::string& name = "main");

private:
    // ---- 表达式 → C表达式字符串 ----
    std::string GenExpr(const tir::PrimExpr& expr);

    // ---- 语句 → C语句 ----
    void GenStmt(const tir::Stmt& stmt);

    // ---- 工具方法 ----
    std::string DTypeToCType(tir::DataType dtype);
    std::string GetVarName(const tir::Var& var);
    void PrintIndent();

    std::ostringstream stream_;
    int indent_{0};
    std::unordered_map<const Object*, std::string> var_name_map_;
    int var_counter_{0};

    static constexpr int kStackAllocThreshold = 4096;
};

}  // namespace codegen
}  // namespace kxc
