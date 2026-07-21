/*! \file include/codegen/codegen_c.h
 * \brief 定义仅用于诊断和导出的 TIR 到 C 源码发射器。
 */

#pragma once

#include <sstream>
#include <string>
#include <unordered_map>

#include "base/object.h"
#include "tir/stmt.h"

namespace kxc {
namespace codegen {

/*!
 * \brief 将 PrimFunc 发射为可读 C 源码，不生成或加载可执行模块。
 *
 * 该类型不属于 Compiler 的 backend 选择集合。它只服务于调试、IR 检查和
 * 源码导出，调用方不能把返回字符串解释为已经完成编译的运行时产物。
 */
class CSourceEmitter {
public:
    /*! \brief 清空内部状态并将指定 PrimFunc 发射为完整 C 函数源码。 */
    std::string Generate(const tir::PrimFunc& func, const std::string& name = "main");

private:
    /*! \brief 递归生成单个 TIR 表达式对应的 C 表达式文本。 */
    std::string GenExpr(const tir::PrimExpr& expr);

    /*! \brief 递归发射单个 TIR 语句及其嵌套语句。 */
    void GenStmt(const tir::Stmt& stmt);

    /*! \brief 将受支持的 TIR dtype 映射为 C 标量类型名称。 */
    std::string DTypeToCType(tir::DataType dtype);
    /*! \brief 为 TIR Var 分配并缓存本次发射过程中的稳定 C 标识符。 */
    std::string GetVarName(const tir::Var& var);
    /*! \brief 按当前缩进层级向输出流写入空格。 */
    void PrintIndent();

    std::ostringstream stream_;
    int indent_{0};
    std::unordered_map<const Object*, std::string> var_name_map_;
    int var_counter_{0};

};

}  // namespace codegen
}  // namespace kxc
