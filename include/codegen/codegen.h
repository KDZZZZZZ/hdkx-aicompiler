/*! \file include/codegen/codegen.h
 * \brief 定义 codegen 后端、C/LLVM codegen、JIT 和 compiled kernel 抽象。
 */

#pragma once

#include <memory>
#include <sstream>
#include <string>

#include "base/container.h"
#include "base/pass.h"
#include "codegen/backend.h"
#include "tir/stmt.h"

namespace kxc {
namespace codegen {

/*! \brief 代码生成产物，承载源码/模块指针和函数签名元信息。 */
struct CodeGenResult {
    CodeGenBackend backend;
    std::string source;          // 生成的源码（C/LLVM IR文本）
    void* module_ptr{nullptr};   // LLVM Module指针（backend=kLLVM时）
    Map<String, String> metadata;  // 函数签名等元信息
};

/*! \brief 代码生成器基类，遍历 TIR 树生成目标后端代码。 */
class CodeGenBase : public TIRExprFunctor<void>, public TIRStmtFunctor<void> {
public:
    virtual ~CodeGenBase() = default;

    /*! \brief 将 PrimFunc 转换为目标后端的代码生成结果。 */
    virtual CodeGenResult Generate(const tir::PrimFunc& func) = 0;

protected:
    /*! \brief 文本后端输出流，主要用于 C/CUDA 代码生成。 */
    std::ostringstream stream_;
    int indent_{0};

    void PrintIndent();
    void IncIndent() { indent_ += 2; }
    void DecIndent() { indent_ -= 2; }
};

}  // namespace codegen
}  // namespace kxc
