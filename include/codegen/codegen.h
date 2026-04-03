#pragma once

#include <memory>
#include <sstream>
#include <string>

#include "base/container.h"
#include "base/pass.h"
#include "tir/stmt.h"

namespace kxc {
namespace codegen {

// 代码生成后端类型
enum class CodeGenBackend {
    kLLVM,  // LLVM IR → JIT（默认，高性能）
    kC,     // C源码 → gcc（备用，可调试）
    kCUDA,  // CUDA（GPU，未来扩展）
};

// 代码生成产物
struct CodeGenResult {
    CodeGenBackend backend;
    std::string source;          // 生成的源码（C/LLVM IR文本）
    void* module_ptr{nullptr};   // LLVM Module指针（backend=kLLVM时）
    Map<String, String> metadata;  // 函数签名等元信息
};

// 代码生成器基类
// 继承TIRExprFunctor和TIRStmtFunctor，遍历TIR树生成目标代码
class CodeGenBase : public TIRExprFunctor<void>, public TIRStmtFunctor<void> {
public:
    virtual ~CodeGenBase() = default;

    // 主入口：将PrimFunc转换为目标代码
    virtual CodeGenResult Generate(const tir::PrimFunc& func) = 0;

protected:
    // 输出流管理（用于C/CUDA代码生成）
    std::ostringstream stream_;
    int indent_{0};

    void PrintIndent();
    void IncIndent() { indent_ += 2; }
    void DecIndent() { indent_ -= 2; }
};

}  // namespace codegen
}  // namespace kxc
