#pragma once

#ifdef KXC_USE_LLVM

#include <memory>
#include <string>
#include <unordered_map>

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

#include "base/pass.h"
#include "codegen/codegen.h"
#include "tir/stmt.h"

namespace kxc {
namespace codegen {

class CodeGenLLVM {
public:
    explicit CodeGenLLVM(llvm::LLVMContext& ctx);

    // 将PrimFunc编译为LLVM Module中的一个Function
    void AddFunction(const tir::PrimFunc& func, const std::string& name = "main");

    // 获取生成的LLVM Module（所有权转移）
    std::unique_ptr<llvm::Module> TakeModule();

    // 将Module打印为LLVM IR文本（用于调试）
    std::string DumpIR() const;

private:
    // ---- 表达式code generation ----
    llvm::Value* GenExpr(const tir::PrimExpr& expr);
    llvm::Value* GenIntImm(const tir::IntImmNode* op);
    llvm::Value* GenFloatImm(const tir::FloatImmNode* op);
    llvm::Value* GenVar(const tir::VarNode* op);
    llvm::Value* GenBinaryOp(const tir::BinaryOpNode* op, const tir::PrimExpr& ref);
    llvm::Value* GenLoad(const tir::LoadNode* op);
    llvm::Value* GenCall(const tir::CallNode* op);
    llvm::Value* GenSelect(const tir::SelectNode* op);
    llvm::Value* GenNot(const tir::NotNode* op);

    // ---- 语句code generation ----
    void GenStmt(const tir::Stmt& stmt);
    void GenFor(const tir::ForNode* op);
    void GenStore(const tir::StoreNode* op);
    void GenAllocate(const tir::AllocateNode* op);
    void GenIfThenElse(const tir::IfThenElseNode* op);
    void GenLetStmt(const tir::LetStmtNode* op);
    void GenSeqStmt(const tir::SeqStmtNode* op);
    void GenEvaluate(const tir::EvaluateNode* op);

    // ---- 工具方法 ----
    llvm::Type* GetLLVMType(tir::DataType dtype);
    llvm::FunctionType* CreateFuncType(const tir::PrimFunc& func);
    llvm::Function* GetOrDeclareIntrinsic(const std::string& name, llvm::Type* type);

    llvm::LLVMContext& ctx_;
    std::unique_ptr<llvm::Module> module_;
    llvm::IRBuilder<> builder_;
    llvm::Function* current_func_{nullptr};

    // TIR Var → LLVM Value* 映射
    std::unordered_map<const Object*, llvm::Value*> var_map_;
};

}  // namespace codegen
}  // namespace kxc

#endif  // KXC_USE_LLVM
