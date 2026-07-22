/*! \file src/codegen/llvm/internal/codegen_llvm.h
 * \brief 定义 codegen 后端、C/LLVM codegen、JIT 和 compiled kernel 抽象。
 */

#pragma once

#if KXC_USE_LLVM

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

#include "kxc/runtime/kernel_abi.h"
#include "kxc/tir/stmt.h"

namespace kxc {
namespace codegen {

/*! \brief 将 TIR PrimFunc lowering 为 LLVM IR module 的代码生成器。 */
class CodeGenLLVM {
public:
    /*! \brief 使用外部 LLVMContext 创建 codegen；调用方负责 context 生命周期。 */
    explicit CodeGenLLVM(llvm::LLVMContext& ctx);

    /*! \brief 将 PrimFunc 编译为 LLVM Module 中的一个 Function。 */
    void AddFunction(const tir::PrimFunc& func, const std::string& name = "main");

    /*! \brief Adds several independently addressable PrimFuncs to one module. */
    void AddFunctions(
        const std::vector<std::pair<tir::PrimFunc, std::string>>& functions);

    /*! \brief 取出生成的 LLVM Module，所有权转移给调用方。 */
    std::unique_ptr<llvm::Module> TakeModule();

    /*! \brief 将当前 Module 打印为 LLVM IR 文本，用于调试和 profiling artifact。 */
    std::string DumpIR() const;

private:
    /*! \brief 生成 TIR 表达式对应的 LLVM Value。 */
    llvm::Value* GenExpr(const tir::PrimExpr& expr);
    llvm::Value* GenExprInContext(const tir::PrimExpr& expr, const std::string& context);
    llvm::Value* GenIntImm(const tir::IntImmNode* op);
    llvm::Value* GenFloatImm(const tir::FloatImmNode* op);
    llvm::Value* GenVar(const tir::VarNode* op);
    llvm::Value* GenBinaryOp(const tir::BinaryOpNode* op, const tir::PrimExpr& ref);
    llvm::Value* GenLoad(const tir::LoadNode* op);
    llvm::Value* GenCall(const tir::CallNode* op);
    llvm::Value* GenSelect(const tir::SelectNode* op);
    llvm::Value* GenNot(const tir::NotNode* op);
    llvm::Type* CommonNumericType(llvm::Value* lhs, llvm::Value* rhs) const;
    llvm::Type* CommonIntegerType(llvm::Value* lhs, llvm::Value* rhs) const;
    llvm::Value* CastValue(llvm::Value* value, llvm::Type* target_type,
                           bool is_signed, const std::string& name);
    llvm::Value* CastToBool(llvm::Value* value, const std::string& name);
    void PromoteBinaryOperands(llvm::Value** lhs, llvm::Value** rhs,
                               const tir::DataType& lhs_dtype,
                               const tir::DataType& rhs_dtype,
                               const std::string& name);

    /*! \brief 生成 TIR 语句对应的 LLVM IR 指令序列。 */
    void GenStmt(const tir::Stmt& stmt);
    void GenFor(const tir::ForNode* op);
    void GenStore(const tir::StoreNode* op);
    void GenAllocate(const tir::AllocateNode* op);
    void GenIfThenElse(const tir::IfThenElseNode* op);
    void GenLetStmt(const tir::LetStmtNode* op);
    void GenSeqStmt(const tir::SeqStmtNode* op);
    void GenEvaluate(const tir::EvaluateNode* op);

    /*! \brief 将 TIR DataType 映射为 LLVM Type。 */
    llvm::Type* GetLLVMType(tir::DataType dtype);
    llvm::FunctionType* CreateFuncType(const tir::PrimFunc& func);
    llvm::Function* GetOrDeclareIntrinsic(const std::string& name, llvm::Type* type);

    llvm::LLVMContext& ctx_;
    std::unique_ptr<llvm::Module> module_;
    llvm::IRBuilder<> builder_;
    llvm::Function* current_func_{nullptr};

    // TIR Var → LLVM Value* 映射
    std::unordered_map<const Object*, llvm::Value*> var_map_;
    std::string expr_context_;
};

}  // namespace codegen
}  // namespace kxc

#endif  // KXC_USE_LLVM
