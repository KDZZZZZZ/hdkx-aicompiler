/*! \file src/codegen/codegen_llvm.cc
 * \brief 实现 C/LLVM codegen、LLVM JIT 和 compiled kernel 调用封装。
 */

#if KXC_USE_LLVM

#include "codegen/codegen_llvm.h"

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>

#include <algorithm>
#include <stdexcept>

namespace kxc {
namespace codegen {

CodeGenLLVM::CodeGenLLVM(llvm::LLVMContext& ctx)
    : ctx_(ctx),
      module_(std::make_unique<llvm::Module>("kxc_module", ctx)),
      builder_(ctx) {}

// ==================== 类型映射 ====================

llvm::Type* CodeGenLLVM::GetLLVMType(tir::DataType dtype) {
    if (dtype.code == 2) {  // Float
        if (dtype.bits == 32) return llvm::Type::getFloatTy(ctx_);
        if (dtype.bits == 64) return llvm::Type::getDoubleTy(ctx_);
        if (dtype.bits == 16) return llvm::Type::getHalfTy(ctx_);
    }
    if (dtype.code == 0 || dtype.code == 1) {  // Int / UInt
        return llvm::Type::getIntNTy(ctx_, dtype.bits);
    }
    if (dtype.code == 4) {  // Void
        return llvm::Type::getVoidTy(ctx_);
    }
    throw std::runtime_error("CodeGenLLVM: unsupported dtype code=" +
                             std::to_string(dtype.code) + " bits=" +
                             std::to_string(dtype.bits));
}

llvm::FunctionType* CodeGenLLVM::CreateFuncType(const tir::PrimFunc& func) {
    // ABI约定: int32_t kernel(void** packed_args)
    // packed_args[i] 指向第i个buffer的数据
    llvm::Type* void_ptr = llvm::PointerType::get(ctx_, 0);
    llvm::Type* void_ptr_ptr = llvm::PointerType::get(ctx_, 0);
    llvm::Type* ret_type = llvm::Type::getInt32Ty(ctx_);
    return llvm::FunctionType::get(ret_type, {void_ptr_ptr}, false);
}

// ==================== 主入口 ====================

void CodeGenLLVM::AddFunction(const tir::PrimFunc& func, const std::string& name) {
    var_map_.clear();

    llvm::FunctionType* ft = CreateFuncType(func);
    current_func_ = llvm::Function::Create(
        ft, llvm::Function::ExternalLinkage, name, module_.get());

    // packed_args参数
    llvm::Value* packed_args = current_func_->getArg(0);
    packed_args->setName("packed_args");

    // 创建entry block
    llvm::BasicBlock* entry = llvm::BasicBlock::Create(ctx_, "entry", current_func_);
    builder_.SetInsertPoint(entry);

    // 从packed_args中提取每个buffer参数
    for (size_t i = 0; i < func->params.size(); ++i) {
        const auto& param = func->params[i];

        // 查找buffer信息
        tir::DataType elem_dtype = tir::DataType::Float(32);
        if (func->buffer_map.count(param)) {
            elem_dtype = func->buffer_map.at(param)->dtype;
        } else {
            elem_dtype = param->dtype;
        }

        // packed_args[i] → void* → typed_ptr
        llvm::Value* idx = llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx_), i);
        llvm::Value* slot_ptr = builder_.CreateGEP(
            llvm::PointerType::get(ctx_, 0), packed_args, idx,
            "arg_slot_" + std::to_string(i));
        llvm::Value* void_ptr = builder_.CreateLoad(
            llvm::PointerType::get(ctx_, 0), slot_ptr,
            "arg_ptr_" + std::to_string(i));

        var_map_[param.get()] = void_ptr;
    }

    // 生成函数体
    GenStmt(func->body);

    // 返回0 (成功)
    if (!builder_.GetInsertBlock()->getTerminator()) {
        builder_.CreateRet(llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx_), 0));
    }

    // 验证
    std::string err;
    llvm::raw_string_ostream err_stream(err);
    if (llvm::verifyFunction(*current_func_, &err_stream)) {
        throw std::runtime_error("LLVM function verification failed: " + err);
    }
}

std::unique_ptr<llvm::Module> CodeGenLLVM::TakeModule() {
    return std::move(module_);
}

std::string CodeGenLLVM::DumpIR() const {
    std::string ir;
    llvm::raw_string_ostream os(ir);
    module_->print(os, nullptr);
    return ir;
}

// ==================== 表达式生成 ====================

llvm::Value* CodeGenLLVM::GenExpr(const tir::PrimExpr& expr) {
    if (!expr.defined()) {
        std::string message = "CodeGenLLVM: undefined expression";
        if (!expr_context_.empty()) {
            message += " in " + expr_context_;
        }
        throw std::runtime_error(message);
    }

    if (auto* n = expr.As<tir::IntImmNode>()) return GenIntImm(n);
    if (auto* n = expr.As<tir::FloatImmNode>()) return GenFloatImm(n);
    if (auto* n = expr.As<tir::VarNode>()) return GenVar(n);
    if (auto* n = expr.As<tir::BinaryOpNode>()) return GenBinaryOp(n, expr);
    if (auto* n = expr.As<tir::LoadNode>()) return GenLoad(n);
    if (auto* n = expr.As<tir::CallNode>()) return GenCall(n);
    if (auto* n = expr.As<tir::SelectNode>()) return GenSelect(n);
    if (auto* n = expr.As<tir::NotNode>()) return GenNot(n);

    throw std::runtime_error("CodeGenLLVM: unsupported expression type");
}

llvm::Value* CodeGenLLVM::GenExprInContext(const tir::PrimExpr& expr,
                                            const std::string& context) {
    std::string previous = expr_context_;
    expr_context_ = context;
    try {
        llvm::Value* value = GenExpr(expr);
        expr_context_ = previous;
        return value;
    } catch (...) {
        expr_context_ = previous;
        throw;
    }
}

llvm::Value* CodeGenLLVM::GenIntImm(const tir::IntImmNode* op) {
    llvm::Type* ty = GetLLVMType(op->dtype);
    return llvm::ConstantInt::get(ty, op->value, /*isSigned=*/op->dtype.code == 0);
}

llvm::Value* CodeGenLLVM::GenFloatImm(const tir::FloatImmNode* op) {
    llvm::Type* ty = GetLLVMType(op->dtype);
    return llvm::ConstantFP::get(ty, op->value);
}

llvm::Value* CodeGenLLVM::GenVar(const tir::VarNode* op) {
    auto it = var_map_.find(op);
    if (it == var_map_.end()) {
        throw std::runtime_error("CodeGenLLVM: undefined variable '" +
                                 op->name_hint + "'");
    }
    return it->second;
}

llvm::Type* CodeGenLLVM::CommonNumericType(llvm::Value* lhs, llvm::Value* rhs) const {
    llvm::Type* lhs_type = lhs->getType();
    llvm::Type* rhs_type = rhs->getType();
    if (lhs_type == rhs_type) {
        return lhs_type;
    }

    if (lhs_type->isFloatingPointTy() || rhs_type->isFloatingPointTy()) {
        if (lhs_type->isDoubleTy() || rhs_type->isDoubleTy()) {
            return llvm::Type::getDoubleTy(ctx_);
        }
        if (lhs_type->isFloatTy() || rhs_type->isFloatTy()) {
            return llvm::Type::getFloatTy(ctx_);
        }
        return llvm::Type::getHalfTy(ctx_);
    }

    return CommonIntegerType(lhs, rhs);
}

llvm::Type* CodeGenLLVM::CommonIntegerType(llvm::Value* lhs, llvm::Value* rhs) const {
    llvm::Type* lhs_type = lhs->getType();
    llvm::Type* rhs_type = rhs->getType();
    if (!lhs_type->isIntegerTy() || !rhs_type->isIntegerTy()) {
        throw std::runtime_error("CodeGenLLVM: expected integer operands");
    }
    unsigned bits = std::max(lhs_type->getIntegerBitWidth(), rhs_type->getIntegerBitWidth());
    return llvm::Type::getIntNTy(ctx_, bits);
}

llvm::Value* CodeGenLLVM::CastValue(llvm::Value* value, llvm::Type* target_type,
                                    bool is_signed, const std::string& name) {
    llvm::Type* source_type = value->getType();
    if (source_type == target_type) {
        return value;
    }

    if (source_type->isIntegerTy() && target_type->isIntegerTy()) {
        return is_signed ? builder_.CreateSExtOrTrunc(value, target_type, name)
                         : builder_.CreateZExtOrTrunc(value, target_type, name);
    }
    if (source_type->isIntegerTy() && target_type->isFloatingPointTy()) {
        return is_signed ? builder_.CreateSIToFP(value, target_type, name)
                         : builder_.CreateUIToFP(value, target_type, name);
    }
    if (source_type->isFloatingPointTy() && target_type->isIntegerTy()) {
        return is_signed ? builder_.CreateFPToSI(value, target_type, name)
                         : builder_.CreateFPToUI(value, target_type, name);
    }
    if (source_type->isFloatingPointTy() && target_type->isFloatingPointTy()) {
        unsigned source_bits = source_type->getPrimitiveSizeInBits();
        unsigned target_bits = target_type->getPrimitiveSizeInBits();
        if (source_bits < target_bits) {
            return builder_.CreateFPExt(value, target_type, name);
        }
        return builder_.CreateFPTrunc(value, target_type, name);
    }

    throw std::runtime_error("CodeGenLLVM: unsupported scalar cast");
}

llvm::Value* CodeGenLLVM::CastToBool(llvm::Value* value, const std::string& name) {
    llvm::Type* type = value->getType();
    if (type->isIntegerTy(1)) {
        return value;
    }
    if (type->isIntegerTy()) {
        llvm::Value* zero = llvm::ConstantInt::get(type, 0);
        return builder_.CreateICmpNE(value, zero, name);
    }
    if (type->isFloatingPointTy()) {
        llvm::Value* zero = llvm::ConstantFP::get(type, 0.0);
        return builder_.CreateFCmpONE(value, zero, name);
    }
    throw std::runtime_error("CodeGenLLVM: cannot cast value to bool");
}

void CodeGenLLVM::PromoteBinaryOperands(llvm::Value** lhs, llvm::Value** rhs,
                                        const tir::DataType& lhs_dtype,
                                        const tir::DataType& rhs_dtype,
                                        const std::string& name) {
    llvm::Type* common_type = CommonNumericType(*lhs, *rhs);
    *lhs = CastValue(*lhs, common_type, lhs_dtype.code == 0, name + ".lhs.cast");
    *rhs = CastValue(*rhs, common_type, rhs_dtype.code == 0, name + ".rhs.cast");
}

llvm::Value* CodeGenLLVM::GenBinaryOp(const tir::BinaryOpNode* op,
                                       const tir::PrimExpr& ref) {
    llvm::Value* a = GenExprInContext(op->a, "binary lhs");
    llvm::Value* b = GenExprInContext(op->b, "binary rhs");

    if (ref.As<tir::AndNode>()) {
        a = CastToBool(a, "and.lhs.bool");
        b = CastToBool(b, "and.rhs.bool");
        return builder_.CreateAnd(a, b, "and");
    }
    if (ref.As<tir::OrNode>()) {
        a = CastToBool(a, "or.lhs.bool");
        b = CastToBool(b, "or.rhs.bool");
        return builder_.CreateOr(a, b, "or");
    }

    PromoteBinaryOperands(&a, &b, op->a->dtype, op->b->dtype, "binary");
    bool is_float = a->getType()->isFloatingPointTy();
    bool is_signed = op->a->dtype.code == 0 || op->b->dtype.code == 0;

    if (ref.As<tir::AddNode>()) {
        return is_float ? builder_.CreateFAdd(a, b, "add")
                        : builder_.CreateAdd(a, b, "add");
    }
    if (ref.As<tir::SubNode>()) {
        return is_float ? builder_.CreateFSub(a, b, "sub")
                        : builder_.CreateSub(a, b, "sub");
    }
    if (ref.As<tir::MulNode>()) {
        return is_float ? builder_.CreateFMul(a, b, "mul")
                        : builder_.CreateMul(a, b, "mul");
    }
    if (ref.As<tir::DivNode>()) {
        if (is_float) return builder_.CreateFDiv(a, b, "div");
        return is_signed ? builder_.CreateSDiv(a, b, "div")
                         : builder_.CreateUDiv(a, b, "div");
    }
    if (ref.As<tir::ModNode>()) {
        if (is_float) return builder_.CreateFRem(a, b, "mod");
        return is_signed ? builder_.CreateSRem(a, b, "mod")
                         : builder_.CreateURem(a, b, "mod");
    }
    if (ref.As<tir::MinNode>()) {
        if (is_float) {
            llvm::Value* cmp = builder_.CreateFCmpOLT(a, b, "cmp");
            return builder_.CreateSelect(cmp, a, b, "min");
        }
        llvm::Value* cmp = is_signed ? builder_.CreateICmpSLT(a, b, "cmp")
                                     : builder_.CreateICmpULT(a, b, "cmp");
        return builder_.CreateSelect(cmp, a, b, "min");
    }
    if (ref.As<tir::MaxNode>()) {
        if (is_float) {
            llvm::Value* cmp = builder_.CreateFCmpOGT(a, b, "cmp");
            return builder_.CreateSelect(cmp, a, b, "max");
        }
        llvm::Value* cmp = is_signed ? builder_.CreateICmpSGT(a, b, "cmp")
                                     : builder_.CreateICmpUGT(a, b, "cmp");
        return builder_.CreateSelect(cmp, a, b, "max");
    }
    if (ref.As<tir::EQNode>()) {
        return is_float ? builder_.CreateFCmpOEQ(a, b, "eq")
                        : builder_.CreateICmpEQ(a, b, "eq");
    }
    if (ref.As<tir::LTNode>()) {
        if (is_float) return builder_.CreateFCmpOLT(a, b, "lt");
        return is_signed ? builder_.CreateICmpSLT(a, b, "lt")
                         : builder_.CreateICmpULT(a, b, "lt");
    }

    throw std::runtime_error("CodeGenLLVM: unsupported binary op");
}

llvm::Value* CodeGenLLVM::GenLoad(const tir::LoadNode* op) {
    llvm::Value* buf_ptr = GenExprInContext(tir::PrimExpr(ObjectRef(op->buffer_var)),
                                            "load buffer");
    llvm::Value* index = GenExprInContext(op->index, "load index");
    llvm::Type* elem_type = GetLLVMType(op->dtype);

    llvm::Value* ptr = builder_.CreateGEP(elem_type, buf_ptr, index, "load_ptr");
    llvm::Value* val = builder_.CreateLoad(elem_type, ptr, "load_val");

    if (op->predicate.defined()) {
        llvm::Value* pred = GenExprInContext(op->predicate, "load predicate");
        pred = CastToBool(pred, "load.predicate.bool");
        llvm::Value* zero = llvm::Constant::getNullValue(elem_type);
        val = builder_.CreateSelect(pred, val, zero, "pred_load");
    }
    return val;
}

llvm::Value* CodeGenLLVM::GenCall(const tir::CallNode* op) {
    // 内置函数映射
    llvm::Type* ret_type = GetLLVMType(op->dtype);

    std::vector<llvm::Value*> args;
    for (const auto& arg : op->args) {
        llvm::Value* arg_value = GenExprInContext(arg, "call argument");
        args.push_back(CastValue(arg_value, ret_type, arg->dtype.code == 0, "call.arg.cast"));
    }

    llvm::Function* callee = GetOrDeclareIntrinsic(op->name, ret_type);
    return builder_.CreateCall(callee, args, "call_" + op->name);
}

llvm::Value* CodeGenLLVM::GenSelect(const tir::SelectNode* op) {
    llvm::Value* cond = GenExprInContext(op->condition, "select condition");
    cond = CastToBool(cond, "select.condition.bool");
    llvm::Value* tv = GenExprInContext(op->true_value, "select true value");
    llvm::Value* fv = GenExprInContext(op->false_value, "select false value");
    PromoteBinaryOperands(&tv, &fv, op->true_value->dtype, op->false_value->dtype,
                          "select.value");
    return builder_.CreateSelect(cond, tv, fv, "sel");
}

llvm::Value* CodeGenLLVM::GenNot(const tir::NotNode* op) {
    llvm::Value* val = GenExprInContext(op->value, "not value");
    val = CastToBool(val, "not.value.bool");
    return builder_.CreateNot(val, "not");
}

llvm::Function* CodeGenLLVM::GetOrDeclareIntrinsic(const std::string& name,
                                                    llvm::Type* type) {
    // 尝试查找已有声明
    llvm::Function* f = module_->getFunction(name);
    if (f) return f;

    // 简单映射: tir内置函数 → C数学函数
    std::string c_name = name;
    if (name == "tir.exp") c_name = "expf";
    else if (name == "tir.log") c_name = "logf";
    else if (name == "tir.sqrt") c_name = "sqrtf";
    else if (name == "tir.tanh") c_name = "tanhf";
    else if (name == "tir.floor") c_name = "floorf";
    else if (name == "tir.ceil") c_name = "ceilf";
    else if (name == "tir.fabs") c_name = "fabsf";

    // 创建函数声明（假设单参数数学函数）
    llvm::FunctionType* ft = llvm::FunctionType::get(type, {type}, false);
    f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                               c_name, module_.get());
    return f;
}

// ==================== 语句生成 ====================

void CodeGenLLVM::GenStmt(const tir::Stmt& stmt) {
    if (!stmt.defined()) return;

    if (auto* n = stmt.As<tir::ForNode>()) return GenFor(n);
    if (auto* n = stmt.As<tir::StoreNode>()) return GenStore(n);
    if (auto* n = stmt.As<tir::AllocateNode>()) return GenAllocate(n);
    if (auto* n = stmt.As<tir::IfThenElseNode>()) return GenIfThenElse(n);
    if (auto* n = stmt.As<tir::LetStmtNode>()) return GenLetStmt(n);
    if (auto* n = stmt.As<tir::SeqStmtNode>()) return GenSeqStmt(n);
    if (auto* n = stmt.As<tir::EvaluateNode>()) return GenEvaluate(n);

    throw std::runtime_error("CodeGenLLVM: unsupported statement type");
}

void CodeGenLLVM::GenFor(const tir::ForNode* op) {
    llvm::Function* func = current_func_;

    // 创建循环的BasicBlock结构
    llvm::BasicBlock* preheader = builder_.GetInsertBlock();
    llvm::BasicBlock* header = llvm::BasicBlock::Create(ctx_, "for.header", func);
    llvm::BasicBlock* body = llvm::BasicBlock::Create(ctx_, "for.body", func);
    llvm::BasicBlock* exit = llvm::BasicBlock::Create(ctx_, "for.exit", func);

    // preheader → header
    llvm::Value* init = GenExprInContext(op->min, "for min");
    llvm::Value* extent = GenExprInContext(op->extent, "for extent");
    llvm::Type* loop_type = CommonIntegerType(init, extent);
    llvm::Type* declared_loop_type = GetLLVMType(op->loop_var->dtype);
    if (!declared_loop_type->isIntegerTy()) {
        throw std::runtime_error("CodeGenLLVM: for loop var must be integer");
    }
    unsigned loop_bits = std::max(loop_type->getIntegerBitWidth(),
                                  declared_loop_type->getIntegerBitWidth());
    loop_type = llvm::Type::getIntNTy(ctx_, loop_bits);
    const bool loop_signed = op->loop_var->dtype.code == 0;
    init = CastValue(init, loop_type, loop_signed, "for.min.cast");
    extent = CastValue(extent, loop_type, loop_signed, "for.extent.cast");
    builder_.CreateBr(header);

    // header: phi, cmp, br
    builder_.SetInsertPoint(header);
    llvm::PHINode* phi = builder_.CreatePHI(loop_type, 2, op->loop_var->name_hint);
    phi->addIncoming(init, preheader);

    var_map_[op->loop_var.get()] = phi;

    llvm::Value* end = builder_.CreateAdd(init, extent, "end");
    llvm::Value* cond = loop_signed ? builder_.CreateICmpSLT(phi, end, "for.cond")
                                    : builder_.CreateICmpULT(phi, end, "for.cond");
    builder_.CreateCondBr(cond, body, exit);

    // body
    builder_.SetInsertPoint(body);
    GenStmt(op->body);

    // increment and loop back
    if (!builder_.GetInsertBlock()->getTerminator()) {
        llvm::Value* next = builder_.CreateAdd(
            phi, llvm::ConstantInt::get(loop_type, 1), "next");
        phi->addIncoming(next, builder_.GetInsertBlock());
        builder_.CreateBr(header);
    }

    // continue at exit
    builder_.SetInsertPoint(exit);
}

void CodeGenLLVM::GenStore(const tir::StoreNode* op) {
    llvm::Value* buf_ptr = GenExprInContext(tir::PrimExpr(ObjectRef(op->buffer_var)),
                                            "store buffer");
    llvm::Value* value = GenExprInContext(op->value, "store value");
    llvm::Value* index = GenExprInContext(op->index, "store index");

    // 确定元素类型
    llvm::Type* elem_type = value->getType();
    llvm::Value* ptr = builder_.CreateGEP(elem_type, buf_ptr, index, "store_ptr");

    if (op->predicate.defined()) {
        llvm::Value* pred = GenExprInContext(op->predicate, "store predicate");
        pred = CastToBool(pred, "store.predicate.bool");
        llvm::BasicBlock* store_bb =
            llvm::BasicBlock::Create(ctx_, "pred.store", current_func_);
        llvm::BasicBlock* merge_bb =
            llvm::BasicBlock::Create(ctx_, "pred.merge", current_func_);
        builder_.CreateCondBr(pred, store_bb, merge_bb);

        builder_.SetInsertPoint(store_bb);
        builder_.CreateStore(value, ptr);
        builder_.CreateBr(merge_bb);

        builder_.SetInsertPoint(merge_bb);
    } else {
        builder_.CreateStore(value, ptr);
    }
}

void CodeGenLLVM::GenAllocate(const tir::AllocateNode* op) {
    // 计算总大小
    llvm::Type* elem_type = GetLLVMType(op->dtype);
    llvm::Value* total_size =
        llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx_), 1);
    for (const auto& ext : op->extents) {
        llvm::Value* e = GenExprInContext(ext, "allocate extent");
        if (e->getType() != llvm::Type::getInt64Ty(ctx_)) {
            e = builder_.CreateSExt(e, llvm::Type::getInt64Ty(ctx_), "ext64");
        }
        total_size = builder_.CreateMul(total_size, e, "alloc_size");
    }

    // 栈分配（alloca）
    llvm::Value* alloc = builder_.CreateAlloca(elem_type, total_size,
                                               op->buffer_var->name_hint);
    var_map_[op->buffer_var.get()] = alloc;

    GenStmt(op->body);
}

void CodeGenLLVM::GenIfThenElse(const tir::IfThenElseNode* op) {
    llvm::Value* cond = GenExprInContext(op->condition, "if condition");
    cond = CastToBool(cond, "if.condition.bool");

    llvm::BasicBlock* then_bb =
        llvm::BasicBlock::Create(ctx_, "if.then", current_func_);
    llvm::BasicBlock* merge_bb =
        llvm::BasicBlock::Create(ctx_, "if.merge", current_func_);
    llvm::BasicBlock* else_bb =
        op->else_case.defined()
            ? llvm::BasicBlock::Create(ctx_, "if.else", current_func_)
            : merge_bb;

    builder_.CreateCondBr(cond, then_bb, else_bb);

    builder_.SetInsertPoint(then_bb);
    GenStmt(op->then_case);
    if (!builder_.GetInsertBlock()->getTerminator()) {
        builder_.CreateBr(merge_bb);
    }

    if (op->else_case.defined()) {
        builder_.SetInsertPoint(else_bb);
        GenStmt(op->else_case);
        if (!builder_.GetInsertBlock()->getTerminator()) {
            builder_.CreateBr(merge_bb);
        }
    }

    builder_.SetInsertPoint(merge_bb);
}

void CodeGenLLVM::GenLetStmt(const tir::LetStmtNode* op) {
    llvm::Value* val = GenExprInContext(op->value, "let value");
    var_map_[op->var.get()] = val;
    GenStmt(op->body);
}

void CodeGenLLVM::GenSeqStmt(const tir::SeqStmtNode* op) {
    for (const auto& stmt : op->seq) {
        GenStmt(stmt);
    }
}

void CodeGenLLVM::GenEvaluate(const tir::EvaluateNode* op) {
    if (!op->value.defined()) {
        return;
    }
    GenExprInContext(op->value, "evaluate value");
}

}  // namespace codegen
}  // namespace kxc

#endif  // KXC_USE_LLVM
