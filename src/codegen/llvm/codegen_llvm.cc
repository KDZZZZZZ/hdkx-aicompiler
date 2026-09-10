/*! \file src/codegen/llvm/codegen_llvm.cc
 * \brief 实现 C/LLVM codegen、LLVM JIT 和 compiled kernel 调用封装。
 */

#if KXC_USE_LLVM

#include "internal/codegen_llvm.h"

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Intrinsics.h>
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
    (void)func;
    // 私有 ABI 固定为 i32 kernel(void** packed_args, uint64_t count)。
    llvm::Type* void_ptr_ptr = llvm::PointerType::get(ctx_, 0);
    llvm::Type* count_type = llvm::Type::getInt64Ty(ctx_);
    llvm::Type* ret_type = llvm::Type::getInt32Ty(ctx_);
    return llvm::FunctionType::get(ret_type, {void_ptr_ptr, count_type}, false);
}

// ==================== 主入口 ====================

void CodeGenLLVM::AddFunction(const tir::PrimFunc& func, const std::string& name) {
    var_map_.clear();

    llvm::FunctionType* ft = CreateFuncType(func);
    current_func_ = llvm::Function::Create(
        ft, llvm::Function::ExternalLinkage, name, module_.get());

    // 两个入口参数共同描述 call frame；任何 slot 读取都必须晚于门禁。
    llvm::Value* packed_args = current_func_->getArg(0);
    packed_args->setName("packed_args");
    llvm::Value* argument_count = current_func_->getArg(1);
    argument_count->setName("argument_count");

    // 参数数量不匹配返回 -1；非空帧却传入 null 返回 -2。
    llvm::BasicBlock* entry = llvm::BasicBlock::Create(ctx_, "entry", current_func_);
    llvm::BasicBlock* check_pointer =
        llvm::BasicBlock::Create(ctx_, "check_pointer", current_func_);
    llvm::BasicBlock* invalid_count =
        llvm::BasicBlock::Create(ctx_, "invalid_count", current_func_);
    llvm::BasicBlock* invalid_pointer =
        llvm::BasicBlock::Create(ctx_, "invalid_pointer", current_func_);
    llvm::BasicBlock* body = llvm::BasicBlock::Create(ctx_, "body", current_func_);
    builder_.SetInsertPoint(entry);
    llvm::Value* expected_count = llvm::ConstantInt::get(
        llvm::Type::getInt64Ty(ctx_), static_cast<uint64_t>(func->params.size()));
    builder_.CreateCondBr(builder_.CreateICmpEQ(argument_count, expected_count),
                          check_pointer, invalid_count);

    builder_.SetInsertPoint(invalid_count);
    builder_.CreateRet(llvm::ConstantInt::getSigned(
        llvm::Type::getInt32Ty(ctx_), -1));

    builder_.SetInsertPoint(check_pointer);
    llvm::Value* null_frame = llvm::ConstantPointerNull::get(
        llvm::PointerType::get(ctx_, 0));
    llvm::Value* frame_is_valid =
        func->params.empty()
            ? llvm::ConstantInt::getTrue(ctx_)
            : builder_.CreateICmpNE(packed_args, null_frame);
    builder_.CreateCondBr(frame_is_valid, body, invalid_pointer);

    builder_.SetInsertPoint(invalid_pointer);
    builder_.CreateRet(llvm::ConstantInt::getSigned(
        llvm::Type::getInt32Ty(ctx_), -2));

    builder_.SetInsertPoint(body);

    // 只有 count 与 frame 指针均合法后，才提取每个 buffer 地址。
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

void CodeGenLLVM::AddFunctions(
    const std::vector<std::pair<tir::PrimFunc, std::string>>& functions) {
    if (functions.empty()) {
        throw std::invalid_argument(
            "CodeGenLLVM AddFunctions requires at least one PrimFunc");
    }
    for (const auto& function : functions) {
        AddFunction(function.first, function.second);
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
    llvm::Type* ret_type = GetLLVMType(op->dtype);

    if (op->name == "tanh" || op->name == "erf") {
        if (op->args.size() != 1 || op->dtype != tir::DataType::Float(32) ||
            op->args[0]->dtype != op->dtype) {
            throw std::runtime_error("CodeGenLLVM: " + op->name +
                                     " requires exactly one float32 argument and result");
        }
        // Exact C ABI; the existing ORC host-symbol generator resolves libm.
        // Do not approximate tanh with exp (overflow) or erf with a polynomial.
        auto callee = module_->getOrInsertFunction(
            op->name + "f", llvm::FunctionType::get(ret_type, {ret_type}, false));
        return builder_.CreateCall(callee, {GenExprInContext(op->args[0], "libm argument")},
                                   "call_" + op->name);
    }

    if (op->name == "cast") {
        if (op->args.size() != 1) {
            throw std::runtime_error("CodeGenLLVM: cast expects exactly one argument");
        }
        llvm::Value* value = GenExprInContext(op->args[0], "cast argument");
        bool is_signed = op->args[0]->dtype.code == 0;
        if (op->args[0]->dtype.code == 2 && op->dtype.code != 2) {
            is_signed = op->dtype.code == 0;
        }
        return CastValue(value, ret_type, is_signed, "cast");
    }

    if (op->name == "pow") {
        // M5 S2 backend 分派合同：参数数量、操作数类型和声明函数在此显式固定；
        // 不是只让某个名字出现在 IR 里。非有限结果按 llvm.pow/C99 pow 声明处理
        // （如 pow(+0,-1)=+inf、pow(负底数, 非整数)=NaN）。
        if (op->args.size() != 2) {
            throw std::runtime_error("CodeGenLLVM: pow expects exactly two arguments, got " +
                                     std::to_string(op->args.size()));
        }
        if (op->dtype.code != 2 || op->args[0]->dtype.code != 2 ||
            op->args[1]->dtype.code != 2) {
            throw std::runtime_error(
                "CodeGenLLVM: pow requires floating-point base and exponent");
        }
        if (op->args[0]->dtype.bits != op->dtype.bits ||
            op->args[1]->dtype.bits != op->dtype.bits) {
            throw std::runtime_error(
                "CodeGenLLVM: pow operand bits must match the declared result dtype");
        }
        llvm::Value* base = GenExprInContext(op->args[0], "pow base");
        llvm::Value* exponent = GenExprInContext(op->args[1], "pow exponent");
        llvm::Function* callee = GetOrDeclareIntrinsic("pow", ret_type);
        return builder_.CreateCall(callee, {base, exponent}, "call_pow");
    }

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

    llvm::BasicBlock* then_bb =
        llvm::BasicBlock::Create(ctx_, "select.then", current_func_);
    llvm::BasicBlock* else_bb =
        llvm::BasicBlock::Create(ctx_, "select.else", current_func_);
    llvm::BasicBlock* merge_bb =
        llvm::BasicBlock::Create(ctx_, "select.merge", current_func_);
    builder_.CreateCondBr(cond, then_bb, else_bb);

    builder_.SetInsertPoint(then_bb);
    llvm::Value* tv = GenExprInContext(op->true_value, "select true value");
    llvm::BasicBlock* then_end = builder_.GetInsertBlock();
    if (!then_end->getTerminator()) {
        builder_.CreateBr(merge_bb);
    }

    builder_.SetInsertPoint(else_bb);
    llvm::Value* fv = GenExprInContext(op->false_value, "select false value");
    llvm::BasicBlock* else_end = builder_.GetInsertBlock();
    if (!else_end->getTerminator()) {
        builder_.CreateBr(merge_bb);
    }

    llvm::Type* result_type = CommonNumericType(tv, fv);
    auto cast_in_block = [&](llvm::Value* value, llvm::BasicBlock* block,
                             const tir::DataType& dtype,
                             const std::string& name) -> llvm::Value* {
        if (value->getType() == result_type) {
            return value;
        }
        llvm::IRBuilder<>::InsertPoint saved = builder_.saveIP();
        if (llvm::Instruction* terminator = block->getTerminator()) {
            builder_.SetInsertPoint(terminator);
        } else {
            builder_.SetInsertPoint(block);
        }
        llvm::Value* casted = CastValue(value, result_type, dtype.code == 0, name);
        builder_.restoreIP(saved);
        return casted;
    };
    tv = cast_in_block(tv, then_end, op->true_value->dtype, "select.true.cast");
    fv = cast_in_block(fv, else_end, op->false_value->dtype, "select.false.cast");

    builder_.SetInsertPoint(merge_bb);
    llvm::PHINode* phi = builder_.CreatePHI(result_type, 2, "sel");
    phi->addIncoming(tv, then_end);
    phi->addIncoming(fv, else_end);
    return phi;
}

llvm::Value* CodeGenLLVM::GenNot(const tir::NotNode* op) {
    llvm::Value* val = GenExprInContext(op->value, "not value");
    val = CastToBool(val, "not.value.bool");
    return builder_.CreateNot(val, "not");
}

llvm::Function* CodeGenLLVM::GetOrDeclareIntrinsic(const std::string& name,
                                                    llvm::Type* type) {
    if (!type->isFloatingPointTy()) {
        throw std::runtime_error("CodeGenLLVM: math call '" + name +
                                 "' requires floating-point dtype");
    }

    llvm::Intrinsic::ID id = llvm::Intrinsic::not_intrinsic;
    if (name == "exp" || name == "tir.exp") {
        id = llvm::Intrinsic::exp;
    } else if (name == "log" || name == "tir.log") {
        id = llvm::Intrinsic::log;
    } else if (name == "sqrt" || name == "tir.sqrt") {
        id = llvm::Intrinsic::sqrt;
    } else if (name == "floor" || name == "tir.floor") {
        id = llvm::Intrinsic::floor;
    } else if (name == "ceil" || name == "tir.ceil") {
        id = llvm::Intrinsic::ceil;
    } else if (name == "fabs" || name == "tir.fabs") {
        id = llvm::Intrinsic::fabs;
    } else if (name == "pow" || name == "tir.pow") {
        id = llvm::Intrinsic::pow;
    }

    if (id == llvm::Intrinsic::not_intrinsic) {
        throw std::runtime_error("CodeGenLLVM: unsupported call '" + name + "'");
    }
    llvm::FunctionCallee callee =
        llvm::Intrinsic::getOrInsertDeclaration(module_.get(), id, {type});
    return llvm::cast<llvm::Function>(callee.getCallee());
}

}  // namespace codegen
}  // namespace kxc

#endif  // KXC_USE_LLVM
