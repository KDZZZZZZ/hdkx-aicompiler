/*! \file src/codegen/llvm/codegen_llvm_stmt.cc
 * \brief Lowers TIR statements into LLVM control flow and memory operations.
 */

#if KXC_USE_LLVM

#include "internal/codegen_llvm.h"

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Type.h>

#include <algorithm>
#include <stdexcept>

namespace kxc::codegen {
namespace {

uint64_t StorageBytes(tir::DataType dtype) {
  return std::max<uint64_t>(1, (static_cast<uint64_t>(dtype.bits) + 7) / 8);
}

}  // namespace

void CodeGenLLVM::GenStmt(const tir::Stmt& stmt) {
  if (!stmt.defined()) return;
  if (auto* node = stmt.As<tir::ForNode>()) return GenFor(node);
  if (auto* node = stmt.As<tir::StoreNode>()) return GenStore(node);
  if (auto* node = stmt.As<tir::AllocateNode>()) return GenAllocate(node);
  if (auto* node = stmt.As<tir::IfThenElseNode>()) return GenIfThenElse(node);
  if (auto* node = stmt.As<tir::LetStmtNode>()) return GenLetStmt(node);
  if (auto* node = stmt.As<tir::SeqStmtNode>()) return GenSeqStmt(node);
  if (auto* node = stmt.As<tir::EvaluateNode>()) return GenEvaluate(node);
  throw std::runtime_error("CodeGenLLVM: unsupported statement type");
}

void CodeGenLLVM::GenFor(const tir::ForNode* op) {
  llvm::Function* function = current_func_;
  llvm::BasicBlock* preheader = builder_.GetInsertBlock();
  llvm::BasicBlock* header = llvm::BasicBlock::Create(ctx_, "for.header", function);
  llvm::BasicBlock* body = llvm::BasicBlock::Create(ctx_, "for.body", function);
  llvm::BasicBlock* exit = llvm::BasicBlock::Create(ctx_, "for.exit", function);

  llvm::Value* init = GenExprInContext(op->min, "for min");
  llvm::Value* extent = GenExprInContext(op->extent, "for extent");
  llvm::Type* loop_type = CommonIntegerType(init, extent);
  llvm::Type* declared_loop_type = GetLLVMType(op->loop_var->dtype);
  if (!declared_loop_type->isIntegerTy()) {
    throw std::runtime_error("CodeGenLLVM: for loop var must be integer");
  }
  const unsigned loop_bits =
      std::max(loop_type->getIntegerBitWidth(), declared_loop_type->getIntegerBitWidth());
  loop_type = llvm::Type::getIntNTy(ctx_, loop_bits);
  const bool loop_signed = op->loop_var->dtype.code == 0;
  init = CastValue(init, loop_type, loop_signed, "for.min.cast");
  extent = CastValue(extent, loop_type, loop_signed, "for.extent.cast");
  builder_.CreateBr(header);

  builder_.SetInsertPoint(header);
  llvm::PHINode* phi = builder_.CreatePHI(loop_type, 2, op->loop_var->name_hint);
  phi->addIncoming(init, preheader);
  var_map_[op->loop_var.get()] = phi;
  llvm::Value* end = builder_.CreateAdd(init, extent, "end");
  llvm::Value* condition =
      loop_signed ? builder_.CreateICmpSLT(phi, end, "for.cond")
                  : builder_.CreateICmpULT(phi, end, "for.cond");
  builder_.CreateCondBr(condition, body, exit);

  builder_.SetInsertPoint(body);
  GenStmt(op->body);
  if (!builder_.GetInsertBlock()->getTerminator()) {
    llvm::Value* next =
        builder_.CreateAdd(phi, llvm::ConstantInt::get(loop_type, 1), "next");
    phi->addIncoming(next, builder_.GetInsertBlock());
    builder_.CreateBr(header);
  }
  builder_.SetInsertPoint(exit);
}

void CodeGenLLVM::GenStore(const tir::StoreNode* op) {
  llvm::Value* buffer =
      GenExprInContext(tir::PrimExpr(ObjectRef(op->buffer_var)), "store buffer");
  llvm::Value* value = GenExprInContext(op->value, "store value");
  llvm::Value* index = GenExprInContext(op->index, "store index");
  llvm::Value* pointer = builder_.CreateGEP(value->getType(), buffer, index, "store_ptr");

  if (!op->predicate.defined()) {
    builder_.CreateStore(value, pointer);
    return;
  }
  llvm::Value* predicate =
      CastToBool(GenExprInContext(op->predicate, "store predicate"),
                 "store.predicate.bool");
  llvm::BasicBlock* store =
      llvm::BasicBlock::Create(ctx_, "pred.store", current_func_);
  llvm::BasicBlock* merge =
      llvm::BasicBlock::Create(ctx_, "pred.merge", current_func_);
  builder_.CreateCondBr(predicate, store, merge);
  builder_.SetInsertPoint(store);
  builder_.CreateStore(value, pointer);
  builder_.CreateBr(merge);
  builder_.SetInsertPoint(merge);
}

void CodeGenLLVM::GenAllocate(const tir::AllocateNode* op) {
  llvm::Value* total_size = llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx_), 1);
  for (const auto& extent : op->extents) {
    llvm::Value* value = GenExprInContext(extent, "allocate extent");
    if (value->getType() != llvm::Type::getInt64Ty(ctx_)) {
      value = builder_.CreateSExt(value, llvm::Type::getInt64Ty(ctx_), "ext64");
    }
    total_size = builder_.CreateMul(total_size, value, "alloc_size");
  }

  llvm::Type* i64 = llvm::Type::getInt64Ty(ctx_);
  llvm::Type* void_pointer = llvm::PointerType::get(ctx_, 0);
  llvm::Value* bytes_per_element = llvm::ConstantInt::get(i64, StorageBytes(op->dtype));
  llvm::Value* total_bytes =
      builder_.CreateMul(total_size, bytes_per_element, "alloc_bytes");
  llvm::FunctionType* malloc_type =
      llvm::FunctionType::get(void_pointer, {i64}, false);
  llvm::FunctionCallee malloc_function =
      module_->getOrInsertFunction("malloc", malloc_type);
  llvm::Value* allocation = builder_.CreateCall(
      malloc_function, {total_bytes}, op->buffer_var->name_hint + ".heap");

  auto previous = var_map_.find(op->buffer_var.get());
  llvm::Value* previous_value = previous == var_map_.end() ? nullptr : previous->second;
  var_map_[op->buffer_var.get()] = allocation;
  GenStmt(op->body);

  if (!builder_.GetInsertBlock()->getTerminator()) {
    llvm::FunctionType* free_type =
        llvm::FunctionType::get(llvm::Type::getVoidTy(ctx_), {void_pointer}, false);
    llvm::FunctionCallee free_function = module_->getOrInsertFunction("free", free_type);
    builder_.CreateCall(free_function, {allocation});
  }
  if (previous_value) {
    var_map_[op->buffer_var.get()] = previous_value;
  } else {
    var_map_.erase(op->buffer_var.get());
  }
}

void CodeGenLLVM::GenIfThenElse(const tir::IfThenElseNode* op) {
  llvm::Value* condition =
      CastToBool(GenExprInContext(op->condition, "if condition"), "if.condition.bool");
  llvm::BasicBlock* then_block =
      llvm::BasicBlock::Create(ctx_, "if.then", current_func_);
  llvm::BasicBlock* merge_block =
      llvm::BasicBlock::Create(ctx_, "if.merge", current_func_);
  llvm::BasicBlock* else_block =
      op->else_case.defined()
          ? llvm::BasicBlock::Create(ctx_, "if.else", current_func_)
          : merge_block;
  builder_.CreateCondBr(condition, then_block, else_block);

  builder_.SetInsertPoint(then_block);
  GenStmt(op->then_case);
  if (!builder_.GetInsertBlock()->getTerminator()) builder_.CreateBr(merge_block);
  if (op->else_case.defined()) {
    builder_.SetInsertPoint(else_block);
    GenStmt(op->else_case);
    if (!builder_.GetInsertBlock()->getTerminator()) builder_.CreateBr(merge_block);
  }
  builder_.SetInsertPoint(merge_block);
}

void CodeGenLLVM::GenLetStmt(const tir::LetStmtNode* op) {
  var_map_[op->var.get()] = GenExprInContext(op->value, "let value");
  GenStmt(op->body);
}

void CodeGenLLVM::GenSeqStmt(const tir::SeqStmtNode* op) {
  for (const auto& statement : op->seq) GenStmt(statement);
}

void CodeGenLLVM::GenEvaluate(const tir::EvaluateNode* op) {
  if (op->value.defined()) GenExprInContext(op->value, "evaluate value");
}

}  // namespace kxc::codegen

#endif  // KXC_USE_LLVM
