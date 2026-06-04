/*! \file include/tir/pass_utils.h
 * \brief 定义 TIR PrimExpr、Stmt、PrimFunc 和 pass 工具。
 */

#pragma once

#include "tir/stmt.h"

namespace kxc {
namespace tir {
namespace pass_utils {

bool TryGetConstValue(const PrimExpr& expr, double* out_value);
bool TryGetConstInt64(const PrimExpr& expr, int64_t* out_value);
bool IsConstZero(const PrimExpr& expr);
bool IsConstOne(const PrimExpr& expr);

bool IsNoOpStmt(const Stmt& stmt);
bool ContainsFor(const Stmt& stmt);
bool IsSimpleStraightLineStmt(const Stmt& stmt);

// Flattens nested SeqStmt and drops no-op statements.
void AppendFlattenedStmt(const Stmt& stmt, Array<Stmt>* out);

// Returns undefined stmt for empty list, single stmt for size 1, otherwise SeqStmt.
Stmt MakeSeqStmtOrUnit(const Array<Stmt>& seq);

bool CanNarrowInt64ToInt32(int64_t value);
PrimExpr NarrowIntImmToInt32IfPossible(const PrimExpr& expr);

}  // namespace pass_utils
}  // namespace tir
}  // namespace kxc
