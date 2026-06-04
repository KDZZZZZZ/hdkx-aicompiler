/*! \file include/tir/transforms/simplify_expr.h
 * \brief 声明 TIR 优化 pass。
 */

#pragma once

#include "tir/stmt.h"

namespace kxc {
namespace tir {

PrimFunc SimplifyExprPass(const PrimFunc& func);

}  // namespace tir
}  // namespace kxc
