/*! \file src/tir/internal/static_integer.h
 * \brief Shared checked evaluation of existing static TIR integer expressions.
 */
#pragma once

#include "kxc/tir/expr.h"

namespace kxc::tir::internal {
bool EvaluateStaticInt64(const PrimExpr& expression, int64_t* result);
}
