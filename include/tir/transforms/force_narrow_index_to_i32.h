/*! \file include/tir/transforms/force_narrow_index_to_i32.h
 * \brief 声明 TIR 优化 pass。
 */

#pragma once

#include "tir/stmt.h"

namespace kxc {
namespace tir {

PrimFunc ForceNarrowIndexToI32Pass(const PrimFunc& func);

}  // namespace tir
}  // namespace kxc

