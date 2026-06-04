/*! \file include/tir/transforms/remove_no_op.h
 * \brief 声明 TIR 优化 pass。
 */

#pragma once

#include "tir/stmt.h"

namespace kxc {
namespace tir {

PrimFunc RemoveNoOpPass(const PrimFunc& func);

}  // namespace tir
}  // namespace kxc
