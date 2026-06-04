/*! \file include/tir/transforms/fold_constant.h
 * \brief 声明 TIR 优化 pass。
 */

#pragma once

#include "tir/stmt.h"

namespace kxc {
namespace tir {

PrimFunc FoldConstantPass(const PrimFunc& func);

}  // namespace tir
}  // namespace kxc
