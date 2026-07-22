/*! \file include/kxc/tir/transforms/vectorize_loop.h
 * \brief 声明 TIR 优化 pass。
 */

#pragma once

#include "kxc/tir/stmt.h"

namespace kxc {
namespace tir {

PrimFunc VectorizeLoopPass(const PrimFunc& func);

}  // namespace tir
}  // namespace kxc
