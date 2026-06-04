/*! \file include/tir/transforms/unroll_loop.h
 * \brief 声明 TIR 优化 pass。
 */

#pragma once

#include "tir/stmt.h"

namespace kxc {
namespace tir {

PrimFunc UnrollLoopPass(const PrimFunc& func);

}  // namespace tir
}  // namespace kxc

