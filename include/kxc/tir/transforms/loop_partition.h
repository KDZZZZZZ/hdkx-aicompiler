/*! \file include/kxc/tir/transforms/loop_partition.h
 * \brief 声明 TIR 优化 pass。
 */

#pragma once

#include "kxc/tir/stmt.h"

namespace kxc {
namespace tir {

PrimFunc LoopPartitionPass(const PrimFunc& func);

}  // namespace tir
}  // namespace kxc
