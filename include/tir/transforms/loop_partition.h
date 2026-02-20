#pragma once

#include "tir/stmt.h"

namespace kxc {
namespace tir {

PrimFunc LoopPartitionPass(const PrimFunc& func);

}  // namespace tir
}  // namespace kxc

