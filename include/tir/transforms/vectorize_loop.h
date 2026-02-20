#pragma once

#include "tir/stmt.h"

namespace kxc {
namespace tir {

PrimFunc VectorizeLoopPass(const PrimFunc& func);

}  // namespace tir
}  // namespace kxc

