#pragma once

#include "tir/stmt.h"

namespace kxc {
namespace tir {

PrimFunc UnrollLoopPass(const PrimFunc& func);

}  // namespace tir
}  // namespace kxc

