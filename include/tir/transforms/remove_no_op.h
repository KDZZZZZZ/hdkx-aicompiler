#pragma once

#include "tir/stmt.h"

namespace kxc {
namespace tir {

PrimFunc RemoveNoOpPass(const PrimFunc& func);

}  // namespace tir
}  // namespace kxc
