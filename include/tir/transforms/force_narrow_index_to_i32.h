#pragma once

#include "tir/stmt.h"

namespace kxc {
namespace tir {

PrimFunc ForceNarrowIndexToI32Pass(const PrimFunc& func);

}  // namespace tir
}  // namespace kxc

