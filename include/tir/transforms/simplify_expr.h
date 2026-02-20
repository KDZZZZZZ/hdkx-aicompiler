#pragma once

#include "tir/stmt.h"

namespace kxc {
namespace tir {

PrimFunc SimplifyExprPass(const PrimFunc& func);

}  // namespace tir
}  // namespace kxc
