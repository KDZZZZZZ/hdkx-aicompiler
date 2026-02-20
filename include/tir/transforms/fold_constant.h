#pragma once

#include "tir/stmt.h"

namespace kxc {
namespace tir {

PrimFunc FoldConstantPass(const PrimFunc& func);

}  // namespace tir
}  // namespace kxc
