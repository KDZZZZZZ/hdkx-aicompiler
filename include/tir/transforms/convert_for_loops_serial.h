#pragma once

#include "tir/stmt.h"

namespace kxc {
namespace tir {

PrimFunc ConvertForLoopsSerialPass(const PrimFunc& func);

}  // namespace tir
}  // namespace kxc
