#pragma once

#include "base/container.h"
#include "tir/stmt.h"

namespace kxc {
namespace tir {

PrimFunc RunTIRPassPipeline(const PrimFunc& func, const Array<String>& pass_names);

}  // namespace tir
}  // namespace kxc
