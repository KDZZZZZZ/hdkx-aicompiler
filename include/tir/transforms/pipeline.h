/*! \file include/tir/transforms/pipeline.h
 * \brief 声明 TIR 优化 pass。
 */

#pragma once

#include "base/container.h"
#include "tir/stmt.h"

namespace kxc {
namespace tir {

PrimFunc RunTIRPassPipeline(const PrimFunc& func, const Array<String>& pass_names);

}  // namespace tir
}  // namespace kxc
