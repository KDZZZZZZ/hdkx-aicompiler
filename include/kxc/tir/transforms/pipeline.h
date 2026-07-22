/*! \file include/kxc/tir/transforms/pipeline.h
 * \brief 声明 TIR 优化 pass。
 */

#pragma once

#include "kxc/support/container.h"
#include "kxc/tir/stmt.h"
#include "kxc/tir/transforms/bind_cuda_threads.h"

namespace kxc {
namespace tir {

PrimFunc RunTIRPassPipeline(const PrimFunc& func, const Array<String>& pass_names);

}  // namespace tir
}  // namespace kxc
