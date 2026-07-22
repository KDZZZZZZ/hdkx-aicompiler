/*! \file include/kxc/tir/transforms/convert_for_loops_serial.h
 * \brief 声明 TIR 优化 pass。
 */

#pragma once

#include "kxc/tir/stmt.h"

namespace kxc {
namespace tir {

PrimFunc ConvertForLoopsSerialPass(const PrimFunc& func);

}  // namespace tir
}  // namespace kxc
