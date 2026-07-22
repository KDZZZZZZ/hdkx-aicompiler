#pragma once

#include "kxc/target/target.h"
#include "kxc/runtime/kernel_abi.h"
#include "kxc/tir/stmt.h"

namespace kxc::codegen {

KernelSignature BuildKernelSignature(const tir::PrimFunc& function,
                                     const Map<String, runtime::NDArray>& constants,
                                     const Target& target,
                                     String symbol);

}  // namespace kxc::codegen
