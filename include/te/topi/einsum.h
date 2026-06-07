/*! \file include/te/topi/einsum.h
 * \brief 定义 TOPI 风格的 tensor compute helper。
 */

#pragma once
#include "te/te.h"
#include "te/topi/tags.h"
#include "base/container.h"
#include <stdexcept>
#include <string>
#include <vector>

namespace kxc {
namespace te {
namespace topi {

inline Tensor einsum(std::string equation, Array<Tensor> operands, std::string name = "einsum", std::string tag = kMatMul) {
    (void)equation;
    (void)operands;
    (void)name;
    (void)tag;
    throw std::runtime_error("topi::einsum is not supported by the current TE/TIR lowering");
}

} // namespace topi
} // namespace te
} // namespace kxc
