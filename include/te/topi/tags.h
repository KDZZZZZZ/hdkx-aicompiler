/*! \file include/te/topi/tags.h
 * \brief 定义 TOPI 风格的 tensor compute helper。
 */

#pragma once
#include <string>

namespace kxc {
namespace te {
namespace topi {

// Tag constants for different types of operations
constexpr const char* kElementWise = "elemwise";
constexpr const char* kBroadcast = "broadcast";
constexpr const char* kInjective = "injective";
constexpr const char* kCommReduce = "comm_reduce";
constexpr const char* kMatMul = "matmul";
constexpr const char* kConv2d = "conv2d";
constexpr const char* kPool = "pool";

} // namespace topi
} // namespace te
} // namespace kxc
