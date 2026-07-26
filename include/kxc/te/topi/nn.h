/*! \file include/kxc/te/topi/nn.h
 * \brief 定义 TOPI 风格的 tensor compute helper。
 */

#pragma once
#include "kxc/te/te.h"
#include "kxc/te/topi/broadcast.h"
#include "kxc/te/topi/tags.h"
#include "kxc/te/topi/utils.h"
#include "kxc/te/topi/window.h"
#include "kxc/support/container.h"
#include <vector>
#include <stdexcept>

namespace kxc {
namespace te {
namespace topi {

// Relu
Tensor relu(const Tensor& x, std::string name = "relu", std::string tag = kElementWise);

// Leaky Relu
Tensor leaky_relu(const Tensor& x, double alpha, std::string name = "leaky_relu", std::string tag = kElementWise);

// Dense (Matrix Multiplication)
// A: [M, K], B: [N, K] (Transposed B by default in many frameworks, or [K, N])
// Let's assume A: [M, K], B: [N, K] -> Output: [M, N]
Tensor dense(const Tensor& A, const Tensor& B, const Tensor& bias = Tensor(), std::string name = "dense", std::string tag = kMatMul);

// MatMul (ONNX/NumPy batch broadcasting)
// A: [..., M, K], B: [..., K, N] -> Output: [..., M, N]
Tensor matmul(const Tensor& A, const Tensor& B, std::string name = "matmul", std::string tag = kMatMul);

// Conv2D NCHW
// Data: [N, C, H, W]
// Weight: [O, C, KH, KW]
// Stride: [sh, sw], Dilation: [dh, dw]
// padding 已由调用方通过 ExpandPadding2D 展开为四边，允许非对称。
Tensor conv2d_nchw(const Tensor& data, const Tensor& kernel, AxisPair2D strides,
                   Padding2D padding, AxisPair2D dilation,
                   std::string name = "conv2d_nchw", std::string tag = kConv2d);

// Transitional expanded-padding overload retained for compatibility.
Tensor conv2d_nchw(const Tensor& data, const Tensor& kernel, int stride_h, int stride_w,
                   Padding2D padding, int dilation_h, int dilation_w,
                   std::string name = "conv2d_nchw", std::string tag = kConv2d);

// Historic symmetric-padding overload retained for source and binary compatibility.
Tensor conv2d_nchw(const Tensor& data, const Tensor& kernel, int stride_h, int stride_w,
                   int pad_h, int pad_w, int dilation_h, int dilation_w,
                   std::string name = "conv2d_nchw", std::string tag = kConv2d);

// Pool2D
// dilation 与 Relay 类型推导读取的 MaxPool2DAttrs::dilation 必须一致，否则推导出的
// 输出 shape 与 compute 出的不一致。
Tensor pool2d(const Tensor& data, AxisPair2D kernel_size, AxisPair2D stride,
              Padding2D padding, AxisPair2D dilation, std::string pool_type,
              bool ceil_mode = false, std::string name = "pool2d",
              std::string tag = kPool);

// Transitional expanded-padding/dilation overload retained for compatibility.
Tensor pool2d(const Tensor& data, Array<int> kernel_size, Array<int> stride,
              Padding2D padding, Array<int> dilation, std::string pool_type,
              bool ceil_mode = false, std::string name = "pool2d",
              std::string tag = kPool);

// Historic raw-padding/default-dilation overload retained for compatibility.
Tensor pool2d(const Tensor& data, Array<int> kernel_size, Array<int> stride,
              Array<int> padding, std::string pool_type, bool ceil_mode = false,
              std::string name = "pool2d", std::string tag = kPool);

Tensor global_avg_pool2d(const Tensor& data, std::string name = "global_avg_pool2d",
                                std::string tag = kPool);

} // namespace topi
} // namespace te
} // namespace kxc
