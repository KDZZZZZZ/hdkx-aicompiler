# TOPI 算子组合指南

本指南将说明如何使用 `include/te/topi` 中的 **Tensor Operator Inventory (TOPI)** 来构建复杂的算子和神经网络层。TOPI 提供了一个预定义算子库，可以高效地组合这些算子来创建复杂的计算图。

## 目录

1. [简介](#简介)
2. [基础算子](#基础算子)
3. [神经网络原语](#神经网络原语)
4. [张量变换](#张量变换)
5. [算子组合](#算子组合)
6. [示例](#示例)

---

## 简介

TOPI 是基于 Tensor Expression (TE) API 实现的一组高层算子集合。您无需为每个操作编写原始的 `te::compute` 定义，而是可以直接使用 TOPI 函数，它们会自动处理：
- 自动形状推导（例如：广播机制）。
- 索引映射和迭代逻辑。
- 常见的计算模式，如规约（Reduction）和卷积（Convolution）。

使用 TOPI 前，请包含必要的头文件：
```cpp
#include "te/topi/broadcast.h"
#include "te/topi/elemwise.h"
#include "te/topi/reduction.h"
#include "te/topi/nn.h"
#include "te/topi/transform.h"
```

## 基础算子

### 广播 (Broadcasting)
处理不同形状张量之间的二元运算（遵循 NumPy/PyTorch 的广播规则）。

**头文件：** `te/topi/broadcast.h`

| 函数 | 描述 |
|----------|-------------|
| `add(A, B)` | 带广播机制的逐元素加法 |
| `subtract(A, B)` | 带广播机制的逐元素减法 |
| `multiply(A, B)` | 带广播机制的逐元素乘法 |
| `divide(A, B)` | 带广播机制的逐元素除法 |
| `maximum(A, B)` | 带广播机制的逐元素最大值 |
| `minimum(A, B)` | 带广播机制的逐元素最小值 |

### 逐元素操作 (Element-wise Operations)
对张量的每个元素应用特定函数。

**头文件：** `te/topi/elemwise.h`

| 函数 | 描述 |
|----------|-------------|
| `relu(x)` | 修正线性单元：`max(x, 0)` |
| `sigmoid(x)` | `1 / (1 + exp(-x))` |
| `exp(x)`, `log(x)` | 指数和对数 |
| `sqrt(x)` | 平方根 |
| `clip(x, min, max)` | 将数值截断在 min 和 max 之间 |
| `cast(x, dtype)` | 将元素转换为新的数据类型 |

### 规约 (Reduction)
沿指定轴对张量进行规约操作。

**头文件：** `te/topi/reduction.h`

| 函数 | 描述 |
|----------|-------------|
| `sum(data, axis, keepdims)` | 沿指定轴求和 |
| `max(data, axis, keepdims)` | 沿指定轴求最大值 |
| `min(data, axis, keepdims)` | 沿指定轴求最小值 |

---

## 神经网络原语

深度学习模型中常见的高层运算层。

**头文件：** `te/topi/nn.h`

### 全连接 (Dense / Matrix Multiplication)
计算 `A * B^T + Bias`。

```cpp
Tensor dense(const Tensor& A, const Tensor& B, const Tensor& bias = Tensor(), ...);
```
- **A**: 输入张量 [M, K]
- **B**: 权重张量 [N, K] (注意：通常预期权重是转置过的)
- **bias**: 可选的偏置张量 [N]

### 二维卷积 (Conv2D NCHW)
标准的二维卷积操作。

```cpp
Tensor conv2d_nchw(const Tensor& data, const Tensor& kernel, int stride_h, int stride_w, int pad_h, int pad_w, ...);
```
- **data**: 输入 [N, C, H, W]
- **kernel**: 权重 [O, I, KH, KW]

### 池化 (Pooling)
```cpp
Tensor pool2d(const Tensor& data, vector<int> kernel, vector<int> stride, vector<int> pad, string type, ...);
```
- **type**: "max" (最大池化) 或 "avg" (平均池化)

---

## 张量变换

改变张量的形状或维度。

**头文件：** `te/topi/transform.h`

| 函数 | 描述 |
|----------|-------------|
| `transpose(x, axes)` | 维度置换 (Permute) |
| `expand_dims(x, axis)` | 在指定位置插入大小为 1 的维度 |
| `squeeze(x, axes)` | 移除大小为 1 的维度 |
| `concatenate(inputs, axis)` | 沿指定轴拼接张量 |

---

## 算子组合

TOPI 的强大之处在于组合这些原语。由于每个 TOPI 函数都返回一个 `Tensor`（它是 `te::compute` 操作的包装），你可以直接链式调用它们。

### 模式 1: Conv2d + BatchNorm + ReLU (融合)
虽然 TOPI 提供独立的算子，但你可以将它们链接起来定义一个块。注意，对于图层面的融合，通常依赖编译器的 Pass 机制，但在 TOPI 中定义计算序列非常直观。

```cpp
// 1. 卷积
Tensor conv = topi::conv2d_nchw(input, weight, 1, 1, 1, 1, 1, 1);

// 2. 偏置加法 (使用广播加法)
// 偏置形状 [O], 卷积输出 [N, O, OH, OW]
// 我们的通用广播机制支持右对齐。
// 为了将 [O] 广播到 [N, O, OH, OW]，我们可能需要显式的 reshape 或 expand_dims 来匹配语义。
// 假设我们需要使用 expand_dims 来匹配通道维度。
Tensor bias_reshaped = topi::expand_dims(topi::expand_dims(bias, 1), 2); // [O, 1, 1]
Tensor biased = topi::add(conv, bias_reshaped);

// 3. ReLU 激活
Tensor output = topi::relu(biased);
```

### 模式 2: Softmax
Softmax 由 `exp`, `sum`, 和 `divide` 组合而成。

```cpp
// Softmax(x) = exp(x) / sum(exp(x))
Tensor softmax(Tensor x, int axis) {
    // 1. 求最大值以保持数值稳定性 (x - max(x))
    Tensor max_elem = topi::max(x, {axis}, true);
    Tensor shifted = topi::subtract(x, max_elem);
    
    // 2. 指数运算
    Tensor exps = topi::exp(shifted);
    
    // 3. 求和
    Tensor sum_exps = topi::sum(exps, {axis}, true);
    
    // 4. 除法
    return topi::divide(exps, sum_exps);
}
```

---

## 示例

### 示例: 构建简单的多层感知机 (MLP)

```cpp
#include "te/topi/nn.h"
#include "te/topi/broadcast.h"

using namespace kxc::te;

Tensor build_mlp(Tensor input, Tensor w1, Tensor b1, Tensor w2, Tensor b2) {
    // 第一层: Dense -> ReLU
    Tensor dense1 = topi::dense(input, w1, b1);
    Tensor relu1 = topi::relu(dense1);
    
    // 第二层: Dense (输出层)
    Tensor output = topi::dense(relu1, w2, b2);
    
    return output;
}

int main() {
    auto x = placeholder({32, 784}, DataType::Float(32), "x");
    auto w1 = placeholder({128, 784}, DataType::Float(32), "w1");
    auto b1 = placeholder({128}, DataType::Float(32), "b1");
    auto w2 = placeholder({10, 128}, DataType::Float(32), "w2");
    auto b2 = placeholder({10}, DataType::Float(32), "b2");
    
    Tensor mlp = build_mlp(x, w1, b1, w2, b2);
    
    // 创建调度 (Schedule), 构建 (Build) 等...
}
```

### 最佳实践

1.  **关注形状 (Shape Awareness)**: 始终跟踪你的张量形状。TOPI 算子会推导输出形状，但如果输入形状不匹配（例如在 `dense` 或 `concatenate` 中），会导致运行时断言错误或未定义行为。
2.  **标签 (Tagging)**: TOPI 算子会附加标签（如 `kBroadcast`, `kCommReduce`）。这些标签对于后续的 `AutoSchedule` 或手动调度非常重要，用于应用正确的优化策略（如分块 tiling、向量化 vectorization）。
3.  **数据类型 (DType)**: 确保输入具有兼容的数据类型。在进行二元运算之前，如有必要，请使用 `topi::cast` 进行类型转换。
