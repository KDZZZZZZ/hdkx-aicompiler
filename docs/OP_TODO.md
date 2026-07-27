# Transformer 模型算子清单

> **仅作参考快照。** 这是某个 ONNX Encoder 模型的算子清单，用于估算前端与算子覆盖差距；它不是支持矩阵、实施计划或生产能力声明。当前能力以 [Relay 算子支持矩阵](OP_SUPPORT_MATRIX.md) 和 `test/nlp_validation/transformer_capability_matrix.json` 为准。

算子名保留 ONNX 原名。输入数和输出数来自该模型快照，不代表算子的通用元数约束。

| ONNX 算子 | 数量 | 输入数 | 输出数 | 示例属性 |
|---|---:|---|---:|---|
| `Add` | 88 | 2 | 1 | 无 |
| `Cast` | 15 | 1 | 1 | `to: 1` |
| `Concat` | 25 | 3、4 | 1 | `axis: 0` |
| `Constant` | 180 | 0 | 1 | `value: Tensor([])` |
| `ConstantOfShape` | 1 | 1 | 1 | `value: Tensor([1])` |
| `Div` | 25 | 2 | 1 | 无 |
| `Equal` | 1 | 2 | 1 | 无 |
| `Erf` | 6 | 1 | 1 | 无 |
| `Expand` | 1 | 2 | 1 | 无 |
| `Gather` | 12 | 2 | 1 | `axis: 0` |
| `MatMul` | 49 | 2 | 1 | 无 |
| `Mul` | 38 | 2 | 1 | 无 |
| `Pow` | 13 | 2 | 1 | 无 |
| `ReduceMean` | 26 | 1 | 1 | `axes: [-1]` |
| `Reshape` | 25 | 2 | 1 | `allowzero: 0` |
| `Shape` | 17 | 1 | 1 | 无 |
| `Slice` | 7 | 3、5 | 1 | 无 |
| `Softmax` | 6 | 1 | 1 | `axis: -1` |
| `Split` | 1 | 2 | 2 | `axis: -1` |
| `Sqrt` | 31 | 1 | 1 | 无 |
| `Squeeze` | 2 | 2 | 1 | 无 |
| `Sub` | 14 | 2 | 1 | 无 |
| `Transpose` | 24 | 1 | 1 | `perm: [0, 2, 1, 3]` |
| `Unsqueeze` | 30 | 2 | 1 | 无 |
| `Where` | 2 | 3 | 1 | 无 |
