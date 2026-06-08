# TOPI 组合指南

本页保留 TOPI 的阅读入口。新增 Relay 算子时，请直接使用新的 API reference：

- [relay-op-integration/03-topi-te.md](relay-op-integration/03-topi-te.md)
- [relay-op-integration/04-lowering-backend.md](relay-op-integration/04-lowering-backend.md)

## TOPI 是什么

TOPI 是基于 TE 的 tensor compute helper 集合，位于 `include/te/topi`。它用来搭建 Relay lowering 中的 compute 图，但 TOPI helper 存在不等于对应 Relay op 已支持。

## 当前 helper 分类

| 分类 | Header | 典型 API |
| --- | --- | --- |
| broadcast | `include/te/topi/broadcast.h` | `add`, `subtract`, `multiply`, `divide`, `maximum`, `minimum`, `broadcast_to` |
| elemwise | `include/te/topi/elemwise.h` | `exp`, `log`, `sqrt`, `floor`, `ceil`, `sigmoid`, `identity`, `negative`, `clip`, `cast` |
| reduction | `include/te/topi/reduction.h` | `sum`, `max`, `min`; `prod` 当前显式 unsupported |
| transform | `include/te/topi/transform.h` | `transpose`, `expand_dims`, `squeeze`, `concatenate` |
| nn | `include/te/topi/nn.h` | `relu`, `leaky_relu`, `dense`, `matmul`, `conv2d_nchw`, `pool2d`, `global_avg_pool2d` |

## 使用边界

| 情况 | 做法 |
| --- | --- |
| helper 语义完全匹配 Relay op | 在 `FRelayToTE` 中直接调用 |
| 需要复用或索引逻辑复杂 | 新增 TOPI helper |
| 只服务一个 op 且逻辑很短 | 可以在 op 注册文件中写局部 `te::compute` |
| 当前 TIR/LLVM 不支持生成的节点 | 先扩后端和 numeric test |
| 当前不能真实实现 | 显式抛出 unsupported，不返回空 tensor，不写假实现 |

完整 API、参数、返回值和限制见 [relay-op-integration/03-topi-te.md](relay-op-integration/03-topi-te.md)。
