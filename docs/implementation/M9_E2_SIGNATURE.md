# M9 E2 decode 签名审计（给 M2 的绑定目标）

> 状态：已完成（2026-09-07）。依据 E0 通过的动态 decode 导出（SHA `6d3f4262…`，见 [M9_E0_RECEIPT.md](M9_E0_RECEIPT.md)）。本文把 `past/present` 的签名固定下来，M2 的状态合同按此绑定；真实图绑定仍以 E1/E2 的编译运行 receipt 为准。

## 1. decode 图的输入输出签名（opset 17，动态轴）

**输入（17 个，顺序固定）**：

| 序 | 名称 | 形状 | 说明 |
|---|---|---|---|
| 0 | `input_ids` | `[batch, 1]` | int64，当前步 token |
| 1 | `past_k_0` | `[batch, past, 4, 96]` | 第 0 层 K |
| 2 | `past_v_0` | `[batch, past, 4, 96]` | 第 0 层 V |
| … | …（k/v 交错） | … | 第 1–7 层同构，共 16 个 past 张量 |

**输出（17 个，顺序固定）**：

| 序 | 名称 | 形状 | 说明 |
|---|---|---|---|
| 0 | `logits` | `[batch, 1, 6400]` | 当前步 next-token logits |
| 1 | `present_k_0` | `[batch, total, 4, 96]` | 第 0 层 K（含新步） |
| 2 | `present_v_0` | `[batch, total, 4, 96]` | 第 0 层 V |
| … | …（k/v 交错） | … | 共 16 个 present 张量 |

## 2. 固定的关系

- **布局**：`[batch, seq, kv_heads=4, head_dim=96]`；kv_heads/head_dim/层数(8) 固定，`batch`、`past`、`total` 为有界动态轴。
- **`total = past + current`**，decode 每步 `current = 1`；E0 数值门禁已按 `total = past + 1` 逐形状验证（17/9/34）。
- **层序**：k/v 按 `(layer, {k, v})` 扁平交错；`present_i` 与 `past_i` 同层同布局，层间互不别名。
- prefill 图（无 past 输入）：`input_ids [batch, seq]` → `logits [batch, seq, 6400]` + 16 个初始 KV（形状 `[batch, seq, 4, 96]`），即 decode 第一步的 `past` 种子。

## 3. 对 M2 的绑定要求

1. 会话持有的缓存按"每层 K/V 两块、布局 `[batch, capacity, 4, 96]`"组织；`capacity >= total`，有效长度 = `total`。
2. prefill 把输出 KV 写入缓存（extent=seq）；每个 decode 步追加 1 个 token（extent+1）并把 `present_i` 对应的有效区交给图；图级 `total` 由 extent 派生，不由 importer 或 runtime 各自推导。
3. `logits` 的形状/布局和 16 个 present 的层序必须与上表逐一对应；任何顺序/别名错位在 launch 前拒绝。
4. 真实 MiniMind 图绑定（E1/E2 编译运行）依赖 M4 的 `Pow/Neg/Sigmoid/Expand/Unsqueeze`（见 E0 receipt §5）与 M3 的受限 shape 绑定；在这些落地前，受控外部-KV fixture 按本文签名构造。

## 4. 实测补充（2026-09-08）：静态导出下的多步行为

真实 decode 图现在已经能走完整生产链路（见 [E1 receipt](M9_E1_RECEIPT.md) 的同一条 importer → Relay → LLVM → RuntimeSession）。用 prefill 的 `present` 作为 decode 的 `past` 做真实自回归续接，逐步实测：

| 步 | 图 | past → present | 实算节点 | 与 ONNX 参考的最坏逐元素差 |
|---|---|---|---:|---|
| prefill | `--static` | — → 16 | 650 | 8.82149e-06 |
| decode 第 1 步 | `--static`（默认 `--past 16`） | 16 → 17 | 666 | 6.85453e-06 |
| decode 第 2 步 | `--static --past 17` | 17 → 18 | 666 | 5.54323e-06 |

**两条对 M2 有直接影响的事实：**

1. **`present` 的前缀与输入 `past` 逐元素完全相同**（两步都验证过）。真实图对 cache 只追加、不改写历史。因此把 `past` 输入与 `present` 输出绑到**同一块存储**在语义上是安全的——重叠区本来就一致，那次拷贝是冗余的。这正是 §3 第 2 条要求的原址更新可以成立的依据。

2. **静态导出下每一步需要一个独立产物**。`past` 的长度被烤进形状：`--past 16` 的图是 `past[1,16,4,96] → present[1,17,4,96]`，`--past 17` 的图是 `past[1,17,4,96] → present[1,18,4,96]`，节点数同为 1173 但形状不同，因此是两个不同的编译产物。

第 2 条决定了 **G2 第 2 项不能用静态导出达成**：该项要求同一 session、同一产物、cache 地址不变、runtime 期间不发生隐式编译，而静态图每步换产物与之直接冲突。目前得到的是 WAVE_2 §4 允许的「真实图 + 外部-KV 受控证据」，不是第 2 项本身。

要真正通过第 2 项，二选一：

- **有界动态 past**：把 `past`/`total` 导成有界符号轴，一个产物覆盖所有步（M3 的受限 shape 绑定）；或
- **定容 state + 有效长度**：图按 `capacity` 读、按 extent 掩码，`past` 输入与 `present` 输出绑同一块 state（M2 §3 第 1、2 条）。事实 1 说明这条路径的语义前提已经成立。

复现：`export_minimind_onnx.py --static --past N` 导出对应步的图，`make_minimind_l1a_fixture.py --graph decode` 生成 fixture，`minimind_l1a_llvm_test` 执行。
