# 第二波并行计划：MiniMind-L1

第一波已经在 `dev` 完成：G0 固定了 CPU/LLVM 与 bounded 基线，G1 合入了执行观测、静态 ONNX 子集和 Equal→Where 组合证据。现在目标模型也已经明确：MiniMind-O 是北极星，纯文本 MiniMind 是当前验收目标。仓库虽然有 MiniMind 的静态 prefill/decode 导出产物，但动态 decoder 的 `past_key_values` 导出、KV 更新、变长执行和 host 生成循环尚未通过。因此第二波不再重复第一波的 A/B/C 任务，而是围绕 L1a → L1b 建立真正的模型闭环。

这一波的入口是 [M9 模型与导出验收](M9_MINIMIND_TARGET.md)。M9 的 E0 只需锁定并审计导出合同即可先行；导出合同稳定后，M2、M3、M4/M5 可以并行。M10 作为独立 E 线审计仓库里已经存在的结构化控制流：`KXC_ENABLE_CONTROL_RUNTIME` 默认关闭，gate-on 才能验证静态精确 `If` 和有界 `While` 的真实 LLVM 路径。M10 不阻塞 L1a；只有在 M2/M3 的 state/extent 合同落定后，才评估它是否用于 L1b 生成循环。M1 的第一波实现保留为基础，第二波只补 MiniMind 所需的关联字段和真实模型 profile，不重新设计事件系统。

> 状态：**已执行，G2 部分通过（2026-09-08）**。逐项结果、L1a 的确切阻塞点与跨线冲突处置见 [G2 验收记录](G2_RECORD.md)；第一波证据见 [G0](G0_BASELINE.md) / [G1](G1_RECORD.md)。原始分派内容保留如下。模块总览见 [README](README.md)，目标模型阶梯见 [PROJECT_GOAL](../PROJECT_GOAL.md) §2.2。

## 1. 先过模型门禁

| 检查点 | 负责人 | 通过标准 | 不通过时的处理 |
|---|---|---|---|
| E0 导出 | M9 | 源码/依赖/配置/opset/轴/输出顺序可复现，inventory 和 SHA 有记录 | 只保留失败报告；M2/M3 不按猜测实现动态图 |
| E1 L1a prefill | M9 + M4/M5 | 真实静态 prefill 导入、LLVM 执行、数值对齐 | 继续补实际缺失算子，不把节点名交集当成通过 |
| E2 decode 签名 | M9 + M2 | past/present 层序、布局、total 关系固定 | 暂用受控外部-KV fixture，动态 export 保持未验收 |

## 2. 可并行的五条线

| 线 | 模块与具体工作 | 交付物 | 不改的区域 |
|---|---|---|---|
| A：模型入口 | M9 E0/E1；真实 ONNX inventory、签名、L1a smoke/e2e、OP_TODO 更新 | 导出 metadata、inventory、L1a receipt、失败样例 | 不改 RuntimeSession 状态和 shape evaluator |
| B：KV 状态 | M2 S1；容量、cursor、valid extent、append/read、同 session prefill→decode | 生产编译的 state plan、LLVM 原址更新、三步 decode fixture | 不在 runtime 内临时编译或复制 shape VM |
| C：形状与注意力 | M3 S1/S2；Shape 值、Reshape/Expand/Squeeze/Unsqueeze 受限链和有界 attention | 两个合法 shape 的同产物证据、launch 前 guards | 不放开未知 rank、数据相关任意 shape |
| D：前端/算子 | M4/M5；按 M9 实际 inventory 补 `Pow`、`Expand`、`Neg`、`Sigmoid`、`Unsqueeze` 及多输出/属性边界 | 节点级导入报告、Relay/LLVM 数值与负例 | 不新建 registry，不提前接 MiniMind-O 音频算子 |
| E：结构化控制流 | M10 C0/C1；审计 `ControlPlan`/Relay lowering/runtime owner，在 gate-on LLVM 构建中验证 `If` 两分支、`While` 0/1/多次迭代及上限拒绝；C2 给出 host loop 与 bounded graph loop 的选择 | 控制流能力矩阵、可复现 gate-on receipt、L1b 生成循环决策记录 | 不放开 CUDA、runtime extent、KV/state 或第二套 RuntimeSession；不阻塞 L1a |

M1 作为横向验证人：为 A/B/C/D 的真实运行接入现有 bundle，补 run 与 model/export receipt 的关联；若不需要新字段，只增加测试和诊断规则。M6/M7/M8 不占用这波核心共享文件，可继续做小型设计审查，但不得阻塞 L1。

## 3. 所有权和依赖

M9 独占 `python/tools/export_minimind_onnx.py`、`python/tools/onnx_op_inventory.py`、MiniMind 测试输入和 `docs/OP_TODO.md`；M2 独占 `src/runtime` 的状态合同和 KV 测试；M3 独占 shape/lowering 的动态合同和 shape 测试；M4/M5 各自维护 importer 或算子契约，公共 generated 文件由集成负责人按提交顺序重新生成。每条线使用独立 worktree/build 目录，禁止直接在共享 build 目录重配置。

依赖顺序是：M9 E0 → M9 E1 与 D 线；M9 E2 签名 → M2；M10 C1 可在 M9 E0 后独立进行，M10 C2 评估真实 decode 时再依赖 M2 的 state owner 与 M3 的 extent/shape 绑定；M2 的 extent ABI + M3 的 shape 绑定 → L1b。M1 的 profile 只为验收提供证据，不把观测字段反向塞进 kernel ABI。任何 route miss、unsupported op、非法 shape 或不支持的控制流都必须在 launch 前报错。

## 4. G2 组合验收

1. 从同一份锁定导出生成 L1a prefill，真实 importer → Relay → LLVM → RuntimeSession 运行，输出与 PyTorch/NumPy 参考逐元素对齐。
2. 同一 session 完成 prefill 后追加至少三步 decode；每步检查 cache 地址不变、valid extent 递增、无效容量哨兵不影响 logits。
3. 用两个合法 bounded shape 重复执行，确认不发生隐式编译，非法范围/轴/容量/输出数量均零 launch。
4. 生成最小 greedy token 序列；bundle 能按 export receipt、run_id、kernel 和 state extent 关联运行证据。
5. M10 控制流 gate-on 证据独立完成：`If` 两分支、`While` 0/1/多次迭代、`max_trip_count` 越界和 gate-off 拒绝均有结果；同时记录 L1b 采用 host loop 还是暂不接入 bounded graph loop。
6. 重新运行 contract/pass/NLP/include/public-header/docs 检查和完整 CPU/LLVM 回归，更新能力矩阵逐格证据。

G2 通过后才进入 L1b 的采样策略、热替换运行证据和性能融合；MiniMind-V/MiniMind-O 留在后续波次。若动态 export 仍失败，G2 可以只宣布 L1a + 外部-KV 受控证据，明确 L1b blocked by export/state，不得改写目标。
