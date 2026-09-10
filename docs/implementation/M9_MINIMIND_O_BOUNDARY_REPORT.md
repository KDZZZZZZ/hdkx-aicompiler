# MiniMind-O L3 完成边界技术报告

核对日期：2026-09-10。本文把“MiniMind-O 完全成功”固定为一个可审计的编译器/运行时验收合同，避免把静态 demo、单帧输出或 Thinker-only 证据误写成 L3 完成。它是范围与验收模块的技术报告，不宣称当前 L3 已实现。

## 方法

采用“组件清单 + 状态 ABI + bounded profile + 运行时观测 + 实时窗口”的组合门禁：先锁定六类真实组件的导出 receipt，再逐类证明 importer/lowering/backend/numeric；将 Thinker/Talker 的 KV 和 Mimi 的 ring buffer 统一挂到现有 `ExecutablePlan`/`RuntimeSession` 状态合同；用显式调度合同连接两个自回归链；最后在固定硬件上以连续 1,000 个 80 ms 窗口验证 steady-state 预算。所有超范围 shape、容量、设备、代际和证据篡改都在提交 kernel 前拒绝。

本次模型基线是 dense `minimind-3o`，主验收设备为 RTX 4070 Ti SUPER 16 GB；CPU/LLVM承担组件语义与数值回归，不要求 CPU 语音实时。覆盖文本、图文、语音输入到文本及流式语音输出，以及参考音色条件。输入范围必须在导出前冻结为有限 profile，覆盖最小/中间/最大/越界值，禁止为通过测试事后缩小。上游的组件、流式 Mimi 和音色条件依据其[官方说明](https://github.com/jingyaogong/minimind-o#-模型细节)核对；这里的 p99 与 1,000 帧是本项目制定的工程验收条件。

## 完成判据

| 门禁 | 必须证明的结果 | 证据要求 |
|---|---|---|
| L3-C1/E0 | Thinker、Talker、SenseVoice、SigLIP2、Mimi、CAM++ 的源码/配置/导出可复现 | 每图 metadata、签名、opset、SHA256、export receipt |
| L3-C1/E1 | 六类图均走生产 importer → Relay → TIR → LLVM/CUDA | 每类正例、属性/shape 负例、backend receipt |
| L3-C1/E2 | KV 与 Mimi ring buffer 的容量、游标、extent、reset、耗尽行为由唯一 state owner 持有 | 连续帧、环回、错误执行后状态不变、地址/extent 记录 |
| L3-C1/E3 | Thinker/Talker 双自回归交接由显式调度和有限 profile 编排 | token/frame 交接、无隐式编译、profile miss 零提交 |
| L3-C1/E4 | steady-state 与边界 profile 对齐独立参考 | logits、tokens、state、音频误差统计和固定 seed |
| L3-C1/E5 | 连续 1,000 帧 p99 ≤80 ms、RTF ≤1、积压不持续增长 | kernel/copy/state/scheduler 全链路 bundle；排除冷编译；报告超时数、首音频延迟和峰值显存 |
| L3-C1/E6 | stateful plan 在帧边界安全换代并可回滚 | generation 1→2→1、same route/ABI、health receipt、负例 |
| L3-C1/E7 | agent 可消费并审计全部证据 | frame/stage/generation/state/kernel/copy/stream 字段与篡改审计 |

## 当前结论

当前仓库已具备 MiniMind Thinker（纯文本 L1）和 MiniMind-V 固定单图/双图 L2 的部分基础；Mimi 流式 ring buffer、Talker 双自回归调度、六类 L3 真实导出、连续实时预算和 stateful L3 热替换尚未通过。因此当前不能宣称 MiniMind-O 完全成功。生产服务层的麦克风、网络、VAD 驱动的 barge-in 和近双工 UI 不在本合同内；运行时 cancel/reset/会话隔离仍在范围内。训练、MoE、无限输入范围不属于该模型验收。其他支柱的分布式和通用热替换目标继续独立验收。

## 复核命令

范围变更后至少运行：

```bash
python3 tools/architecture/check_docs.py --root .
python3 -m json.tool test/nlp_validation/transformer_capability_matrix.json >/dev/null
```

本报告的验收结论只在上述门禁、实际模型 receipt 和完整硬件 bundle 同时存在时更新。
