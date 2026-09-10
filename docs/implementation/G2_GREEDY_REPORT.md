# G2 技术报告：host greedy 生成与 bundle 关联

> 状态：**已完成（2026-09-08）**。本报告覆盖 G2 第 4 项的最小 host-side greedy 证据；真实 prefill 输出与会话持有 K/V 的后续归并见 [M2 报告](M2_MINIMIND_STATE_REPORT.md)。

## 目标

把定容 decode fixture 从“随机 token 回放”提升为可复验的生成循环：prefill 的最后一个位置和每一步 decode 的最后一个位置都由 host 做确定性 argmax，并把导出回执、阶段、代际、计划 ABI 和 state extent 关联到 bundle 的同一运行事件。

## 方法

1. fixture 生成器在 ONNX ReferenceEvaluator 的 prefill logits 上取最后位置 argmax，随后每步从上一轮 logits 取 argmax；落盘 `prefill_logits.bin`、`sampling.txt=greedy_argmax`、`steps.txt` 和定容图 SHA256 回执。
2. C++ LLVM 测试不信任 `steps.txt` 的 token：先执行真实 LLVM prefill，从其最后一行 logits 计算首 token，再从被测 RuntimeSession 的 logits 计算后续 token，逐行断言与 fixture 序列一致。
3. 每次 `RuntimeSession::Run` 传入 `export_receipt`、`stage`、`generation`、`plan_abi`、`state_extent`、`state_version`、`token_index`、`token_id`。plan ABI 由既有 `BuildPlanAbiFingerprint` 计算；观测器把 metadata 复制到 run、kernel submit/exec 和 alloc/copy 事件。

## 改动

- `python/tools/make_minimind_decode_loop_fixture.py`
  - 增加 ONNX SHA256 回执、prefill 最后一行 logits 和 greedy 策略记录。
  - token 由参考 logits argmax 产生，不再使用随机 token。
- `test/minimind_decode_loop_llvm_test.cpp`
  - 加入 host argmax、策略/回执校验和每步 token 选择断言。
  - 使用真实 Compiler → LLVM → RuntimeSession 运行，并为每步写入模型/状态 metadata。

## 验证与效果

使用已有 capacity=32、8 层 MiniMind 图生成新 fixture，并执行：

```bash
PYTHONPATH=python out/venv/bin/python \
  python/tools/make_minimind_decode_loop_fixture.py \
  --onnx out/minimind_onnx_cap32 --out out/fx_decode_stateful --steps 4 --seed 0
KXC_MINIMIND_DECODE_LOOP_DIR=out/fx_decode_stateful \
  ./out/build/dev-ninja-cpu/minimind_decode_loop_llvm_test
```

结果：prefill **650 个 kernel**，定容 decode **683 个 kernel/步**，同一 decode 产物完成 **4 步**，extent `16 → 20`，greedy token 序列和 ONNX 参考逐步一致；decode logits 最坏差 `5.42402e-06`，无效区两种哨兵填充值下逐位相同。decode bundle 有 **6 个 runtime run**（4 步加 2 次哨兵重放）、4098 对 submit/exec、96 次状态 copy，共 12506 条事件。其中 12396 条属于这些 runtime run，关联字段缺失数为 0。另有 prefill bundle 的 1 个 run、650 对 submit/exec；跨 bundle 使用 trace/session/run 与 receipt 关联。

## 后续边界

当前验收固定 batch=1 和固定步数，EOS、top-k 和随机采样尚未纳入；这是 M10 已选择的最小 host loop。cache 更新现由 RuntimeSession 执行，后续继续扩展有界形状和请求级生成能力。
