# M9 L2：MiniMind-V 固定双图联合推理技术报告

2026-09-10。固定双图 MiniMind-V 已接入现有 ONNX → Relay → LLVM → RuntimeSession 链路。每张图固定为 `256×256`、64 个视觉 token；两个图像槽位由固定 marker 和 2 个分隔 token 隔开。实际 CPU/LLVM 测试覆盖完整八层语言模型、prefill 输出、16 份 KV、四步容量 decode 和 RuntimeSession 状态，不引入新的 agent/调度 IR。

## 方法

导出器把图像数量作为显式 profile 参数。`--image-count 2` 生成：

| 字段 | 值 |
|---|---:|
| `input_ids` | `int64[1,134]` |
| 图像 marker | `12`，区间 `[2,66)`、`[68,132)` |
| `pixel_values` | `float32[1,2,3,256,256]` |
| 词表 | `6400` |
| KV capacity | `146`，decode extent `134 → 138` |

profile、marker 区间、词元范围和像素 dtype/shape 在 Python fixture 和 C++ 生产调用者分别校验。C++ 同时兼容旧的单图四字段 profile；新 profile 使用六字段 `marker start tokens count vocab sequence`。缺任一图 marker、在槽外出现 marker、错误像素 ABI、非法容量或 profile 与序列长度不一致，都会在执行前拒绝。

上游 `vision_proj` 的展平输出按 `[batch,image_count,64,hidden]` 还原后逐图比较；三个输入场景（确定性图文、图像归零、改变文本）均核对前缀/视觉 token/后缀 embedding。独立 ONNX `ReferenceEvaluator` 再核对导出参考，生产 LLVM prefill 的真实 K/V 通过 `InitializeState` 交给同一 decode session。每步 host 只做 greedy argmax，RuntimeSession 持有容量状态并追加有效 extent。

## 效果与证据

- 导出：prefill 1,909 个原始 ONNX 节点，折叠后导入 1,154 个 Relay 节点；capacity decode 1,167 个原始节点，导入 683 个 Relay 节点。
- 独立 fixture：三组输入的完整 logits 和 16 份 K/V 均与未修改上游模型一致；ONNX 与上游 prefill 最大绝对差约 `8.17e-6`，四步 logits 最大差约 `3.52e-6`。
- 真实 KXC LLVM：三组 VLM case 均执行同一生产 prefill/decode 入口；case 0 的 prefill/所有 K/V 最大差 `1.12057e-5`，四步 logits 最大差 `5.42402e-6`，低于 `1e-4` 门禁；完整 16 份状态最大差 `1.12057e-5`。
- 状态与缓存：容量为 146，prefill 先写入 extent 134，连续四步追加到 138；缓存地址保持不变，步骤间 primitive-cache 计数不变。把无效容量区填为 `0.0` 与 `-1234.5` 的最后一步重放得到逐位相同 logits，证明 mask 真的屏蔽了无效区。

复现命令：

```bash
OPENBLAS_NUM_THREADS=1 out/venv/bin/python python/tools/export_minimind_v_joint_onnx.py \
  --src out/minimind-v-source --vision-config out/minimind-v-config/config.json \
  --config-revision 9465d1dc89db6bc6227c5b6b0e0ca9b940325d62 \
  --out out/minimind_v_multi --image-count 2
OPENBLAS_NUM_THREADS=1 out/venv/bin/python python/tools/make_minimind_v_joint_fixture.py \
  --export out/minimind_v_multi --out out/fx_minimind_v_multi
cmake --build out/build/bounded-llvm --target minimind_decode_loop_llvm_test -j2
KXC_MINIMIND_VLM_DIR=$PWD/out/fx_minimind_v_multi \
  ctest --test-dir out/build/bounded-llvm --output-on-failure --no-tests=error -j1 \
  -R '^minimind_vlm_case[012]_llvm_test$'
```

Python profile 正/负例、导出完整性、ONNX shape inference 和 fixture 哈希检查均通过。CTest 三个真实 VLM case 均通过（总计 220.22s）；同轮 bounded CPU 回归为 70 pass、1 skip（分布式 fixture 未配置，按既有门禁跳过）。Windows/Tailscale GPU 没有参与这次证据；该模块没有 CUDA 数值或热替换结论。

## 边界

本模块验收的是固定两图、固定 64-token 图像槽位、固定 2-token 分隔符和固定文本前后缀布局。profile helper 只接受能放入语言模型 2048 位置上限的固定图像数，但除单图/双图外的数量没有本次端到端证据；任意图像位置、变长图文、跨请求多图状态、GPU、多机分布式 decode 和新 IR 仍未完成。
