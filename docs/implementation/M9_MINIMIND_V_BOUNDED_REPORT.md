# M9 L2：MiniMind-V 运行时图像位置与有界图文 prefill 技术报告

2026-09-10。一份 bounded LLVM prefill 产物现在服务 0～3 张图、任意图像位置和 1～224 的文本长度。之前的固定单图/双图导出把上游 Python marker 扫描固化进图里，每种图数和布局各需一个产物。本模块把扫描移到显式 host 步骤，视觉 token 以运行时输入进入同一张嵌入表，语言侧复用 L1 的受限形状证明、容量 decode 和会话 KV。测试只在 CPU/LLVM 上执行；GPU 验证按用户决定延后。

## 方法

**图像位置作为运行时输入。** 上游 `count_vision_proj` 扫描 token 中的 marker 连续段，再用 `torch.cat` 把第 k 段替换为第 k 张图的视觉 token。本模块的 host 函数 `slot_ids` 执行同一扫描：第 k 个恰好 64 个 marker 的连续段改写为下标 `6400 + 64k + [0,64)`。prefill 图的嵌入改为

```
table  = Concat(embed_tokens.weight [6400,768], visual_slots [192,768])   # [6592,768]
hidden = Gather(table, slot_ids [1,S])
```

`visual_slots` 按三张图定容，思路与定容 KV 相同：未用的行填哨兵，不参与计算。Gather 本身就是 L1 已验证的运行时索引 embedding；拼接后的表维度仍是静态常量（`DimExpr::Add` 折叠两个常量得 6592），所以无需放宽 bounded Gather 的静态表准入。唯一的符号轴是 `input_ids` 的 S，上界 224。

**三段产物，同一套权重。** 导出器复用固定联合导出的 `build_vlm_pair`（打补丁、seed 0、SigLIP2 配置），从同一个模型导出：

| 产物 | 合同 | 编译方式 | 每次调用 |
|---|---|---|---:|
| `vision` | 单图 `[1,3,256,256]` → `[1,64,768]` | 静态，沿用固定 Shape 证明 | 472 |
| `prefill_slots` | `input_ids [1,S]`、`visual_slots [192,768]`，`1≤S≤224` | 受限形状 `CompileBounded` | 743（742 带运行时 extent） |
| `decode_capacity` | 容量 240，`position`/`attention_mask` 驱动 extent | 静态 + `BindStateOutputs` | 683 |

视觉编码器保持固定输入合同：每张图形状相同，做成 bounded 没有收益。`fold_fixed_shape_queries` 只用于这一段；prefill 走 shape-source 导入，保留 S 相关的形状控制交给 C++ 受限准备。

**Host 放置合同。** 图像必须是恰好 64 个 marker 的连续段，最多 3 张，词元须在 `[0,6400)`，`1≤S≤224`。编译后的 Gather 对越界下标填零（KXC 的既有扩展），因此这些检查必须在任何张量产生之前完成，不能依赖 kernel。Python（`minimind_v_slots.py`）与 C++ 调用者各实现一次，C++ 结果须与 fixture 中 Python 的改写逐位相同。

## 效果与证据

**导出与独立参考。** 四组输入均用未修改的上游 `MiniMindVLM` 生成参考：

| case | 图数 N | S | 布局（文本段长度，段间各一张图） |
|---|---:|---:|---|
| 0 | 0 | 9 | 纯文本 |
| 1 | 1 | 68 | 2 · 图 · 2（与原固定单图同布局） |
| 2 | 2 | 143 | 5 · 图 · 3 · 图 · 7 |
| 3 | 3 | 207 | 1 · 图 · 1 · 图 · 4 · 图 · 9 |

- PyTorch 层：slot 适配器 prefill 与上游在四组输入上**逐位相同**（最大差 0）；容量 decode 对上游最大差 `2.38e-6`；逐图视觉与上游批量视觉最大差 0。
- ONNX 层：独立 `ReferenceEvaluator` 与上游的最大差为视觉 `1.16e-6`、prefill `4.86e-6`、decode `4.86e-6`。prefill 导出以 S=207 追踪，S=9 等其余长度同样对齐，说明导出没有把样本长度固定进图。
- 规模：prefill 原图 2,075 节点，静态折叠 736 个、物化 1.63 MB 后剩 1,339，导入 1,475 个 Relay 节点；视觉 703 → 484；decode 1,167 → 683。

**KXC CPU/LLVM（`minimind_vlm_bounded_llvm_test`）。** 三个产物各编译一次，此后所有 case 前后 primitive-cache 计数不变：

| 项 | 结果 |
|---|---|
| 视觉 | 同一产物 6 次运行（1+2+3 张图），对上游视觉 token 最大差 `1.75461e-6` |
| prefill | 同一 bounded 产物 7 次运行；全部 17 个输出（logits + 16 KV）对上游最大差 `1.27554e-5` |
| 未用槽位 | N<3 的三组把未用行从 `7.0` 改为 `-1234.5` 重放，2,759,680 个输出值逐位相同 |
| decode | 每组 prefill 的实际 K/V 经 `InitializeState` 进入容量 240 的会话状态，四步 greedy 共 16 步；logits 最大差 `5.21541e-6`，完整缓存 `1.18762e-5`；地址不变，extent 每步加一 |
| 拒绝 | host：63 个 marker、4 张图（先按 S 上界拒绝，放宽 S 后按槽位上限拒绝）、越界词元、S=225；运行时：S=225、S=0、batch 2、int32 ids、槽位 191 行、float64 槽位、缺输入，七次均零 kernel、零分配 |
| 观测 | bundle 记录 `model/stage/images/sequence/shape_case/state_extent/export_receipt`；29 次成功运行、7 次拒绝；18,961 次 kernel 提交恰为 6×472 + 7×743 + 16×683 |

单次 Debug 运行 206 秒，峰值 RSS 5.59 GB；这是本机资源记录，不是性能基准。

## 复现

```bash
cd python/tools
../../out/venv/bin/python export_minimind_v_bounded_onnx.py --src ../../out/minimind-v-source \
  --vision-config ../../out/minimind-v-config/config.json \
  --config-revision 9465d1dc89db6bc6227c5b6b0e0ca9b940325d62 --out ../../out/minimind_v_bounded
../../out/venv/bin/python make_minimind_v_bounded_fixture.py \
  --export ../../out/minimind_v_bounded --out ../../out/fx_minimind_v_bounded
cd ../..
cmake --build out/build/adaptive-bounded-llvm --target minimind_vlm_bounded_llvm_test
KXC_MINIMIND_VLM_BOUNDED_DIR=$PWD/out/fx_minimind_v_bounded \
  ctest --test-dir out/build/adaptive-bounded-llvm -R '^minimind_vlm_bounded_llvm_test$' --output-on-failure
PYTHONPATH=python out/venv/bin/python -m pytest -q test/minimind_v_slot_ids_test.py
```

未设置环境变量时 CTest 报告 Skipped，不计作通过。`build_vlm_pair` 从固定联合导出中抽出后，重新导出单图联合模型，prefill/decode 的 SHA256、全部参考文件和补丁哈希与原 receipt 完全一致。

## 边界

- 图数上限为 3、S 上界 224，均由导出前冻结的 profile 决定；超出范围在执行前拒绝，不会隐式编译。图数是槽位上的取值，不是形状轴。
- 每张图固定为 `256×256` 预处理后的 float32 像素；图像解码、缩放和归一化在编译器边界外。
- batch 固定为 1；跨请求共享槽位或批处理多图请求没有实现。
- 权重为 seed 0 随机初始化，只证明编译语义，不证明识图质量。
- 视觉编码器本身不是 bounded：shape-source 导入拒绝 Tanh/Erf，bounded LayerNorm 与卷积也未准入；本模块不需要它们。
- 没有 CUDA 数值、热替换或分布式 decode 结论。
