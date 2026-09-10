# M9 E1 receipt（L1a 静态 prefill）

> 状态：**已完成（2026-09-08）**。任务书见 [M9_MINIMIND_TARGET.md](M9_MINIMIND_TARGET.md) §E1；导出合同见 [E0 receipt](M9_E0_RECEIPT.md)，decode 签名审计见 [E2 签名](M9_E2_SIGNATURE.md)。本 receipt 只覆盖 **prefill**；decode 的运行证据属 E2/L1b，不在此列。
>
> 本文的作用是把 L1a 的 ABI 钉死：M2 的状态合同和 M3 的 extent 绑定按此对齐。没有本 receipt 的图不得被当作固定 ABI 引用。

## 1. 被验收的图

| 项 | 值 |
|---|---|
| 模型源码 | `github.com/jingyaogong/minimind` @ `6fc918b` |
| 源码补丁 | `python/tools/minimind_noninplace_mask.patch`（非原地 mask，1 文件 +5/-1） |
| 配置 | hidden 768 / **8 层** / 8 heads / 4 kv_heads / head_dim 96 / vocab 6400 / 参数 63.91M |
| 权重 | 随机初始化，`torch.manual_seed(0)`（本模块只验图结构与数值一致性，不验模型质量） |
| 导出 | opset 17，`dynamic_axes` 关闭（`--static`），batch=1、seq=16 |
| 产物 | `out/minimind_onnx_noninplace/minimind_prefill_static.onnx`，SHA256 前 16 位 `eb1c60c73980b0e5` |

该 SHA 与 [E0 receipt](M9_E0_RECEIPT.md) §2 记录的非原地 mask 产物一致。

## 2. 导入链路与规模

真实 ONNX → `python/kxc_onnx`（含导入前常量折叠）→ Relay → LLVM → `RuntimeSession`。全程使用生产链路，无手写等价 Relay 图。

| 阶段 | 数值 |
|---|---|
| 源图 | 1141 节点 / 60 initializer |
| 常量折叠 | 折掉 **491** 节点（`Constant` 388、`Identity` 31、`ConstantOfShape` 16、`Mul` 16、`Equal` 16、`Where` 16、`Trilu` 8） |
| 物化前沿 | 386 个值 / 257.2 KiB（纯中间静态值不进 initializer） |
| 实算节点 | **650** —— 与 E0 receipt 记录的折叠后规模一致 |
| Relay 图 | 650 节点 / 446 参数 |
| 编译产物 | 650 个 kernel call / 238 个常量 |

折叠用 ONNX 自己的 `ReferenceEvaluator` 求值，因此折叠语义即 ONNX 语义。折叠边界是 fail-closed 的：只折每个输入都静态的节点；产出图输出的节点不折；只物化被保留节点消费的前沿；超字节预算（默认 256 MiB）即拒绝。

**`ConstantOfShape` 与 `Trilu` 没有也不需要 Relay 算子**——它们是导出器留下的静态 mask 构造，折叠后整体消失。`Expand` 的控制形状同样折成 initializer 常量，直接落到既有的静态 `expand`，无需接形状值链。

## 3. 固定的 ABI

**输入（1 个）**：`input_ids`，`int64[1, 16]`。

**输出（17 个，顺序固定）**：`logits` `float32[1, 16, 6400]`，随后按层交错 `present_k_i` / `present_v_i`，各 `float32[1, 16, 4, 96]`，`i` 从 0 到 7。

顺序、命名与形状由 `test/minimind_l1a_llvm_test.cpp` 逐项钉住，任何漂移都会失败。

## 4. 数值证据

参考值由 ONNX `ReferenceEvaluator`（onnx 1.22.0 / numpy 2.5.3 / Python 3.12.3）在折叠后的同一张图上求得，与被测的 importer→Relay→LLVM 路径完全独立，因此比较是交叉验证而非自证。

输入 token（`seed=0`，`rng.integers(0, 6400)`）：

```
[5443, 4076, 3271, 1726, 1970, 262, 481, 105, 1121, 5204, 4156, 5841, 3223, 3882, 6212, 4668]
```

**全部 17 个输出逐元素比较**，编译配置 CPU/LLVM、opt level 0、标准 Relay 流水线（`fold_constant` + `simplify_expr`）：

| 规模 | 实算节点 | 最坏逐元素绝对差 | 编译+执行耗时 |
|---|---:|---|---:|
| 1 层 | 90 | 4.91738e-06 | 1s |
| 2 层 | 170 | 6.92904e-06 | 2s |
| 3 层 | 250 | 8.16584e-06 | 3s |
| 4 层 | 330 | 8.70228e-06 | 3s |
| **8 层（锁定配置）** | **650** | **8.82149e-06** | **6s** |

误差随层数缓慢增长且停在 1e-05 以下，量级与 float32 舍入一致；差异来源是 MatMul 归约顺序，不是语义偏差。8 层峰值 RSS 1.49 GB。

> 缩减层数的模型（`--layers`）只用于确认误差随规模的走向，**不是 E0 合同的证据**。锁定配置是 8 层那一行。

测试容差定为 `1e-4`，比实测最坏值高一个数量级：够容纳更深的模型，又不至于放过真实的数值退化。

## 5. 复现方式

```bash
# 1) 导出（需先应用非原地 mask 补丁）
out/venv/bin/python python/tools/export_minimind_onnx.py --static \
  --out out/minimind_onnx_noninplace

# 2) 生成 fixture（导入规范、参数、输入、ONNX 参考输出）
out/venv/bin/python python/tools/make_minimind_l1a_fixture.py \
  --onnx out/minimind_onnx_noninplace --out out/minimind_fixture_L8

# 3) 运行验收
KXC_MINIMIND_IMPORT_DIR=$PWD/out/minimind_fixture_L8 \
  ./out/build/dev-ninja-cpu/minimind_l1a_llvm_test
```

产物约 280 MB，不入库；环境变量未设置时测试跳过，不构成构建硬依赖。

## 6. 达成本 receipt 途中修掉的东西

导入侧四个缺口：`Identity`（31，归一为参数别名）、`ConstantOfShape`(16)/`Trilu`(8)/`Expand` 形状链(16)（由常量折叠一次消解）、`Gather` 运行时索引（embedding 查表）。另有恒等 `Cast`(57) 与推导链缺失的 `Add`/`Softmax`/`Transpose`。

编译侧一个系统性缺陷：**Relay 表达式是 DAG，但四处遍历按树走**，Transformer 残差的共享导致指数展开。四处（IR 打印器、`RelayPass::Mutate`、`RequireTyped`、`ANFNormalizer::CollectNames`）修复后，8 层从「OOM，峰值 >10 GB」变为「6 秒，1.49 GB」。详见 [G2 记录](G2_RECORD.md) §3.3。

## 7. 交接

- **给 M2**：本文 §3 的 17 输出即 present 侧 ABI。注意 prefill 的 present 是**图输出**，而 decode 的 past 是**图输入**（见 [E2 签名](M9_E2_SIGNATURE.md)）——两者是独立张量，M2 若要证明同一块 state 原址更新，需要把 past 输入与 present 输出绑到同一块存储并证明地址不变，而不是加一次拷贝。
- **给 M3**：本图的形状全部静态；bounded extent 绑定在本 receipt 范围内不涉及。
- **规模基线的前提**：§4 的耗时与 RSS 是 batch=1、seq=16、静态 prefill 的实测值。换序列长度、换 batch 或引入 decode 的 past 张量之后必须重测，不得直接外推。
