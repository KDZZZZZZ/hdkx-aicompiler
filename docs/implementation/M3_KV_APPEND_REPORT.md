# M3 技术报告：有界 KV 长度追加与派生形状

日期：2026-09-09。范围：CPU/LLVM，现有 Relay/TE/RuntimeSession 的 fresh-output 路径。

## 达到的效果

一份编译产物可以把 `[B,P,2,3]` 的缓存内容与定长片段拼接，输出长度为 `P+C`，并让后续算子和形状控制读取实际输出长度。B∈[1,3]，P∈[0,8]；C 是编译时常量。本次执行了尾部追加 C=1、头部插入 C=2、空片段 C=0，以及连续两次追加得到 P+2。

四条图各自编译一次，再由同一 RuntimeSession 执行 `(B,P)=(1,0),(1,1),(2,4),(3,8)`。共 **16 次运行、48 次 LLVM kernel 调用**，所有复制结果、ReLU 结果、ShapeOf 和 Shape→Gather 控制值均与独立参考逐元素完全一致。B>1 的结果检查了每个 batch 的行跨度，空缓存和空输出也实际执行。

这是原始 MiniMind decode 所需的 `past → present` 长度变化基础。它当前分配新输出，尚未把这条路径绑定到持久 KV 容量或有效长度。本报告不把它计为完整变长 decode、原地 KV 更新或请求级批处理完成。

## 使用的方法

### 沿用 DimExpr 的证明和模块形状合同

图输入仍只能用直接 Symbol 或 Const 绑定。内部值允许 **一个 Symbol 加非负常量**；识别时由 DimExpr 求出常量项，再与 `DimExpr::Add(Symbol, Const)` 做规范结构相等比较。求值结果本身不能替代结构证明，因此乘法、多个符号相加、减法和输入形状反解仍不进入该子集。

每个 unit 优先引用与输出维度完全相同的输入轴；拼接产生新长度时，在既有 `DynamicShapeExpr::InputAxis` 中记录常量 offset。只有 Concatenate 能引入新的输出 offset；不能把 ReLU 等形状透明算子的输出伪造成 P+1。后续 ShapeOf/Shape→Gather 可以直接引用已证明的 P+C 输入轴。

局部输入 guard 使用完整维度表达式区分 P、P+1 等值。相同表达式继续复用已有相等 guard，范围由 DimExpr 在原符号上下界求值；派生维度不错误继承原符号的整除条件。派生长度超过 int32 循环域时，在编译/cache 之前拒绝。

输出分配继续由既有 ModuleInvocationContract 执行：把带 offset 的局部轴投影为已有 `ModuleShapeExpr::Add(InputAxis, Const)`，同时把 offset 纳入最大输出字节数。内核的有序 extent 参数仍只传输入轴长度，运行时无需增加一个 P+C 标量或新的 shape evaluator。

### 在 TE/TIR 中执行实际拼接

Concatenate 的类型推导和 TOPI 现在允许一个动态拼接轴与静态片段。非拼接轴仍须由 producer 证明相同；两侧拼接轴均动态时拒绝。规则保留了 [ONNX Concat 的轴和非轴维度要求](https://onnx.ai/onnx/operators/onnx__Concat.html)，支持范围小于完整 ONNX。

TE 的输出循环长度是输入 extent load 加常量，分支选择及源地址也使用真实输入长度。惰性 Select 保证空片段不会触发无效分支的 load。新增的内部结构识别函数只接受生成的 uint64 extent load 转为 int64 后加非负 int64 常量；外来 buffer、负 offset、乘法、两个 load 相加均拒绝。原来的“直接 load”识别接口保留原语义。

同一识别规则接入输出形状核对、tensor/归约域验证、buffer shape、KernelSignature 构建、LoweredFunction 校验、schedule identity 和中间存储上界。带 offset 的循环必须有该次 lowering 的输入上界，并证明 `upper + offset ≤ INT32_MAX`。没有通过修改元数据来冒充执行能力。

### 身份版本

- Concatenate schema：v2 → **v3**。
- bounded applicability：v10 → **v11**。
- unit shape contract：v6 → **v7**，canonical 字节包含 offset。
- bounded TE schedule：`bounded-dynamic-serial-v2` → **v3**，循环轴身份包含 extent 参数序号和 offset。
- preparation 保持 v3，ModuleInvocationContract 保持 ABI v4；后者已有 Add 表达式，无需新增 runtime 表达式语义。

## 验证

专项测试为 [bounded_kv_append_llvm_test.cpp](../../test/bounded_kv_append_llvm_test.cpp)，由默认关闭的 bounded gate 下的 CTest 注册执行，无下载或模型文件依赖。

| 检查 | 结果 |
|---|---|
| 三种单次拼接 | 每图四组 B/P、12 次 LLVM 调用；拼接、ReLU、ShapeOf 结果精确 |
| 连续两次追加 | 四组 B/P、12 次 LLVM 调用；P+2 的 payload、batch 跨度和折叠 Shape→Gather 值精确 |
| 运行反例 | 18 次：P 上界、B 上界、batch 不一致、片段长度、rank、dtype；runtime alloc/submit 均无增加，全部 primitive cache 统计不变 |
| 准备反例 | 两个动态拼接轴、非轴表达式不一致、P+C 超出 int32；cache 不变 |
| extent 结构 | 正向/反向加法通过；负常量、乘法、双 load 和外来 buffer 拒绝，旧直接-load 接口仍拒绝加法 |
| 专项回归 | 8/8 通过，包括 restricted shape、bounded graph、shape value、attention、RoPE、InferType 和 TE schedule |
| 默认 CPU/LLVM | Debug，动态 gate 关闭，48/48 CTest 通过，39.70 秒 |
| bounded CPU/LLVM | Debug，61/61 CTest 通过，115.18 秒；LLVM 20.1.2 |
| 真实模型回归 | 显式启用已有六组模型 fixture：投影、拆头、GQA、RoPE、第一层 attention 和完整八层 prefill；最后一项共 3,710 次调用、17 输出最大误差仍为 8.46386e-6，因果性检查通过 |
| Python | 276/276 通过，1.24 秒；本次未扩展 ONNX int64 Sub |
| 契约与集成 | Relay 37/37、pass 20/20 及生成物检查通过；279 个源码 include 扫描、89 个安装头和 10 个实验头编译、52 份文档检查通过；静态库 1,845 个 strong symbol，无重复定义 |

执行命令和日志：

```bash
cmake --build out/build/bounded-llvm -j4
KXC_MINIMIND_BOUNDED_PREFILL_DIR="$PWD/out/fx_minimind_bounded_prefill" \
KXC_MINIMIND_ATTENTION_DIR="$PWD/out/fx_minimind_attention" \
KXC_MINIMIND_ROPE_DIR="$PWD/out/fx_minimind_rope" \
KXC_MINIMIND_GQA_DIR="$PWD/out/fx_minimind_gqa" \
KXC_MINIMIND_HEADS_DIR="$PWD/out/fx_minimind_heads" \
KXC_MINIMIND_PROJECTION_DIR="$PWD/out/fx_minimind_projection_scalar" \
ctest --test-dir out/build/bounded-llvm --output-on-failure --no-tests=error
cmake --build out/build/dev-ninja-cpu -j4
ctest --test-dir out/build/dev-ninja-cpu --output-on-failure --no-tests=error
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 out/venv/bin/python -m pytest -q test/ python/
```

静态 MiniMind prefill/decode-loop 和 ResNet 的可选实际模型分支本轮未开启，不能把其历史证据写成本轮重跑；CUDA 未执行。

日志位于 `/tmp/kxc-m3-append-{focused,bounded-ctest,default-ctest,python}.log`。事件包位于 `out/build/bounded-llvm/out/bounded_kv_append_profile/events.jsonl`，包含 stage、B、P、append_count 和 kernel 执行事件；48 对 submit/exec、144 次 alloc、16 次成功 Run 与 18 次输入拒绝均有记录。这些小张量检查不构成性能 benchmark。

## 三轮 Ponytail QA

1. **复用检查**：扩展 DimExpr 投影、DynamicUnitShapeContract、已有 ModuleShapeExpr::Add 和 Concatenate；没有新 IR、状态容器、执行器或路由器。
2. **权威与身份检查**：图输入绑定未放宽；offset 由 producer 证明，module 分配和内核循环引用相同输入轴；schema、applicability、unit 与 schedule 版本覆盖行为变化。
3. **消费者与反例检查**：字段已由真实 LLVM 循环、输出分配和连续算子消费；零长度、多 batch、累计 offset、错误输入和 cache/alloc/submit 副作用均有实际检查。源码中的旧直接-load 接口继续服务要求直接引用的调用方。

## 完整 decode 的剩余工作

实际原始导出 `out/minimind_onnx_bounded/minimind_decode.onnx` 的 SHA-256 为 `c0733d61304970a37fff14fec8d56336baf945576398a0f5e0d14d8acd1d5fb2`，有 2,184 个节点，纯静态折叠后 1,423 个节点。输入为 `input_ids[B,1]` 和 16 个 `past_[kv]_[0..7][B,P,4,96]`，输出为 logits 和 16 个长度 P+1 的 present。

本轮复核该图时发现，除拼接外还需要位置表窗口 `P:P+1`、形状控制中的 Add/Sub 及随后完整图证明。当前 Python shape-source 导入在实际节点 `/model/model/layers.0/self_attn/Sub` 处明确拒绝 int64 减法。该控制值来自 attention 总长度减当前长度，即 `(P+1)-1`；其后仍需保持与真实 P 的证明关系。没有冻结代表长度来绕过它。后续继续完成这条实际图，再接 RuntimeSession 的容量、有效长度和 prefill 状态交接。现有静态容量型 decode 的历史验收保持原口径，见 [M2 报告](M2_MINIMIND_STATE_REPORT.md)。
