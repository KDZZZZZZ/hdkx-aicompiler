# CUDA 有界多阶段归约与注意力技术报告

2026-09-09。状态：本模块完成，能力边界见文末。生产 CUDA 数值、设备活动、最终回归和证据归档均已通过。目标依据：[PROJECT_GOAL](../PROJECT_GOAL.md)。

## 达到的效果

同一份 CUDA 产物现在可以执行变长 Softmax、显式 masked Softmax、固定归约轴的 ReduceMean/RMS 归一化，以及 `QKᵀ → 广播 mask → Softmax → V` 的三头注意力链。它们从 `Compiler::CompileBounded` 进入普通 `RuntimeSession::RunAsync`，实际运行改变 B/S 或 B/Q/T，不触发编译和 cache mutation。

Windows RTX 4070 Ti SUPER / CUDA 12.9 上，新增两组图共 **28 次运行、41,604 个值**：归约组合最大绝对误差 `1.19209289551e-7`，注意力最大误差 `2.38418579102e-7`，均低于测试阈值 `1e-6`。这补上完整变长 GPU 模型所需的多阶段归约；完整八层 bounded MiniMind、索引/形状重排、KV 状态和请求批处理仍是后续验收。

## 方法与实现

### 复用已有所有者

[上一模块](GPU_BOUNDED_CORE_REPORT.md)已经打通实际 extent 的 uint64 设备 ABI、按上界启动与实际形状寻址，以及上界进入缓存身份。本模块不增加形状求值器、参数 ABI、运行时状态表、launch owner 或 IR。

唯一实现变化位于 `BindCudaThreads` 内的有界地址证明。原有多项式解析、整数宽度检查、输出单元独占、支配初始化与全域读地址检查被复用；在其上补上多阶段的生产顺序和临时存储投影。已有静态多阶段证明继续使用原路径。

参考采用两个不变量：MLIR Linalg 区分独立迭代和归约依赖，循环变换须保持依赖顺序；CUDA 自动数组属于线程私有作用域，实际可能落到设备 local memory，不能据此宣称它比全局内存快。[MLIR Linalg](https://mlir.llvm.org/docs/Dialects/Linalg/)、[CUDA Local Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#local-memory)。

### 多阶段划分与行内地址证明

1. 读取最外层的无条件 `Allocate` 和顺序执行的完整 stage。参数与 scratch 必须唯一，输入只读，每个 scratch/public output 恰有一个完整 producer。
2. 每个 stage 的数据循环须逐维等于该输出的实际形状；Store 地址仍须等于其紧凑行主序编号。归约只能读取已初始化的同一输出单元。
3. 所有 stage 共有的前缀坐标决定独立行。比如 Softmax 的 `[B,Q,T]` 中，max/sum 是 `[B,Q,1]`，前缀为 `[B,Q]`；一个线程按顺序完成该行全部 stage。
4. 读取其他 stage 结果时，必须证明 producer 已完成，且地址恰含本线程的行偏移。减去这个偏移后，地址不能再含行变量，且剩余索引须落在实际行域内。只证明整块数组内没有越界是不够的。
5. 仅对通过证明的 scratch 地址消去行坐标；公共输入/输出仍使用原实际地址。保留的循环使用本次真实长度，不按上界补算。

前缀之外的坐标保持串行。因此 `[B,S,7]` 在 axis=1 做 Softmax 时，一个线程拥有一整个 `[S,7]` 平面；本模块没有把最后的 7 列也拆给线程。这个保守范围足以执行本次注意力与归一化，后续并行化须继续证明索引和依赖。

### 静态容量与实际长度

scratch 的逻辑形状保持符号维度，证明完成后，CUDA 私有数组按剩余各维的编译上界分配；寻址仍按实际紧凑 strides。每线程所有数组合计不超过已有 **64 KiB** 预算。零上界的独立域没有 tensor 访问；可证明零次的内部循环被移除。数组的物理容量可以大于本次实际使用区间。

例如测试中的三阶段 copy→sum→normalize，输入为 `[N,S]`、N≤19、S≤17。每个线程只需长度 17 的 float32 临时行和一个 scalar，合计 **72 字节**，而不是整个 `[19,17]` scratch。symbolic S 改变行内循环与地址，不改变编译产物。

超过预算、缺 producer、重复 producer、先读后写、跨行读取、窄整数或未知间接地址都在 backend 编译/缓存发布前拒绝。没有 runtime fallback。

### 身份与能力边界

CUDA pass schema 升为 **6**，backend identity 升为 **`cuda-nvrtc-driver-v6`**。已有 canonical schedule 继续编码 runtime extent 顺序和有限上界；可调用参数 ABI 不变。CPU policy、shape 求值和持久状态所有权不变。

有界 ReduceMean 继续要求归约维度的长度静态且为正；RMS 归一化使用现有 `mul → reduce_mean → add → sqrt → divide`。独立的 `nn_layer_norm` bounded admission 尚未开放，本报告不把 RMS 组合等同于该算子的动态支持。

## 数值与执行证据

生产 consumer 为 [bounded_cuda_test.cpp](../../test/bounded_cuda_test.cpp)，沿用已有 CTest 和 [CUPTI 验证器](../../test/cuda_profile_bundle_test.py)。没有新硬件测试入口或测试专用 launcher。

| 新增图 | 形状与覆盖 | 结果 |
|---|---|---|
| 九个 primitive 的归约组合 | 输入 `[B,S,7]`，B∈[0,19]、S∈[1,18]；两个 Softmax 轴、广播 bool mask、全 False 行、零输入 RMS、最大边界、空 batch、两条非默认流 | 16 次运行；38,664 个值；最大误差 `1.19209289551e-7` |
| 五个 primitive 的注意力 | Q=`[B,3,Q,7]`、K=`[B,3,T,7]`、V=`[B,3,T,5]`；B≤3、Q≤19、1≤T≤23；mask 在 head 轴广播；覆盖最大值、Q≠T、空 Q、空 B | 12 次运行；2,940 个值；最大误差 `2.38418579102e-7` |

普通 Softmax 输入先加 1000，独立主机参考使用 double 稳定 softmax。RMS 的参考直接计算平方均值与归一化。注意力参考直接计算 QK、mask、稳定 softmax 和乘 V，不复用编译器或 TIR。

归约图前七组 shape 各运行两次并比较全部五个输出；最后一组的两次运行专门把 masked 位置改为 NaN/+Inf/-Inf，只核验 masked 输出的 210 个值，其余 tuple 分支会有意消费非有限数据，不计入数值正确性声明。masked 输出在 False 位置和全 False 行均须**精确正零**，不能仅落入误差阈值。

临时输入句柄在等待 completion 前释放；两个 session/stream 交替执行；保留输出在 graph/session/cache 释放后再次检查。所有正常运行与前置拒绝期间，primitive cache 的十项统计不变。新增两组各有四项非法输入，包括超上界、零 softmax 归约长度和错误实参数量，均为零提交。

## 设备活动与反例

| 新增图 | CUPTI kernels | uint64 extent H2D | 字节数 |
|---|---:|---:|---:|
| 归约组合 | 144 | 288 | 2,304 |
| 注意力 | 60 | 168 | 1,344 |
| 新增合计 | 204 | 456 | 3,648 |

每个 kernel 均关联实际 run/call_index、launch 和完成；每次 scalar DMA 必须在其 consumer kernel 的设备执行开始前完成。每个 call 的固定 grid/block 与编译上界吻合。注意力各 call 使用 2/3/3/3/3 个 scalar，分别对应其实际 ABI。

连同前一模块两组基础图，当前 consumer 共 62 次有 profile 的运行、286 条 kernel、638 次 scalar DMA、22 次零提交拒绝和 32 项回执篡改拒绝。另有两次上界缓存区分的 ReLU 运行未计入 CUPTI：全部数值总计 52,576 个，其中 profile 主图为 51,934 个。

新增 TIR 正例核验私有 72 字节 scratch、输入 PrimFunc 不变与两个零上界情形；13 个反例覆盖阶段顺序、缺失/重复 producer、跨行与尾部越界、错误 Store、未初始化读取、错误 predicate/shape、超预算和间接索引。普通 Softmax 的超大上界和 Gather 的动态间接索引也通过真实编译入口拒绝。

## 复现与回归

沿用 [基础模块配置](GPU_BOUNDED_CORE_REPORT.md#最终硬件回归与复现)：CUDA 12.9、四个 dynamic/restricted/exact/bounded gates 开启，Windows LLVM=OFF。运行：

```sh
cmake --build <cuda-build> --config Release --target bounded_cuda_test
ctest --test-dir <cuda-build> -C Release -V -R '^bounded_cuda_test$'
```

初次完整 consumer 通过 1/1、2.23 秒；原始记录在 Windows `logs/bounded-multistage-stage4-*`，Bundle 为 `build-bounded-cuda/bounded-cuda-evidence/bounded-ivalqc2o/`。初次接线曾因测试使用了不存在的 `multiply` 算子名失败，修正为契约声明的 `mul` 后通过；未新增别名或改变算子注册。

最终 Windows 回归已完成，所有 CUDA consumer 均重新构建：

| CUDA 配置 | 结果 |
|---|---|
| bounded 四门禁开启 | 16/16，48.07 秒 |
| 默认四门禁关闭 | 13/13，44.37 秒 |

两套回归均实际执行完整静态 MiniMind prefill、KV 状态与四步 decode，没有把 fixture 缺失或硬件跳过计为通过。首次尝试全目标构建时，可选 `onnx_importer_fixture` 因 Windows 源码目录没有 `resnet18.onnx` 失败；最终构建显式覆盖上述 CUDA 回归的全部可执行目标。原始失败日志与最终日志一起保存，没有修改 ResNet 或删除该测试。

最终四组 bounded Bundle 为 `build-bounded-cuda/bounded-cuda-evidence/bounded-btmse_cv/{elementwise,matmul,reductions,attention}`。原始归档 `out/windows-gpu/bounded-multistage-evidence.tar.gz` 为 8,740,420 字节，SHA-256：`0b8c12af1325f33ce6688b76a43681676cfc3043f3a1487110631f1972929756`。取回后逐字节核验归档；26 个非空 Bundle 通过 schema 与真实 agent 诊断消费，6 个刻意隔离的 foreign profile 保持空记录，单独登记。四组 bounded 另执行 32 项篡改拒绝。分析仅修改解压副本中的诊断，所有原始 `events.jsonl` 的 SHA-256 保持不变。

执行源码 overlay 为 `bounded-multistage-verified.tar.gz`，SHA-256：`5a7c1b258eace2d5725e226cb43cfcb42c5093468c325b1c318a1ef014d13ca2`。当时完整 699 文件 manifest 的 SHA-256 为 `e61281aa9a12975feeae1ac04e5c22b6df88a4e85b8b4aac9b45e27335b4c8f8`；Windows 源码逐项核验无差异。后续仅报告文字与最终回执单独同步，不改变已执行的生产代码或 consumer。

CPU/LLVM 三套配置均完成全量构建与 CTest：

| CPU 配置 | 结果 | 实际模型负载 |
|---|---|---:|
| 默认 | 55/55，148.73 秒 | 5 |
| adaptive | 55/55，217.21 秒 | 6 |
| bounded | 70/70，230.41 秒 | 8 |

模型负载包含固定视觉、三种图文输入和文本容量 decode；adaptive 额外执行真实 MiniMind 两代替换；bounded 额外执行完整变长 prefill/decode 和请求批处理。已逐个核验数值/提交完成标记且无 fixture 跳过；bounded 的请求批处理嵌入 decode CTest，因此是 7 个模型 CTest、8 项模型负载。原始回执为 `out/windows-gpu/logs/cpu/kxc-bounded-multistage-*`，审计为 `out/windows-gpu/logs/bounded-multistage-cpu-regression-audit.json`。

生成的 Relay/Pass 契约保持最新，40 个算子和 20 个 Pass 校验通过；284 个文件通过 include-layer 检查；90 个安装头文件与 10 个实验头文件通过清单和独立编译检查；diagnosis pytest 为 4/4。NLP 矩阵保留局部 `implemented` gate，新增报告、生产 consumer、设备验证器和原有证据必须同时存在；58 项错误升级、错误 gate 或证据缺失均被拒绝，GPU 请求批处理继续关闭。

Linux 的 LLVM 20.1.2 + CUDA 12.9、四个 shape gates 开启的混合构建已完成。session、pipeline、kernel signature、CUDA 结构证明、TE 和 profiling 六项通过；GPU consumer 因本机无可用设备返回 77，明确记为跳过。真实 GPU 证据来自上述 Windows 主机。默认 CPU 与混合后端静态库分别检查 1,603/1,894 个外部强符号，均无重复。

最终文档共 74 篇，通过本地链接和陈旧路径检查。最终报告文字 overlay、两端 699 文件一致性与归档摘要记录在 `out/windows-gpu/logs/bounded-multistage-completion-audit.json`；其中单独保留执行时 manifest 与最终文档 manifest，避免把后续文字更新混作已经重跑的代码。

## 三轮 Ponytail QA

| 轮次 | 结论 |
|---|---|
| A：复用 | 复用原有多项式证明、runtime-extent ABI、ShapeProgram、RuntimeSession、CUPTI validator 和 64 KiB 预算；不加新 IR/API |
| B：唯一权威与身份 | 所有线程/行/scratch 决策仍在 BindCudaThreads；上界已进入 schedule identity，本次以 pass/backend v6 区分行为 |
| C：真实 consumer | 普通编译入口执行多阶段归约和完整 QK-mask-Softmax-V 链；非法输入、跨行读取、错误 producer 和证据篡改都有拒绝检查 |

仍未完成的范围：完整 bounded MiniMind 模型、动态 Gather/形状重排、bounded CUDA KV 状态与请求批处理、bounded nn_layer_norm、带状态热替换，以及 cooperative/tiled/tensor-core 调度优化。当前没有吞吐加速结论，也没有 Compute Sanitizer 插桩通过结论。总 project goal 保持进行中。
