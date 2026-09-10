# CUDA 多阶段归一化技术报告

2026-09-09。Softmax、MaskedSoftmax、LayerNorm、ReduceMean 已在 Windows RTX 4070 Ti SUPER 上通过 Compiler → NVRTC → Driver → RuntimeSession 的实际数值测试；MatMul → Softmax → MatMul 的三头组合也已执行。最终 CUDA 专项 **6/6 通过**，本轮新增 60 组数值、25,460 个输出元素。本模块提供静态归一化的正确性基础，不包含协作归约或模型性能优化。

## 原问题与采用的方法

前一模块能把一个输出元素及其串行归约交给一个 CUDA 线程，但 Softmax 等算子包含多个生产阶段和内部 Allocate，仍无法通过调度。直接为每个线程保留整张中间张量既不能证明阶段间独立，也会浪费临时存储。

本轮继续使用既有 [BindCudaThreads](../../src/tir/transforms/bind_cuda_threads.cc)，不新增 IR、TE 调度原语或启动入口。它读取 TE→TIR 已有的 Allocate、For、Seq、Store、Load，根据数据依赖确定线程的独立坐标，再把各阶段限制在该线程拥有的部分。

1. 每个阶段必须完整写出一个静态、稠密张量；循环与形状一致，归约只能读取已初始化的当前输出元素。标量归约的初始化和更新可能被 SeqStmt 展平，因此只把连续写同一标量的语句恢复成一个待证明阶段。
2. 检查单一生产者、阶段顺序和已声明的 buffer。只内联“只读参数 Load 或其 cast”的临时复制，包括 LayerNorm 的 data/scale/bias float64 转换，以及广播 mask 复制。内联前仍检查原始阶段顺序和消费者索引范围；不重复计算归约或其他中间结果。
3. 剩余阶段的输出秩必须一致。凡是所有阶段都具有相同非单位长度的坐标，都作为候选独立坐标。例如 Softmax `[2,3,5]` 沿 axis=1 归约时，axis=0、2 合成 10 个线程，而每个线程按步长 5 负责 3 个输出。
4. 对所有被写 buffer 的读写地址作非负仿射和整数溢出证明。移除线程坐标后，把系数和常量拆成原形状的各维坐标，并检查每一维不会进位到另一线程的坐标。仅检查最终扁平地址不越界是不够的：测试中的 `r*4` 访问仍在总容量内，却会跨越本应由 `r*5` 保持的线程边界，因此必须拒绝。
5. 用其余坐标重建私有地址，把独立维度缩为 1，并保持阶段的原始顺序。总私有临时存储限制为每线程 64 KiB；超限在代码生成前拒绝。归约继续在线程内串行执行，共有坐标经原有 int64 工作量、block/thread 绑定和尾块 guard 生成唯一启动元数据。

这参考了 [TVM DLight general reduction](https://raw.githubusercontent.com/apache/tvm/main/python/tvm/s_tir/dlight/gpu/general_reduction.py) 从生产者/消费者访问关系对齐空间与归约坐标的做法。KXC 使用现有 TIR 自己证明私有存储，没有移植其共享内存、compute_at 或线程协作框架。

## 实际效果与接口衔接

中间轴 Softmax 的四个私有 buffer 为 `[1,1,1]`、`[1,3,1]`、`[1,3,1]`、`[1,1,1]`，float32 每线程合计 **32 字节**。一个线程负责一条归约行；每个算子的多阶段计算仍在一个普通 kernel 中完成。没有 GPU 阶段间全局临时分配或额外 RuntimeSession 调度器。

[CUDA emitter](../../src/codegen/cuda/codegen_cuda.cc) 原来只识别 `tir.exp` 等拼写，实际 TE 输出的是 `exp`、`sqrt`。现在这两种现有拼写进入同一组 float32/64 数学函数映射，并检查单参数和一致 dtype；其他调用仍拒绝。使用常规 CUDA 数学函数，未启用 fast-math。

挂载链为：既有 Relay/TE 计算 → TE→TIR → pass schema **v3** → CUDA backend **cuda-nvrtc-driver-v3** → 既有 Compiler 与 RuntimeSession。两个身份都由现有 pipeline/artifact key 消费；pipeline 测试拒绝伪装成旧 schema v2 的计划。CPU 默认 TE policy 和 LLVM lowering 语义未改变。

硬件测试同时发现并修复了同步复制/清零与非阻塞消费流之间的竞争，具体方法、96 次数据检查和限制见独立 [同步内存报告](GPU_SYNC_MEMORY_REPORT.md)。该修复属于 runtime DeviceAPI 的完成语义，未在算子测试中插入同步来掩盖问题。

## 数值与验证范围

环境为 Windows 原生 Release、MSVC 19.29、CUDA 12.9 Update 1、驱动 576.57、RTX 4070 Ti SUPER，LLVM 关闭。主机参考使用独立循环和 double 中间结果，不调用 KXC 的 TE 或 GPU 输出作为真值。

| 场景 | 覆盖 |
|---|---|
| Softmax float32/64 | `[257,7]` 尾块、三维首轴/中间轴、rank-1、单行长度 257；普通 logits 和正负类型极值 |
| MaskedSoftmax float32/64 | 同形 runtime bool mask、`[1,3,1]` 广播、标量 mask；同一 session 四次运行，含全 False、全 True、部分 mask、极值；被遮罩 NaN/Inf 不污染结果，零值要求正零 |
| LayerNorm | float32 输入与 scale/bias、float64 累加；257 行、二维后缀和 rank-1；常量行、大均值小方差及两种 epsilon |
| ReduceMean float32/64 | 单轴、多轴、全归约；keepdims true/false 和标量输出 |
| 注意力组合 | 三个 head，Q `[3,5,7]`、K `[3,7,11]`、V `[3,11,4]`；普通生产 MatMul→Softmax→MatMul |

Softmax/MaskedSoftmax float32 使用 `2e-5*(1+abs(reference))`，LayerNorm/ReduceMean float32 为 `1e-5`，float64 为 `1e-12`；所有有限参考比较都拒绝 NaN/Inf。极值用例验证稳定性，普通 logits 用例验证非平凡概率分布。组合图是确定性算子链，不代表完整 MiniMind 模型。

| 本轮实际数值 | 组数 | 输出元素总数 | 最大绝对误差 |
|---|---:|---:|---:|
| Softmax | 20 | 8,732 | `2.98023e-8` |
| MaskedSoftmax | 24 | 14,752 | `5.96046e-8` |
| LayerNorm | 3 | 1,836 | `1.81899e-12` |
| ReduceMean | 12 | 80 | `1.49012e-8` |
| 三头注意力组合 | 1 | 60 | `1.11759e-8` |

Windows 最终专项包含 pipeline、调度、Device/NDArray、codegen、runtime profiling 和 CUPTI smoke，共 6/6、1.94 秒；codegen 25 个子项通过，连同旧有 17 组回归共 **77 组、28,417 个输出元素**。Windows 的 LLVM profiling 子项因构建关闭 LLVM 而跳过，不计入 LLVM 证据。测试耗时包含缓存和驱动开销，不能用来推导算子加速比。

完整 CPU/LLVM 回归全部通过：默认 **55/55、142.41 秒**，adaptive **55/55、212.56 秒**，bounded **70/70、225.70 秒**。三套均实际执行完整视觉、三组固定单图联合模型和 state/decode fixture；adaptive 还执行实际模型热替换，bounded 执行完整八层变长 prefill/decode（含模型请求批处理）。逐测试日志已核对这些模型用例没有 SKIP。未设置的 ResNet18 执行和独立 L1a import opt-in 仍跳过；默认/adaptive 的 bounded-only 用例按构建门禁跳过，均不计为本轮模型证据。

40 个 Relay operator、20 个 pass 合同与生成物新鲜度、NLP 门禁、284 个 include 边界、90 个已安装及 10 个实验 public header 的独立编译、67 篇文档链接、git diff 检查均通过。另以篡改矩阵确认：文本升为 validated、任意替换局部门禁、开放整模型三类修改均被拒绝。

复核文件位于 `out/windows-gpu/logs/`：`cuda-multistage-final-ctest.log`、`cuda-multistage-final-build.log`、`cuda-multistage-numeric-audit.json`、`cuda-multistage-cpu-regression-audit.json`、`cuda-multistage-symbol-audit.json`。CPU/CUDA 静态库分别有 1603/1614 个强定义，重复为 0；本地 CUDA 12.9 只做构建，真实设备执行来自 Windows。

本轮最终 13 个源码文件逐项 SHA-256 校验后才在 Windows 构建。增量源码包 `out/windows-gpu/cuda-multistage-final.tar.gz` 的 SHA-256 为 `2ebc5d989b006fee8e88a70379a1f2de625683098f5fb36443617bfd4ca80deb`，最终 Windows CTest 日志为 `b5a7a4882a5a2303df981b2fbba842b5b9907fc65e93d2b15c128b4833cb8b06`。初始归约失败、内存修复后暴露的 exp 拼写错误以及最终通过日志均保留，未覆盖前一模块的证据。

## 三轮审查与尚未开放的范围

第一轮检查真实 lowering 和整数/存储证明：七组结构正例通过；覆盖跨坐标访问、先读后写、缺少初始化、条件初始化、零次归约、重复生产者、条件 Allocate、内联复制的前向读取、未声明 Load 和 64 KiB 超限。原始 PrimFunc 在变换后保持不变。

第二轮以 Windows 实际编译和执行检查 producer/consumer 衔接，发现并修复同步内存竞争及数学调用拼写不匹配。第三轮检查最终源码、CPU/LLVM 三配置回归、契约生成物、能力矩阵、架构边界和报告引用。首次默认 CPU 回归为 54/55：旧 executable_capability_test 仍要求小型 LayerNorm 被拒绝，现已改成长度 10,000、必须因 64 KiB 预算而拒绝的负例。该契约测试在 Linux 与 Windows 均通过（Windows 额外 1/1、0.20 秒），初始失败回执保留。

矩阵只为 stable_softmax、masked_softmax_all_masked、normalization 增加限定范围的 `implemented` CUDA 局部实测证据；不会通过文本把它们升为整体 `validated`。完整 prefill/decode、GPU KV state、请求批处理、Gather 间接索引、bounded CUDA 和完整模型 profiling 继续关闭。

没有共享内存、warp reduction、Tensor Core、自动调优或吞吐结论；单行归约可只有一个活跃线程，长行可能因私有存储上限而明确拒绝。Compute Sanitizer 仍受既有环境插桩问题阻挡，未获得“零内存错误”的回执。后续优化需保持这些数值与拒绝边界，而非把本轮局部成功解释为任意 CUDA 图均已支持。
