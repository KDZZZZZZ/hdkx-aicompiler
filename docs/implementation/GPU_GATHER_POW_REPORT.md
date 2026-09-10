# CUDA Gather 与 Pow 技术报告

2026-09-09。静态 Gather 已通过正式 Compiler → NVRTC → Driver → RuntimeSession，在 Windows RTX 4070 Ti SUPER 上完成 **120 组、30,460 个元素的精确比较**。Pow 补齐 CUDA 数学函数发射，另外 3 组、28 个元素全部符合参考。它们补齐了完整八层 MiniMind 静态 prefill 原先剩余的 1 个 Gather、33 个 Pow；完整模型数值见 [prefill 报告](GPU_MINIMIND_PREFILL_REPORT.md)。

## 原问题与方法

既有 Gather TE 已经生成负索引归一化和越界填零，但 CUDA 调度会统一拒绝含 Load 的索引。不能直接移除拒绝条件，否则动态索引、整数溢出及不完整 guard 都可能形成越界访问。

本轮在 [BindCudaThreads](../../src/tir/transforms/bind_cuda_threads.cc) 的单阶段输出所有权证明之后加入内部只读地址证明，继续使用既有 For、Load、Select、整数运算和静态 buffer 合同。TE compute、算子注册、运行时 ABI 和启动入口没有增加另一套实现。

1. 索引来源须为只读 int32/int64 输入，索引数组本身的地址先通过既有非负仿射证明。用 buffer 身份及规范化后的仿射系数标识同一次读取；分开构建的 `i` 与 `i*1+0` 可复用事实，`i+1` 不可。
2. 按 Select、And、Or 和谓词的实际短路路径传播整数区间。合取收紧范围，分支汇合使用保守包围区间；只跳过已经证明不可达的分支。事实仅用于当前表达式，不跨 Store 保留。
3. 对 Gather 的 `-A <= index < A` 有效分支，分别证明负索引加 A 和非负索引的结果均在 `[0,A)`，再证明最终扁平地址小于表元素总量。索引值仍在运行时读取；编译不扫描或缓存输入索引。
4. 每一步加、减、乘、除、余数和 cast 都检查区间、TIR dtype 与 CUDA C 整数运算宽度；溢出、丢失信息的 guard 转换及无法建立的界限在发射前拒绝。大表的扁平地址保留必要的 int64。
5. 完美外层循环中的零 extent 表示整个计算为空。保留一个合法的一块启动和工作量 guard，线程不访问零字节 buffer；空表且输出非空时则执行已有 typed-zero 分支。

参考 [ONNX Runtime CUDA Gather](https://raw.githubusercontent.com/microsoft/onnxruntime/main/onnxruntime/core/providers/cuda/tensor/gather_impl.cu) 的负索引、越界零值及大表 int64 地址处理，以及 [TVM 整数区间分析](https://raw.githubusercontent.com/apache/tvm/main/src/arith/int_set.cc) 的区间交并思路。KXC 的证明直接消费现有 TIR，没有移植索引运行库、约束求解框架或新 IR。

[CUDA emitter](../../src/codegen/cuda/codegen_cuda.cc) 将 `pow` / `tir.pow` 映射到 `powf` / `pow`，检查两个参数及相同 float32/64 dtype。其他数学调用仍是一参数合同，未开启 fast-math。**Relay Pow 的原有 float32 限制保持有效**；float64 只在现有 TIR/backend 层验证，不据此扩展 Relay 契约。

## 效果与挂载

运行时索引、负索引、重复索引、越界和空张量均由一个普通 Gather kernel 处理，无 CPU fallback。输出域仍由原有 CUDA pass 分配给独立线程，源表和索引表只读。完整模型的静态源码审计从 **616/650** 提升为 **650/650**。

行为变化进入原有身份链：pass schema **v4**，backend **cuda-nvrtc-driver-v4**，生成物通过正式 generator 更新；pipeline 测试拒绝旧 schema v3 冒充当前合同。没有新公开 API、注册入口或调度 IR。

| 实际硬件用例 | 覆盖与结果 |
|---|---|
| Gather 数据类型 | float32、float64、int32、int64、bool，各配 int32/int64 索引 |
| 形状 | 三维中间轴、257 项尾块、标量索引、二维索引、负 axis、空表轴、空输出 |
| 数值与重复执行 | 正/负合法值、重复值、正/负越界、INT32/64_MIN/MAX；同一 session 两次更换索引；120 组、30,460 个元素精确相等 |
| Relay Pow float32 | 二维 `[3,4]` 输入、同形指数与标量广播；负底数整数指数、正底数分数指数、负指数、零；2 组、24 元素误差 0 |
| TIR Pow float64 | 直接 NVRTC/Driver kernel，4 个元素误差 0 |
| 源码负例 | Pow 参数数量和 dtype 不符均拒绝；旧未绑定 TIR 等拒绝仍有效 |

调度结构测试包含 int32/int64 的六种真实 Gather lowering，其中 `[262144,8960]` 表仅作结构证明，**没有在 GPU 分配这张大表**。完整、等价仿射和范围内 narrowing 三种 guard 正例通过；只有上界、只有下界、不同 Load、先 narrowing 再 guard、先溢出再 guard、错误表容量六种反例均拒绝。原有无 guard 间接读取、跨线程读取及 64 KiB 私有存储超限反例保留。

## 验证与三轮审查

最终 Windows CUDA 专项 **8/8、5.28 秒**；codegen 数值共 **200 组、58,905 个元素**，包含前两轮归约/归一化回归。CUPTI smoke 与完整模型均实际执行。构建为 MSVC 19.29、CUDA 12.9 Update 1、驱动 576.57、LLVM 关闭；Linux CUDA 构建只作编译验证。

A：复用既有静态整数/仿射读取及分支 TIR，没有另建 evaluator 服务或算子专用 launcher。B：线程绑定、launch metadata 仍只有原 pass 一个权威，schema/backend 身份同步改变。C：生产正例、边界反例和完整模型均实际消费新增能力，未使用空接口或 CPU 替代。

首轮硬件运行发现测试误把 Array 对象身份当作形状内容比较，已改为逐维比较；随后确认 Relay Pow 拒绝 float64 正是原合同，测试改为 float32 Relay 与 float64 TIR 分层验证，未放宽算子。初始失败日志均保留。

三套完整 CPU/LLVM 回归、检查器和源码哈希的最终摘要见 [模型报告](GPU_MINIMIND_PREFILL_REPORT.md)。原始回执位于 `out/windows-gpu/logs/cuda-gather-*`；最终 12 个源码文件先逐项 SHA-256 对齐再在 Windows 构建，清单为 `out/windows-gpu/cuda-gather-final-manifest.json`。

## 范围限制

新增证明仅用于静态单阶段只读间接访问；多阶段间接存储、任意嵌套索引、动态 CUDA extent、原地 scatter/更新、设备端 KV 状态仍未开放。无法证明的安全表达式也可能被保守拒绝。单线程归约策略、64 KiB 私有临时存储上限和 Slice 的既有步长限制不因本模块改变。

这组小形状数值不提供吞吐或加速比结论；Compute Sanitizer 的既有 Windows 插桩问题仍未解决，不能把数值通过解释为内存插桩通过。
