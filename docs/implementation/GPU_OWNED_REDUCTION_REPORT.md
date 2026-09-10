# CUDA 多维输出与线程内归约技术报告

2026-09-09。现有 CUDA 生产链已能执行静态 MatMul、Dense、批次广播 MatMul，以及多维 Where、Slice、Concatenate。Windows RTX 4070 Ti SUPER 的专项 **6/6 通过**，包含真实 NVRTC 编译、Driver 启动和 RuntimeSession 输出。每个线程负责一个输出元素，并在该线程内串行完成归约；本轮提供正确性基础，尚未做矩阵乘分块或性能优化。

## 问题与采用的方法

此前 `BindCudaThreads` 只接受一个外层循环和独立 Store，任何嵌套循环、写后读或内部临时分配都会被拒绝。普通 TE MatMul 本来已正确表示为“输出数据循环 → 初始化 → 归约循环”，但 CUDA 调度无法区分独立输出坐标和共享同一输出的归约坐标，因此在 NVRTC 前被拦截。多维逐元素计算也被同一限制挡住。

本轮只扩展既有 [BindCudaThreads](../../src/tir/transforms/bind_cuda_threads.cc)。TE 默认 CUDA 调度仍保持 serial，既有 `te::Program`、TE→TIR、PrimitiveUnit、Compiler、KernelSignature 和 RuntimeSession 继续承担原来的职责。没有新增 IR、TE bind、调度 registry 或模型专用执行入口。

证明采用以下约束：

1. 取函数最外层完美嵌套的静态数据循环，计算行主序 stride，并展平为一个输出域。标量域的工作量为 1。
2. 每个 Store 的地址必须是同一组数据坐标的行主序仿射式，不能依赖归约坐标。不同线程因而写入不同元素；写入地址还必须落在声明的静态、稠密输出 buffer 内。
3. 仅承认无条件或常真谓词 Store 的初始化。读取任一被写 buffer 时，必须已有支配该读取的初始化，且读取地址必须等于当前线程拥有的输出地址。条件初始化、零次循环内初始化、跨线程读取和重绑定变量均不能通过。
4. 保留归约 For 在该线程内串行执行。检查数据轴、归约边界、地址每个中间表达式及总工作量的整数范围；通过证明的循环上下限统一发射为循环变量的类型，避免 C++ 先在较窄类型中计算 `min + extent`。
5. `blockIdx.x * block_size + threadIdx.x` 使用 int64。按现有策略选择最多 256 个线程，向上取整 grid，并在任何初始化、Load 或 Store 之前检查尾块 guard。

例如输出为 `[17,19]` 时，线程 t 负责行 `t / 19`、列 `t % 19`，独自在该元素上累加 33 个乘积。共启动两个 256 线程块；编号大于等于 323 的线程直接退出。

这借鉴了 [TVM 归约 lowering](https://apache.googlesource.com/tvm/+/ab25b49225cfc4b91171f111578bfb5906cabae1/src/tir/transforms/lower_cross_thread_reduction.cc) 对归约输出索引与主导写入的约束；KXC 本轮只实现线程内归约，没有移植跨线程 all-reduce、warp 协作或 TVM 的完整调度框架。

## 实测发现并修复的源码发射问题

MatMul 以零初始化，首先通过了实际编译和数值测试。新增 max 归约以负无穷初始化，暴露出 [CUDA emitter](../../src/codegen/cuda/codegen_cuda.cc) 原来直接输出 `INFINITY`/`NAN`，但 NVRTC 的自包含源码没有定义这些宏：实际失败为 `identifier "INFINITY" is undefined`。

现在 float32/64 的非有限常量使用 `__int_as_float` / `__longlong_as_double` 构造 IEEE 值。NVIDIA 的 [CUDA 12.9 类型重解释文档](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__CAST.html) 明确说明这些 device intrinsic 无需额外头文件。测试同时检查 NaN 类别、正负无穷和负零符号；不依赖宿主头文件路径，也不宣称保留任意 NaN payload。

新增测试最初还尝试了 Slice 步长 2，正确地被现有 Relay 合同拒绝。本轮测试随后改为合同允许的 `+1`；没有扩大 Slice 属性范围。

## 契约、身份与三轮审计

| 审计 | 本轮结果 |
|---|---|
| A：复用现有机制 | 复用 TIR For/Seq/Store、表达式 visitor、静态 buffer 求值、既有 ThreadBinding 与 typed launch metadata；小型仿射证明仅用于所有权和溢出判断，不替代 ShapeProgram |
| B：权威与 identity | `tir.bind_cuda_threads.schema_version` 从 1 到 2，重新生成 pass contract；CUDA backend version 从 `cuda-nvrtc-driver-v1` 到 v2。两者经既有 normalized pipeline / artifact key 消费，没有旁路缓存 |
| C：消费者与正负例 | 生产 MatMul/Dense 和多维算子实际经过 Compiler → TE/TIR → BindCudaThreads → NVRTC → Driver → RuntimeSession；独立 TIR sum/max 补充归约证明覆盖，明确区分证据层级 |

[pipeline_resolver_test](../../test/pipeline_resolver_test.cpp) 验证 schema v2 进入实际执行步骤，并拒绝伪装成旧版 schema v1 的执行计划。原有 artifact identity 测试继续检查 backend 与 pipeline 变化必须导致 cache miss。

## 数值结果与效果

环境沿用 [Windows GPU 接入报告](GPU_WINDOWS_VALIDATION_REPORT.md)：RTX 4070 Ti SUPER、驱动 576.57、CUDA 12.9 Update 1、MSVC 19.29、Release、`KXC_ENABLE_LLVM=OFF`。主机参考使用独立索引循环，矩阵乘以 double 累加后比较；所有输入来自确定性正负值序列。本轮没有运行训练权重或完整模型。

| 实际执行场景 | 元素数 | 最大绝对误差 |
|---|---:|---:|
| float32 MatMul `[17,33] × [33,19]` | 323 | `2.38419e-7` |
| float32 Dense，权重 `[19,33]` | 323 | `2.38419e-7` |
| float64 MatMul / Dense，各一例 | 各 323 | 各 `3.33067e-16` |
| float32 MatMul `[2,1,5,9] × [1,4,9,7]` → `[2,4,5,7]` | 280 | `5.96046e-8` |
| float32 三维 Where，bool `[1,3,1]` mask 和值广播 | 30 | 0 |
| float32 三维 Slice，axis=-1、step=+1 | 18 | 0 |
| float32 三维 binary Concatenate，axis=-2 | 40 | 0 |
| float32 scalar add | 1 | 0 |
| 直接 TIR float32 `[17,19]` 输出、长度 33 的 sum / max | 各 323 | `1.49012e-7` / 0 |
| 直接 TIR float64 同一二维 sum / max | 各 323 | 0 / 0 |
| 直接 TIR float32/64 标量 sum / max | 各 1 | 全部 0 |

323 和 280 个输出都跨过 256 线程块边界，验证非整块尾部。标量归约从 `2147483647` 开始、长度 33，输入访问减去该起点：只分配 33 个输入，但实际验证 64 位循环跨越 int32 最大值时仍正确。float32 容差为 `1e-5 * (1 + abs(reference))`，float64 为 `1e-12 * (1 + abs(reference))`，max 要求精确相等；有限数值测试显式拒绝 NaN/Inf。

上述能力的直接效果是原来被调度拒绝的普通计算现在能够产生并执行 CUDA kernel。每个输出仍串行读取并累加 K 个乘积，没有共享内存分块、Tensor Core、warp reduction 或跨阶段融合；测试总时间不能作为模型加速或吞吐结论。

## 验证与边界

Windows 最终源码专项 6/6，总耗时 0.83 秒，包括 `pipeline_resolver_test`、`cuda_schedule_test`、`device_runtime_test`、`codegen_cuda_test`、`runtime_profiling_test`、`cupti_smoke_test`。其中 codegen 的 19 个子项全部通过，17 组数值记录共比较 2957 个元素；profiling 中的 LLVM Where 子项按构建选项跳过，不计 Windows LLVM 数值证据。

| Linux 验证 | 结果与覆盖 |
|---|---|
| 默认 CPU/LLVM | **55/55，145.01 秒**；完整视觉、三组图文联合、原有 prefill/state/decode fixture |
| adaptive CPU/LLVM | **55/55，211.38 秒**；上述 fixture 加真实 MiniMind prefill 的 baseline/candidate/rollback |
| bounded CPU/LLVM | **70/70，224.10 秒**；上述模型加投影、拆头、RoPE、GQA、因果 attention、完整有界 prefill/decode、KV 状态与真实请求批处理 |
| CUDA 12.9 编译/链接 | Linux runtime archive、codegen、schedule 与 pipeline 测试目标通过；本机驱动未恢复，不计 Linux GPU 启动证据 |
| 集成检查 | Relay/Pass 生成物 freshness、40 个算子与 20 个 Pass 合同、NLP gate、284 文件 include 层级、90 个安装头及 10 个实验头独立编译通过 |
| 强定义审计 | CPU/CUDA archive 分别 1603/1614 个强定义，重复为 0 |

三套回归均设置完整视觉、图文联合与 L1 decode fixture；adaptive 另设 `KXC_MINIMIND_ADAPTIVE_PREFILL_DIR`，bounded 设置投影、拆头、RoPE、GQA、attention 和完整 prefill/decode 的专用 fixture。bounded 的实际八层请求批处理仍得到 7 个请求步合为 5 次运行，所有 logits 和 16 份 KV 状态与独立执行逐位相等。未设置的 opt-in ResNet18 执行和独立 L1a 导入 fixture 继续跳过，不计这些额外场景。

首次 adaptive 全量回归在并行运行完整模型时为 54/55：内核 OOM 日志明确记录 `adaptive_runtime_test` 被杀，并非数值断言失败。相同 fixture 单独重跑通过（68.16 秒，最大 RSS 12,473,672 KiB）。因此为该测试加入 CTest 原生 `RUN_SERIAL`，保留完整模型输入，避免它与其他模型/JIT 同时占用内存；最终 adaptive 全量回归的 55 项全部通过，完整模型热替换子项耗时 67.94 秒。

以下能力继续明确拒绝或保持未验证：

- Softmax、masked softmax、LayerNorm、ReduceMean 等需要多个 TE 阶段和内部 `Allocate` 的组合归约；本轮没有用单线程执行整张图来绕过存储证明。
- Gather 间接 Load、运行时长度的 CUDA 归约、一般条件控制流、任意输出布局、跨线程更新、未经初始化的输出读取。
- float16/bfloat16、空输出、一般形状或算子组合、完整 Transformer/MiniMind-V、CUDA KV/bounded 状态和 GPU 性能。
- Compute Sanitizer 仍受此前独立最小程序也复现的插桩故障影响；本轮没有得到 memcheck 零错误/零泄漏结果。详见前一份环境报告。

[能力矩阵](../../test/nlp_validation/transformer_capability_matrix.json) 仅把 batched_matmul 的 CUDA 格提升为 `implemented` 的静态局部实测门禁，更新 mask_select、slice_concat 的多维证据。三格必须含本报告和生产测试路径，checker 仍禁止凭文本把 CUDA 提升成 `validated`；其他 CUDA 模型门禁保持关闭。

## 复现与回执

在既有 Windows 环境中：

```powershell
$root = 'F:\kxc-gpu\20260909-joint'
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9'
$env:PATH = "$env:CUDA_PATH\bin;$env:CUDA_PATH\extras\CUPTI\lib64;$env:PATH"
& 'C:\Program Files\CMake\bin\cmake.exe' --build "$root\build" --config Release --parallel 6 --target `
  codegen_cuda_test cuda_schedule_test pipeline_resolver_test `
  device_runtime_test runtime_profiling_test cupti_smoke_test
& 'C:\Program Files\CMake\bin\ctest.exe' --test-dir "$root\build" -C Release --no-tests=error -V -j1 `
  -R '^(device_runtime_test|cuda_schedule_test|pipeline_resolver_test|codegen_cuda_test|runtime_profiling_test|cupti_smoke_test)$'
```

本轮在上一份报告已验证的 682 文件快照上更新 10 份源码/合同/测试/头文件/CMake。最终补丁包 `out/windows-gpu/cuda-owned-overlay-final.tar.gz` SHA256 为 `550f0c9d70f529d0cb58322ac5fd2e7acdddc5a783d6a56ba10b6a8f8ec2ba3b`；合并本轮清单与上轮 CUPTI 修复的哈希后，远端再次核验全部 682 份文件，差异为 0，随后才构建并测试。

本地 `out/windows-gpu/logs/cuda-owned-*` 保留初次 Slice 合同拒绝、第二次 NVRTC 无穷宏失败、最终构建与 6/6 运行日志。最终 Windows CTest 日志 SHA256 为 `784e5e1e75dcdc4ffa0de106c1366d02f788fa8cbd957d9e55fc8215e943c81b`；`cuda-owned-receipt.json` 记录源码核验和六个二进制哈希，`out/windows-gpu/cuda-owned-numeric-audit.json` 记录 17 组数值与同源日志哈希。

结构与 Linux CUDA 编译日志分别为 `/tmp/kxc-cuda-owned-structural-build.log`、`/tmp/kxc-cuda-owned-cuda-build.log`。CPU 三配置回归使用 `/tmp/kxc-cuda-owned-{default,adaptive,bounded}-{build,ctest}.log`，并归档到 `out/windows-gpu/logs/cpu/`；该目录也保留 adaptive 初次并行 OOM 与单独复测的记录。
