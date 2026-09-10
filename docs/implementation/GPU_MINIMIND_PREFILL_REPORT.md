# 完整 MiniMind 静态 CUDA Prefill 技术报告

2026-09-09。完整八层 MiniMind 已在 Windows RTX 4070 Ti SUPER 上通过正式导入、编译和执行链：**650 个 kernel，logits 与全部 16 个 K/V 输出，两次运行共比较 401,408 个元素，最大绝对误差 1.001358032e-5**。这是固定 B=1、S=16 的纯文本 prefill 数值验收；模型使用导出时随机初始化权重，不评价生成质量。

## 模型、方法与真实执行路径

复用既有 `make_minimind_decode_loop_fixture.py` 的完整 prefill fixture，不构造手写注意力替代图。导出来自 MiniMind commit `6fc918beb68a0d8c40452338df6319fe168014ba`，seed=0，约 63.91M 参数，hidden=768，8 个 query heads、4 个 KV heads、head_dim=96、vocab=6400、SwiGLU 中间宽度 2432。导出 metadata 记录 PyTorch 2.14.0+cpu、ONNX 1.22.0、opset 17、关闭 flash attention。

源文件 `out/minimind_onnx_cap32/minimind_prefill_static.onnx` 的 SHA-256 已重新核对：

```text
eb1c60c73980b0e56693876906e4e6d1a861b2609bd40469e2637d09b0eef455
```

导入 spec、参数、token 输入、完整 logits、16 份 K/V 参考和布局/receipt 共 **23 个文件**经 Tailscale/SSH 传入 Windows，逐项 SHA-256 校验后才执行。参数 spec 文件为 275,518,060 字节；参考由独立 ONNX ReferenceEvaluator 产生。重复运行使用同一组 token，未把重复运行声称为不同模型输入覆盖。

新增 [minimind_prefill_cuda_test](../../test/minimind_prefill_cuda_test.cpp)：

1. 验证完整模型的 token 输入和 17 输出接口，执行原有 InferType、常量折叠和简化。
2. 通过 `Compiler::Compile`、实际 CUDA Target 和 opt_level=3 编译；要求 650 个计划调用及相同数量的 artifact pins。常量由正式编译模块放置到 GPU。
3. 将 int64 `input_ids[1,16]` 上传，创建一个 RuntimeSession 和非默认 CUDA stream，连续两次 RunAsync 并等待 completion。
4. 每次逐维检查 CUDA device、float32 dtype、logits `[1,16,6400]` 和各层 K/V `[1,16,4,96]`，比较全部元素；有限值和绝对误差 `<=1e-3` 都必须成立。
5. 第二次输出必须拥有独立 storage；十项 primitive cache 统计在运行期间完全不变，证明该入口没有查缓存或隐式编译。返回的 K/V 是普通输出，未绑定为持久状态。

实际编译缺口由 [Gather/Pow 模块](GPU_GATHER_POW_REPORT.md) 补齐。各层 RMSNorm、RoPE、GQA、因果 attention、SwiGLU 和输出投影都经过已有生产原语；没有新增模型运行时、CUDA 库调用旁路或 IR。

## 数值效果

| 输出 | 每次元素数 | 两次运行中的最大绝对误差 |
|---|---:|---:|
| 完整 logits | 102,400 | `5.569308996e-6` |
| 八层 K | 49,152 | `1.001358032e-5` |
| 八层 V | 49,152 | `4.798173904e-6` |
| 合计 | 200,704 | `1.001358032e-5` |

两次运行各输出的最大误差相同，全部远低于现有模型绝对误差阈值。最终 Windows 专项 **8/8、5.28 秒**；模型单项 3.13 秒，编译观测约 1.916 秒，两次带观测的 RunAsync+Wait 约 60.45/55.38 毫秒。这些是一次专项日志中的耗时，包含观测、分配及主机开销，不是性能基准，也不能据此推导对框架的加速比。

## Windows 栈与验收门禁

首次完整图在编译阶段发生 Win32 异常，本机 JIT 调试器启动失败后竟返回 exit code 0，导致 CTest 显示 Passed，却没有一条模型数值记录。应用事件日志确认了原生异常。关闭 CUPTI 后仍复现；只对诊断副本把栈预留改为 8 MiB 后，全部输出立即完成比较。据此定位为默认栈不足；没有取得原生调用栈或精确异常码。

正式 [CMake](../../CMakeLists.txt) 仅为 MSVC 的该完整模型测试增加 `/STACK:8388608`。这是可执行文件的栈预留，不修改系统注册表、显示驱动或 GPU 内存；MSVC 默认 1 MiB 及选项语义见 [Microsoft 文档](https://learn.microsoft.com/en-us/cpp/build/reference/stack-stack-allocations?view=msvc-170)。Windows 外部调用方运行同等深度的现有递归图遍历时，也应提供足够的调用线程栈；本模块没有把所有编译器递归遍历改成迭代。

CTest 还要求完整模型最终数值通过标记出现，codegen 测试要求最后一组完成标记，避免把静默提前退出记为成功。未设置可选 fixture 时返回 77，由 CTest 明确标为 Skipped；设置 fixture 后文件缺失、设备不可用或数值失败都必须报错。正式验收日志含 34 条模型数值记录及最终完成标记，不是初始空输出的 Passed。

## Profiling 的效果与限制

后续更新：以下保留本轮未修复时的原始证据。设备 run/span 传播、独立 profile 路由和时钟原点问题已由后续 [M1 CUDA 模块](M1_CUDA_CORRELATION_REPORT.md) 修复并重新验证，原始 bundle 未覆盖。

既有 observer 记录 **2 个模型 run、1,300 对 kernel_submit/kernel_exec、1,300 个分配事件**；模型 run 和主机 kernel 事件包含 export receipt、stage=prefill、sequence_length=16、repeat 与 run_id。CUPTI 同时采集到 **1,300 条真实 cuda_kernel 设备活动**，整份 bundle 34,773 条事件，severity 全为 info、status 全为 ok。

但本次 **1,300 条 CUPTI kernel 活动的 run_id/parent_span_id 为空**。主机 observer 的关联尚未完整传播到设备活动，不能用同名 kernel 猜测关联，也不能把 host_observed_complete 的 duration 当设备执行时间。本模块保留原始 bundle 作为后续 M1 GPU 关联修复的具体输入；不宣布模型级 GPU 时序关联已经完成。

## 回归、复核与三轮审查

完整 CPU/LLVM 回归全部通过：默认 **55/55、144.82 秒**，adaptive **55/55、212.83 秒**，bounded **70/70、225.11 秒**。逐测试日志确认三套均实际执行完整视觉、三组固定单图联合模型及文本状态/decode；adaptive 运行实际模型热替换，bounded 运行完整变长 prefill/decode 与请求批处理，无模型 SKIP。未设置的独立 L1a/ResNet18 opt-in，以及默认/adaptive 的 bounded-only 子项按原门禁跳过，不计入这些模型证据。

40 个 Relay operator、20 个 pass 合同及生成物新鲜度、NLP 门禁、284 个 include 边界、90 个 installed 与 10 个 experimental header 独立编译、69 篇文档链接和 git diff 检查均通过。7 种矩阵篡改均被拒绝，含文本升为 validated、错误局部门禁、开放未执行 decode/state、移除完整模型 consumer。Windows 独立 CTest 反例确认：静默 exit 0 无完成标记会失败；真实模型缺少可选 fixture 会明确 Skipped。

回执目录为 `out/windows-gpu/logs/`，包括所有初始失败、关闭 CUPTI/扩大栈的诊断、正式 build/CTest、数值及 profiling 审计。完整模型 bundle 为 `out/windows-gpu/fixtures/minimind-static/cuda_prefill_profile_bundle/`，fixture 清单为 `out/windows-gpu/minimind-static-fixture-manifest.json`，最终源码清单为 `out/windows-gpu/cuda-gather-final-manifest.json`。

A：复用现有完整 ONNX fixture、导入/简化和运行会话，未新增模型框架。B：backend/pass 身份由 Gather/Pow 模块进入既有 artifact key，执行 metadata 仅用于观测。C：从真实 Windows kernel 到每个输出、缓存不变、独立输出 lifetime 和失败日志逐层核对；发现原生异常的“零退出码”后补齐证据门禁，而不是接受原 CTest 计数。

## 后续边界

能力矩阵仅增加 `prefill_exact.cuda` 的固定模型局部 `implemented` 证据，仍禁止用文本升为整体 `validated`。GPU decode、会话 KV 状态、bounded CUDA、请求批处理、多图/变长 MiniMind-V、L3 语音链和完整设备计时关联仍需各自实现与验收。Compute Sanitizer 的环境插桩问题保持原记录。

最终源码包 `out/windows-gpu/cuda-gather-final.tar.gz` 的 SHA-256 为 `49d96b84f8b77b8153a6906086d681e442b688a66e8a915f47393d3cf5df5403`；Windows CTest 日志为 `ad971bec9e5a00e68d742db3bddad2ce7b415867855bf8ca5f180e35999294c2`。CPU/CUDA 静态库分别有 1603/1614 个强定义，重复数均为 0。完整审计摘要为 `out/windows-gpu/logs/cuda-gather-completion-audit.json`。
