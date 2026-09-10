# CUDA 会话 KV 状态与多步 decode 技术报告

2026-09-09。完整八层 MiniMind 已在 Windows RTX 4070 Ti SUPER 上完成实际 prefill → 会话 KV 初始化 → 四步 greedy decode。缓存容量为 32，有效长度从 16 增至 20；四步 logits 最大绝对误差为 `4.41074371338e-6`，16 份完整 KV 的最大误差为 `1.10864639282e-5`，均低于 `1e-4`。每步使用同一编译产物和同一份会话缓存，没有运行时编译。

## 方法与最小改动

原有 [CPU 状态合同](M2_MINIMIND_STATE_REPORT.md)已经把固定物理容量、可变有效长度、初始化和追加放在 ExecutablePlan/RuntimeSession 中。实现并不依赖 CPU 指针访问：初始化和追加统一调用 `CopyStateRange` → `StorageCopySync`，构造时填充使用既有 DeviceAPI。真正阻止 CUDA 接入的是计划的 CPU-only 校验和模块的 LLVM-only 校验。

本模块只为 `kStaticStatefulExternalV1` 扩展这两个校验：state 与追加源必须是同一 CPU:0 或同一 CUDA 设备上的 float32 连续张量；模块 backend/device 必须对应 LLVM/CPU 或 CUDA/CUDA。常量、可写输入、alias/donation、容量、维度和签名校验全部保留。bounded 和动态 extent ABI 的 CUDA 拒绝仍然存在。

采用 ONNX Runtime [past/present 缓存说明](https://onnxruntime.ai/docs/genai/howto/past-present-share-buffer.html)中的容量与有效长度分离不变量。KXC 本次保留私有 present 输出，显式复制其追加段到会话状态，不新增分页、共享缓存或第二张模型状态表。

执行顺序直接沿用现有 consumer：

1. 在首个 launch 前校验输入、设备、stream、容量及先前状态，并绑定会话持有的缓存。
2. 所有 kernel 在调用者指定 stream 上读取旧缓存，生成 logits 与私有 present。
3. 等待最后一个 kernel 完成，再同步 D2D 复制每个状态的追加段；多个外层维度按原布局逐行复制。
4. 所有复制成功后提交有效长度。`RunAsync` 返回时 completion 已 ready，下一步可使用另一个 stream。

这是明确的同步提交合同，不是 GPU 状态排队器。kernel 执行及复制失败继续向调用者传播；已经提交工作的失败会毒化会话，不能继续使用部分写入的状态。异步对象、模块、Storage、ValueTable 和状态所有权仍由原有机制管理。

`InitializeState` 接收已完成、同设备且布局匹配的 prefill 输出，复制有效前缀到独立状态 allocation。普通调用者只传 ids/position/mask，并只接收 logits。当前位置作为显式数据输入，变化不引起编译或 profile 路由。

计划身份仍为既有 static external v10；state 绑定、容量、填充值、设备与 backend 已进入 canonical identity，内存计划算法和 kernel ABI 未变化，因此不增加版本。真实 GPU consumer 检查绑定前后 ABI 不同，四步 ABI 相同，而两个不同填充值的 replay ABI 各自不同。

## 独立数值与真实 consumer

[完整模型测试](../../test/minimind_decode_loop_llvm_test.cpp)增加 `--cuda`，复用原有 Compiler、ONNX 导入、状态绑定与 greedy driver。prefill 和 decode 使用两个独立 ProfileContext，避免全局 bundle 环境变量混淆两个模块。

模型为固定随机权重的完整八层架构，B=1、4 个 KV head、head_dim=96，16 份缓存布局为 `[1,32,4,96]`。prefill 的全部 17 个输出先与 ONNX ReferenceEvaluator 对照，再将实际 GPU K/V 初始化到 decode 会话；原输出与完成句柄随后释放。四步交替使用两个非默认 CUDA stream，每步检查缓存地址不变、extent 增加一、完整缓存和 logits 有限且符合参考。

[fixture 生成器](../../python/tools/make_minimind_decode_loop_fixture.py)保存每步全部 16 份容量缓存参考，参考只来自独立 ONNX 执行。四步输入 token 为 `5756, 5756, 1576, 1576`，每一步都由前一步实际 logits 的 host argmax 验证。无效区保留哨兵 7；另建两个会话，以相同有效前缀和填充值 0 / -1234.5 重放最后一步，所得 6,400 个 logits 逐位相同。全部十项 primitive cache 计数在 decode 和 replay 前后不变。

| 独立参考比较 | float32 元素数 | 最大绝对误差 |
|---|---:|---:|
| 一次 prefill 的 logits 与全部 K/V | 200,704 | `1.04904174805e-5` |
| 四步 decode logits | 25,600 | `4.41074371338e-6` |
| 四步完整容量缓存（含有效和未使用区域） | 786,432 | `1.10864639282e-5` |
| 合计 | 1,012,736 | 均低于 `1e-4` |

[通用 CUDA 状态测试](../../test/cuda_state_runtime_test.cpp)使用 `[2,3,4,5]` 缓存，在 axis=2 初始化一格、再追加三格，覆盖六个外层行。图先生成 present，再计算 `old_cache + token`：全部 360 个输出和 360 个缓存元素逐位正确，证明追加没有提前覆盖后续 kernel 需要的旧状态。

两个会话的缓存地址与内容独立；达到容量后再次执行被拒绝，长度和数据不变。初始化超容量、初值设备错误、重复初始化、运行输入 shape 错误、stream 设备错误、运行超容量，共六类负例。后三类 run 的 submit_count 全为 0。保留的输出和状态 NDArray 在 session/module 析构后仍可读取。计划测试另覆盖 CPU/CUDA 绑定和混合设备拒绝。

## 设备活动、复制与提交顺序

每次实际 decode 计划包含 **682** 个 kernel。导入原始图的 683 个 primitive 全部通过 CUDA lowering/source 审计；生产 consumer 的显式常量折叠与 simplify 后，计划为 682 个，硬件关联按后者验收。

[现有 Bundle 验证器](../../test/cuda_profile_bundle_test.py)新增 state/decode 模式，按 schema v1 读取实际 CUPTI 活动，要求非空最终成功标记、CUPTI available、唯一 run/call 对应及没有 dropped records。主模型四步加两次 replay 共 4,092 条设备 kernel，独立 prefill 650 条；通用状态图另有 6 条。

验证器进一步检查所有状态 D2D 都在本次全部设备 kernel 完成后才开始，copy 完成落在 `RunAsync` 区间内；每个 launch/完成保留实际 stage、extent、token、generation 和 plan ABI。初始化没有借用另一个 run。每个 D2D 与唯一 copy_launch/copy_id/主机完成一一对应。

| 状态复制 | 次数 | 字节数 |
|---|---:|---:|
| 主模型从 prefill 初始化 16 个状态 | 16 | 393,216 |
| 四步 decode 与两次 replay 的追加 | 96 | 147,456 |
| 两个 replay 会话初始化有效前缀 | 32 | 933,888 |
| 主模型 D2D 合计 | 144 | 1,474,560 |
| 通用图两个初始化和三次追加 | 30 | 600 |

非零状态初始填充另有主模型 32 次 H2D（每次 49,152 字节）、通用图 2 次 H2D（每次 480 字节），使用既有原始 DeviceAPI，不具备 Storage copy_id。零填充走 memset。编译期常量快照也单独识别，不混入 KV 初始化/追加计数。

每份状态 Bundle 注入十种破坏：空事件、错误 extent、丢失状态 metadata、错误 kernel parent、重复 kernel、错误 copy parent、错误字节数、缺少复制完成、提前 commit、错误 submit_count。两份状态 Bundle 共拒绝 20 项；prefill 关联验证器另拒绝 6 项。主机观察完成时间与 CUPTI 设备时间分开，不把等待区间当作 kernel 或 DMA 性能。最终主模型 144 次状态 D2D 的设备时长合计 277,735 ns，通用图 30 次合计 25,663 ns；这些局部时长不是模型吞吐基准。

## 复现与回归

Windows 使用 RTX 4070 Ti SUPER、驱动 576.57、CUDA 12.9 Update 1、MSVC 19.29。完整模型栈保留 8 MiB；Linux CUDA 构建只作为编译证据。新 CTest 每次生成独立 `cuda-state-evidence/{state,decode}-*/`，缺设备或可选模型 fixture 返回 77；已有目录不会覆盖，native 进程静默退出不能通过。

```sh
OPENBLAS_NUM_THREADS=1 PYTHONPATH=python out/venv/bin/python \
  python/tools/make_minimind_decode_loop_fixture.py \
  --onnx out/minimind_onnx_cap32 --out out/fx_minimind_cuda_state --steps 4 --seed 0
cmake --build <cuda-build> --config Release \
  --target cuda_state_runtime_test minimind_decode_loop_cuda_test
# KXC_MINIMIND_CUDA_STATE_DIR 指向完整 fixture 目录
ctest --test-dir <cuda-build> -C Release -V \
  -R '^(cuda_state_runtime_test|minimind_decode_loop_cuda_test)$'
```

fixture 共 100 个文件，两端逐项 SHA-256 相同。decode ONNX SHA-256 为 `d9be9abcb40946add9838c0da2d2e4c827abfe2b713413b747646f1d4545895a`；prefill 为 `eb1c60c73980b0e56693876906e4e6d1a861b2609bd40469e2637d09b0eef455`。fixture 数据与模型权重保留在 out，不入源码树。

专项首轮 2/2、28.75 秒通过；最终 Windows 回归 **13/13、40.37 秒**通过，保留完整 prefill、并发 profile 和复制生命周期旧用例，新增真实模型状态、边界及 26 项回执破坏。

| CPU/LLVM 全量配置 | 结果 |
|---|---|
| 默认 | 55/55，156.70 秒 |
| adaptive | 55/55，218.13 秒 |
| bounded | 70/70，233.75 秒 |

实际模型工作负载分别为 5 / 6 / 8 项：视觉、三种图文输入、文本容量 decode 均核验未跳过；adaptive 另执行模型热替换，bounded 另执行变长 prefill/decode 和其中的请求批处理。bounded 的 8 项工作负载由 7 个模型 CTest 承载，未把内嵌请求测试误记为额外 CTest。历史可选模型 fixture 的 SKIP 不用于这些结论。

40 个 Relay operator、20 个 pass 的生成物与合同、284 个 include 边界、90 个 installed/10 个 experimental header 独立编译、72 篇文档检查均通过。23 种能力矩阵篡改被拒绝，包含移除每份必要证据、错误 gate、伪造 validated 和擅自开启 GPU 请求批处理。Python 诊断测试 4/4 通过；Windows 回归的 10 份非空 Bundle 经公开 schema/离线诊断入口读取，3 份 worker Bundle 保持空白。CPU/CUDA 库分别 1,603/1,614 个强定义，无重复。Linux 最终 CUDA consumer 构建及 runtime_profiling_test 通过，两项硬件 consumer 均返回 77，不计为 Linux GPU 成功。

验证时源码 overlay 为 `out/windows-gpu/gpu-state-final.tar.gz`，包含 19 个本模块文件，SHA-256 为 `1a750bcd327304fd977924bb24b0fcc5854a18c97ff62336c42d22d89fee03a5`。两端 696 文件逐项核验相同；最终报告和矩阵说明另经 metadata overlay 同步，最终清单为 `gpu-state-source-manifest.json`，可执行代码未再改变。

原始 Windows 证据归档 `out/windows-gpu/gpu-state-evidence.tar.gz` 共 3,491,488 字节，SHA-256 为 `1d9991e366ca08d5e857853f6d8662244f2a37b4e02c6571e7df2ee215e04939`，提取目录为 `out/windows-gpu/gpu-state-evidence/`。主模型 Bundle 位于 `build/cuda-state-evidence/decode-lba63m59/`，events SHA-256 为 `5e5338b53b41fa2ebf986be10e5fa183742dbca3a0fb2df4e4b64c3e4fefe6cb`；通用图为 `state-qeqxkl87/`，events SHA-256 为 `339d4d270e61cdf77b8901d5aba7886b4569cd305f71084d49b1e5ca7c840b26`。前者 80,669 条记录均正常；后者只有预期的 3 条前置拒绝 run 为 error，提交数为零。

CPU 原始回执在 `out/windows-gpu/logs/cpu/kxc-gpu-state-*`；实际模型、Windows、schema/诊断、矩阵反例、强定义和集成检查记录在 `out/windows-gpu/logs/gpu-state-*-audit.json`。离线诊断生成的文件可重建，原始 archive 和 events 保持可核验。

## 三轮 QA 与适用边界

A：直接复用 state/output binding、extent book、StorageCopySync 和 AsyncOperation，没有新增资源表、状态 API 或调度 IR。B：旧 CPU 字节合同、计划身份和内存计划不变；新 target 已被 device/backend 字段区分，动态 cursor 不进入编译身份。C：同一个生产 Compiler/RuntimeSession 既执行真实模型，也执行多维状态正反例；设备时间验证等待与复制顺序，所有计数来自实际 Bundle。

能力矩阵只将 `decode_external_kv.cuda` 与 `kv_cache.cuda` 标为带指定 local-evidence gate 的 implemented；CPU 数值记录保留原有独立含义。此结果覆盖静态 B1/C32 MiniMind 和通用 axis-2 状态合同，不声称 bounded CUDA、GPU 请求批处理、异步排队、跨设备 KV、paged/ragged 缓存、状态热迁移或训练已支持。主模型持久 KV 占 0.75 MiB，每步追加复制 24 KiB；每步仍计算容量图并生成私有 present，中间输出也仍需分配，不等于 kernel 原地更新或零分配。它也不代表模型质量、吞吐提升或长期 CUPTI 时钟/存储稳定性。现有 Compute Sanitizer 环境限制仍未解决，不声称插桩通过。project goal 继续保持进行中。
