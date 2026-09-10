# CUDA 有界形状基础执行技术报告

> 2026-09-09；状态：本模块已完成，真实 GPU 数值、设备活动与最终回归已通过。目标依据：[PROJECT_GOAL](../PROJECT_GOAL.md)。完成范围为单阶段输出、形状值和线程内 MatMul 归约，完整 bounded Transformer 仍需后续模块。

## 结果与适用范围

`Compiler::CompileBounded` 已能在 CUDA 上生成一份接受多个合法形状的产物。真实 Windows RTX 4070 Ti SUPER / CUDA 12.9 已执行广播 Add→ReLU→Sqrt、`shape_of` 和 rank-2 MatMul：两份图共 34 次运行、10,330 个输出值，和独立主机参考的最大绝对误差为 0。运行及非法输入拒绝期间，primitive cache 的全部十项统计不变。

这解决了原先 bounded 编译入口、TE lowering、scalar ABI 和 fresh-output session 只允许 CPU/LLVM 的限制。输出按本次形状分配紧凑存储，GPU 使用本次维度计算地址。编译范围有限，超范围输入会在提交 kernel 前报错。

新能力保留默认关闭的 `KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH` 门禁。它要求 dynamic module ABI、restricted/exact shape gates，以及实际可用的 LLVM 或 CUDA 后端。Windows 本次单独配置 LLVM=OFF、CUDA=ON，执行未经过 CPU 编译后端。

## 方法与实现

### 沿用形状、参数和启动的现有权威

准备仍经过 `RestrictedSymbolicShapeAdapter → BoundedCompileRequest → PrepareBoundedCompile`。主机用现有 `ShapeProgram` 和 module invocation 合同检查维度上下界、整除、共享符号、输出字节预算，再求出实际 extent。TE 仍使用 `bounded-dynamic-serial-v3`。

TE→TIR 按已有参数顺序传递 `uint64[1]` extent buffer，并增加 `kxc.cuda.runtime_extent_upper_bounds`，给 CUDA 证明提供同序有限上界。`CanonicalTEScheduleContract` 将 CUDA 上界逐项编码进已有 schedule/artifact identity：上界改变可能改变网格，也必须改变缓存身份。CPU 的调度字节不增加该字段。

运行时已有 `LaunchResolved` 会在目标设备分配这些 scalar buffer，复制本次值，并让 completion 保活所有参数 Storage 和模块。本次复用该路径，没有新增 native CUDA API 或第二套 shape 求值器。`RuntimeSession` 只扩大 fresh-output 模式的 CUDA 目标范围；bounded state、请求批处理和动态原地状态的 CPU 边界保留。

设计参考采用两个已有不变量：IREE 将 dispatch 的动态工作量与输入/输出维度显式传入；TVM TensorIR 可以让符号维度由实际 buffer 或函数参数绑定。这里只采用显式参数与唯一形状来源的原则，没有引入它们的 IR 或运行时。[IREE Flow](https://iree.dev/reference/mlir-dialects/Flow/)、[TVM TensorIR Creation](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/tir_creation.html)。

### 按上界启动，按实际形状寻址

`BindCudaThreads` 仍独占线程分配与 launch metadata。它从编译上界计算固定 grid/block，在线程内先判断 `linear < actual_work`，再计算实际行主序坐标。

例如 MatMul 输出的上界为 `[19,17]`，最多 323 个输出，采用两个 256-thread block。本次输出若为 `[3,5]`，只有前 15 个线程执行，坐标为 `row=linear/5`、`column=linear%5`。实际任一输出维度为零时 guard 为假，坐标除法、取模及 tensor Load/Store 都不会执行。K 可以为零，此时每个输出线程只写初始化的零。

没有枚举所有具体 shape，也没有按运行长度编译或选择另一套 kernel。固定上界网格的代价是小形状会启动较多空线程；这是当前实现的明确上限，后续性能优化需有真实测量。

### 证明整个形状域内的所有权和地址安全

新增的局部多项式证明位于原 CUDA pass 内部。符号原子只有已声明 extent 和当前循环变量；接受非负常量、加法、乘法和可证明无损的整数转换。

1. 把每个输出 Store 地址化为符号多项式，要求等于实际输出域的紧凑行主序编号；所有输出必须完整初始化。
2. 输出 Load 只能读该线程相同编号的元素，且必须已有支配它的无条件 Store。动态归约可能执行零次，不能把归约内的初始化当作循环后的保证。
3. 输入地址用各循环的 `extent-1` 代入上界，证明 `buffer_elements-(maximum_address+1)` 的所有系数非负；同时检查整数中间结果和循环终点。
4. 不能证明时在 backend 编译和 cache publication 前拒绝。多项式最多 256 项、每项最多 32 个因子，防止证明本身无限膨胀。

MatMul 的 `A[i*K+r]` 上界为 `(N-1)*K+(K-1)=N*K-1`，`B[r*M+j]` 同理为 `K*M-1`。每个输出线程只初始化并累加自己的 `C[i*M+j]`。这里的形式推导只用于编译期地址证明，运行时仍消费原有实际维度参数。

### 身份与缓存

Pass schema 升至 **5**，CUDA backend identity 升至 **`cuda-nvrtc-driver-v5`**。上界进入现有 canonical schedule contract，shape/plan/module callable ABI 的物理参数类型不变。

身份审计发现并补上了一个关键区别：CPU 的串行动态循环不按上界决定启动范围，CUDA 的固定网格会。因此不能只复用旧的动态 TE identity。新增真实 CUDA 回归将同一 ReLU 图分别编译为 N≤129 和 N≤513，要求缓存键、1/3-block launch 和缓存条目彼此区分，并分别执行 129/513 个值。

## 验证

主要 consumer 是 [bounded_cuda_test.cpp](../../test/bounded_cuda_test.cpp)，从公开编译入口到真实 `RuntimeSession::RunAsync`。CUPTI bundle 由现有 [验证器](../../test/cuda_profile_bundle_test.py) 的 bounded 模式校验；native 进程仅返回 0 不会被算作通过，还必须出现最终数值标记并通过设备活动检查。

| 验证 | 最终数值/设备证据 |
|---|---|
| 广播与形状值 | `x[B,S,7] + y[1,S,1] → ReLU → Sqrt`，同时返回 `shape_of(x)`；B∈[0,19]、S∈[0,18] 且整除 2；8 组形状各运行两次，9,008 个值，最大误差 0 |
| MatMul | `[N,K] @ [K,M]`；N≤19、K≤23、M≤17；9 组形状各运行两次，含最大值、跨 block 尾部、N/M/K=0；1,322 个值，最大误差 0 |
| 两条流与输入保活 | 每图两个独立 session、两条非默认流；临时输入句柄在等待完成前销毁；输出存储不与仍保留的旧输出复用 |
| 设备关联 | 64+18=82 条 CUDA kernel 活动，逐条关联到实际 run、call_index、launch 和完成事件；各 call 的 grid/block 跨形状保持一致 |
| 实际维度传递 | 128+54=182 次 H2D scalar DMA，共 1,456 字节；每个 scalar 恰为 8 字节，每次复制在其 consumer kernel 的设备执行前完成 |
| 输入拒绝 | 9+5=14 次错误输入：范围、整除、共享符号、rank、固定轴、设备/stream 和实参数量；submit_count 全为 0，无 kernel 或 extent DMA |
| 缓存 | 主图编译后全部十项缓存统计在正常运行与非法输入期间不变；N≤129/513 的两份 ReLU 产物分别执行 129/513 个值，两次 miss、两个条目且无交叉命中 |
| 输出保活 | graph、session 和 primitive cache 释放后，仍保留的非空输出再次对齐独立参考 |
| 不支持路径 | 实际 bounded Softmax 的动态 scratch 与 runtime Gather 的间接索引，在 `BindCudaThreads` 拒绝，未到 backend 编译或缓存发布 |
| 证明负例 | synthetic CUDA 验证缺失/错误上界、地址越界、跨线程输出读取、未初始化读取、可写输入、间接地址、窄整数及启动范围溢出等 17 个拒绝案例 |
| 证据负例 | 在两份真实 bundle 上分别篡改空记录、父关联、shape、grid、scalar 字节/先后顺序、重复记录和错误提交计数，共 16 次拒绝 |

加上缓存区分回归，本模块合计执行 36 次、核验 10,972 个值；CUPTI 的 82 条 kernel 和 182 次 extent DMA 对应其中两份主图的 34 次运行，额外两次 ReLU 没有计入这份 profile。

### 最终硬件回归与复现

Windows 使用 RTX 4070 Ti SUPER、驱动 576.57、CUDA 12.9 Update 1 和 MSVC 19.29。bounded 配置关闭 LLVM，打开下列四个已有门禁；默认 CUDA 配置保留关闭。新 consumer 不依赖模型 fixture。

```sh
cmake -S . -B <cuda-build> -DKXC_ENABLE_LLVM=OFF -DKXC_ENABLE_CUDA=ON \
  -DKXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI=ON \
  -DKXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE=ON \
  -DKXC_ENABLE_SHAPE_PRODUCTION_EXACT=ON \
  -DKXC_ENABLE_BOUNDED_DYNAMIC_GRAPH=ON
cmake --build <cuda-build> --config Release --target bounded_cuda_test
ctest --test-dir <cuda-build> -C Release -V -R '^bounded_cuda_test$'
```

| Windows CUDA 配置 | 最终结果 |
|---|---|
| bounded | 16/16，42.31 秒 |
| 默认 | 13/13，45.64 秒 |

两套均保留完整静态八层 MiniMind prefill、KV 状态交接与四步 decode，以及并发 profile、复制生命周期等旧用例。bounded 主图的最终 Bundle 为 `out/windows-gpu/bounded-cuda-evidence/build-bounded-cuda/bounded-cuda-evidence/bounded-sjkq284g/{elementwise,matmul}/`，包含最终上界缓存修复后的源码验收。

原始 Windows 回执归档为 `out/windows-gpu/bounded-cuda-evidence.tar.gz`，8,590,490 字节，SHA-256 为 `ea2f060be40f5012d551867ec725a082b047996b5b292dde5a86acfdb934ee41`。代码与 CUDA consumer 对应的 overlay 为 `bounded-cuda-verified.tar.gz`，SHA-256 为 `c2fc25af163b4cdb7f5b61ef5e0ccd59e331fcb2bc680862a7527c8288c952cd`；归档同时保留当时 698 文件的逐项源码核验。后续报告及 session 构造测试的旧断言修订单独记录，不改变已验证的生产实现或 CUDA consumer。

初次 `bounded-cuda-stage2-evidence/` 只用于保留发现和修复过程。全量验收曾发现三条测试与新能力不符：kernel signature 和 session 两处旧断言要求动态 CUDA 必须拒绝；新 consumer 的负例只捕获 `invalid_argument`，而 compiler phase 会将拒绝包装为 `runtime_error`。前两处改为验证合法 CUDA 合同，后者捕获标准异常并继续检查 `BindCudaThreads` 拒绝来源。初次失败日志保留为 `*-initial.log`，最终通过记录与其分开。

### CPU 回归与证据消费

| CPU/LLVM 全量配置 | 结果 |
|---|---|
| 默认 | 55/55，146.34 秒 |
| adaptive | 55/55，223.60 秒 |
| bounded | 70/70，251.15 秒 |

实际模型工作负载分别为 5/6/8 项：固定视觉、三种图文输入和文本容量 decode 均未跳过；adaptive 额外执行完整模型两代替换，bounded 额外执行完整变长 prefill/decode 与内嵌请求批处理。后者有 7 个模型 CTest，不能把内嵌批处理误计为第 8 个 CTest。原始回执为 `out/windows-gpu/logs/cpu/kxc-bounded-cuda-*`，逐项数值与未跳过审计为 `bounded-cuda-cpu-regression-audit.json`。

40 个 Relay operator、20 个 pass 的生成物和合同、284 个 include 边界、90 个 installed/10 个 experimental header 独立编译、73 篇文档检查及 Python 诊断 4/4 均通过。矩阵审计拒绝 19 项篡改，保持 bounded rank-2 CUDA 的局部证据边界。

两套 Windows 回归的 24 份非空 Bundle 均经过公开 schema 和离线诊断入口读取；另外 6 份 foreign-scope Bundle 按旧复制归属测试要求保持空白。两份 bounded 主图只有预期的 9/5 次前置拒绝 run 为 error，submit_count 为零；再次执行的 16 项证据篡改检查全部拒绝。具体计数、原始 events SHA-256 和诊断类别记录在 `out/windows-gpu/logs/bounded-cuda-agent-consumer-audit.json`，离线诊断未改变原始 events 或压缩归档。

Linux 的 LLVM 20.1.2 + CUDA 12.9、四个 shape gates 同时开启的构建已完成。session、pipeline、kernel signature、CUDA 结构证明、TE 数值与 runtime profiling 六项通过；真实 bounded CUDA consumer 返回 77（无可用设备），明确不计为 Linux GPU 成功。默认 CPU 库与双后端 bounded 库分别核验 1,603/1,894 个外部强定义，均无重复。

最终报告、索引及 session 测试修订单独打包为 `out/windows-gpu/bounded-cuda-metadata.tar.gz`。最终 698 文件清单为 `bounded-cuda-source-manifest.json`；overlay、逐文件 SHA-256、两端核验与全部验收入口记录在 `out/windows-gpu/logs/bounded-cuda-completion-audit.json`。这些是本地与 Windows 验证机之间的证据同步，不代表 GitHub 已推送。

## 三轮 Ponytail QA

| 轮次 | 审计结论 |
|---|---|
| A：复用 | 使用既有 ShapeProgram、uint64 scalar ABI、参数保活、TIR pass、canonical encoder 和 CUPTI validator；只增加地址证明所需上界 |
| B：权威与身份 | 没有新 registry/evaluator/launch owner；CUDA 上界必须进入真实缓存键，已补回归；CPU identity 不增加 CUDA 字段 |
| C：真实消费与拒绝 | 主图经过实际 CUDA Compiler/RuntimeSession；Softmax 动态 scratch 和动态 Gather 仍在原 proof 拒绝；默认 CPU、adaptive、bounded 与两套 CUDA 回归均已通过 |

## 明确的剩余范围

- 本模块只开放能够通过上述单阶段证明的有界输出和串行归约。动态多阶段私有 scratch、Softmax/归一化、间接 Gather 和完整模型形状重排尚未完成。
- rank-2 bounded MatMul 的数值不能推广为任意 bounded batched MatMul 或完整 Transformer 已通过；矩阵保持局部 `implemented` 证据。
- bounded CUDA KV 状态、请求批处理、带状态的热替换、协作归约、tensor core/tiled 优化仍是后续工作。
- 本机 Linux GPU 驱动不可用，真实 GPU 证据来自上述 Windows 主机。既有 Compute Sanitizer 环境问题仍未消除；没有把编译或 CPU 测试写成 GPU 插桩通过。
- 这是正确性与接入证据，没有性能提速结论；总 project goal 保持进行中。
