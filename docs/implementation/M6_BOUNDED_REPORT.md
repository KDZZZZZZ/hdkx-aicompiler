# M6：有界形状与会话状态热替换技术报告

2026-09-10，相同有界 profile/状态 ABI 下的 CPU/LLVM 无状态与会话状态热替换已完成。真实小图、完整八层模型、四配置回归、最后连接预检复验及集成审计均通过。静态容量切片的独立证据见 [前一报告](M6_STATEFUL_REPORT.md)。这项完成不包含批处理或 CUDA 热替换。

## 要解决的问题与方法

有界模型已经能用同一产物处理多种合法 B/P，但原 adaptive preparation 只接受静态尺寸，重组未替换单元时也没有携带动态调用合同。因此有界模型不能沿现有 controller 换代并保留会话 KV。

采用两条已有设计的不变量：[JAX 形状多态](https://docs.jax.dev/en/latest/export/shape_poly.html)把维度约束纳入合法输入范围；[IREE 执行模型](https://iree.dev/developers/design-docs/invocation-execution-model/)区分可复用代码与每个 context 的可变状态。KXC 继续要求显式编译和发布，并复用自己的 ShapeProgram、ExecutablePlan 与 RuntimeSession。

## 实现

- 新的 `ProductionCompileRequest` 重载只接受 adapter 铸造的 `BoundedCompileRequest`。准备阶段由原 `PrepareBoundedCompile` 生成权威边界，核对 baseline 的输入范围、输入顺序及逐个 kernel 的读写连接；只按已声明的状态绑定把前缀缓冲区读端还原为原图输入。构造时保留这份不可变准备结果，发布时直接复用。裸 Function 不能授权有界替换。
- 身份仍由 `experimental_identity` 生成。`BuildBoundedShapeProfileKey` 编码实际计划的输入 dtype、rank、固定轴、有限上下界、整除和共享轴相等约束，避免把 `-1` 当作适用范围；`BuildBoundedDispatchKey` 标明显式 profile 选择。实际 P 不进入路由身份，状态物理容量、填充值和追加合同仍在 Plan ABI。
- 普通有界编译与替换共用 `ResolveBoundedExecutionContract` 和 `PrepareCompilerGraph`。有界 `CompilePrimitiveUnits` 接受明确 unit id 列表，复用原校验、lowering、cache 和后端。未选择的 artifact pin 及其模块调用合同保留；不能从 cache 中猜测图级范围。
- 候选组装复用 baseline 的有界计划；发布前严格核对 Plan ABI。每个请求沿现有 `StatefulSession` 保留 RuntimeSession，每步固定 generation lease；前缀打包、kernel 执行、追加复制和 extent 提交继续由原运行时处理。
- adaptive 合同为 v6、preparation 为 v4。有界 fresh/stateful selection 分别使用既有 memory-plan version；普通静态 identity、bounded 编译流水线字节、kernel ABI 与状态 Plan ABI 格式不变。

## 当前效果与验证

小图已证明：只替换 concat→relu 中指定的 relu kernel，另一 kernel 及其调用合同保留；B=2 的两份请求独立，KV 地址和未用容量的 sentinel 保持不变，P 从 1 增长到 6。执行结果、全部状态与独立 RuntimeSession 精确一致，真实完成 Bundle 触发第二代隔离及回滚。无状态入口另外执行 B/P 为 1/0、2/2、3/5 的三组输入。

裸 Function、不同输入上界、不同状态容量及错误输入被拒绝；执行与拒绝均不访问 primitive cache。运行期错误允许记录一条失败 run 事件，必须没有 kernel、分配或复制事件；不能用“事件总数不变”代替这个要求。

输入下界、上界、整除或共享轴相等约束分别改变时，路由身份均改变，未发布路由在查询时拒绝。连接反例把 `(a+b)*b` 改接成 `(a+b)*a`：两个计划都真实执行 LLVM 并得到不同结果；尽管物理签名兼容，错误 baseline 仍在编译前被拒绝，cache 计数不变。最初补充检查误用了继承的对象不等比较，导致内容相同的 kernel 名称被拒绝；现已沿用 KXC String 的内容等号，并由小图正反例复验。初次失败日志保留。

完整模型以真实有界 LLVM prefill 的输出初始化 16 份 K/V，四步 greedy decode 的代际依次为 **1→2→1→1**，有效长度 **4→8**，物理容量地址不变。每步实际调用 774 个 kernel；只有 unit 773 换用第二代产物，其他 773 个 artifact key 保持相等。第二代的实际完成 Bundle 触发一次隔离和回滚。

本次执行 **742 次 prefill + 3,096 次 decode LLVM kernel**。独立 ONNX 参考比较覆盖 prefill 最后一行 logits、全部 16 份种子 KV，以及四步的全部 logits 和有效 KV 前缀，共 **216,320 个值**；最大绝对误差 **8.9407e-6**，低于 `5e-5` 容差。每步另逐值检查未用容量仍为 `1e20f`。greedy token 为 `5059,4840,4840,1763`，由上一步真实 logits 计算。此证据使用锁定导出与固定随机 float32 权重，不代表预训练模型质量；prefill 其他三行 logits 没有被这项测试单独比较。

修正容量反例断言并复用形状准备结果后，定向整模型检查通过：GNU time 记录 **222.39 秒、峰值 RSS 11,329,952 KiB**。时间包含导入、形状证明、显式编译、发布、推理、参考比较与回滚，不是推理延迟。前一次运行在全部四步数值检查之后因旧断言失败，耗时 262.45 秒；保留该失败日志，不能算作成功测试或严谨性能基准。

| 检查 | 状态 |
|---|---|
| adaptive + bounded + LLVM 组合构建 | 通过 |
| 原静态热替换及新增有界小图 | 通过 |
| 完整八层 bounded prefill/KV/四步 decode 换代 | 通过，216,320 个参考值；容量耗尽记录零提交错误 |
| 默认、adaptive、bounded、组合开关回归 | 分别通过 55/55、56/56、70/70、74/74；最后补充的连接预检重建后，两个执行消费者复验 2/2 通过 |
| 生成物、公共头文件、依赖层、文档、符号审计 | 两份生成契约新鲜；Relay 40/40、Pass 20/20；四配置公共头文件通过；依赖层 284 文件、文档 80 文件、NLP 检查与诊断 4/4 通过；五个库无重复强符号 |
| CUDA 编译兼容性 | runtime 库与三个相关 consumer 目标重建通过；RuntimeSession 单测通过，三项硬件门禁因设备不可用返回 77，不计作 GPU 通过 |
| CUDA 热替换设备执行 | 未执行 |

四配置 CTest 实际用时依次为 **146.97、260.94、231.22、597.94 秒**。组合全量通过之后新增了有界连接预检，因此另重建并复验 `adaptive_runtime_test` 与 `adaptive_bounded_state_model_test`，两项均实际消费模型 fixture，共 **276.51 秒**；不把旧的 74 项结果写成最后修改后又完整运行一遍。该预检代码仅在 adaptive 与 bounded 同时开启时编译，其余三个配置不执行此分支。

独立 Bundle 审计核对 64 次前缀复制、共 **540,672 字节**在 kernel 前完成，64 次追加复制、共 **98,304 字节**在全部 kernel 后发生；第二代没有分配新的状态或前缀缓冲区。9 项事件篡改检查均被拒绝，容量耗尽的错误 run 提交数为 0。五个库的外部强符号数为 **1,607 / 1,731 / 1,891 / 1,971 / 1,902**，分别对应默认、adaptive、bounded、组合与 bounded-CUDA；重复定义均为 0，两个 RuntimeSession 入口各自只有一个定义。CUDA 配置显式重建了 `kxc_runtime` 静态库，未用旧归档代替当前对象。

复现入口见 [构建说明](../BUILDING.md)。本地 `out/windows-gpu/bounded-hot-swap-*` 保存回归驱动、定向复验、CPU/事件/符号审计与源码及证据归档；`logs/cpu/kxc-bounded-hot-swap-*` 保留完整输出和最后补充检查的初次失败。归档用于固定本次证据，不代表已提交或推送 GitHub。

## 三轮实现审查与边界

1. A：复用 adapter 铸造的请求、原有有界编译准备、单元编译、完整调用合同与运行时状态；没有新增 shape evaluator 或 KV 索引。
2. B：profile 编码适用范围，Plan ABI 编码调用/状态边界，selection 编码所选产物；容器只按完整显式身份查找，不执行重叠范围搜索或扩大 bucket。
3. C：真实小图与完整模型消费编译、发布、运行和回滚入口，并有独立参考及拒绝反例；四配置、两个最终执行消费者、事件顺序和集成审计均通过。

本切片不包含请求批处理热替换、跨不同容量或布局的 KV 迁移、自动 profile 搜索或隐式编译。CUDA、跨机器状态、新 IR 与模型加速均不因本次 CPU 验证通过而宣称完成。
