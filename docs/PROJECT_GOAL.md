# 项目目标

> **状态：** 唯一目标权威文档
> **更新时间：** 2026-09-07
> **规则：** 本文定义仓库"要成为什么"。[架构总览](ARCHITECTURE.md) 定义"当前是什么"。
> 某项能力被本文列为目标，**不等于它已经实现**；当前事实一律以架构总览、机器可读契约和可复现测试为准。

> **范围决策记录**
> - 2026-09-06：排除 eager / define-by-run 路线（§4）。
> - 2026-09-06：延后新 agent / 调度友好 IR 的设计，含 parser 与 Python 编译入口暴露（§2.4）。
> - 2026-09-07：确定目标模型阶梯 —— MiniMind-O 为北极星，MiniMind（纯文本）为当前验收目标（§2.2）。

## 1. 一句话目标

> **一台能被 agent 读懂、能按观测结果自我调整的 Transformer 推理编译器与运行时。**

这句话是本仓库所有取舍的最终依据。任何工作若不能归入下列五根支柱之一，或不能说明它如何服务于其中之一，就不属于本仓库范围。

---

## 2. 五根支柱

### 2.1 动静兼备 —— 以热替换为唯一机制

**定义**：同时支持静态精确编译与运行期形态变化；但形态变化**只允许通过在已验证变体之间的有界替换实现**，不允许通过运行时隐式编译实现。

> **动静边界解释（2026-09-07，与 §8 一并解释）**：同一变体内的合法 shape 由已验证的 bounded 合同处理（一份产物服务多个合法 shape，属静态编译范畴）；变体之间的切换由有限路由与热替换处理。两者都不得隐式编译：超出 bounded 合同的形状必须先经过可验证的产物准备，不得在运行时悄悄换实现。

**为什么是热替换**：它是三种动静结合机制中唯一与本仓库 fail-closed 原则相容的一种。

| 机制 | 运行时是否隐式编译 | 与 fail-closed 原则 | 结论 |
|---|---|---|---|
| eager / define-by-run | 是，未命中即现场编译 | 正面冲突 | **排除**（见 §4） |
| tracing + guard 子图 | 是，guard 失效即重编译 | 冲突较小但存在 | 非首选 |
| **有界热替换** | **否**，变体离线编译、运行时只切换 | **相容** | **采用** |

**范围**：全图有界替换，旧代际通过租约保活，替换过程复用既有原语编译与图组装链路。

**验收**：替换动作本身必须有运行时证据，而不只是 preparation 阶段的证据。

### 2.2 Transformer 推理

**定义**：以自回归 Transformer 推理为首要目标负载，覆盖 **prefill** 与 **decode** 两个阶段。

**包含**：
- 变长序列输入 —— 以固定秩、有界、带整除约束的动态形状实现，不是任意动态；
- KV cache —— 需要运行时状态，是当前端到端链路的硬阻塞；
- 批处理 —— 含动态批处理方向。

#### 目标模型阶梯（2026-09-07 决定）

首要目标负载不再是匿名的"某个 Transformer"，而是 [MiniMind 系列](https://github.com/jingyaogong/minimind)三个具体模型。选择它的唯一理由是**规模与本仓库的覆盖能力相称**：MiniMind-O 是目前少见的、小到能被一个自研编译器完整覆盖的 omni 模型，因此"端到端跑通"是一个可达成而非象征性的目标。

三级构成一条**真子集阶梯**，后一级复用前一级的全部成果：

| 级别 | 模型 | 新增基座需求 | 状态 |
|---|---|---|---|
| **L1** | [MiniMind](https://github.com/jingyaogong/minimind)（纯文本 decoder，约 64M） | KV cache（§6.2）、shape-as-value（§6.3）、采样与生成循环驱动 | **当前验收目标** |
| **L2** | [MiniMind-V](https://github.com/jingyaogong/minimind-v)（+ SigLIP2 视觉编码） | 接近于零：SigLIP2 为 256×256 固定输入 → 64 patch token 的静态图，复用现有 `nn_conv2d` / `nn_layer_norm` / `softmax` / `matmul` 路径 | 后续 |
| **L3** | [MiniMind-O](https://github.com/jingyaogong/minimind-o)（+ 语音输入输出，Thinker–Talker） | Conv1d / ConvTranspose1d、流式卷积状态、双自回归调度、实时帧预算、多流会话 | **北极星** |

**为什么 L1 就是通往 L3 的第一段**：MiniMind-O 的 Thinker 即 MiniMind backbone，同作者、同 block 定义。选 L1 作为当前验收目标不是放弃 L3，而是走它的第一段；这条路径上没有一段工作会被废弃。

**L3 的已知代价必须写明，不得用"0.1B 小模型"掩盖**：

- 标题的 0.1B 仅指可训练主干（Thinker 63.9M + Talker 47.1M）。端到端部署面还包含冻结的 SenseVoice-Small（234M）、SigLIP2（94.6M）、Mimi 编解码器（96.2M）与 CAM++，合计约 540M 参数、5 种异构架构。**编译器覆盖面按后者计。**
- Mimi 的增量解码需要每个卷积层维护流式 ring buffer。这是**比 KV cache 更强的一类运行时状态**，不在当前动态执行合同内。
- Talker 以 12.5 Hz 帧率输出，即每 80 ms 必须完成一次 Talker 前向加 Mimi 增量解码。这是**实时预算**，其验证依赖执行侧观测（§2.5）。
- barge-in 与近双工要求会话级中断与输入输出并发。当前 `RuntimeSession` 是单会话顺序执行。**这部分属于服务系统工程，尚未纳入五根支柱**；进入 L3 前必须先决定它是新增支柱还是移出范围（§7）。

**目标模型算子清单的证据规则**：[模型算子清单](OP_TODO.md)必须来自**对目标模型的实际 ONNX 导出**，标注模型版本、导出参数与 opset，不得使用来源不明的快照。导出本身的可行性（decoder 带 `past_key_values` 的 dynamic axes 导出）是 L1 的第一个待验证项，未验证前 M2 / M3 的方案均建立在假设上。

**验收**：以 `test/nlp_validation/transformer_capability_matrix.json` 为唯一权威，12 项能力 × 8 个层级（frontend / relay / lowering / llvm / cuda / runtime / numeric / profile）逐格标注，不得以整体"支持"替代逐格证据。

#### 结构化控制流能力边界（横向能力）

Transformer 图中可能出现条件分支和生成循环，仓库已有一条独立的结构化控制流实现，但它不是普通 `Compiler::Compile` 的默认能力。`KXC_ENABLE_CONTROL_RUNTIME` 默认关闭；开启后，`Compiler::CompileControlFlowExact` 可把静态精确的 Relay `If` 和有界、条件先于循环体执行的 `While` 编译成真实 LLVM 原语，再由 `ControlRuntimeSession` 在 CPU:0、默认流上执行。控制计划已经有 branch/loop region、Phi 绑定、循环携带值、`max_trip_count` 和严格校验，相关 schema、lowering、运行时与 CTest 也已经存在。

这条路径当前的边界必须进入目标定义：固定 rank/shape 和静态 kernel 签名；CPU/LLVM 与默认流；CPU 标量布尔谓词；非负的循环上限；fresh-output kernel effect。它明确拒绝 CUDA、非默认设备或异步流、运行时 extent、KV/持久状态、alias/donation/storage reuse，以及任意数据相关形状。关闭门禁时的“明确拒绝”测试不等于生产能力，reference executor 也不等于 LLVM 证据。

对 MiniMind 的关系分两步处理：L1a 静态 prefill 不依赖控制流路径，L1b 的首个生成循环先由 host driver 明确编排；随后用 M10 评估固定步数或导出图中的 `While` 是否值得接入。若要让控制流承载真实 decode，必须先由 M2/M3 为同一运行时 owner 定义 KV 状态和 extent ABI，不能在 `ControlRuntimeSession` 旁边再造一套模型专用状态引擎。MiniMind-O 的双自回归、80 ms 帧预算和多流近双工也不由现有控制流合同自动获得。

**控制流验收**：在 gate-on LLVM 构建中分别证明 `If` 两个分支、`While` 的 0/1/多次迭代和上限拒绝，并与 reference 结果对齐；在 gate-off 构建中证明入口在执行前拒绝。每个结果单独记录到能力矩阵和 profile receipt，不能把“源码已存在”写成“MiniMind 已支持”。

### 2.3 分布式执行

**定义**：多 worker 的已编译计划分发与执行，含放置策略与集合通信。

**边界**：分布式不改变单目标编译的语义与身份；已编译产物在分布式下的行为必须与单机一致。

**当前风险**：该方向已投入约 2,487 行，但**测试覆盖为零**。按本仓库的证据标准，它当前不构成任何能力证据。补齐证据或明确降级，二者必居其一。

### 2.4 agent 与调度友好的中间表达

**定义**：IR 与契约必须可被**非人类消费者**可靠地读取、修改并交回编译器。

> **范围决定（2026-09-06）：重新设计一套新的 agent / 调度友好 IR —— 延后。**
> 本支柱当前**不包含**新表示层的设计与实现。agent 表面以**现有** Relay / TIR、机器可读契约与 canonical bytes 承载。

**当前在范围内（已具备，需保持）**：

1. **确定性** —— 同一输入、目标与流水线必须产生同一 IR 与同一身份。canonical bytes 与规范化顺序不得依赖哈希遍历顺序、对象地址或缓存状态。这条是所有 agent 消费的前提，任何新工作都不得削弱它。
2. **机器可读契约** —— 算子与 Pass 元数据以 `contracts/relay_op_contract.json`、`contracts/pass_contract.json` 表达，生成与校验双向闭环。

**延后期间 agent 如何参与**：

契约层已经是双向闭环（可生成、可校验），因此**契约而非 IR 是当前 agent 的写入路径**：agent 读观测层与契约，改契约，由生成器落到代码。这条路径不需要 IR parser，也不需要新 IR。

**已延后（随新 IR 设计一并推迟）**：

- **IR round-trip / parser** —— 现状是只有 printer（`relay/printer`、`tir/printer`）、无 parser，IR 对 agent 只读。为现有 IR 补 parser 本身不是重新设计，但其价值取决于"agent 改写 IR"这一闭环的最终形态，故一并延后，避免做完即废。
- **调度信息在 IR 中的表达** —— 见 §7。
- **Python 绑定暴露编译入口** —— 与上述闭环绑定，一并延后。

**重新启动的条件**（满足其一即应重新评估）：

- 观测层与契约层的组合已不足以支撑 agent 做出需要的干预；
- §7 的调度语义已确定，且现有 IR 无法表达；
- 出现明确的外部消费者需要 IR 级读写。

### 2.5 agent 友好的观测层

**定义**：覆盖编译与执行全链路的结构化、可关联、版本化观测数据，供 agent 与工具直接消费。

**已具备的形态**：Profile Bundle（`manifest.json` + `events.jsonl` + `summary.json` + `trace.json`），schema 版本化，`span_id` / `parent_span_id` 构成调用树，`event_type` 分类；配套离线诊断引擎与性能工作台。

**当前缺口**：第一波已把 RuntimeSession 的运行、内核提交/完成、分配和拷贝接入 Span，并用 LLVM 组合图验证；但这些事件还没有与 MiniMind 的导出 receipt、prefill/decode、KV extent 和 generation 稳定关联，能力矩阵中 12 项能力的 `profile` 列也不能因此整体改为已验证。

**为什么它是支柱而不是配套设施**：热替换和 MiniMind-O 的实时预算都必须依据执行结果决策。第一波已经打通基础事件，后续要把事件与模型阶段、状态和代际关联，才能形成可消费的决策依据。

---

## 3. 支柱之间的依赖

以下依赖关系决定了工作顺序，不是可选的组织方式：

- **观测层（2.5）→ 热替换（2.1）**：替换决策的输入来自执行侧观测。第一波基础 Span 已存在，仍需把它和 generation/ABI/模型阶段关联后，才能让热替换消费真实证据。
- **KV cache 状态支持 → Transformer（2.2）**：prefill 与 decode 的完整链路依赖运行时状态。当前动态执行模式明确拒绝 state / alias / donation / storage reuse，与本支柱直接冲突，必须解决。
- **结构化控制流 → Transformer（2.2）**：现有控制流路径只消费静态精确值和 CPU 标量谓词，明确拒绝 runtime extent、持久状态和 CUDA。它可独立做 gate-on 证据，但只有在 M2 的 KV owner 与 M3 的 extent/shape 合同落定后，才能评估是否用于真实 decode；不能绕过普通 `ExecutablePlan`/`RuntimeSession` 的状态权威。
- **契约闭环（2.4）→ agent 参与的一切**：agent 的写入路径是机器可读契约（生成与校验双向闭环），不是 IR parser。parser 已按 §2.4 延后，不再列作阻塞；闭环依赖的是契约可生成、可校验、可落码。
- **分布式（2.3）→ 证据标准**：在补齐测试之前，它不能作为任何能力声明的依据。

---

## 4. 明确的非目标

以下内容不属于本仓库范围。列出是为了防止范围漂移，不是对其价值的否定。

| 非目标 | 原因 |
|---|---|
| **eager / define-by-run（经典动态图）** | 其本质是运行时隐式编译，与 fail-closed 核心原则正面冲突；且 Transformer 推理的形状高度结构化，属有界动态，不需要任意动态图。若未来目标转向**建模灵活性**（研究新架构、训练），需重新评估本条 |
| 训练与自动微分 | 目标是推理 |
| 自动调优 / 自动候选搜索 | 调度候选须显式、可验证、可复现 |
| 通用动态秩、参差（ragged）、数据相关任意形状 | 变长以 padding + mask + 有界 bucketing 实现 |
| 隐式编译副作用与隐式回退 | 不支持的语义必须在执行前明确报错 |
| 成为 TVM 的直接替代品 | 本仓库是工程与研究代码库 |

---

## 5. 当前状态快照（2026-09-07）

| 支柱 | 当前状态 | 最大缺口 |
|---|---|---|
| 2.1 动静兼备（热替换） | 约 868 行，门禁默认 OFF，仅 preparation 有测试 | 替换动作本身缺运行时证据 |
| 2.2 Transformer 推理 | 目标已明确为 MiniMind 阶梯；第一波有静态 Transformer/ONNX 组合证据，LLVM 层 8 项 `implemented`、4 项 `unsupported` 仍只是矩阵状态；控制流源码与测试已具备但 gate 默认 OFF | MiniMind L1a 真实 prefill 尚未端到端通过；L1b 仍缺 KV 更新、动态有效长度和 host 生成循环；控制流尚缺当前 gate-on LLVM receipt，且不能直接承载 state/extent |
| 2.3 分布式 | 约 2,487 行 | 零测试、零证据 |
| 2.4 agent / 调度友好 IR | 契约与 canonical bytes 已具备并在用；agent 经**契约**写入 | 新 IR 设计与 parser **已延后**（§2.4）；当前无阻塞项 |
| 2.5 agent 友好观测层 | Bundle 格式与诊断引擎已具备；第一波已接通 RuntimeSession 的 run/kernel/alloc/copy 事件并有 LLVM 组合证据 | MiniMind 的导出 receipt、prefill/decode、KV extent 和 generation 关联尚未完成；能力矩阵 profile 列仍不能整体升级 |

---

## 6. 优先级顺序

1. **锁定 MiniMind 导出并完成 L1a prefill** —— 先把模型版本、opset、past/present 轴和实际算子清单变成唯一可复现输入，再补真实导入与 LLVM 运行缺口。
2. **验证并挂载结构化控制流路径** —— 用独立的 gate-on LLVM 证据确认 `If`/bounded `While` 的真实编译与执行，并给出它对 L1 生成循环的适用性结论；这一步不阻塞静态 L1a。
3. **KV cache 的运行时状态支持** —— 支柱 2.2 L1b 端到端链路的硬阻塞；要证明同一 session 的追加、有效长度和多步 decode，并决定控制流是否能复用同一状态 owner。
4. **shape-as-value 基座** —— 解锁 MiniMind 变长所需的 Shape/Reshape/Expand/Unsqueeze 等受限控制值，并扩展有界 attention；若控制流使用 shape 值，必须共用这份 extent 合同。
5. **用第一波 profiling 支撑模型验收和热替换** —— 补导出 receipt、stage、extent、generation 关联，使支柱 2.5/2.1 从基础设施变成可消费证据。
6. **分布式补齐证据或明确降级** —— 消除支柱 2.3 的零 compiled-kernel 证据状态。
7. **跨算子融合（`te::Program`）** —— 用真实 MiniMind profile 选择首个性能切片，排在 L1 能力打通之后。

**已移出本序列**：IR parser / round-trip、Python 编译入口暴露、新 agent 友好 IR 的设计。理由见 §2.4。

纯外挂算子（`Erf` / `Pow` / `Equal` / `Constant`）不占本序列位置：基座已就绪，可随时并行推进。

---

## 7. 待决问题

**"调度友好"中的调度指哪一层？** 两种含义对 IR 的要求差异很大：

- **循环调度**（TE 层的 split / tile / fuse / vectorize）—— IR 需暴露循环结构、迭代域与依赖事实；
- **任务与请求调度**（分布式分发、continuous batching）—— IR 需暴露任务粒度、资源需求与边界契约。

结合支柱 2.3 与动态批处理方向，当前倾向于后者为主。

**本问题随 §2.4 的 IR 设计一并延后**，不再阻塞其他工作。延后期间应做的是**积累证据而非做决定**：记录调度器与 agent 实际需要而现有 IR 无法提供的信息，作为将来设计的输入。

在此期间维持一条约束：**不得在 IR 层做不可逆的结构决策。**

---

## 8. 与既有工作的关系

| 既有工作 | 定位 |
|---|---|
| Bounded 动态图（5 个未合入提交） | 服务支柱 2.2 的变长输入，方向正确，应合入主线 |
| [TE Program IR 设计提案](TE_PROGRAM_IR.md) | 服务支柱 2.1 与 2.2 的**性能**手段（跨算子融合），动机不是 agent 友好，**不在 §2.4 的延后范围内**；优先级见 §6 第 6 条 |
| 新 agent / 调度友好 IR 的设计 | **已延后**，见 §2.4 |
| IR parser / round-trip、Python 编译入口暴露 | **已延后**，与上一项绑定 |
| 受限符号形状 / 精确 profile 路由 | 服务支柱 2.2；与 bounded 动态图存在职责重叠，需合并 |
| `compiler/control_flow/` + `runtime/control_*` | 横向控制流能力：默认关闭的静态精确 `If` / 有界 `While` 路径；先做 gate-on 证据与 MiniMind 生成循环适用性评估，再决定是否接入 M2/M3 的状态/extent 合同 |
| eager 底座 + tracing 子图（A + B 路线） | **移出范围**，理由见 §4 |
| `distributed/` | 服务支柱 2.3，见 §2.3 的证据要求 |
| [代码库审计与模块清单](CODEBASE_INVENTORY.md) | 现状快照，本文的事实依据之一 |
