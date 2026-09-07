# 代码库清单：基座层与外挂层

> **状态：** 现状快照
> **更新时间：** 2026-09-06
> **审计基点：** `dev` @ `f379ebf`
> **标尺：** [项目目标](PROJECT_GOAL.md)。本文按目标重新划分代码，不重复架构描述；架构权威仍是 [架构总览](ARCHITECTURE.md)。

## 规模概览

| 范围 | 行数 | 文件数 |
|---|---|---|
| `src/` + `include/` | 46,004 | 268 |
| `test/` | 20,541 | 46 |
| `python/` | 4,821 | 27 |

---

## 划分原则

本文用两层划分代替"有用 / 多余"的二分：

- **基座层** —— 决定**什么是可能的**。只有改动它，才能解锁新的能力形态。单位成本高，难并行，是排期的真正约束。
- **外挂层** —— 在基座已支持的形态内做量的扩展。基座确定后单位成本低、可并行、可交给自动化。

**判据**：一项工作若需要改变契约、ABI、执行模式或产物身份，属于基座；若只需在既有契约内填格子，属于外挂。

**关键推论**：一个算子属于哪一层，**取决于基座是否已支持它所需的形态**。同一个算子在静态形状下是外挂，在动态形状下可能是基座工作。第 2.3 节按此重新分类了缺失算子。

---

# 第一部分：基座层

## 1.1 已建成的基座

### 表示与契约地基

| 模块 | 行数 | 职责 | 服务支柱 |
|---|---|---|---|
| `support/` | 497 | `Object` / `ObjectRef` 引用计数对象系统、`TypeRegistry`、Arena、容器、规范化哈希 | 全部 |
| `ffi/` | 323 | 类型擦除的 `PackedFunc` 调用约定与全局注册表 | 4、5 |
| `ir/` | 34 | Relay 与 TIR 共享的 `Type` / `Span` 根句柄 | 全部 |
| `target/` | 246 | 后端选择与不可变能力快照的唯一权威；`VirtualDevice` 放置 | 1、3 |
| `pass/` | 465 | `PassContext`、Pass 解析、顺序与不变量契约 | 4 |
| `contracts/` + `python/tools/` | 2,205 | 机器可读算子与 Pass 契约，生成与校验双向闭环 | **4（核心）** |

### 编译主干

| 模块 | 行数 | 职责 | 服务支柱 |
|---|---|---|---|
| `relay/` 本体 | 6,133（含算子实现） | 图级 IR、类型推断、ANF 与优化 Pass、printer。**数学语义的唯一权威** | 2、4 |
| `te/` | 1,486 | Tensor Expression 计算 DAG 与 TOPI 计算库；调度原语仅 `split`/`reorder`/`vectorize`/`unroll`/`parallel` | 2 |
| `tir/` | 3,041 | 低层 IR：表达式、语句、访问者、变换、printer | 2 |
| `compiler/` 主干 | 约 9,300 | 值图构建与校验、partition（1 Call = 1 `PrimitiveUnit`）、lowering、原语编译、**原语缓存**（请求合并 / 失败传播 / 背压 / 淘汰 / `ArtifactPin`）、语义键与产物身份、Kernel ABI | 1、2 |

### 执行地基

| 模块 | 行数 | 职责 | 服务支柱 |
|---|---|---|---|
| `runtime/` | 6,083 | `CompiledModule`、`ExecutablePlan`、静态内存规划、`NDArray` / `Storage`（引用计数 + 共享存储 + 视图）、`Device` / `DeviceAPI` / 流与事件、`KernelABI`、模块调用契约、`RuntimeSession` | 1、2、3 |
| `codegen/` | 2,466 | LLVM（1,130，主力 CPU，需 LLVM ≥ 20）、CUDA（840，保守子集）、C（329，可读源码） | 2 |

### 观测地基

| 模块 | 行数 | 职责 | 服务支柱 |
|---|---|---|---|
| `profiling/` | 1,315 | Span 采集机制，覆盖编译准备、原语编译、组装、Pass、形状与自适应编译；可选 CUPTI | **5** |
| Bundle 格式 | — | `manifest.json` + `events.jsonl` + `summary.json` + `trace.json`，schema 版本化，`span_id` / `parent_span_id` 构成树 | **5** |
| `python/kxc_agent/` | 1,055 | 离线诊断引擎（454 行）、规则库、bundle 加载与比较 | **5** |
| `tools/workbench/` | — | React/TS 性能工作台与 CLI | 5 |

### 动静与形状地基

| 模块 | 行数 | 门禁 | 职责 | 服务支柱 |
|---|---|---|---|---|
| 自适应热替换 | 868 | `KXC_ENABLE_ADAPTIVE_HOT_SWAP`（OFF） | 有界全图替换控制面，旧代际租约保活 | **1（核心机制）** |
| 形状决策层 | 约 3,400 | `SHAPE_PRODUCTION_EXACT` / `RESTRICTED_SYMBOLIC_SHAPE`（OFF） | 形状代数、符号模板到精确值的决策、精确 profile 路由 | 2 |
| Bounded 动态图 | 约 4,800 | `KXC_ENABLE_BOUNDED_DYNAMIC_GRAPH`（OFF，**未合入 dev**） | 一份产物服务多个合法 shape：固定秩、有界、整除约束 | **2（变长）** |
| 控制流运行时 | 约 1,400 | `KXC_ENABLE_CONTROL_RUNTIME`（OFF） | 静态 If 与条件前置有界 While 的编译期控制拓扑 | 2 |

### 分布式地基

| 模块 | 行数 | 职责 | 服务支柱 |
|---|---|---|---|
| `distributed/` | 2,487 | Disco 风格会话、DRef、worker、放置策略、CPU 集合通信、执行计划与 JSON 序列化 | **3** |

## 1.2 基座缺口

这是决定进度的部分。**外挂层的工作量大但可并行；下列每一项都会阻塞一整类能力。**

| 缺口 | 阻塞什么 | 现状证据 |
|---|---|---|
| **shape-as-value** | `Shape` / `Expand` / `ConstantOfShape` 三个算子；动态形状下的 `Squeeze` / `Unsqueeze`；ONNX Transformer 图的整个动态形状链 | 全仓无任何相关实现 |
| **运行时状态（KV cache）** | 支柱 2 的 decode 阶段与端到端链路 | `kv_cache` 能力全线仅有契约；`kDynamicFreshOutputV1` 明确拒绝 state / alias / donation / storage reuse |
| **执行侧观测 Span** | 支柱 5 的可信度；支柱 1 的决策输入 | 能力矩阵 `profile` 列 **12/12 unsupported**；架构文档声明不覆盖 RuntimeSession 内核、分配与拷贝路径 |
| ~~IR parser / round-trip~~ | — | **已延后**，见 [项目目标](PROJECT_GOAL.md) §2.4。现状仍是有 printer 无 parser，但不作为当前缺口计入 |
| **导入层多输出支持** | `Split` 及一切多输出算子 | 导入器自述"only single-output nodes are supported in the static-shape MVP" |
| ~~编译入口的 Python 暴露~~ | — | **已延后**，与上一项绑定 |
| **热替换的运行时证据** | 支柱 1 从"准备就绪"到"可验证可用" | 仅 `adaptive_preparation_test`，替换动作本身无运行时测试 |
| **分布式证据** | 支柱 3 的全部能力声明 | 2,487 行，**零测试**；且 executor 对已编译模块的 kernel launch 分支**尚未实现** —— 缺实现且缺证据，不是只缺测试 |
| **CUDA 能力面** | 支柱 2 的 GPU 路径 | 能力矩阵 `cuda` 列 12 项中 10 项 unsupported |

> **关于支柱 4**：新 agent / 调度友好 IR 的设计已延后。延后期间 agent 的写入路径是**机器可读契约**（`contracts/*.json` 生成与校验双向闭环），不是 IR。因此上表中两项划线条目当前不构成阻塞。

## 1.3 基座层的重复与待整理

| 项 | 行数 | 说明 |
|---|---|---|
| 形状决策层五模块 | 约 3,400 | `shape.cc` 736 / `shape_exact` 620 / `restricted_symbolic_shape` 603 / `shape_specialization` 284 / `shape_control` 272。四种命名的边界靠注释区分而非类型；未合入的 bounded 动态图将引入第六个（`dynamic_shape_contract` 663 行）。**服务同一支柱，用了本该更少的代码** |
| `experimental_identity` | 569 | 大于生产版 `identity.cc`（322 行）。形状路由与自适应准备等门禁路径会调用 experimental_identity builders，非零生产消费者；直接删除前必须迁移消费者并保持身份兼容 |

---

# 第二部分：外挂层

## 2.1 外挂层有三个必须同步的表面

> **更新（2026-09-07，第一波集成后）**：契约 24→**25**（新增 `equal`）、Relay 算子实现 24→**25**、importer 映射 15→**16**（新增 `Equal`；另有 `Constant` 节点经常量物化路径接受，不入映射表）。下文保留更新前的统计口径供追溯。

| 表面 | 当前数量 | 位置 |
|---|---|---|
| 机器可读契约 | 25 | `contracts/relay_op_contract.json` |
| Relay 算子实现 | 25 | `src/relay/op/` |
| ONNX 导入映射 | 16 | `python/kxc_onnx/importer.py` |

**仍不同步的部分收窄为**：以下 Relay 算子当前无法从任何 ONNX 名称到达：

```
nn_avg_pool2d   nn_dense
```

（`nn_dense` 可能经 `Gemm` 间接到达，需核实；`nn_avg_pool2d` 仅 `AvgPool` 名称未映射。第一波前不可达的 `cast` `divide` `mul` `reduce_mean` `reshape` `sqrt` `subtract` 已由静态导入接通。）扩算子时三个表面必须一起推进。

**按模型交集统计（与上表的"Relay 算子不可达"是两个不同口径）**：[OP_TODO](OP_TODO.md) 快照的 25 个模型 ONNX 名称中，importer 现在接受 **17** 个（既有交集 8：`Add` `Concat` `Gather` `MatMul` `Slice` `Softmax` `Transpose` `Where`；第一波新增 9：`Cast` `Constant` `Div` `Equal` `Mul` `ReduceMean` `Reshape` `Sqrt` `Sub`），仍缺 **8** 个名称（`ConstantOfShape` `Erf` `Expand` `Pow` `Shape` `Split` `Squeeze` `Unsqueeze`）。且名称交集不等于语义覆盖：交集中 `Concat` 只接受两个输入（模型出现三/四输入）、`Gather` 要求常量索引、`ReduceMean` 的 keepdims 中间轴存在已知数值缺陷（见 [G1 记录](implementation/G1_RECORD.md)）。

## 2.2 现有 24 个算子的用途归属

- **服务 Transformer（18 个）**：`add` `cast` `concatenate` `divide` `gather` `matmul` `mul` `reduce_mean` `reshape` `slice` `softmax` `sqrt` `subtract` `transpose` `where` `nn_dense` `nn_gemm` `nn_layer_norm`
- **基础能力验证（6 个）**：`nn_conv2d` `nn_max_pool2d` `nn_avg_pool2d` `nn_global_avg_pool2d` `nn_flatten` `nn_relu` —— 见 2.4

## 2.3 缺失算子：按"实际属于哪一层"分类

目标 Transformer 模型用到 25 个 ONNX 算子，第一波后已覆盖 17 个（更新前 15 个）。仍缺的 8 个中，**只有 2 个是纯外挂**：

| ONNX 算子 | 模型中出现 | 所需基座形态 | 基座就绪 | 实际归属 |
|---|---:|---|---|---|
| `Erf` | 6 | 一元 elementwise | ✅ | **纯外挂** |
| `Pow` | 13 | 二元 elementwise | ✅ | **纯外挂** |
| ~~`Equal`~~ | 1 | 布尔输出 elementwise | ✅ | **已完成**（第一波 C 线 + B 线 ONNX 接线，LLVM 数值与 Equal→Where 组合证据见 [G1 记录](implementation/G1_RECORD.md)） |
| ~~`Constant`~~ | 180 | 常量物化（导入层） | ✅ | **已完成**（第一波 B 线静态导入） |
| `Squeeze` | 2 | 静态：reshape；动态：shape-as-value | 静态 ✅ / 动态 ❌ | **取决于形状模式** |
| `Unsqueeze` | 30 | 同上 | 静态 ✅ / 动态 ❌ | **取决于形状模式** |
| `Split` | 1 | 多输出导入 | Relay ✅ / 导入 ❌ | **基座**（导入层） |
| `Shape` | 17 | shape-as-value | ❌ | **基座** |
| `Expand` | 1 | shape-as-value + 广播 | ❌ | **基座** |
| `ConstantOfShape` | 1 | shape-as-value | ❌ | **基座** |

**结论**：`Erf` / `Pow` 可以立刻做，成本低（`Equal` / `Constant` 已在第一波完成）。其余六个中有四个必须先补 shape-as-value 基座 —— 它们是"伪装成算子缺口的基座缺口"，而且正是让 ONNX Transformer 图具备变长能力的那部分。

## 2.4 视觉算子：第一步基础能力验证，不是多余

`nn_conv2d` 等 6 个算子与 ResNet 工具链虽不服务目标模型，但它们验证了基座的多项形态，而这些形态 Transformer 路径同样依赖：

| 被验证的基座形态 | 载体 | Transformer 是否依赖 |
|---|---|---|
| 窗口迭代与边界处理 | `te/topi/window.h`（130 行） | 间接（未来注意力窗口 / 滑动） |
| 归约调度 | `te/topi/reduction.cc`（75 行） | **是**（softmax、layer_norm、reduce_mean） |
| 多维布局与轴变换 | `convolution.cc` 88、`pooling.cc` 116 | **是** |
| 端到端导入—编译—执行链路 | ResNet18 路径 | **是**（同一条主干） |

它们同时是当前**第二条端到端验证链**。在 Transformer 变长链路加宽到能独立支撑验证之前，保留它们作为参照系。

**处置**：保留，不列入清理。可考虑移出默认构建以减少噪音，但不删除。

---

# 第三部分：证据现状

## 已跑通的端到端链路

`onnx_importer_test.cpp` 的 `TestRunExactTransformerProtobufLLVM`：

```text
真实 ONNX protobuf -> 导入 -> 编译成 9 个 LLVM 单元 -> RuntimeSession 执行 -> 数值校验
```

覆盖算子：`Concat` `Gather` `LayerNormalization` `MatMul` `Slice` `Softmax` `Transpose` `Where`。

**这是可复现的执行证据，不是契约。** 骨架已成立，但很窄：静态精确形状、8 个算子、仅 LLVM。所有后续工作应体现为**把这条链路加宽**。

## 能力矩阵分层统计

以 `test/nlp_validation/transformer_capability_matrix.json` 为准，12 项能力：

| 层 | implemented | validated | contracted | unsupported |
|---|---:|---:|---:|---:|
| llvm | 8 | — | — | 4 |
| cuda | 2 | — | — | **10** |
| runtime | 8 | — | 2 | 2 |
| numeric | 5 | 4 | — | 3 |
| profile | — | — | — | **12** |

---

# 第四部分：待清理

明确不服务任何支柱，且不构成基础能力验证。

| 项 | 行数 | 依据 |
|---|---|---|
| ~~`experimental_identity`~~ | 569 | **移出直接删除项**：形状路由与自适应代码会调用其 builders（门禁路径），零生产消费者的前提不成立；如需整理必须先迁移消费者并保持身份兼容 |
| 九个陈旧清理分支 | — | 删除目标已不在 `dev`：`task_plan.h`、`task_executor.h`、`adaptive_hot_swap_v2.*`、`docs/handoffs/`、`docs/plans/` 等均已不存在。落后 `dev` 108～122 个提交 |
| `include/kxc/te/topi.h` | — | 伞头文件无人使用（TOPI 本体大量使用，勿误删） |
| `src/relay/distributed/plan_adapter.{h,cc}` | — | 无任何 include 者 |
| DeviceInfo | 约 60 | 仅 `device_info_test.cpp` 消费 |
| 文件头 `\brief` 复制粘贴 | 63 个文件 | 8 组重复，其中 11 个文件共用同一句与内容无关的描述 |

**定位待定**：C 后端（329 行）。它输出可读源码，与支柱 4 的"agent 可读"沾边，但 agent 应消费 IR 与契约而非生成的 C 源码。需明确它服务哪根支柱，否则归入本节。

---

# 第五部分：结论

1. **基座缺口是唯一的排期约束。** 外挂层（算子）单位成本低且可并行，但九项基座缺口中每一项都阻塞一整类能力。
2. **算子缺口有 40% 是伪装的基座缺口。** 10 个缺失算子中 4 个需要 shape-as-value，1 个需要导入层多输出支持。先补基座，再补算子，顺序反了会返工。
3. **外挂层的三个表面必须同步推进。** 契约 24 / Relay 24 / 导入 15 的不同步，使已实现算子有相当部分对真实模型不可用。
4. **视觉路径是资产不是负债。** 它验证了归约、布局、端到端链路等 Transformer 同样依赖的基座形态，并提供第二条参照链。
5. **主轴是加宽已有的端到端链路**，而不是另起炉灶。每项工作都应体现为能力矩阵多覆盖一格，且由可复现测试背书。
