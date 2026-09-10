# M10：结构化控制流（If / 有界 While）

仓库里已经有一条完整但默认关闭的控制流路径：`KXC_ENABLE_CONTROL_RUNTIME=OFF` 时，普通 `Compiler::Compile` 会在准备阶段拒绝残留的 Relay `If`/`While`；打开门禁后，`Compiler::CompileControlFlowExact` 能把静态精确的 `If` 和有界、条件先于循环体的 `While` 编译成真实 LLVM 原语，再由 `ControlRuntimeSession` 在 CPU:0、默认流上执行。`ControlPlan` v2、runtime schema v1、Relay lowering、Phi/循环携带值、执行前校验和 control-flow CTest 都已经存在。

这条代码不是可以直接宣称“已支持 MiniMind”的死代码。它目前只接受固定 rank/shape、CPU 标量布尔谓词、静态 kernel 签名和非负 `max_trip_count`；fresh-output kernel effect 也明确拒绝 state、alias、donation、storage reuse 和 runtime extent。CUDA、非默认设备、异步流、KV cache、持久会话状态和任意数据相关形状都不在合同内。MiniMind L1a 静态 prefill 不依赖它，L1b 的第一版生成循环先由 host driver 编排；本模块负责给出控制流的当前生产证据，以及它何时、如何与 M2/M3 的状态和 extent 合同交接。

> **状态（2026-09-10）：** C0/C1/C2 已完成。gate-on LLVM 的 If/While 生产 receipt、gate-off 拒绝和 L1b host-loop 选择见 [控制流 receipt](M10_CONTROL_RECEIPT.md)。C3 的 state/extent 接入未启用：L1b 继续由 host loop 驱动，避免在 `ControlRuntimeSession` 旁新增状态 owner；C4 的 MiniMind-O 需求仍待单独立项。

## 当前情况与要做的模块

| 模块 | 当前情况 | 要做的事 | 首个交付物 |
|---|---|---|---|
| C0 合同和代码审计 | control plan、Relay lowering、runtime executor、feature gate 和测试均已存在 | 唯一 owner、schema 版本、支持/拒绝矩阵已审计 | 控制流能力矩阵和代码审计记录，已完成 |
| C1 gate-on 生产证明 | gate-on LLVM receipt 与 gate-off 拒绝均已完成 | `If` 两分支、`While` 0/1/多次迭代、上限拒绝和非法 plan 零 launch | 可复现的 CTest/数值 receipt，已完成 |
| C2 MiniMind 生成循环评估 | `generate()` 的采样与停止条件在 host 侧，导出的静态图没有控制节点 | 维持 host loop；不把 host 循环写成图已支持 | L1b generation-loop 选择记录，已完成 |
| C3 state/extent 交接 | `ControlRuntimeSession` 只绑定静态值，显式拒绝 runtime extent 和持久 state | 若决定接入 decode，把 state/extent 挂到既有 `ExecutablePlan`/`RuntimeSession` owner，并版本化 ABI/identity；不在控制流 runtime 旁再造状态权威 | 交接设计与负例 |
| C4 MiniMind-O 边界 | Thinker–Talker、Mimi ring buffer、80 ms 帧预算和多流会话尚未接入 | 单独盘点双自回归调度、流式状态和实时观测需求；判断哪些是新 runtime/service 合同 | L3 预研清单，不扩大 L1 合同 |

## C0：先把现有能力说清楚

1. 以 `CMakeLists.txt` 的 `KXC_ENABLE_CONTROL_RUNTIME` 为唯一生产门禁。旧的 Relay control-flow 开关保持移除状态，不能恢复第二个开关。
2. 以 `src/compiler/control_flow/control_plan.*` 的 schema v2 为编译控制面的唯一结构：任务只有 kernel、branch、loop；branch 使用独立 then/else region 和 Phi 绑定，loop 使用 condition/body region、condition value、carried bindings 与 `max_trip_count`。
3. 以 `include/kxc/runtime/control_execution_plan.h` 的 runtime schema v1 为执行面输入。编译 schema 到 runtime schema 的转换必须继续经过既有 adapter，不能让 runtime 读取 Relay、TE、compiler callback 或缓存对象。
4. 记录当前可接受条件：所有 tensor rank/shape 静态精确且维度非负；所有 task 在 CPU:0/默认 stream；predicate 是 CPU 标量 bool；kernel 输出遵循 fresh-output effect；结构化 region 可达且依赖唯一。
5. 记录当前必拒绝条件：动态 extent（`kRuntimeExtent`）、持久 state、alias/donation/reuse、非默认设备/流、CUDA、未知 rank、任意数据相关控制值、缺少真实 primitive artifact 的 branch/loop。
6. 把 `control_plan_test`、`control_plan_reference_executor_test`、`relay_control_plan_test` 与 `control_runtime_integration_test` 的覆盖拆开记录：schema/参考执行器/Relay lowering/生产 LLVM 不能互相代替。

## C1：在同一 LLVM 基线取得生产证据

使用独立 build 目录，避免重配第一波目录：

```bash
cmake -S . -B out/build/control-llvm -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DKXC_ENABLE_CUDA=OFF \
  -DKXC_ENABLE_LLVM=ON \
  -DKXC_ENABLE_CONTROL_RUNTIME=ON \
  -DPython3_EXECUTABLE=out/venv/bin/python
cmake --build out/build/control-llvm -j2
ctest --test-dir out/build/control-llvm -L control-flow --output-on-failure --no-tests=error
```

验收顺序必须能区分编译和运行：

1. 用最小静态 Relay `If` fixture 分别选择 then/else，检查两条真实 primitive module 都被绑定，结果与独立参考逐元素一致。
2. 用 condition-before-body 的 `While` fixture 覆盖 0、1 和多次迭代；检查 carried value、region 依赖和最终结果，不能只跑 reference executor。
3. 构造超过 `max_trip_count` 的输入，确认在执行路径中抛出明确错误；构造非法 shape、设备、predicate dtype、缺 kernel artifact 和不支持 effect 的 plan，确认 launch 计数为零。
4. 在同一提交上运行 gate-off 构建，确认 `CompileControlFlowExact` 和 `ControlRuntimeSession` 在执行前明确拒绝。这个结果只证明门禁行为，不计入 gate-on 数值证据。
5. 将 build 配置、commit、schema/identity 版本、输入 checksum、branch/loop 结果、launch 计数和失败信息写入 receipt。若没有真实 LLVM artifact，状态只能是“结构/参考测试通过”。

## C2：决定 MiniMind L1b 的循环承载方式

首个 L1b 交付采用 host-side deterministic greedy loop：prefill 返回 logits 和 KV 句柄，host 选择 token，随后每步只提交新 token 和同一 session 的 cache，直到固定步数或 EOS。这个循环能先验证 M2 的 append/read、M3 的 `past + current = total` 和 M1 的 generation 关联，不需要把采样器塞进 Relay 控制图。

只有出现以下明确输入时，才评估把 `While` 放入编译图：导出图确实保留控制节点，循环上限可静态证明，谓词是 CPU 标量 bool，循环携带值和输出 shape 在现有 ABI 中可表达，并且没有隐式编译或第二套状态 owner。固定步数的合成图可以作为 C1/C2 证据；它不能替代真实 MiniMind decode。

C2 的结果必须是一个选择记录：

- **host loop（L1 首选）**：控制流只做独立能力验证，L1b 不依赖 `ControlRuntimeSession`；
- **bounded graph loop（后续候选）**：仅在 M2/M3 交接完成、真实导出保留 `While` 且端到端 receipt 通过后启用；
- **暂不接入**：如果现有控制流的静态 shape/state 边界与生成语义不相容，保留 gate-on 能力和明确拒绝，不为 MiniMind 添加特例。

## C3：需要状态时只扩展既有 owner

若 C2 选择真实 decode 控制图，先在 [M2](M2_KV_STATE.md) 定义容量、cursor、valid extent、append/read 和失败后的 session 状态，再在 [M3](M3_SHAPE_VALUES.md) 定义 `batch`、`past`、`current`、`total` 的有界 shape/extent 来源。控制流只消费经过验证的 invocation contract，不自行解释 ShapeProgram。

交接时必须完成以下工作：

1. 在 `ExecutablePlan`/`RuntimeSession` 中表达持久 state 与 extent 的所有权、生命周期、完成可见性和 identity/version；
2. 为 branch/loop 的 carried state 定义 ABI 顺序、别名/写入模式和 preflight 检查；
3. 明确静态控制计划与有状态 plan 的兼容性；不兼容时在准备阶段失败，不在运行时临时复制或编译；
4. 增加同一 session 的 prefill→三步 decode、容量越界、失败后 session 不可继续和 pending completion 保活测试；
5. 若无法满足这些条件，保留 `ControlRuntimeSession` 的 state/extent 拒绝规则，把控制流限定为静态/合成场景。

## C4：把 MiniMind-O 的新增问题单独列出

MiniMind-O 的 Thinker 可以复用 L1 backbone，但 Talker 的双自回归调度、Mimi 每层 ring buffer、12.5 Hz/80 ms 实时预算、barge-in 和近双工会话都超出现有控制流合同。C4 只输出算子、状态、调度和观测的需求清单，不提前把这些语义塞入 L1 的 branch/loop schema。进入 L3 前应先决定它们属于新的 runtime/service 能力，还是继续由 host/service 调度。

## 代码落点与所有权

| 位置 | 责任 |
|---|---|
| `CMakeLists.txt`、`include/kxc/compiler/compiler.h` | feature gate、公开入口和 gate-off 行为 |
| `src/compiler/control_flow/control_plan.*` | schema v2、结构化任务、验证和确定性内容 |
| `src/compiler/control_flow/relay_control_plan.*` | Relay `If`/`While` lowering、类型/region/Phi 规则 |
| `src/compiler/control_flow/production_control_flow.*` | gate-on 准备、primitive 编译、LLVM artifact 绑定 |
| `include/kxc/runtime/control_execution_plan.h`、`src/runtime/control_execution_plan.cc` | runtime schema v1、静态 ABI 和拒绝规则 |
| `include/kxc/runtime/control_session.h`、`src/runtime/control_session.cc` | CPU 默认流执行、preflight、branch/loop 运行和 completion 生命周期 |
| `test/control_plan_test.cpp`、`test/relay_control_plan_test.cpp`、`test/control_runtime_integration_test.cpp` | 结构、Relay、生产门禁与数值证据 |
| [M2](M2_KV_STATE.md)、[M3](M3_SHAPE_VALUES.md) | 未来 state/extent owner；M10 不复制其合同 |

控制流能力不新建模型名 registry，不把 `ControlRuntimeSession` 变成第二个通用 `RuntimeSession`，也不通过 profiling 事件改变编译或分支选择。任何 ABI、schema、effect 或 identity 变化必须同时更新对应版本、正例和执行前负例。

## 验证与退出条件

公共检查：

```bash
python3 tools/architecture/check_docs.py --root .
python3 tools/architecture/check_include_layers.py --root .
python3 tools/architecture/check_public_headers.py --root . --compile
git diff --check
```

M10 C1 完成需要：

- [ ] gate-on LLVM 构建中 `If` 两分支有真实 module/artifact 和数值结果；
- [ ] gate-on LLVM 构建中 `While` 0/1/多次迭代与上限拒绝均有结果；
- [ ] gate-off 入口拒绝、非法 plan 零 launch，并与 gate-on 证据分开记录；
- [ ] schema v2 → runtime v1、branch/loop/Phi/loop-carried 和 effect 边界有稳定 receipt；
- [ ] 明确 L1b 使用 host loop 还是 bounded graph loop；未选择的路径保留拒绝证据；
- [ ] 若接入 M2/M3，state/extent 的 owner、ABI、identity 和同 session decode 证据已完成；否则文档明确写出未接入原因；
- [ ] C4 的 MiniMind-O 需求不改变 L1 合同，能力矩阵逐格更新。

未满足这些条件时，控制流只能写成“源码和部分测试已存在、默认未启用”，不能写成“MiniMind 或 MiniMind-O 已支持控制流”。
