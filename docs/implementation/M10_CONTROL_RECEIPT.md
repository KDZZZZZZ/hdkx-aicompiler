# M10 控制流 receipt（C0 审计 + C1 gate-on 证据 + C2 决策）

> 状态：已完成 C0/C1/C2（2026-09-07）。任务书见 [M10_STRUCTURED_CONTROL.md](M10_STRUCTURED_CONTROL.md)，分派见第二波并行计划（WAVE_2）。代码基点 `23b3f1c`。

## C0：合同与代码审计结论

- **唯一生产门禁**：`CMakeLists.txt` 的 `KXC_ENABLE_CONTROL_RUNTIME`（默认 OFF）；仓库内无第二个控制流开关、注册表或执行器（grep 核对：该选项仅 4 处引用——option 定义、两处编译定义传递、gate 判定）。
- **编译控制面唯一结构**：`src/compiler/control_flow/control_plan.{h,cc}` schema v2（任务仅 kernel/branch/loop；branch 有独立 then/else region 与 Phi 绑定；loop 有 condition/body region、condition value、carried bindings、`max_trip_count`）。
- **执行面唯一输入**：`include/kxc/runtime/control_execution_plan.h` runtime schema v1；编译 schema → runtime schema 经既有 adapter（`src/runtime/control_execution_plan.cc`），runtime 不读 Relay/TE/compiler callback/缓存对象。
- **当前接受条件**：全部 tensor rank/shape 静态精确且维度非负；任务在 CPU:0/默认流；predicate 是 CPU 标量 bool；kernel 输出 fresh-output effect；结构化 region 可达且依赖唯一。
- **当前必拒绝条件**：动态 extent（`kRuntimeExtent`）、持久 state、alias/donation/storage reuse、非默认设备/流（如 stream="borrowed"）、CUDA、未知 rank、任意数据相关控制值、缺真实 primitive artifact 的 branch/loop。执行前校验命中时 launch 计数为零（下述 receipt 的 `invalid_plan_launches=0`）。
- **既有测试分层**（互相不可替代）：`control_plan_test`=schema/验证；`control_plan_reference_executor_test`=独立参考执行器；`relay_control_plan_test`=Relay If/While lowering；`control_runtime_integration_test`=生产 gate 链路；本次新增 `control_llvm_receipt_test`=同一基线的 gate-on LLVM 数值 receipt（C1）。

## C1：gate-on LLVM 生产证据（同一构建、同一 commit）

构建：`KXC_ENABLE_CONTROL_RUNTIME=ON, KXC_USE_LLVM=1, CUDA=OFF`（`out/build/control-llvm`），Ninja Debug，基点 `23b3f1c`。证据由 `test/control_llvm_receipt_test.cpp` 产出（M10_RECEIPT 行），ctest 标签 `control-flow;control-runtime;cpu`。

| 证据 | 实测结果 |
|---|---|
| Relay `If(p, add(x,y), mul(x,y))` 双分支 | 两个 region 各绑定真实 primitive entry：`kxc_unit_0_add` / `kxc_unit_1_mul`；choice=then 时 launch 计数 then/else = `1/0`，choice=else 时 `0/1`；输出与独立参考逐元素一致（`if_output` checksum 记录） |
| `While`（condition-before-body，`max_trip_count=3`） | 0 次：`while_body_launches=0`、carried result=7；1 次：launches=1、result=8；3 次：launches=3、result=10（carried value 每次迭代更新，region 依赖按事件序记录） |
| 超 `max_trip_count` | 执行路径抛出 `ControlRuntimeSession loop exceeded max_trip_count`（`while_bound_rejected`） |
| 非法 plan（runtime extent / 非默认流） | plan validation 拒绝，`invalid_plan_launches=0` |
| gate-off 入口拒绝 | 默认构建（gate off）中 `CompileControlFlowExact` 抛 `disabled by KXC_ENABLE_CONTROL_RUNTIME`（执行前拒绝；只证门禁行为，不计入数值证据） |
| gate-off 全量回归 | 46/46 通过（含新 receipt 测试在 gate-off 分支的拒绝断言） |
| gate-on 全量回归 | 46/46 通过 |

**已记录边界（fail-closed，不改）**：纯 passthrough 的 `If(p, x, y)`（两个分支都不含 kernel）不产生任何 PrimitiveUnit，在 `CompilePrimitiveUnits requires at least one PrimitiveUnit` 合同下被拒绝。receipt 测试的 gate 用例因此使用 add/mul 非退化分支。若未来要支持 passthrough 分支（Phi 纯转发），属 lowering 能力扩展，需带正例+负例另行切片。

## C2：L1b 生成循环选择记录

**选择：host-side deterministic greedy loop（L1 首选）。** 已在 [G2 greedy 报告](G2_GREEDY_REPORT.md) 中用真实定容 LLVM fixture 验证。

- 理由：`generate()` 的采样与停止条件在 host 侧；导出的静态图无控制节点；host 循环能直接验证 M2 的 append/read、M3 的 `total = past + current` 与 M1 的 generation 关联，不需要把采样器塞进 Relay 控制图；`ControlRuntimeSession` 当前拒绝 state 与 runtime extent，接入真实 decode 必须等 M2 的 state owner 与 M3 的 extent ABI（C3 交接）。
- **bounded graph loop = 后续候选**：仅在真实导出保留 `While`、循环上限可静态证明、谓词为 CPU 标量 bool、循环携带值/输出 shape 可表达于现有 ABI、且 M2/M3 交接完成后评估；本波保留其拒绝证据（上表）。
- 本选择不声称"图已支持生成循环"；MiniMind L1b 的每步 decode 由 host 驱动。真实 fixture 已完成固定 4 步 greedy argmax、ONNX 数值对齐和 bundle metadata 关联（见 [G2 greedy 报告](G2_GREEDY_REPORT.md)）。

C4（MiniMind-O 双自回归调度、Mimi ring buffer、80ms 预算）需求清单另行立项，不进入 L1 合同。
