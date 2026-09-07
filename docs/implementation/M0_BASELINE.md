# M0：共同基线与证据口径

第一波开始前，主线 `dev` 以静态精确形状为主，有界动态能力散落在一个未合入分支；模型清单还是匿名 Encoder 快照。G0 已核查分支、LLVM、feature gate、动态合同和文档口径，并把结果写入 [G0_BASELINE.md](G0_BASELINE.md)；随后 G1 已在 `dev` 集成第一波代码。现在 M0 的作用是维护这个共同事实，不再是下一波的开发阻塞。

本模块不增加 KV cache、执行观测或新算子。后续任何 MiniMind 工作都必须从 G0/G1 的已验证后继开始，并继续区分 bounded fresh-output、静态 ONNX 和真实 stateful decode。

> 状态：**已完成**（2026-09-07）。第一波入口证据见 [G0](G0_BASELINE.md)，组合验收见 [G1](G1_RECORD.md)；新的分派从 [WAVE_2](WAVE_2.md) 开始。当前目标模型以 [PROJECT_GOAL.md](../PROJECT_GOAL.md) §2.2 为准。

## 已固定的事实

| 事项 | G0/G1 结论 | 后续维护方式 |
|---|---|---|
| 基线提交 | G0 记录了 `dynamic-graph-stage2-integration` @ `f9f955b` 与当时 `dev` @ `f379ebf` 的关系；G1 集成点为 `dev` @ `796fb9f`，文档收尾为 `3c636fa` | 新工作只引用 G0/G1 和当前 `dev`，不盲目重放旧 commit 列表 |
| bounded 首切片 | CPU/LLVM、fixed-rank、有界、fresh-output；通过多个合法 shape，不含 state/alias/donation/reuse | 不能升级成“动态 Transformer 已完成”；M2/M3 需另立 state/shape 证据 |
| 模型覆盖 | 目标已改为 MiniMind；实际导出分别有 dynamic/static 原始图和折叠后的 L1a 统计 | 由 [M9](M9_MINIMIND_TARGET.md) 锁定导出参数，按 [OP_TODO](../OP_TODO.md) 的 raw/folded 两种口径维护 |
| LLVM | G0 默认路径与 bounded 路径都发现 LLVM 20.1.2 并实际运行 LLVM 测试 | 后续没有 LLVM 数值结果就不能更新 LLVM 能力格子 |
| 通用状态 | 已有 `is_state`、alias、持久 buffer、pending async 拒绝 | M2 只补 MiniMind 的 KV 更新和动态有效长度，不另建状态权威 |
| identity | shape/adaptive 路径仍有 `experimental_identity` builders 消费者 | 不直接删除；任何合同变化必须由已有 identity/version 表达 |
| 分布式 | compiled module kernel launch 仍未实现，也没有多 worker 数值证据 | M7 先补证据或明确降级，不借单机测试填分布式格子 |

## 已完成的边界统一

1. [PROJECT_GOAL.md](../PROJECT_GOAL.md) §2.4 已把契约闭环作为 agent 当前写入路径，parser/新 IR 延后，不再把 parser 当作其他模块的阻塞。
2. §2.1 与 §8 已统一动静边界：变体内合法 shape 由已验证 bounded 合同处理，变体间切换由有限路由/热替换处理；两者都不得隐式编译。
3. G0/G1 记录明确区分实现状态、参考计算、真实 LLVM/RuntimeSession 证据和 profiling 证据。第一波不把测试数量或名称交集写成完整模型支持。

## 已执行步骤与维护动作

1. 在隔离目录记录工作区改动和各分支 commit，核查五个动态提交及其消费者。
2. 对照主线确认 shape 求值、模块调用合同、内存计划和 identity 的消费者，重点检查 guards、参数顺序和缓存复用。
3. 跑默认 CPU/LLVM 与 bounded gates 配置，并保留开关关闭时的拒绝行为。
4. 修正文档统计，记录 LLVM 发现、feature flags、实际测试和未覆盖范围。
5. 以 G0 作为 A/B/C 的共同代码基点，以 G1 记录合并后的 runtime/ONNX/Equal 证据。
6. 维护动作：若编译器边界、目标模型或测试配置变化，先更新 G0/G1 的事实，再更新模块计划；不要用 M0 重新开启第一波。

## 重点代码与测试

- [RuntimeSession](../../src/runtime/session.cc)、[执行计划](../../include/kxc/runtime/executable_plan.h)、[模块调用合同](../../include/kxc/runtime/compiled_module.h)。
- [compiler](../../src/compiler/compiler.cc)、[形状路由](../../src/compiler/shape/shape_control.cc)、[受限形状适配](../../src/compiler/shape/restricted_symbolic_shape.cc)、[identity](../../src/compiler/identity/experimental_identity.cc)。
- G0 中的 `bounded_dynamic_graph_llvm_test`、`compiled_module_dynamic_llvm_test`、shape/identity/TE 测试，以及 G1 的 runtime/ONNX/Equal 测试。

G0 使用过的 bounded 配置如下；复核基线时仍需确认 LLVM 被发现，不能只看 `KXC_ENABLE_LLVM=ON`：

```bash
cmake -S . -B out/build/bounded-llvm -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug -DKXC_ENABLE_CUDA=OFF -DKXC_ENABLE_LLVM=ON \
  -DKXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI=ON \
  -DKXC_ENABLE_SHAPE_PRODUCTION_EXACT=ON \
  -DKXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE=ON \
  -DKXC_ENABLE_BOUNDED_DYNAMIC_GRAPH=ON
cmake --build out/build/bounded-llvm -j2
ctest --test-dir out/build/bounded-llvm --output-on-failure --no-tests=error
```

## G0 验收（已完成）

- [x] 同一编译产物对两组不同合法 shape 得到正确结果，产物身份保持稳定。
- [x] 非法范围、整除关系和图语义在 cache publication/backend launch 前拒绝，失败不隐式编译。
- [x] fresh-output 对 state/alias/reuse 的拒绝仍在；没有为了通过集成而放宽旧合同。
- [x] 默认路径和启用 gates 路径均有实际测试结果，LLVM 测试未被静默跳过。
- [x] 文档、矩阵和检查器未夸大能力；所有 G0 结论有 commit 和命令。
- [x] A/B/C 获得同一个基点及明确的公共文件协调人。

详细命令、测试数量和已知限制以 [G0 基线记录](G0_BASELINE.md) 为准；第一波组合证据以 [G1 验收记录](G1_RECORD.md) 为准。

## 风险与交接

主要风险是把一组“存在于分支中的测试源码”或“导出图的算子名称”当成已运行能力。遇到新模型失败，先定位合同和真实消费者；不能通过关闭 LLVM、删除负例或打开隐式回退把能力标为完成。

M0 只按实际兼容性变化更新既有版本，不额外发明新的 identity。其三项持续检查是：沿用既有 evaluator 和 builder；不新增第二份状态/形状权威；新增入口都由 production 编译和运行测试消费。
