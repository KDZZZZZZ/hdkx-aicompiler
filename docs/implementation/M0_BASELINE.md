# M0：共同基线与证据口径

现在主线 `dev` 的编译运行路径以静态精确形状为主。有界动态执行在另一个分支中，包含五个提交；两篇新文档还把部分实验分支行为、默认主线路径和参考测试混在了一起。若直接分派任务，不同人可能在“是否已经支持动态执行”这件事上得到不同答案。

本模块要把大家工作的起点固定下来：核查并验证已有动态提交，纠正文档中会误导排期的统计，记录一个明确的基线 commit 和测试配置。它不增加 KV cache、执行观测或新算子；验收结果是一份真实、可重跑的共同起点。

> 状态：待实施。第一波的前置检查点 G0。当前事实以[架构总览](../ARCHITECTURE.md)为准；返回[并行计划](WAVE_1.md)。

## 当前代码与需要核实的事实

| 事项 | 当前核对结果 | 本模块动作 |
|---|---|---|
| 集成分支 | `dynamic-graph-stage2-integration` @ `f9f955b`，相对 `dev` @ `f379ebf` 有 5 个提交 | 开始实施时重新验证分支关系，不依靠旧 commit 列表盲目重放 |
| bounded 首切片 | 一份 CPU/LLVM elementwise 产物运行多个合法 shape；fresh-output，不含 state/alias/reuse | 专项运行后记录范围，不能升级成“动态 Transformer 已完成” |
| 模型覆盖 | 25 个目标 ONNX 名称中 importer 交集为 8，缺失 17；8 个交集也有语义限制 | 修正[代码库清单](../CODEBASE_INVENTORY.md)，按模型交集统计 |
| LLVM 矩阵 | 8 项 implemented，不是“7 项已通过” | 修正[项目目标](../PROJECT_GOAL.md)快照并区分实现与运行证据 |
| 通用状态 | 已有 is_state、alias、持久 buffer、pending async 拒绝 | 把“运行时完全没有状态”收窄为动态 KV 协议缺口 |
| identity 清理 | 形状与自适应代码会调用 experimental_identity builders | 从直接删除项移出；如需整理必须迁移消费者和身份兼容 |
| 分布式执行 | executor 的 compiled module kernel launch 尚未实现 | 明确“缺实现且缺证据”，不是只缺测试 |

待核对的既有提交从旧到新为 `ae175bb`、`0aa3adb`、`f6258a8`、`d9b9cbb`、`f9f955b`。这份列表用于定位，不授权覆盖主线或删除其他分支。

## 文档中两处边界需要统一

1. 项目目标 §2.4 已延后 parser，并允许 agent 经契约写入；§3 的“IR round-trip 是 agent 参与的一切的前提”应改为当前契约闭环的依赖，不再把 parser 列作阻塞。
2. §2.1 的“形态变化只允许热替换”与 §8 要求合入“一份产物服务多个 shape”需要共同解释。本计划拟采用：变体内的合法 shape 由已验证 bounded 合同处理；变体之间的切换由有限路由/热替换处理；两者都不得隐式编译。将这个解释写回目标边界后，再把它当作后续模块的共同前提。

## 实施步骤

1. 记录工作区改动和各分支 commit，使用隔离工作目录核查已有五个提交。保留用户未提交文档和机器本地 skill，不混入实现 PR。
2. 对照主线逐项确认 shape 求值、模块调用合同、内存计划和 identity 的消费者；重点检查 scalar 顺序、完整 guards、参数重定位和缓存复用。
3. 如果基点发生变化，在任务分支整合最新主线并解决具体冲突。五个提交作为一个已有纵向能力审查，避免只移入 API 或前半段提交。
4. 跑默认 CPU/LLVM 基线，再用下列 bounded 配置跑专项。开关关闭时也验证原有拒绝行为。
5. 把真实行为、文档校正和测试结果放入同一个集成评审中；状态矩阵与检查器一并核对。通用有界 elementwise 不能把 dynamic_batching 全行变成支持。
6. 记录 G0 commit、配置及测试结果，交给 A/B/C。后续集成只接受基于这个 commit 或其已验证后继的分支。

## 重点代码与测试

- [RuntimeSession](../../src/runtime/session.cc)、[执行计划](../../include/kxc/runtime/executable_plan.h)、[模块调用合同](../../include/kxc/runtime/compiled_module.h)。
- [compiler](../../src/compiler/compiler.cc)、[形状路由](../../src/compiler/shape/shape_control.cc)、[受限形状适配](../../src/compiler/shape/restricted_symbolic_shape.cc)、[identity](../../src/compiler/identity/experimental_identity.cc)。
- 集成分支新增的 `src/compiler/shape/dynamic_shape_contract.cc`、`test/bounded_dynamic_graph_llvm_test.cpp` 目前不在 `dev`，引用它们不表示已合入。

动态配置只在上述提交已经进入实施工作树后使用：

```bash
cmake -S . -B out/build/bounded-llvm -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DKXC_ENABLE_CUDA=OFF -DKXC_ENABLE_LLVM=ON \
  -DKXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI=ON \
  -DKXC_ENABLE_SHAPE_PRODUCTION_EXACT=ON \
  -DKXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE=ON \
  -DKXC_ENABLE_BOUNDED_DYNAMIC_GRAPH=ON
cmake --build out/build/bounded-llvm -j2
ctest --test-dir out/build/bounded-llvm -N
ctest --test-dir out/build/bounded-llvm --output-on-failure --no-tests=error \
  -R 'bounded_dynamic_graph_llvm_test|runtime_session_test|executable_plan_test|compiled_module_dynamic_llvm_test|shape_.*test|restricted_symbolic_shape_test|compiler_identity_test|te_schedule_test'
ctest --test-dir out/build/bounded-llvm --output-on-failure --no-tests=error
```

核对 `ctest -N` 中确实存在预期用例。LLVM 的发现与安装方法见[构建说明](../BUILDING.md)；`KXC_ENABLE_LLVM=ON` 本身不是发现成功的证明。

## G0 验收

- [ ] 同一编译产物对两组不同合法 shape 得到正确结果，产物身份保持稳定。
- [ ] 非法范围、整除关系和图语义在 cache publication/backend launch 前拒绝，失败不隐式编译。
- [ ] fresh-output 对 state/alias/reuse 的拒绝仍在；没有为了通过集成而放宽旧合同。
- [ ] 默认路径和启用 gates 路径均有实际测试结果，LLVM 测试未被静默跳过。
- [ ] 文档、矩阵和检查器未夸大能力；所有 G0 结论有 commit 和命令。
- [ ] A/B/C 获得同一个基点及明确的公共文件协调人。

## 风险与交接

主要风险是把一组“存在于分支中的测试源码”当成已经运行过的能力。遇到失败，先定位该层的合同和消费者；不能通过关闭 LLVM、删除负例或打开隐式回退把 G0 标为完成。

M0 只按实际兼容性变化更新既有版本，不额外发明新的 identity。其三项设计检查是：沿用既有 evaluator 和 builder；不新增第二份状态/形状权威；新增入口都由 production 编译和运行测试消费。
