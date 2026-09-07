# G0 基线记录

> 状态：已完成（2026-09-07）。本文是[第一波并行计划](WAVE_1.md)第 0 节要求产出的唯一基点记录，A/B/C 三条线以本 commit 为共同代码基点。

## 1. 基点与分支关系

- 主线基点：`dev` @ `f379ebf`（默认静态精确形状路径）。
- 动态集成分支：`dynamic-graph-stage2-integration` @ `f9f955b`，相对 `dev` @ `f379ebf` 恰好领先 5 个提交（merge-base 即 `f379ebf`），从旧到新为 `ae175bb`、`0aa3adb`、`f6258a8`、`d9b9cbb`、`f9f955b`，合计 47 个文件、+4778/-409。与 [M0](M0_BASELINE.md) 的记载一致。
- 集成评审提交：本 commit（`docs/implementation/` 计划、文档事实修正与本记录一次性入库）。这是文档与事实修正的 docs commit，不改变任何运行时代码；A/B/C 的实现分支各自从它切出。

## 2. 工具链与 LLVM 发现

- cmake 3.28.3 + Ninja 1.11.1，GCC/Debug，8 核。
- `cmake --preset dev-ninja-cpu` 配置输出：`LLVM support: enabled (version 20.1.2)`，`CUDA support: disabled by KXC_ENABLE_CUDA=OFF`。LLVM ≥ 20 要求满足，发现成功（系统另有 llvm-config 18.1.3，未被采用）。

## 3. 实际测试清单

### 3.1 默认路径（gates 全关，`dev` @ `f379ebf`，out/build/dev-ninja-cpu）

```bash
cmake --preset dev-ninja-cpu && cmake --build --preset dev-ninja-cpu -j2
ctest --test-dir out/build/dev-ninja-cpu --output-on-failure --no-tests=error
```

- 结果：**42/42 通过，0 失败**（35.7 s），含 `op_numeric_llvm_test`（LLVM 数值，未跳过）、`runtime_session_test`、`profile_bundle_test`、`onnx_importer_test`。
- 公共检查：`check_relay_op_contract` 20/20、`check_pass_contract` 0 issues、`check_nlp_gpu_validation`（含 negative gates 全部 PASS）、`check_include_layers`（272 文件）、`check_docs`（16 文件）、`check_public_headers --compile`（87 安装头 + 9 experimental 头）、`git diff --check` 干净。

### 3.2 bounded 路径（gates 开启，`f9f955b`，隔离 worktree `out/worktrees/g0-bounded`）

```bash
cmake -S . -B out/build/bounded-llvm -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DKXC_ENABLE_CUDA=OFF -DKXC_ENABLE_LLVM=ON \
  -DKXC_ENABLE_DYNAMIC_COMPILED_MODULE_ABI=ON \
  -DKXC_ENABLE_SHAPE_PRODUCTION_EXACT=ON \
  -DKXC_ENABLE_RESTRICTED_SYMBOLIC_SHAPE=ON \
  -DKXC_ENABLE_BOUNDED_DYNAMIC_GRAPH=ON
cmake --build out/build/bounded-llvm -j2
ctest --test-dir out/build/bounded-llvm -N          # 44 个用例，含 bounded_dynamic_graph_llvm_test (#15)
ctest --test-dir out/build/bounded-llvm -R 'bounded_dynamic_graph_llvm_test|runtime_session_test|executable_plan_test|compiled_module_dynamic_llvm_test|shape_.*test|restricted_symbolic_shape_test|compiler_identity_test|te_schedule_test'
ctest --test-dir out/build/bounded-llvm              # 全量
```

- 专项：**12/12 通过**（`bounded_dynamic_graph_llvm_test`、`compiled_module_dynamic_llvm_test`、runtime/plan/shape 系、identity、TE schedule）。
- 全量：**44/44 通过，0 失败**。
- 失败或跳过项：**无**。未发现被静默跳过的 LLVM 用例。

## 4. G0 验收对照

- [x] 同一编译产物对两组不同合法 shape 得到正确结果：`bounded_dynamic_graph_llvm_test` 通过，产物身份稳定。
- [x] 非法范围/整除/图语义在 publication 与 launch 前拒绝：该测试的 `rejected_without_compile` 断言（含 state/alias 拒绝）通过，失败不隐式编译。
- [x] fresh-output 对 state/alias/reuse 的拒绝仍在：`compiled_module_dynamic_llvm_test`（`kDynamicFreshOutputV1`）通过，未放宽旧合同。
- [x] 默认与 gates 路径均有结果，LLVM 未被静默跳过。
- [x] 文档、矩阵与检查器未夸大能力：本次一并修正（见第 5 节），全部检查器通过。

## 5. 文档事实修正（随本 commit 入库）

1. [项目目标](../PROJECT_GOAL.md)：§2.2 快照的"LLVM 层 7 项通过"改为矩阵口径"8 项 `implemented`、4 项 `unsupported`"，并注明 `implemented` 不等于运行证据；"state 被动态路径拒绝"收窄为"通用 state/alias/重复执行机制已存在，KV 更新语义与动态有效长度是协议缺口"。§3 中"IR round-trip → agent 参与的一切"改为契约闭环依赖（parser 已延后，不再列作阻塞）。§2.1 补充与 §8 的共同解释：变体内合法 shape 由 bounded 合同处理、变体间切换由有限路由/热替换处理，均不得隐式编译。
2. [代码库清单](../CODEBASE_INVENTORY.md)：新增按模型交集统计（25 个模型名称与 importer 交集 8、缺失 17，并列出两个口径的区别与交集中的语义限制）；`experimental_identity` 由"零生产消费者"修正为"形状路由与自适应准备等门禁路径消费其 builders，删除前必须迁移"，并从直接删除项移出；分布式由"零测试"补强为"kernel launch 尚未实现，缺实现且缺证据"。

## 6. 范围限定（A/B/C 必须遵守的口径）

- bounded 能力仅为 **CPU/LLVM elementwise、fresh-output、固定秩有界 shape**：不含 state、alias、donation、storage reuse，不构成"动态 Transformer 已完成"。
- 通用运行时已有 is_state/alias/持久 buffer/pending async 拒绝；缺的是 KV 更新语义与动态有效长度（M2 处理）。
- 分布式 kernel launch 未实现；identity builders 有真实消费者，不得直接删除。
- 分派见 [WAVE_1.md](WAVE_1.md)；后续集成只接受基于本 commit（或其已验证后继）的分支。
