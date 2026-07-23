# Compiler Foundation 分支与集成规则

## 1. 基线

- 上游事实基线：`dev@3b95aca188122ff52ebdb2f43d390d21273ff3e2`
- 本轮集成分支：`integration/compiler-foundation`
- W0 规划共同基线：`e295a73343f82a7852eeae4ba6d11e3d892adc52`
- W1 固定基线分支：`baseline/compiler-foundation-w1@525950a`
- W1 已集成六条 isolated/default-off baseline，并通过 Region Task DAG OFF/ON 的 38/38 CPU CTest；详细证据见 [`../../handoffs/compiler-foundation/INTEGRATION.md`](../../handoffs/compiler-foundation/INTEGRATION.md)。
- W2 feature branches 从 `integration/compiler-foundation@525950a` 分叉，专门处理跨轨 production adapter，不回写 W1 固定基线。

`dev` 不接收并行开发中的中间状态。所有能力分支先进入
`integration/compiler-foundation`，通过统一门禁后，才具备提交 GitHub PR 到
`dev` 的条件。

## 2. 分支和 worktree

| 能力 | 分支 | worktree |
|---|---|---|
| Core contract / identity / cache | `feature/compiler-foundation-core` | `hdkx-aicompiler-wt-core` |
| Shape / specialization | `feature/compiler-foundation-shape` | `hdkx-aicompiler-wt-shape` |
| Adaptive compilation / hot swap | `feature/compiler-foundation-adaptive` | `hdkx-aicompiler-wt-adaptive` |
| Dynamic graph / control flow | `feature/compiler-foundation-control-flow` | `hdkx-aicompiler-wt-control-flow` |
| Region / ExecutionPlan / runtime | `feature/compiler-foundation-runtime-plan` | `hdkx-aicompiler-wt-runtime-plan` |
| NLP / GPU validation | `feature/compiler-foundation-nlp-gpu` | `hdkx-aicompiler-wt-nlp-gpu` |

W2 跨轨分支：

| 能力 | 分支 | worktree |
|---|---|---|
| Shape production exact adapter | `feature/compiler-foundation-shape-production` | `hdkx-aicompiler-wt-shape-production` |
| Adaptive production exact adapter | `feature/compiler-foundation-adaptive-production` | `hdkx-aicompiler-wt-adaptive-production` |
| Control runtime adapter/executor | `feature/compiler-foundation-control-runtime` | `hdkx-aicompiler-wt-control-runtime` |
| Runtime manifest/observability | `feature/compiler-foundation-runtime-observability` | `hdkx-aicompiler-wt-runtime-observability` |
| Transformer operator slices | `feature/compiler-foundation-nlp-transformer` | `hdkx-aicompiler-wt-nlp-transformer` |

## 3. 开发规则

1. 一个 worktree 只写自己的 feature branch。
2. 使用 Conventional Commits；一个提交只表达一个可独立审查的变化。
3. feature branch 不直接 merge、rebase 或 push 集成分支。
4. 不允许多个轨道通过未版本化的私有结构建立依赖；跨轨契约使用冻结 DTO、mock 或 fake。
5. 保留现有 static exact fallback；未通过 verifier、guard 或 feature gate 的能力不得进入生产默认路径。
6. 当前源码和可复现测试是 capability 事实源；handoff 或计划文字不能代替测试。
7. 不通过删除测试、放宽错误检查或伪造 skip 来取得绿色结果。
8. 每个轨道在 `docs/handoffs/compiler-foundation/` 下维护独立 handoff，记录提交、测试、限制和集成要求。

## 4. 合并顺序

开发并行，生产集成按依赖门禁进行：

1. Core contract、identity 和 cache correctness。
2. Shape exact path、static exact hot swap、静态 control-flow IR 和 task DAG contract 可分别审查。
3. 处理公共 DTO/CMake 冲突并运行统一回归。
4. 接入 shape-aware routing、control task 和 dynamic allocation 等交叉能力。
5. 使用 NLP/GPU fixture 验证垂直切片，不因 fixture 存在而宣称生产支持。

主集成者可使用 `git merge --no-ff` 保留能力分支边界；若分支尚未满足自己的
Done 条件，可以只 cherry-pick 已验证的独立提交，但必须在 handoff 中记录未合入项。

## 5. 合并门禁

每个候选分支至少满足：

- 工作树干净，提交可审查。
- `git diff --check` 通过。
- 新增 API 有正向、边界和负向测试。
- 相关现有回归测试通过。
- 并发代码有生命周期、停止、失败和竞争测试。
- cache/identity 代码使用完整相等性，不依赖 hash-only identity。
- Shape 复用没有 `cached_dims >= query_dims` 一类模糊正确性判断。
- `RuntimeSession` 仍是静态强类型数据面，不持有 Compiler、预测器或后台编译线程。
- handoff 明确列出未完成能力；不得把 feature-gated mock 写成当前能力。

集成分支完成后，统一运行 operator/pass contract、compiler、runtime、Shape、控制流、
并发和 NLP 数值测试，再决定是否建立远端 PR。当前流程不自动 push。
