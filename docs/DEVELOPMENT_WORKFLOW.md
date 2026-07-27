# 开发流程

仓库采用短生命周期任务分支和评审 Pull Request。
`dev` 是共享集成分支；`main` 是稳定分支，只通过评审发布流程推进。

## 开始任务

1. 选择或创建一个范围、验收标准和验证方式都明确的 Issue。
2. 从 `origin/dev` 更新本地 `dev`。
3. 从该提交创建命名清晰的任务分支。
4. 将改动限制在对应 Issue 的范围内。

示例：

```powershell
git fetch origin
git switch dev
git pull --ff-only origin dev
git switch -c pass-analysis-contract
```

除非用户明确要求，不要给分支名添加 Agent 或厂商前缀。

## 实施规则

- 保留与任务无关的工作区改动。
- 保持公共契约、实现、生成文件、测试与文档同步。
- 优先复用现有工具；没有必要时，优先删除而不是新增抽象。
- 未经明确要求，不要引入新依赖。
- 不支持的语义必须明确报错，不得引入静默回退，包括目标、算子、形状和运行时回退路径。
- 不要把图内 ID、对象地址或链接符号作为语义产物身份。

架构变更必须同步更新 [架构总览](ARCHITECTURE.md)。
算子与 Pass 变更必须遵循 [编译器扩展契约](COMPILER_EXTENSION_CONTRACT.md)。

## 验证

先运行能证明行为变化的最小测试，再执行对应的仓库契约检查：

```powershell
python tools/architecture/check_docs.py --root .
python tools/architecture/check_include_layers.py --root .
python python/tools/check_relay_op_contract.py --root . `
  --matrix contracts/relay_op_contract.json
python python/tools/check_pass_contract.py --root . `
  --matrix contracts/pass_contract.json
git diff --check
```

代码改动还需执行：

```powershell
cmake --build out/build/dev-mingw-cpu --parallel
ctest --test-dir out/build/dev-mingw-cpu `
  --output-on-failure --no-tests=error
```

LLVM、CUDA、CUPTI、控制流、形状或自适应功能变更必须执行对应测试矩阵。
被跳过的后端必须记录为验证缺口。

## 发起 Pull Request

发起 PR 前：

1. 执行 `git fetch origin`，校验当前分支与 `origin/dev` 的关系；
2. 运行 `git diff --check`；
3. 检查 `git status --short`；
4. 只清理本任务产生的生成文件；
5. 不要强推。

PR 说明应包含：

- 关联 Issue；
- 行为与边界变化；
- 完整测试命令与结果；
- 跳过的检查及原因；
- 风险或兼容性说明；
- 文档变更说明。

只有在 Issue 的验收标准全部达成后，才能使用 `Closes #<number>`。
只完成父级清单或部分架构片段不能直接关闭 Issue。

## 合并与清理

只有评审通过且必需检查全部完成后，才能合并到 `dev`。不要对共享分支强推。
任务分支合并后：

- 确认分支顶端提交已包含在 `dev` 中；
- 确认使用该分支的工作区干净；
- 删除本地与远端任务分支；
- 保留 `main`、`dev`、正在使用的活动分支和尚未合并的工作分支。

稳定发布通过仓库发布流程，将经过评审的 `dev` 内容迁移到 `main`。
