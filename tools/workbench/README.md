# KXC 性能工作台

性能工作台提供本地 React/TypeScript 界面和 CLI，用于查看 KXC Profile Bundle。它只展示 Bundle 中实际存在的数据，不会根据缺失事件推断编译器或后端能力。

性能分析边界见 [架构总览](../../docs/ARCHITECTURE.md)。

## 要求

- Node.js 22.6 或更新版本；
- npm；
- 一个 KXC Profile Bundle，或仓库中已经提交的合成 Fixture。

## 浏览器界面

```bash
npm install --prefix tools/workbench
npm run fixture --prefix tools/workbench
npm run dev --prefix tools/workbench
```

打开 `http://localhost:5273`。顶部栏可以加载 `tools/workbench/fixtures/bundles/` 下的样例，也可以通过浏览器 File System Access API 加载本地 Bundle。

构建和测试：

```bash
npm run typecheck --prefix tools/workbench
npm test --prefix tools/workbench
npm run build --prefix tools/workbench
```

## CLI

CLI 可以在无浏览器环境下读取 Bundle：

```bash
node tools/workbench/cli/kxc-wb.mjs query phases \
  --bundle <bundle-directory>

node tools/workbench/cli/kxc-wb.mjs query pass-ranking \
  --bundle <bundle-directory> --json

node tools/workbench/cli/kxc-wb.mjs compare \
  --baseline <baseline-directory> \
  --candidate <candidate-directory> \
  --threshold 1.2 --json
```

运行 `node tools/workbench/cli/kxc-wb.mjs --help` 查看筛选参数和界面控制命令。`--json` 只输出一个 JSON 对象，不混入日志文本，便于自动化处理。

## Bundle 契约

解析器和查询层位于 [src/kxc](src/kxc)。主要规则如下：

- 事件时间戳使用 `ts_ns`；
- 流水线事件使用 `relay_pipeline`、`tir_pipeline` 等组件名；
- 可选字段可以缺失或为空；
- 只有当 Bundle 包含对应事件和关联字段时，缓存、内核、算子与形状视图才可用；
- 数据不完整时必须显示明确的“不可用”或“部分可用”状态，不能显示成看似可信的空图表。

`events.jsonl` 是更完整的事件来源。`trace.json` 是兼容 Chrome/Perfetto 的投影视图，可能缺少部分关联字段。`summary.json` 是聚合结果，不能替代事件级证据。

## Fixture 状态

`fixtures/bundles/` 下的 Bundle 是确定性的界面与查询回归数据。部分 Fixture 会刻意模拟旧版缓存、运行时和内核事件，用于测试对比与稀疏数据行为。它们不能证明当前编译器、CUDA、分布式或数值执行能力。

生成 Fixture：

```bash
node tools/workbench/scripts/make-fixture.mjs
```

Fixture 的来源与场景见 [fixtures/README.md](fixtures/README.md)。

## 源码布局

```text
cli/         无界面查询、对比与界面控制命令
scripts/     确定性 Fixture 生成器与开发辅助脚本
server/      可选的本地界面控制服务
src/kxc/     Bundle 解析与查询契约
src/state/   工作区状态、布局、撤销与重做
src/tiles/   可视化卡片与就绪状态注册表
src/ui/      外壳、列布局、命令与键盘交互
test/        查询、状态与应用测试
```

性能工作台只负责观察。它不能重写 Bundle、静默修复无效 Schema，也不能把不完整 Fixture 当成后端能力证据。
