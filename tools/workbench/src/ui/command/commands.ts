/**
 * 命令面板命令定义（需求 §15）。
 *
 * 命令按分组展示，支持模糊搜索。
 * 部分命令（搜索 Pass/Op/Kernel）需要实查 bundle 的 facets。
 *
 * 导出纯函数 buildCommands(ctx) 便于测试。
 */

import { getTileSpec } from '../../tiles/registry'
import { getQueryClient } from '../../kxc/query-client'
import type {
  TileType,
  AnalysisContext,
  Workspace,
  Column,
  Tile,
  WorkbenchMode,
} from '../../state/types'
import type { MetaResult } from '../../kxc/query-protocol'

export interface Command {
  id: string
  label: string
  group: string
  description: string
  shortcut?: string
  run: () => void
  canExecute?: boolean
}

export interface CommandContext {
  store: any // useWorkbench 状态
  workspace: Workspace | null
  focusedColumn: Column | null
  focusedTile: Tile | null
  bundleMeta: MetaResult | null
  globalContext: AnalysisContext
  mode: WorkbenchMode
}

/**
 * 构建命令列表。
 */
export function buildCommands(ctx: CommandContext): Command[] {
  const commands: Command[] = []

  // ===== 打开图表 =====
  const tileGroup = '打开图表'
  const allTiles: TileType[] = [
    'kpi',
    'phase_breakdown',
    'hotspot_topn',
    'perfetto_timeline',
    'pass_waterfall',
    'pass_ranking',
    'shape_cache_heatmap',
    'kernel_duration_dist',
    'kernel_topn',
    'diagnostics',
    'regression_delta',
    'logs',
    'event_table',
  ]

  for (const tileType of allTiles) {
    const spec = getTileSpec(tileType)
    if (!spec) continue
    commands.push({
      id: `open-tile-${tileType}`,
      label: spec.title,
      group: tileGroup,
      description: spec.purpose,
      run: () => {
        if (ctx.focusedColumn) {
          ctx.store.addTile(ctx.focusedColumn.id, tileType)
          ctx.store.showToast('info', `已添加 ${spec.title}`)
        }
      },
      canExecute: !!ctx.focusedColumn,
    })
  }

  // ===== Workspace 操作 =====
  const workspaceGroup = '工作空间'

  commands.push(
    {
      id: 'new-workspace',
      label: '新建 Workspace',
      group: workspaceGroup,
      description: '创建新的分析空间',
      run: () => {
        const id = ctx.store.createWorkspace()
        ctx.store.switchWorkspace(id)
        ctx.store.showToast('info', '已创建新 Workspace')
      },
    },
    {
      id: 'rename-workspace',
      label: '重命名 Workspace',
      group: workspaceGroup,
      description: '给当前 Workspace 改名',
      run: () => {
        if (!ctx.workspace) return
        const name = prompt('新名称：', ctx.workspace.name)
        if (name) {
          ctx.store.renameWorkspace(ctx.workspace.id, name)
        }
      },
      canExecute: !!ctx.workspace,
    },
    {
      id: 'duplicate-workspace',
      label: '复制 Workspace',
      group: workspaceGroup,
      description: '复制当前 Workspace 及其所有列和图表',
      run: () => {
        if (!ctx.workspace) return
        const newId = ctx.store.duplicateWorkspace(ctx.workspace.id)
        ctx.store.switchWorkspace(newId)
        ctx.store.showToast('info', '已复制 Workspace')
      },
      canExecute: !!ctx.workspace,
    },
    {
      id: 'fork-workspace',
      label: 'Fork Workspace',
      group: workspaceGroup,
      description: '从当前分支创建新的独立 Workspace',
      run: () => {
        if (!ctx.focusedColumn) return
        const newId = ctx.store.forkWorkspaceFromColumn(ctx.focusedColumn.id)
        ctx.store.switchWorkspace(newId)
        ctx.store.showToast('info', '已 Fork Workspace')
      },
      canExecute: !!ctx.focusedColumn,
    }
  )

  // ===== Bundle 操作 =====
  const bundleGroup = 'Bundle'

  commands.push(
    {
      id: 'switch-bundle',
      label: '切换 Bundle',
      group: bundleGroup,
      description: '从已加载的 Bundle 中选择',
      run: () => {
        const bundleIds = Object.keys(ctx.store.bundles)
        if (bundleIds.length === 0) {
          ctx.store.showToast('warn', '还没有加载任何 Bundle')
          return
        }
        // 简单实现：显示第一个，真实场景需要选择对话框
        const bundleId = bundleIds[0]
        ctx.store.bindBundle(bundleId)
        ctx.store.showToast('info', '已切换 Bundle')
      },
      canExecute: Object.keys(ctx.store.bundles).length > 0,
    },
    {
      id: 'set-baseline',
      label: '设置 Baseline',
      group: bundleGroup,
      description: '选择对比基线 Bundle',
      run: () => {
        const bundleIds = Object.keys(ctx.store.bundles)
        if (bundleIds.length < 2) {
          ctx.store.showToast('warn', '需要至少两个已加载的 Bundle')
          return
        }
        const baselineId = bundleIds[0]
        ctx.store.bindBaseline(baselineId)
        ctx.store.showToast('info', '已设置 Baseline')
      },
      canExecute: Object.keys(ctx.store.bundles).length >= 2,
    }
  )

  // ===== 搜索与过滤 =====
  const searchGroup = '搜索与过滤'

  // 如果 bundle 已加载，查询 facets 列表
  // TODO: 实现真实的 facets 查询，当前只显示占位符
  // 待实现：调用 queryClient.query(bundleId, { kind: 'facets' }, filter)

  // Pass 搜索
  if (ctx.bundleMeta) {
    // 占位符命令，真实实现需查询 facets
    commands.push({
      id: 'search-passes-hint',
      label: '搜索 Pass (待实现)',
      group: searchGroup,
      description: '过滤到特定 Pass',
      run: () => {
        ctx.store.showToast('info', '搜索 Pass 功能待实现')
      },
      canExecute: true,
    })
  }

  // Op 搜索
  if (ctx.bundleMeta) {
    commands.push({
      id: 'search-ops-hint',
      label: '搜索 Op (待实现)',
      group: searchGroup,
      description: '过滤到特定 Op',
      run: () => {
        ctx.store.showToast('info', '搜索 Op 功能待实现')
      },
      canExecute: true,
    })
  }

  // Kernel 搜索
  if (ctx.bundleMeta) {
    commands.push({
      id: 'search-kernels-hint',
      label: '搜索 Kernel (待实现)',
      group: searchGroup,
      description: '过滤到特定 Kernel',
      run: () => {
        ctx.store.showToast('info', '搜索 Kernel 功能待实现')
      },
      canExecute: true,
    })
  }

  // ===== 布局操作 =====
  const layoutGroup = '布局操作'

  commands.push(
    {
      id: 'consume-left',
      label: 'Consume 到左列',
      group: layoutGroup,
      description: '将当前 Tile 收入左侧列',
      shortcut: '[',
      run: () => {
        if (ctx.focusedTile) {
          ctx.store.consume(ctx.focusedTile.id, 'left')
        }
      },
      canExecute: !!ctx.focusedTile,
    },
    {
      id: 'consume-right',
      label: 'Consume 到右列',
      group: layoutGroup,
      description: '将当前 Tile 收入右侧列',
      shortcut: ']',
      run: () => {
        if (ctx.focusedTile) {
          ctx.store.consume(ctx.focusedTile.id, 'right')
        }
      },
      canExecute: !!ctx.focusedTile,
    },
    {
      id: 'expel',
      label: 'Expel',
      group: layoutGroup,
      description: '将当前 Tile 从列中拆出',
      shortcut: 'E',
      run: () => {
        if (ctx.focusedTile) {
          ctx.store.expel(ctx.focusedTile.id)
        }
      },
      canExecute: !!ctx.focusedTile,
    },
    {
      id: 'pin',
      label: 'Pin 列',
      group: layoutGroup,
      description: '固定当前列，避免被自动替换',
      shortcut: 'P',
      run: () => {
        if (ctx.focusedColumn) {
          ctx.store.togglePin(ctx.focusedColumn.id)
        }
      },
      canExecute: !!ctx.focusedColumn,
    }
  )

  // ===== 模式切换 =====
  const modeGroup = '模式'

  commands.push(
    {
      id: 'toggle-gather',
      label: '进出 Gather 模式',
      group: modeGroup,
      description: '聚合分散的图表到同一屏幕',
      shortcut: 'G',
      run: () => {
        if (ctx.mode === 'gather') {
          ctx.store.exitGather()
          ctx.store.setMode('strip')
        } else {
          ctx.store.setMode('gather')
        }
      },
    },
    {
      id: 'toggle-overview',
      label: '进出 Overview 模式',
      group: modeGroup,
      description: '查看所有 Workspace 和 Column 的空间分布',
      shortcut: 'O',
      run: () => {
        ctx.store.setMode(ctx.mode === 'overview' ? 'strip' : 'overview')
      },
    },
    {
      id: 'toggle-focus',
      label: '进出 Focus 模式',
      group: modeGroup,
      description: '全屏查看单个 Tile',
      shortcut: 'F',
      run: () => {
        if (ctx.focusedTile) {
          ctx.store.enterFocus('tile', ctx.focusedTile.id)
        }
      },
      canExecute: !!ctx.focusedTile,
    },
    {
      id: 'toggle-compare',
      label: '开启对比模式',
      group: modeGroup,
      description: '并行分析两个 Bundle',
      shortcut: 'C',
      run: () => {
        if (ctx.workspace) {
          ctx.store.setCompareEnabled(!ctx.workspace.compareEnabled)
        }
      },
      canExecute: !!ctx.workspace,
    }
  )

  // ===== 编辑操作 =====
  const editGroup = '编辑'

  commands.push(
    {
      id: 'undo',
      label: '撤销',
      group: editGroup,
      description: '撤销上一操作',
      shortcut: 'Ctrl+Z',
      run: () => ctx.store.undo(),
      canExecute: ctx.store.canUndo(),
    },
    {
      id: 'redo',
      label: '重做',
      group: editGroup,
      description: '重做被撤销的操作',
      shortcut: 'Ctrl+Shift+Z',
      run: () => ctx.store.redo(),
      canExecute: ctx.store.canRedo(),
    }
  )

  // ===== 导出 =====
  const exportGroup = '导出'

  commands.push(
    {
      id: 'export-snapshot',
      label: '导出当前视图',
      group: exportGroup,
      description: '将整个 Workspace 和所有 Column 导出为 JSON',
      run: () => {
        // 这会在另一个模块中实现
        ctx.store.showToast('info', '导出功能将在下一步实现')
      },
    },
    {
      id: 'save-layout',
      label: '保存布局快照',
      group: exportGroup,
      description: '将当前布局保存为命名快照',
      run: () => {
        const name = prompt('布局名称：')
        if (name) {
          ctx.store.showToast('info', `将保存布局 "${name}"`)
        }
      },
    }
  )

  return commands.filter((c) => c.canExecute !== false)
}

/**
 * 模糊搜索匹配。
 * 简单的子序列匹配：输入字符序列是否以相同顺序出现在标签/描述中。
 */
export function fuzzyMatch(query: string, target: string): boolean {
  if (!query) return true
  query = query.toLowerCase()
  target = target.toLowerCase()

  let queryIdx = 0
  for (let i = 0; i < target.length && queryIdx < query.length; i++) {
    if (target[i] === query[queryIdx]) {
      queryIdx++
    }
  }
  return queryIdx === query.length
}

/**
 * 对命令进行模糊搜索。
 */
export function searchCommands(commands: Command[], query: string): Command[] {
  if (!query) return commands

  return commands.filter((cmd) => fuzzyMatch(query, cmd.label) || fuzzyMatch(query, cmd.description))
}
