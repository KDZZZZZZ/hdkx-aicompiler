/**
 * Agent 桥接：让 CLI（以及背后的 Claude）操作正在浏览的这个页面。
 *
 * 下行走 SSE（浏览器内置 EventSource），上行走普通 fetch POST。
 * 控制服务见 server/control-server.mjs。
 *
 * 两条设计原则：
 * 1. **命令白名单**。只暴露下面 COMMANDS 里显式列出的动作，不做
 *    "把 store 整个 action 表反射出去"这种事——那等于把内部实现变成外部契约，
 *    以后重构 store 就会悄悄破坏 agent。
 * 2. **状态由页面推送，服务端不持有真相**。agent 读到的永远是页面此刻的实际状态，
 *    而不是"命令发出后应该变成的样子"。命令失败时它能立刻发现。
 */

import { useEffect, useRef, useState } from 'react'
import { useWorkbench } from '../../state/store'
import { useBundleActions } from '../bundle/useBundleActions'
import { getTileSpec, ALL_TILE_SPECS } from '../../tiles/registry'
import type { TileType, WidthTier, WorkbenchMode } from '../../state/types'

const DEFAULT_PORT = 5274
const controlBase = (): string =>
  `http://127.0.0.1:${(import.meta.env.VITE_KXC_WB_PORT as string) || DEFAULT_PORT}`

interface CommandMessage {
  id: string
  cmd: string
  args: Record<string, unknown>
}

type CommandResult = { ok: true; result?: unknown } | { ok: false; error: string }
type CommandHandler = (
  args: Record<string, unknown>,
  deps: CommandDeps,
) => CommandResult | Promise<CommandResult>

/** 需要 hook 能力的命令（如加载 bundle）通过这里拿到入口。 */
interface CommandDeps {
  loadFixtureById: (id: string) => Promise<void>
  openDirectory: () => Promise<void>
  fixtures: Array<{ id: string; path: string; variant: string; event_count: number }>
}

const str = (v: unknown): string | null => (typeof v === 'string' && v ? v : null)

function isTileType(v: unknown): v is TileType {
  return typeof v === 'string' && ALL_TILE_SPECS.some((s) => s.type === v)
}

/**
 * 命令白名单。每条都要么成功、要么给出**能指导下一步的**错误信息——
 * agent 拿到 "未知列 col-3" 比拿到 "失败" 有用得多。
 */
const COMMANDS: Record<string, CommandHandler> = {
  /**
   * 加载 bundle。没有它 agent 就无法自己把数据装进来，
   * 用户还得先手动点一次下拉——那"指挥 Claude 分析"就断在第一步。
   */
  'load-bundle': async (a, deps) => {
    const id = str(a.id) ?? str(a.fixtureId)
    if (!id) {
      return {
        ok: false,
        error: `需要 id；当前可用样例：${deps.fixtures.map((f) => f.id).join(', ') || '（无，请先跑 npm run fixture）'}`,
      }
    }
    const known = deps.fixtures.find((f) => f.id === id)
    if (!known) {
      return {
        ok: false,
        error: `未知样例 ${id}；可用：${deps.fixtures.map((f) => f.id).join(', ')}`,
      }
    }
    await deps.loadFixtureById(id)
    const s = useWorkbench.getState()
    const loaded = Object.keys(s.bundles)
    return { ok: true, result: { loadedBundleIds: loaded } }
  },

  'new-column': (a) => {
    const id = useWorkbench.getState().addColumn({
      title: str(a.title) ?? '新建列',
      width: (str(a.width) as WidthTier) ?? '1/3',
    })
    return { ok: true, result: { columnId: id } }
  },

  'remove-column': (a) => {
    const id = str(a.columnId)
    if (!id) return { ok: false, error: '需要 columnId' }
    if (!useWorkbench.getState().doc.columns[id]) return { ok: false, error: `列不存在: ${id}` }
    useWorkbench.getState().removeColumn(id)
    return { ok: true }
  },

  'add-tile': (a) => {
    const s = useWorkbench.getState()
    const type = a.type
    if (!isTileType(type)) {
      return {
        ok: false,
        error: `未知图表类型 ${String(a.type)}；可用：${ALL_TILE_SPECS.map((t) => t.type).join(', ')}`,
      }
    }
    // 没指定列就用当前聚焦列；一个都没有就先建一列，省得 agent 还要判断。
    let columnId = str(a.columnId) ?? s.activeWorkspace()?.focusedColumnId ?? null
    if (!columnId || !s.doc.columns[columnId]) {
      columnId = s.addColumn({ title: getTileSpec(type).title, width: getTileSpec(type).preferredWidth })
    }
    const tileId = useWorkbench.getState().addTile(columnId, type)
    return { ok: true, result: { tileId, columnId } }
  },

  'remove-tile': (a) => {
    const id = str(a.tileId)
    if (!id) return { ok: false, error: '需要 tileId' }
    if (!useWorkbench.getState().doc.tiles[id]) return { ok: false, error: `Tile 不存在: ${id}` }
    useWorkbench.getState().removeTile(id)
    return { ok: true }
  },

  'set-width': (a) => {
    const id = str(a.columnId)
    const width = str(a.width) as WidthTier | null
    if (!id || !width) return { ok: false, error: '需要 columnId 与 width' }
    if (!['1/3', '1/2', '2/3', 'full', 'fit'].includes(width)) {
      return { ok: false, error: `width 只能是 1/3 | 1/2 | 2/3 | full | fit，收到 ${width}` }
    }
    useWorkbench.getState().setColumnWidth(id, width)
    return { ok: true }
  },

  drill: (a) => {
    const s = useWorkbench.getState()
    const kind = str(a.kind)
    const value = str(a.value)
    if (!kind || !value) return { ok: false, error: '需要 kind（pass/op/kernel/shape/...）与 value' }
    // 没给 tileId 就拿当前聚焦列的第一个 Tile 当来源，保证来源关系可追溯。
    const tileId =
      str(a.tileId) ??
      (() => {
        const w = s.activeWorkspace()
        const col = w?.focusedColumnId ? s.doc.columns[w.focusedColumnId] : null
        return col?.tileIds[0] ?? null
      })()
    if (!tileId) return { ok: false, error: '没有可作为来源的 Tile，请先 add-tile' }
    const colId = s.drillDown(tileId, { kind: kind as 'pass', value }, a.asTab === true)
    if (!colId) return { ok: false, error: '下钻失败：来源 Tile 不存在' }
    return { ok: true, result: { columnId: colId } }
  },

  consume: (a) => {
    const id = str(a.tileId)
    const dir = str(a.direction)
    if (!id || (dir !== 'left' && dir !== 'right')) {
      return { ok: false, error: '需要 tileId 与 direction=left|right' }
    }
    useWorkbench.getState().consume(id, dir)
    return { ok: true }
  },

  expel: (a) => {
    const id = str(a.tileId)
    if (!id) return { ok: false, error: '需要 tileId' }
    useWorkbench.getState().expel(id)
    return { ok: true }
  },

  pin: (a) => {
    const id = str(a.columnId)
    if (!id) return { ok: false, error: '需要 columnId' }
    useWorkbench.getState().togglePin(id)
    return { ok: true }
  },

  focus: (a) => {
    const s = useWorkbench.getState()
    const colId = str(a.columnId)
    const tileId = str(a.tileId)
    if (tileId) {
      s.focusTile(tileId)
      if (a.fullscreen === true) s.enterFocus('tile', tileId)
      return { ok: true }
    }
    if (colId) {
      s.focusColumn(colId)
      if (a.fullscreen === true) s.enterFocus('column', colId)
      return { ok: true }
    }
    return { ok: false, error: '需要 columnId 或 tileId' }
  },

  gather: (a) => {
    const s = useWorkbench.getState()
    const ids = Array.isArray(a.tileIds) ? a.tileIds.filter((x): x is string => typeof x === 'string') : []
    if (ids.length === 0) return { ok: false, error: '需要 tileIds 数组' }
    const missing = ids.filter((id) => !s.doc.tiles[id])
    if (missing.length) return { ok: false, error: `这些 Tile 不存在: ${missing.join(', ')}` }
    s.clearGatherSelection()
    for (const id of ids) s.toggleGatherSelection(id)
    const layoutId = useWorkbench.getState().enterGather(str(a.name) ?? undefined)
    return layoutId ? { ok: true, result: { layoutId } } : { ok: false, error: '进入 Gather 失败' }
  },

  mode: (a) => {
    const m = str(a.mode)
    if (!m || !['strip', 'gather', 'overview', 'focus'].includes(m)) {
      return { ok: false, error: 'mode 只能是 strip | gather | overview | focus' }
    }
    useWorkbench.getState().setMode(m as WorkbenchMode)
    return { ok: true }
  },

  'set-filter': (a) => {
    const patch = a.patch
    if (!patch || typeof patch !== 'object') return { ok: false, error: '需要 patch 对象' }
    useWorkbench.getState().setGlobalFilter(patch as Record<string, never>)
    return { ok: true }
  },

  'clear-filters': () => {
    useWorkbench.getState().clearGlobalFilters()
    return { ok: true }
  },

  'set-bundle': (a) => {
    const id = str(a.bundleId)
    const s = useWorkbench.getState()
    if (!id) return { ok: false, error: '需要 bundleId' }
    if (!s.bundles[id]) {
      return {
        ok: false,
        error: `Bundle 未加载: ${id}；已加载的有 ${Object.keys(s.bundles).join(', ') || '（无）'}`,
      }
    }
    s.bindBundle(id)
    return { ok: true }
  },

  'set-baseline': (a) => {
    const id = str(a.bundleId)
    const s = useWorkbench.getState()
    if (id && !s.bundles[id]) return { ok: false, error: `Bundle 未加载: ${id}` }
    s.bindBaseline(id)
    return { ok: true }
  },

  'new-workspace': (a) => {
    const id = useWorkbench.getState().createWorkspace(str(a.name) ?? undefined)
    return { ok: true, result: { workspaceId: id } }
  },

  'switch-workspace': (a) => {
    const id = str(a.workspaceId)
    if (!id) return { ok: false, error: '需要 workspaceId' }
    if (!useWorkbench.getState().doc.workspaces[id]) return { ok: false, error: `Workspace 不存在: ${id}` }
    useWorkbench.getState().switchWorkspace(id)
    return { ok: true }
  },

  undo: () => {
    useWorkbench.getState().undo()
    return { ok: true }
  },
  redo: () => {
    useWorkbench.getState().redo()
    return { ok: true }
  },
}

export const AGENT_COMMANDS = Object.keys(COMMANDS)

/** 推给 agent 的状态快照：只带定位与决策需要的字段，不带图表数据。 */
function snapshot() {
  const s = useWorkbench.getState()
  const d = s.doc
  const w = d.activeWorkspaceId ? d.workspaces[d.activeWorkspaceId] : null
  return {
    mode: s.mode,
    workspace: w
      ? {
          id: w.id,
          name: w.name,
          bundleId: w.bundleId,
          baselineBundleId: w.baselineBundleId,
          compareEnabled: w.compareEnabled,
          filters: w.filters,
          focusedColumnId: w.focusedColumnId,
        }
      : null,
    workspaces: d.workspaceOrder.map((id) => ({ id, name: d.workspaces[id]?.name ?? '' })),
    columns: (w?.columnIds ?? []).map((cid) => {
      const c = d.columns[cid]
      return {
        id: cid,
        title: c?.title ?? '',
        width: c?.width,
        pinned: c?.pinned ?? false,
        origin: c?.origin,
        context: c?.context ?? {},
        tiles: (c?.tileIds ?? []).map((tid) => {
          const t = d.tiles[tid]
          return t
            ? {
                id: tid,
                type: t.type,
                title: t.title ?? getTileSpec(t.type).title,
                readiness: getTileSpec(t.type).readiness,
              }
            : null
        }).filter(Boolean),
      }
    }),
    bundles: Object.entries(s.bundles).map(([id, b]) => ({
      id,
      label: b.ref.label,
      eventCount: b.meta.eventCount,
      malformedLines: b.meta.malformedLines,
    })),
    canUndo: s.undoStack.length > 0,
    availableTileTypes: ALL_TILE_SPECS.map((t) => ({
      type: t.type,
      title: t.title,
      purpose: t.purpose,
      readiness: t.readiness,
    })),
  }
}

export type AgentConnection = 'off' | 'connecting' | 'connected'

export function useAgentBridge(): AgentConnection {
  const [status, setStatus] = useState<AgentConnection>('connecting')
  const doc = useWorkbench((s) => s.doc)
  const mode = useWorkbench((s) => s.mode)
  const bundles = useWorkbench((s) => s.bundles)
  const connected = useRef(false)

  // bundle 加载能力来自 hook，命令处理器却在 effect 闭包里，
  // 用 ref 中转：始终指向最新一次渲染的入口，避免闭包里拿到过期的 fixtures 列表。
  const bundleActions = useBundleActions()
  const depsRef = useRef<CommandDeps>({
    loadFixtureById: bundleActions.loadFixtureById,
    openDirectory: bundleActions.openDirectory,
    fixtures: bundleActions.fixtures,
  })
  depsRef.current = {
    loadFixtureById: bundleActions.loadFixtureById,
    openDirectory: bundleActions.openDirectory,
    fixtures: bundleActions.fixtures,
  }

  // 命令订阅
  useEffect(() => {
    let es: EventSource | null = null
    let retry: number | null = null
    let closed = false

    // 没有 EventSource 就直接不启用（老浏览器、SSR、测试环境）。
    // agent 桥接是可选能力，缺了它工作台本身照常用。
    if (typeof EventSource === 'undefined') {
      setStatus('off')
      return
    }

    const connect = () => {
      if (closed) return
      es = new EventSource(`${controlBase()}/events`)

      es.addEventListener('hello', () => {
        connected.current = true
        setStatus('connected')
      })

      es.addEventListener('command', (ev) => {
        void (async () => {
          const msg = JSON.parse((ev as MessageEvent).data) as CommandMessage
          let result: CommandResult
          try {
            const handler = COMMANDS[msg.cmd]
            // 有的命令是异步的（加载 bundle 要读文件、解析事件），必须 await，
            // 否则回执会在数据装好之前就发出去，agent 随后读到的是半成品状态。
            result = handler
              ? await handler(msg.args ?? {}, depsRef.current)
              : { ok: false, error: `未知命令 ${msg.cmd}；可用：${AGENT_COMMANDS.join(', ')}` }
          } catch (err) {
            result = { ok: false, error: err instanceof Error ? err.message : String(err) }
          }
          // 回执里直接带上执行后的状态快照。
          // 否则 agent 执行完命令马上读状态会拿到 debounce 之前的旧数据，
          // 从而基于过期信息做下一步决策——这是最容易踩且最难查的坑。
          const after = snapshot()
          void fetch(`${controlBase()}/ack`, {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify({ id: msg.id, ...result, state: after }),
          }).catch(() => {})
          // 同时立刻推一份，让后续独立的 ui state 也读得到最新值。
          void fetch(`${controlBase()}/state`, {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify(after),
          }).catch(() => {})
        })()
      })

      es.onerror = () => {
        connected.current = false
        setStatus('off')
        es?.close()
        // 控制服务没起也不影响正常使用，安静重试就好。
        if (!closed) retry = window.setTimeout(connect, 3000)
      }
    }

    connect()
    return () => {
      closed = true
      if (retry) clearTimeout(retry)
      es?.close()
    }
  }, [])

  // 状态推送：变化后 debounce，避免拖动列宽时刷爆服务端。
  useEffect(() => {
    if (!connected.current) return
    const t = setTimeout(() => {
      void fetch(`${controlBase()}/state`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(snapshot()),
      }).catch(() => {})
    }, 200)
    return () => clearTimeout(t)
  }, [doc, mode, bundles, status])

  return status
}

/** 右下角的连接指示。agent 在操作页面时用户应当看得见。 */
export function AgentStatusBadge({ status }: { status: AgentConnection }): JSX.Element | null {
  if (status !== 'connected') return null
  return (
    <div
      role="status"
      style={{
        position: 'fixed',
        right: 10,
        bottom: 10,
        zIndex: 50,
        padding: '3px 8px',
        borderRadius: 'var(--radius)',
        background: 'var(--bg-raised)',
        border: '1px solid var(--sem-info)',
        color: 'var(--sem-info)',
        fontSize: 'var(--fs-xs)',
        pointerEvents: 'none',
      }}
    >
      ● agent 已连接
    </div>
  )
}
