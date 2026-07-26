/**
 * 桌布状态。
 *
 * 分成两半：
 * - doc：进入 Undo/Redo 与持久化（§16、§17）。
 * - ui ：模式、hover、加载状态、滚动位置等瞬时量，**不进历史**——需求 §17 明确
 *        要求数据加载、Hover 和临时游标不得进入历史。
 *
 * 所有会改变 doc 的操作都必须走 commit()，这样"哪些操作可撤销"只有一处定义。
 */

import { useMemo } from 'react'
import { create } from 'zustand'
import {
  DEFAULT_LINK_GROUP_ID,
  EMPTY_CONTEXT,
  type AnalysisContext,
  type Column,
  type ColumnOrigin,
  type ContextOverride,
  type DiagnosticNote,
  type GatherLayout,
  type LinkGroup,
  type Tile,
  type TileArrange,
  type TileSelection,
  type TileType,
  type WidthTier,
  type WorkbenchDoc,
  type WorkbenchMode,
  type WorkbenchSnapshot,
  type Workspace,
} from './types'
import { autoGatherLayout } from './gather-layout'
import { getTileSpec } from '../tiles/registry'
import type { BundleRef } from '../kxc/bundle-source'
import type { MetaResult } from '../kxc/query-protocol'

// ---------------------------------------------------------------------------
// id 生成：不用 Date.now/random，保证同一操作序列可复现，便于快照测试。
// ---------------------------------------------------------------------------
let idSeq = 0
function nextId(prefix: string): string {
  idSeq += 1
  return `${prefix}-${idSeq.toString(36)}`
}

/**
 * 把计数器推到已有 id 之后。
 *
 * 必须在恢复布局（importSnapshot / localStorage 自动恢复）之后调用：
 * idSeq 是模块级变量，页面一刷新就归零，而恢复回来的文档里已经存在
 * tile-3、col-2 这样的 id。不推进计数器的话，接着新建几个就会分配出
 * 与现存对象相同的 id——两个 Tile 共用一个 key，删一个会连带删掉另一个，
 * 而且要等用户操作到那一步才暴露，极难排查。
 */
function bumpIdSeq(doc: WorkbenchDoc): void {
  let max = idSeq
  // 只认本模块 nextId() 生成的形状。不加这道校验的话，像默认联动组 'global'
  // 这种非生成 id 会被整体按 base36 解析成一个天文数字，把计数器顶飞，
  // 之后所有新 id 都变成 tile-globam 这类乱码。
  const GENERATED = /^(?:ws|col|tile|lg|gather)-([0-9a-z]+)$/
  const scan = (id: string) => {
    const m = GENERATED.exec(id)
    if (!m) return
    const n = parseInt(m[1]!, 36)
    if (Number.isSafeInteger(n) && n > max) max = n
  }
  Object.keys(doc.workspaces).forEach(scan)
  Object.keys(doc.columns).forEach(scan)
  Object.keys(doc.tiles).forEach(scan)
  Object.keys(doc.linkGroups).forEach(scan)
  for (const w of Object.values(doc.workspaces)) {
    for (const g of w.gatherLayouts) scan(g.id)
  }
  idSeq = max
}
/** 逻辑时钟，替代 Date.now() 作为 createdAt/updatedAt。 */
let logicalClock = 0
function tick(): number {
  logicalClock += 1
  return logicalClock
}

// ---------------------------------------------------------------------------
// 已加载 bundle 的登记表（不进 doc：bundle 内容不是布局状态）
// ---------------------------------------------------------------------------
export interface LoadedBundle {
  ref: BundleRef
  meta: MetaResult
}

export interface UiState {
  mode: WorkbenchMode
  /** Focus 模式聚焦的对象。 */
  focusTarget: { kind: 'tile' | 'column'; id: string } | null
  /** 从哪个模式进入 Focus，Esc 时回到那里（§4.4、§14 Esc）。 */
  focusReturnMode: WorkbenchMode
  hover: TileSelection | null
  commandPaletteOpen: boolean
  searchOpen: boolean
  /** Strip 横向滚动位置，px。 */
  scrollLeft: number
  viewportWidth: number
  /** Gather 挑选阶段已选中的 tile。 */
  gatherSelection: string[]
  bundles: Record<string, LoadedBundle>
  loadingBundleId: string | null
  loadError: string | null
  /** 未保存改动标记（§5.2）。 */
  dirty: boolean
  toast: { kind: 'info' | 'warn' | 'error'; text: string } | null
}

export interface WorkbenchState extends UiState {
  doc: WorkbenchDoc
  undoStack: WorkbenchDoc[]
  redoStack: WorkbenchDoc[]
  lastCommitLabel: string | null

  // --- 派生读取 ---
  activeWorkspace: () => Workspace | null
  globalContext: () => AnalysisContext
  columnsOf: (workspaceId: string) => Column[]
  tilesOf: (columnId: string) => Tile[]

  // --- 历史 ---
  undo: () => void
  redo: () => void
  canUndo: () => boolean
  canRedo: () => boolean

  // --- Workspace（§3.1） ---
  createWorkspace: (name?: string) => string
  renameWorkspace: (id: string, name: string) => void
  duplicateWorkspace: (id: string) => string
  deleteWorkspace: (id: string) => void
  switchWorkspace: (id: string) => void
  forkWorkspaceFromColumn: (columnId: string) => string
  reorderWorkspace: (id: string, toIndex: number) => void

  // --- Column（§3.3、§12） ---
  addColumn: (opts?: Partial<Column> & { atIndex?: number }) => string
  removeColumn: (id: string) => void
  moveColumn: (id: string, toIndex: number) => void
  setColumnWidth: (id: string, width: WidthTier) => void
  setColumnArrange: (id: string, arrange: TileArrange) => void
  togglePin: (id: string) => void
  focusColumn: (id: string) => void
  setColumnContext: (id: string, patch: ContextOverride) => void

  // --- Tile（§3.4、§12.1、§12.2） ---
  addTile: (columnId: string, type: TileType, opts?: Partial<Tile>) => string
  removeTile: (id: string) => void
  moveTile: (tileId: string, toColumnId: string, toIndex?: number) => void
  duplicateTile: (id: string) => string
  setTileHeight: (id: string, px: number | null) => void
  setActiveTile: (columnId: string, tileId: string) => void
  focusTile: (id: string) => void
  consume: (tileId: string, direction: 'left' | 'right') => void
  expel: (tileId: string) => void
  setTileContext: (id: string, patch: ContextOverride) => void
  lockTileContext: (id: string, locked: boolean) => void
  resetTileContext: (id: string) => void
  promoteTileContext: (id: string) => void
  setTileSelection: (id: string, sel: TileSelection | null) => void
  setTileCompareMode: (id: string, mode: Tile['compareMode']) => void
  setTileLinkGroup: (id: string, groupId: string) => void

  // --- Drill Down（§7.4） ---
  drillDown: (tileId: string, sel: TileSelection, asTab?: boolean) => string | null

  // --- 过滤器（§7.1、§7.3） ---
  setGlobalFilter: (patch: ContextOverride) => void
  clearGlobalFilters: () => void

  // --- Gather（§4.2） ---
  toggleGatherSelection: (tileId: string) => void
  clearGatherSelection: () => void
  enterGather: (name?: string) => string | null
  exitGather: () => void
  saveGatherLayout: (name: string) => void
  activateGatherLayout: (id: string) => void
  deleteGatherLayout: (id: string) => void
  resetGatherLayout: () => void
  updateGatherSlots: (slots: GatherLayout['slots']) => void
  addTileToGather: (tileId: string) => void
  removeTileFromGather: (tileId: string) => void
  gatherToWorkspace: () => string | null

  // --- Link Group（§7.2） ---
  createLinkGroup: (name: string) => string
  updateLinkGroup: (id: string, patch: Partial<LinkGroup>) => void
  deleteLinkGroup: (id: string) => void

  // --- 模式 ---
  setMode: (mode: WorkbenchMode) => void
  enterFocus: (kind: 'tile' | 'column', id: string) => void
  exitFocus: () => void
  setHover: (sel: TileSelection | null) => void
  setCommandPaletteOpen: (open: boolean) => void
  setScroll: (left: number, viewportWidth: number) => void

  // --- Bundle（§2.1、§4.5） ---
  registerBundle: (ref: BundleRef, meta: MetaResult) => void
  bindBundle: (bundleId: string) => void
  bindBaseline: (bundleId: string | null) => void
  setCompareEnabled: (on: boolean) => void
  unregisterBundle: (bundleId: string) => void
  setLoading: (bundleId: string | null, error?: string | null) => void

  // --- 诊断标注（§11） ---
  setDiagnosticNote: (key: string, patch: Partial<DiagnosticNote>) => void

  // --- 持久化（§16） ---
  exportSnapshot: () => WorkbenchSnapshot
  importSnapshot: (snap: WorkbenchSnapshot) => void
  markSaved: () => void
  showToast: (kind: 'info' | 'warn' | 'error', text: string) => void
}

// ---------------------------------------------------------------------------
// 初始 doc
// ---------------------------------------------------------------------------

function makeWorkspace(name: string): Workspace {
  const t = tick()
  return {
    id: nextId('ws'),
    name,
    description: '',
    bundleId: null,
    baselineBundleId: null,
    filters: {},
    columnIds: [],
    focusedColumnId: null,
    focusedTileId: null,
    gatherLayouts: [],
    activeGatherLayoutId: null,
    focusPosition: 'center_left',
    compareEnabled: false,
    createdAt: t,
    updatedAt: t,
  }
}

function makeColumn(patch: Partial<Column> = {}): Column {
  return {
    id: nextId('col'),
    title: patch.title ?? '新建列',
    context: patch.context ?? {},
    sourceColumnId: patch.sourceColumnId ?? null,
    sourceTileId: patch.sourceTileId ?? null,
    sourceEventId: patch.sourceEventId ?? null,
    width: patch.width ?? '1/3',
    tileIds: patch.tileIds ?? [],
    arrange: patch.arrange ?? 'stack',
    activeTileId: patch.activeTileId ?? null,
    pinned: patch.pinned ?? false,
    compareRole: patch.compareRole ?? null,
    origin: patch.origin ?? 'manual',
    createdAt: tick(),
  }
}

function makeTile(type: TileType, patch: Partial<Tile> = {}): Tile {
  return {
    id: nextId('tile'),
    type,
    title: patch.title ?? null,
    context: patch.context ?? {},
    contextLocked: patch.contextLocked ?? false,
    heightPx: patch.heightPx ?? null,
    selection: patch.selection ?? null,
    linkGroupId: patch.linkGroupId ?? DEFAULT_LINK_GROUP_ID,
    sourceTileId: patch.sourceTileId ?? null,
    compareMode: patch.compareMode ?? 'single',
    createdAt: tick(),
  }
}

function initialDoc(): WorkbenchDoc {
  // 需求 §3.1：必须自动创建一个空 Workspace。
  const ws = makeWorkspace('分析 1')
  return {
    workspaces: { [ws.id]: ws },
    workspaceOrder: [ws.id],
    columns: {},
    tiles: {},
    linkGroups: {
      [DEFAULT_LINK_GROUP_ID]: {
        id: DEFAULT_LINK_GROUP_ID,
        name: '全局联动',
        color: '#5b9dd9',
        shares: {
          timeRange: true,
          hover: true,
          selection: true,
          zoom: true,
          cursor: true,
          filter: true,
          baseline: true,
        },
      },
    },
    activeWorkspaceId: ws.id,
    diagnosticNotes: {},
  }
}

const HISTORY_LIMIT = 100

// ---------------------------------------------------------------------------
// store
// ---------------------------------------------------------------------------

export const useWorkbench = create<WorkbenchState>((set, get) => {
  /**
   * 唯一的 doc 变更入口。任何不走这里的写操作都不会进入 Undo 历史，
   * 这正是我们对 Hover / 加载状态想要的效果。
   */
  function commit(label: string, mutate: (doc: WorkbenchDoc) => void) {
    const prev = get().doc
    const next: WorkbenchDoc = structuredClone(prev)
    mutate(next)
    const ws = next.activeWorkspaceId ? next.workspaces[next.activeWorkspaceId] : null
    if (ws) ws.updatedAt = tick()
    set((s) => ({
      doc: next,
      undoStack: [...s.undoStack, prev].slice(-HISTORY_LIMIT),
      redoStack: [],
      lastCommitLabel: label,
      dirty: true,
    }))
  }

  const ws = () => {
    const d = get().doc
    return d.activeWorkspaceId ? (d.workspaces[d.activeWorkspaceId] ?? null) : null
  }

  /** 把当前 workspace 的 tile 从 gather 引用中摘掉，避免悬空引用。 */
  function dropTileFromGathers(doc: WorkbenchDoc, tileId: string) {
    for (const w of Object.values(doc.workspaces)) {
      for (const g of w.gatherLayouts) {
        g.slots = g.slots.filter((s) => s.tileId !== tileId)
      }
    }
  }

  function activeGather(w: Workspace): GatherLayout | null {
    if (!w.activeGatherLayoutId) return null
    return w.gatherLayouts.find((g) => g.id === w.activeGatherLayoutId) ?? null
  }

  return {
    doc: initialDoc(),
    undoStack: [],
    redoStack: [],
    lastCommitLabel: null,

    mode: 'strip',
    focusTarget: null,
    focusReturnMode: 'strip',
    hover: null,
    commandPaletteOpen: false,
    searchOpen: false,
    scrollLeft: 0,
    viewportWidth: 1600,
    gatherSelection: [],
    bundles: {},
    loadingBundleId: null,
    loadError: null,
    dirty: false,
    toast: null,

    // ---- 派生 ----
    activeWorkspace: ws,
    globalContext: () => {
      const w = ws()
      if (!w) return EMPTY_CONTEXT
      return {
        ...EMPTY_CONTEXT,
        ...w.filters,
        bundleId: w.bundleId,
        baselineBundleId: w.baselineBundleId,
        candidateBundleId: w.compareEnabled ? w.bundleId : null,
      }
    },
    columnsOf: (workspaceId) => {
      const d = get().doc
      const w = d.workspaces[workspaceId]
      if (!w) return []
      return w.columnIds.map((id) => d.columns[id]).filter((c): c is Column => Boolean(c))
    },
    tilesOf: (columnId) => {
      const d = get().doc
      const c = d.columns[columnId]
      if (!c) return []
      return c.tileIds.map((id) => d.tiles[id]).filter((t): t is Tile => Boolean(t))
    },

    // ---- 历史 ----
    undo: () =>
      set((s) => {
        const prev = s.undoStack[s.undoStack.length - 1]
        if (!prev) return s
        return {
          doc: prev,
          undoStack: s.undoStack.slice(0, -1),
          redoStack: [...s.redoStack, s.doc].slice(-HISTORY_LIMIT),
          dirty: true,
        }
      }),
    redo: () =>
      set((s) => {
        const next = s.redoStack[s.redoStack.length - 1]
        if (!next) return s
        return {
          doc: next,
          redoStack: s.redoStack.slice(0, -1),
          undoStack: [...s.undoStack, s.doc].slice(-HISTORY_LIMIT),
          dirty: true,
        }
      }),
    canUndo: () => get().undoStack.length > 0,
    canRedo: () => get().redoStack.length > 0,

    // ---- Workspace ----
    createWorkspace: (name) => {
      const w = makeWorkspace(name ?? `分析 ${get().doc.workspaceOrder.length + 1}`)
      commit('新建 Workspace', (d) => {
        d.workspaces[w.id] = w
        d.workspaceOrder.push(w.id)
        d.activeWorkspaceId = w.id
      })
      return w.id
    },
    renameWorkspace: (id, name) =>
      commit('重命名 Workspace', (d) => {
        const w = d.workspaces[id]
        if (w) w.name = name
      }),
    duplicateWorkspace: (id) => {
      const src = get().doc.workspaces[id]
      if (!src) return id
      const newId = nextId('ws')
      commit('复制 Workspace', (d) => {
        const s = d.workspaces[id]!
        const colMap = new Map<string, string>()
        const tileMap = new Map<string, string>()
        const newCols: string[] = []
        for (const cid of s.columnIds) {
          const c = d.columns[cid]
          if (!c) continue
          const nc = makeColumn({ ...c, tileIds: [] })
          colMap.set(cid, nc.id)
          for (const tid of c.tileIds) {
            const t = d.tiles[tid]
            if (!t) continue
            const nt = makeTile(t.type, { ...t, sourceTileId: t.id })
            tileMap.set(tid, nt.id)
            d.tiles[nt.id] = nt
            nc.tileIds.push(nt.id)
          }
          nc.activeTileId = c.activeTileId ? (tileMap.get(c.activeTileId) ?? null) : null
          d.columns[nc.id] = nc
          newCols.push(nc.id)
        }
        // 修正列间来源引用，指向副本而不是原件。
        for (const cid of newCols) {
          const c = d.columns[cid]!
          if (c.sourceColumnId) c.sourceColumnId = colMap.get(c.sourceColumnId) ?? null
          if (c.sourceTileId) c.sourceTileId = tileMap.get(c.sourceTileId) ?? null
        }
        d.workspaces[newId] = {
          ...structuredClone(s),
          id: newId,
          name: `${s.name} 副本`,
          columnIds: newCols,
          focusedColumnId: s.focusedColumnId ? (colMap.get(s.focusedColumnId) ?? null) : null,
          focusedTileId: null,
          gatherLayouts: s.gatherLayouts.map((g) => ({
            ...g,
            id: nextId('gather'),
            slots: g.slots
              .map((sl) => ({ ...sl, tileId: tileMap.get(sl.tileId) ?? '' }))
              .filter((sl) => sl.tileId),
          })),
          activeGatherLayoutId: null,
          createdAt: tick(),
          updatedAt: tick(),
        }
        d.workspaceOrder.push(newId)
        d.activeWorkspaceId = newId
      })
      return newId
    },
    deleteWorkspace: (id) =>
      commit('删除 Workspace', (d) => {
        const w = d.workspaces[id]
        if (!w) return
        for (const cid of w.columnIds) {
          const c = d.columns[cid]
          if (c) for (const tid of c.tileIds) delete d.tiles[tid]
          delete d.columns[cid]
        }
        delete d.workspaces[id]
        d.workspaceOrder = d.workspaceOrder.filter((x) => x !== id)
        if (d.activeWorkspaceId === id) {
          d.activeWorkspaceId = d.workspaceOrder[0] ?? null
          // 需求 §3.1：始终至少有一个 Workspace。
          if (!d.activeWorkspaceId) {
            const fresh = makeWorkspace('分析 1')
            d.workspaces[fresh.id] = fresh
            d.workspaceOrder.push(fresh.id)
            d.activeWorkspaceId = fresh.id
          }
        }
      }),
    switchWorkspace: (id) =>
      commit('切换 Workspace', (d) => {
        if (d.workspaces[id]) d.activeWorkspaceId = id
      }),
    forkWorkspaceFromColumn: (columnId) => {
      const newId = nextId('ws')
      commit('Fork 为新 Workspace', (d) => {
        const cur = d.activeWorkspaceId ? d.workspaces[d.activeWorkspaceId] : null
        const col = d.columns[columnId]
        if (!cur || !col) return
        const idx = cur.columnIds.indexOf(columnId)
        // 从来源列开始（含）向右的分支被搬进新 Workspace。
        const branch = idx >= 0 ? cur.columnIds.slice(idx) : [columnId]
        cur.columnIds = cur.columnIds.filter((c) => !branch.includes(c))
        if (cur.focusedColumnId && branch.includes(cur.focusedColumnId)) {
          cur.focusedColumnId = cur.columnIds[cur.columnIds.length - 1] ?? null
        }
        const w: Workspace = {
          ...makeWorkspace(`${col.title} 分支`),
          id: newId,
          bundleId: cur.bundleId,
          baselineBundleId: cur.baselineBundleId,
          filters: { ...cur.filters },
          columnIds: branch,
          focusedColumnId: branch[0] ?? null,
        }
        const first = branch[0] ? d.columns[branch[0]] : null
        if (first) first.origin = 'fork'
        d.workspaces[newId] = w
        d.workspaceOrder.push(newId)
        d.activeWorkspaceId = newId
      })
      return newId
    },
    reorderWorkspace: (id, toIndex) =>
      commit('调整 Workspace 顺序', (d) => {
        const from = d.workspaceOrder.indexOf(id)
        if (from < 0) return
        d.workspaceOrder.splice(from, 1)
        d.workspaceOrder.splice(Math.max(0, Math.min(toIndex, d.workspaceOrder.length)), 0, id)
      }),

    // ---- Column ----
    addColumn: (opts = {}) => {
      const col = makeColumn(opts)
      commit('新建 Column', (d) => {
        const w = d.activeWorkspaceId ? d.workspaces[d.activeWorkspaceId] : null
        if (!w) return
        d.columns[col.id] = col
        const at = opts.atIndex ?? w.columnIds.length
        w.columnIds.splice(Math.max(0, Math.min(at, w.columnIds.length)), 0, col.id)
        w.focusedColumnId = col.id
      })
      return col.id
    },
    removeColumn: (id) =>
      commit('删除 Column', (d) => {
        const w = d.activeWorkspaceId ? d.workspaces[d.activeWorkspaceId] : null
        const c = d.columns[id]
        if (!w || !c) return
        for (const tid of c.tileIds) {
          dropTileFromGathers(d, tid)
          delete d.tiles[tid]
        }
        delete d.columns[id]
        const idx = w.columnIds.indexOf(id)
        w.columnIds = w.columnIds.filter((x) => x !== id)
        if (w.focusedColumnId === id) {
          w.focusedColumnId = w.columnIds[Math.max(0, idx - 1)] ?? null
        }
      }),
    moveColumn: (id, toIndex) =>
      commit('移动 Column', (d) => {
        const w = d.activeWorkspaceId ? d.workspaces[d.activeWorkspaceId] : null
        if (!w) return
        const from = w.columnIds.indexOf(id)
        if (from < 0) return
        w.columnIds.splice(from, 1)
        w.columnIds.splice(Math.max(0, Math.min(toIndex, w.columnIds.length)), 0, id)
      }),
    setColumnWidth: (id, width) =>
      commit('调整列宽', (d) => {
        const c = d.columns[id]
        if (c) c.width = width
      }),
    setColumnArrange: (id, arrange) =>
      commit('切换 Stack/Tab', (d) => {
        const c = d.columns[id]
        if (!c) return
        c.arrange = arrange
        if (arrange === 'tabs' && !c.activeTileId) c.activeTileId = c.tileIds[0] ?? null
      }),
    togglePin: (id) =>
      commit('Pin', (d) => {
        const c = d.columns[id]
        if (c) c.pinned = !c.pinned
      }),
    focusColumn: (id) =>
      // 聚焦不改变文档结构，但需求 §16 要求持久化聚焦位置，所以仍写入 doc；
      // 为避免污染 Undo 历史，这里直接 set 而不 commit。
      set((s) => {
        const d = structuredClone(s.doc)
        const w = d.activeWorkspaceId ? d.workspaces[d.activeWorkspaceId] : null
        if (w && d.columns[id]) {
          w.focusedColumnId = id
          w.focusedTileId = d.columns[id]!.tileIds[0] ?? null
        }
        return { doc: d, dirty: true }
      }),
    setColumnContext: (id, patch) =>
      commit('修改列过滤器', (d) => {
        const c = d.columns[id]
        if (c) c.context = { ...c.context, ...patch }
      }),

    // ---- Tile ----
    addTile: (columnId, type, opts = {}) => {
      const tile = makeTile(type, opts)
      commit('添加 Tile', (d) => {
        const c = d.columns[columnId]
        if (!c) return
        d.tiles[tile.id] = tile
        c.tileIds.push(tile.id)
        if (c.arrange === 'tabs') c.activeTileId = tile.id
      })
      return tile.id
    },
    removeTile: (id) =>
      commit('关闭 Tile', (d) => {
        dropTileFromGathers(d, id)
        for (const c of Object.values(d.columns)) {
          if (!c.tileIds.includes(id)) continue
          c.tileIds = c.tileIds.filter((t) => t !== id)
          if (c.activeTileId === id) c.activeTileId = c.tileIds[0] ?? null
        }
        delete d.tiles[id]
      }),
    moveTile: (tileId, toColumnId, toIndex) =>
      commit('移动 Tile', (d) => {
        for (const c of Object.values(d.columns)) {
          if (!c.tileIds.includes(tileId)) continue
          c.tileIds = c.tileIds.filter((t) => t !== tileId)
          if (c.activeTileId === tileId) c.activeTileId = c.tileIds[0] ?? null
        }
        const target = d.columns[toColumnId]
        if (!target) return
        const at = toIndex ?? target.tileIds.length
        target.tileIds.splice(Math.max(0, Math.min(at, target.tileIds.length)), 0, tileId)
        if (target.arrange === 'tabs') target.activeTileId = tileId
      }),
    duplicateTile: (id) => {
      const src = get().doc.tiles[id]
      if (!src) return id
      const copy = makeTile(src.type, { ...src, sourceTileId: src.id })
      commit('复制 Tile', (d) => {
        d.tiles[copy.id] = copy
        for (const c of Object.values(d.columns)) {
          const idx = c.tileIds.indexOf(id)
          if (idx >= 0) {
            c.tileIds.splice(idx + 1, 0, copy.id)
            break
          }
        }
      })
      return copy.id
    },
    setTileHeight: (id, px) =>
      commit('调整 Tile 高度', (d) => {
        const t = d.tiles[id]
        if (t) t.heightPx = px
      }),
    setActiveTile: (columnId, tileId) =>
      set((s) => {
        const d = structuredClone(s.doc)
        const c = d.columns[columnId]
        if (c && c.tileIds.includes(tileId)) c.activeTileId = tileId
        return { doc: d, dirty: true }
      }),
    focusTile: (id) =>
      set((s) => {
        const d = structuredClone(s.doc)
        const w = d.activeWorkspaceId ? d.workspaces[d.activeWorkspaceId] : null
        if (w) {
          w.focusedTileId = id
          const owner = Object.values(d.columns).find((c) => c.tileIds.includes(id))
          if (owner) w.focusedColumnId = owner.id
        }
        return { doc: d, dirty: true }
      }),
    consume: (tileId, direction) =>
      commit(`Consume 到${direction === 'left' ? '左' : '右'}列`, (d) => {
        const w = d.activeWorkspaceId ? d.workspaces[d.activeWorkspaceId] : null
        if (!w) return
        const owner = Object.values(d.columns).find((c) => c.tileIds.includes(tileId))
        if (!owner) return
        const idx = w.columnIds.indexOf(owner.id)
        const targetId = w.columnIds[direction === 'left' ? idx - 1 : idx + 1]
        if (!targetId) return
        const target = d.columns[targetId]
        if (!target) return
        owner.tileIds = owner.tileIds.filter((t) => t !== tileId)
        if (owner.activeTileId === tileId) owner.activeTileId = owner.tileIds[0] ?? null
        target.tileIds.push(tileId)
        // §12.1：根据空间自动选择 Stack 或 Tab。超过 3 个就转 Tab，避免每个都被压扁。
        if (target.tileIds.length > 3) {
          target.arrange = 'tabs'
          target.activeTileId = tileId
        }
        // 空列自动移除，避免 Consume 后留下空壳。
        if (owner.tileIds.length === 0) {
          delete d.columns[owner.id]
          w.columnIds = w.columnIds.filter((x) => x !== owner.id)
          if (w.focusedColumnId === owner.id) w.focusedColumnId = targetId
        }
      }),
    expel: (tileId) =>
      commit('Expel', (d) => {
        const w = d.activeWorkspaceId ? d.workspaces[d.activeWorkspaceId] : null
        if (!w) return
        const owner = Object.values(d.columns).find((c) => c.tileIds.includes(tileId))
        if (!owner || owner.tileIds.length <= 1) return
        const t = d.tiles[tileId]
        owner.tileIds = owner.tileIds.filter((x) => x !== tileId)
        if (owner.activeTileId === tileId) owner.activeTileId = owner.tileIds[0] ?? null
        // §12.2：保持原始宽度偏好，放在当前列相邻位置。
        const spec = t ? getTileSpec(t.type) : null
        const col = makeColumn({
          title: spec?.title ?? '拆出',
          width: spec?.preferredWidth ?? owner.width,
          tileIds: [tileId],
          origin: 'expel',
          sourceColumnId: owner.id,
          sourceTileId: tileId,
          context: { ...owner.context },
        })
        d.columns[col.id] = col
        const idx = w.columnIds.indexOf(owner.id)
        w.columnIds.splice(idx + 1, 0, col.id)
        w.focusedColumnId = col.id
      }),
    setTileContext: (id, patch) =>
      commit('修改 Tile 过滤器', (d) => {
        const t = d.tiles[id]
        if (t) t.context = { ...t.context, ...patch }
      }),
    lockTileContext: (id, locked) =>
      commit('锁定局部过滤器', (d) => {
        const t = d.tiles[id]
        if (t) t.contextLocked = locked
      }),
    resetTileContext: (id) =>
      commit('重置到全局过滤器', (d) => {
        const t = d.tiles[id]
        if (t) {
          t.context = {}
          t.contextLocked = false
        }
      }),
    promoteTileContext: (id) =>
      commit('提升为全局过滤器', (d) => {
        const w = d.activeWorkspaceId ? d.workspaces[d.activeWorkspaceId] : null
        const t = d.tiles[id]
        if (!w || !t) return
        w.filters = { ...w.filters, ...t.context }
        t.context = {}
      }),
    setTileSelection: (id, sel) =>
      // 选择是瞬时状态，不进历史。
      set((s) => {
        const d = structuredClone(s.doc)
        const t = d.tiles[id]
        if (t) t.selection = sel
        return { doc: d }
      }),
    setTileCompareMode: (id, mode) =>
      commit('切换对比模式', (d) => {
        const t = d.tiles[id]
        if (t) t.compareMode = mode
      }),
    setTileLinkGroup: (id, groupId) =>
      commit('修改联动组', (d) => {
        const t = d.tiles[id]
        if (t) t.linkGroupId = groupId
      }),

    // ---- Drill Down ----
    drillDown: (tileId, sel, asTab = false) => {
      const state = get()
      const srcTile = state.doc.tiles[tileId]
      if (!srcTile) return null
      const owner = Object.values(state.doc.columns).find((c) => c.tileIds.includes(tileId))
      if (!owner) return null

      // 由选择对象决定新列该放什么图，这是"下钻到合适的默认列"（§7.4.2）。
      const { title, tiles: tileTypes, context } = drillPlan(sel)

      if (asTab) {
        const ids: string[] = []
        commit('Drill Down 作为 Tab', (d) => {
          const col = d.columns[owner.id]
          if (!col) return
          col.arrange = 'tabs'
          for (const tt of tileTypes) {
            const nt = makeTile(tt, { context, sourceTileId: tileId })
            d.tiles[nt.id] = nt
            col.tileIds.push(nt.id)
            ids.push(nt.id)
          }
          col.activeTileId = ids[0] ?? col.activeTileId
        })
        return ids[0] ?? null
      }

      const newCol = makeColumn({
        title,
        width: tileTypes.length > 1 ? '1/2' : '1/3',
        origin: 'drill_down',
        sourceColumnId: owner.id,
        sourceTileId: tileId,
        sourceEventId: sel.eventId ?? null,
        context: { ...owner.context, ...context },
      })
      commit('Drill Down', (d) => {
        const w = d.activeWorkspaceId ? d.workspaces[d.activeWorkspaceId] : null
        if (!w) return
        const idx = w.columnIds.indexOf(owner.id)
        // §4.1.5：从中间列重新钻取时，默认替换其右侧未固定的列；Pin 列不动。
        const right = w.columnIds.slice(idx + 1)
        const keep = right.filter((cid) => d.columns[cid]?.pinned)
        for (const cid of right) {
          if (keep.includes(cid)) continue
          const c = d.columns[cid]
          if (c) {
            for (const tid of c.tileIds) {
              dropTileFromGathers(d, tid)
              delete d.tiles[tid]
            }
          }
          delete d.columns[cid]
        }
        w.columnIds = [...w.columnIds.slice(0, idx + 1), ...keep]
        for (const tt of tileTypes) {
          const nt = makeTile(tt, { sourceTileId: tileId })
          d.tiles[nt.id] = nt
          newCol.tileIds.push(nt.id)
        }
        d.columns[newCol.id] = newCol
        w.columnIds.push(newCol.id)
        w.focusedColumnId = newCol.id
        w.focusedTileId = newCol.tileIds[0] ?? null
      })
      return newCol.id
    },

    // ---- 过滤器 ----
    setGlobalFilter: (patch) =>
      commit('修改全局过滤器', (d) => {
        const w = d.activeWorkspaceId ? d.workspaces[d.activeWorkspaceId] : null
        if (w) w.filters = { ...w.filters, ...patch }
      }),
    clearGlobalFilters: () =>
      commit('清空全局过滤器', (d) => {
        const w = d.activeWorkspaceId ? d.workspaces[d.activeWorkspaceId] : null
        if (w) w.filters = {}
      }),

    // ---- Gather ----
    toggleGatherSelection: (tileId) =>
      set((s) => ({
        gatherSelection: s.gatherSelection.includes(tileId)
          ? s.gatherSelection.filter((t) => t !== tileId)
          : [...s.gatherSelection, tileId],
      })),
    clearGatherSelection: () => set({ gatherSelection: [] }),
    enterGather: (name) => {
      const s = get()
      const w = ws()
      if (!w) return null
      const picked = s.gatherSelection.length > 0 ? s.gatherSelection : defaultGatherPick(s)
      if (picked.length === 0) {
        set({ toast: { kind: 'warn', text: '没有可聚合的 Tile：先在 Strip 里选中若干 Tile' } })
        return null
      }
      const id = nextId('gather')
      commit('进入 Gather', (d) => {
        const wd = d.workspaces[w.id]
        if (!wd) return
        const inputs = picked
          .map((tid) => d.tiles[tid])
          .filter((t): t is Tile => Boolean(t))
          .map((t) => ({ tileId: t.id, type: t.type }))
        wd.gatherLayouts.push({
          id,
          name: name ?? `聚合 ${wd.gatherLayouts.length + 1}`,
          slots: autoGatherLayout(inputs),
          auto: true,
          createdAt: tick(),
        })
        wd.activeGatherLayoutId = id
      })
      set({ mode: 'gather', gatherSelection: [] })
      return id
    },
    exitGather: () => set({ mode: 'strip' }),
    saveGatherLayout: (name) =>
      commit('保存 Gather 布局', (d) => {
        const w = d.activeWorkspaceId ? d.workspaces[d.activeWorkspaceId] : null
        const g = w ? w.gatherLayouts.find((x) => x.id === w.activeGatherLayoutId) : null
        if (g) g.name = name
      }),
    activateGatherLayout: (id) => {
      commit('切换 Gather 布局', (d) => {
        const w = d.activeWorkspaceId ? d.workspaces[d.activeWorkspaceId] : null
        if (w && w.gatherLayouts.some((g) => g.id === id)) w.activeGatherLayoutId = id
      })
      set({ mode: 'gather' })
    },
    deleteGatherLayout: (id) =>
      commit('删除 Gather 布局', (d) => {
        const w = d.activeWorkspaceId ? d.workspaces[d.activeWorkspaceId] : null
        if (!w) return
        w.gatherLayouts = w.gatherLayouts.filter((g) => g.id !== id)
        if (w.activeGatherLayoutId === id) w.activeGatherLayoutId = w.gatherLayouts[0]?.id ?? null
      }),
    resetGatherLayout: () =>
      commit('恢复自动布局', (d) => {
        const w = d.activeWorkspaceId ? d.workspaces[d.activeWorkspaceId] : null
        const g = w ? w.gatherLayouts.find((x) => x.id === w.activeGatherLayoutId) : null
        if (!g) return
        const inputs = g.slots
          .map((sl) => d.tiles[sl.tileId])
          .filter((t): t is Tile => Boolean(t))
          .map((t) => ({
            tileId: t.id,
            type: t.type,
            pinned: g.slots.find((sl) => sl.tileId === t.id)?.pinned,
          }))
        g.slots = autoGatherLayout(inputs)
        g.auto = true
      }),
    updateGatherSlots: (slots) =>
      commit('调整 Gather 布局', (d) => {
        const w = d.activeWorkspaceId ? d.workspaces[d.activeWorkspaceId] : null
        const g = w ? w.gatherLayouts.find((x) => x.id === w.activeGatherLayoutId) : null
        if (!g) return
        g.slots = slots
        g.auto = false
      }),
    addTileToGather: (tileId) =>
      commit('添加引用 Tile', (d) => {
        const w = d.activeWorkspaceId ? d.workspaces[d.activeWorkspaceId] : null
        const g = w ? w.gatherLayouts.find((x) => x.id === w.activeGatherLayoutId) : null
        if (!g || g.slots.some((s) => s.tileId === tileId)) return
        const inputs = [...g.slots.map((s) => ({ tileId: s.tileId, type: d.tiles[s.tileId]!.type, pinned: s.pinned }))]
        const t = d.tiles[tileId]
        if (!t) return
        inputs.push({ tileId, type: t.type, pinned: false })
        g.slots = autoGatherLayout(inputs)
      }),
    removeTileFromGather: (tileId) =>
      // §4.2.5：Gather 中关闭 Tile 不得关闭原始 Tile——这里只删引用。
      commit('移除引用 Tile', (d) => {
        const w = d.activeWorkspaceId ? d.workspaces[d.activeWorkspaceId] : null
        const g = w ? w.gatherLayouts.find((x) => x.id === w.activeGatherLayoutId) : null
        if (g) g.slots = g.slots.filter((s) => s.tileId !== tileId)
      }),
    gatherToWorkspace: () => {
      const w = ws()
      const g = w ? activeGather(w) : null
      if (!w || !g) return null
      const newId = nextId('ws')
      commit('Gather 复制为 Workspace', (d) => {
        const nw = makeWorkspace(`${g.name}（独立）`)
        nw.id = newId
        nw.bundleId = w.bundleId
        nw.baselineBundleId = w.baselineBundleId
        nw.filters = { ...w.filters }
        // 每个格子变成一列，Tab 组内的 tile 进同一列。
        const byCell = new Map<string, string[]>()
        for (const s of g.slots) {
          const key = `${s.col}:${s.row}`
          byCell.set(key, [...(byCell.get(key) ?? []), s.tileId])
        }
        for (const [key, tileIds] of byCell) {
          const first = g.slots.find((s) => `${s.col}:${s.row}` === key)
          const copies: string[] = []
          for (const tid of tileIds) {
            const t = d.tiles[tid]
            if (!t) continue
            const nt = makeTile(t.type, { ...t, sourceTileId: t.id })
            d.tiles[nt.id] = nt
            copies.push(nt.id)
          }
          if (copies.length === 0) continue
          const col = makeColumn({
            title: '聚合列',
            width: first?.width ?? '1/3',
            tileIds: copies,
            arrange: copies.length > 1 ? 'tabs' : 'stack',
            activeTileId: copies[0] ?? null,
            origin: 'fork',
          })
          d.columns[col.id] = col
          nw.columnIds.push(col.id)
        }
        nw.focusedColumnId = nw.columnIds[0] ?? null
        d.workspaces[newId] = nw
        d.workspaceOrder.push(newId)
        d.activeWorkspaceId = newId
      })
      set({ mode: 'strip' })
      return newId
    },

    // ---- Link Group ----
    createLinkGroup: (name) => {
      const id = nextId('lg')
      const palette = ['#e0a458', '#7fb069', '#c96a6a', '#9b7fc4', '#4fa3a5']
      commit('新建联动组', (d) => {
        d.linkGroups[id] = {
          id,
          name,
          color: palette[Object.keys(d.linkGroups).length % palette.length]!,
          shares: {
            timeRange: true,
            hover: true,
            selection: true,
            zoom: true,
            cursor: true,
            filter: true,
            baseline: true,
          },
        }
      })
      return id
    },
    updateLinkGroup: (id, patch) =>
      commit('修改联动组', (d) => {
        const g = d.linkGroups[id]
        if (g) Object.assign(g, patch)
      }),
    deleteLinkGroup: (id) =>
      commit('删除联动组', (d) => {
        if (id === DEFAULT_LINK_GROUP_ID) return
        delete d.linkGroups[id]
        for (const t of Object.values(d.tiles)) {
          if (t.linkGroupId === id) t.linkGroupId = DEFAULT_LINK_GROUP_ID
        }
      }),

    // ---- 模式 ----
    setMode: (mode) => set((s) => ({ mode, focusReturnMode: mode === 'focus' ? s.mode : s.focusReturnMode })),
    enterFocus: (kind, id) =>
      set((s) => ({ mode: 'focus', focusTarget: { kind, id }, focusReturnMode: s.mode === 'focus' ? s.focusReturnMode : s.mode })),
    exitFocus: () => set((s) => ({ mode: s.focusReturnMode, focusTarget: null })),
    setHover: (sel) => set({ hover: sel }),
    setCommandPaletteOpen: (open) => set({ commandPaletteOpen: open }),
    setScroll: (left, viewportWidth) => set({ scrollLeft: left, viewportWidth }),

    // ---- Bundle ----
    registerBundle: (ref, meta) =>
      set((s) => ({
        bundles: { ...s.bundles, [ref.id]: { ref, meta } },
        loadingBundleId: null,
        loadError: null,
      })),
    bindBundle: (bundleId) =>
      commit('切换 Bundle', (d) => {
        const w = d.activeWorkspaceId ? d.workspaces[d.activeWorkspaceId] : null
        if (w) w.bundleId = bundleId
      }),
    bindBaseline: (bundleId) =>
      commit('设置 Baseline', (d) => {
        const w = d.activeWorkspaceId ? d.workspaces[d.activeWorkspaceId] : null
        if (w) {
          w.baselineBundleId = bundleId
          if (bundleId) w.compareEnabled = true
        }
      }),
    setCompareEnabled: (on) =>
      commit('切换 Compare', (d) => {
        const w = d.activeWorkspaceId ? d.workspaces[d.activeWorkspaceId] : null
        if (w) w.compareEnabled = on
      }),
    unregisterBundle: (bundleId) =>
      set((s) => {
        const next = { ...s.bundles }
        delete next[bundleId]
        return { bundles: next }
      }),
    setLoading: (bundleId, error = null) => set({ loadingBundleId: bundleId, loadError: error }),

    // ---- 诊断标注 ----
    setDiagnosticNote: (key, patch) =>
      commit('标注诊断', (d) => {
        const cur = d.diagnosticNotes[key] ?? { state: 'none', note: '', updatedAt: 0 }
        d.diagnosticNotes[key] = { ...cur, ...patch, updatedAt: tick() }
      }),

    // ---- 持久化 ----
    exportSnapshot: () => {
      const s = get()
      return {
        kind: 'kxc-workbench-layout',
        version: 1,
        savedAt: tick(),
        doc: s.doc,
        bundleRefs: Object.values(s.bundles).map((b) => ({
          id: b.ref.id,
          label: b.ref.label,
          hint: b.ref.hint,
        })),
      }
    },
    importSnapshot: (snap) => {
      if (snap.kind !== 'kxc-workbench-layout') {
        set({ toast: { kind: 'error', text: '不是有效的桌布布局文件' } })
        return
      }
      // 恢复的文档带着旧 id，计数器必须跳过它们，否则后续新建会撞号。
      bumpIdSeq(snap.doc)
      set((s) => ({
        doc: snap.doc,
        undoStack: [...s.undoStack, s.doc].slice(-HISTORY_LIMIT),
        redoStack: [],
        dirty: false,
        toast: { kind: 'info', text: '布局已恢复；若 Bundle 未加载请重新绑定' },
      }))
    },
    markSaved: () => set({ dirty: false }),
    showToast: (kind, text) => set({ toast: { kind, text } }),
  }
})

// ---------------------------------------------------------------------------
// 派生读取 hook
// ---------------------------------------------------------------------------

/**
 * 订阅全局分析上下文。
 *
 * **不要写 `useWorkbench((s) => s.globalContext())`**。globalContext() 每次调用都
 * 返回一个新对象，而 zustand 走 useSyncExternalStore 用 Object.is 比较快照，
 * 新对象会被判定为"状态又变了"，从而触发无限重渲染（React 抛
 * "Maximum update depth exceeded"）。
 * 这里改成订阅稳定的 workspace 引用，再在 useMemo 里派生。
 */
export function useGlobalContext(): AnalysisContext {
  const workspace = useWorkbench((s) =>
    s.doc.activeWorkspaceId ? (s.doc.workspaces[s.doc.activeWorkspaceId] ?? null) : null,
  )
  return useMemo(() => {
    if (!workspace) return EMPTY_CONTEXT
    return {
      ...EMPTY_CONTEXT,
      ...workspace.filters,
      bundleId: workspace.bundleId,
      baselineBundleId: workspace.baselineBundleId,
      candidateBundleId: workspace.compareEnabled ? workspace.bundleId : null,
    }
  }, [workspace])
}

/** 订阅某一列的 Tile 列表。同样避免在 selector 里 map 出新数组。 */
export function useColumnTiles(columnId: string): Tile[] {
  const tileIds = useWorkbench((s) => s.doc.columns[columnId]?.tileIds)
  const tiles = useWorkbench((s) => s.doc.tiles)
  return useMemo(
    () => (tileIds ?? []).map((id) => tiles[id]).filter((t): t is Tile => Boolean(t)),
    [tileIds, tiles],
  )
}

// ---------------------------------------------------------------------------
// 辅助
// ---------------------------------------------------------------------------

/** Drill Down 时按选中对象类型决定新列内容（§7.4.2）。 */
function drillPlan(sel: TileSelection): {
  title: string
  tiles: TileType[]
  context: ContextOverride
} {
  switch (sel.kind) {
    case 'pass':
      return {
        title: `Pass: ${sel.value}`,
        tiles: ['pass_waterfall', 'event_table'],
        context: { pass: sel.value },
      }
    case 'op':
      return {
        title: `算子: ${sel.value}`,
        tiles: ['hotspot_topn', 'event_table'],
        context: { op: sel.value },
      }
    case 'kernel':
      return {
        title: `Kernel: ${sel.value}`,
        tiles: ['kernel_duration_dist', 'event_table'],
        context: { kernel: sel.value },
      }
    case 'shape':
      return {
        title: `Shape: ${sel.value}`,
        tiles: ['shape_cache_heatmap', 'event_table'],
        context: { shapeSignature: sel.value },
      }
    case 'device':
      return { title: `设备: ${sel.value}`, tiles: ['event_table'], context: { device: sel.value } }
    case 'worker':
      return {
        title: `Worker: ${sel.value}`,
        tiles: ['event_table'],
        context: { workerId: Number(sel.value) },
      }
    case 'severity':
      return { title: `级别: ${sel.value}`, tiles: ['logs'], context: { severity: sel.value } }
    case 'range':
      return {
        title: '时间范围',
        tiles: ['perfetto_timeline', 'event_table'],
        context: { timeRange: sel.range ?? null },
      }
    case 'event':
    default:
      return { title: `事件 ${sel.value}`, tiles: ['event_table'], context: {} }
  }
}

/** 没有显式选中时，Gather 默认收拢当前列与相邻列的可聚合 Tile。 */
function defaultGatherPick(s: WorkbenchState): string[] {
  const w = s.doc.activeWorkspaceId ? s.doc.workspaces[s.doc.activeWorkspaceId] : null
  if (!w) return []
  const out: string[] = []
  for (const cid of w.columnIds) {
    const c = s.doc.columns[cid]
    if (!c) continue
    for (const tid of c.tileIds) {
      const t = s.doc.tiles[tid]
      if (t && getTileSpec(t.type).gatherable) out.push(tid)
      if (out.length >= 6) return out
    }
  }
  return out
}
