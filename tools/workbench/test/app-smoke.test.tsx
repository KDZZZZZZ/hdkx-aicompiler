/**
 * 应用冒烟测试：真的把 App 挂到 DOM 上跑一遍。
 *
 * 「能构建」和「能运行」是两回事——类型检查抓不到 doc.tiles 被当成数组这种错误，
 * 只有真挂载才会炸出来。这里用 jsdom 兜住最基本的一层。
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { createRoot, type Root } from 'react-dom/client'
import { act } from 'react'
import { useWorkbench } from '../src/state/store'
import { autoGatherLayout, groupSlots } from '../src/state/gather-layout'
import type { TileType } from '../src/state/types'

// ECharts 在 jsdom 里没有 canvas，直接 init 会炸。图表渲染本身不是本测试的目标，
// 这里只需要保证组件挂载路径通畅。
vi.mock('echarts', () => {
  const chart = {
    setOption: vi.fn(),
    resize: vi.fn(),
    dispose: vi.fn(),
    on: vi.fn(),
    off: vi.fn(),
    getZr: () => ({ on: vi.fn(), off: vi.fn() }),
  }
  return { init: vi.fn(() => chart), dispose: vi.fn(), use: vi.fn(), default: { init: vi.fn(() => chart) } }
})

// Worker 在 jsdom 里不可用；查询层的正确性已由 query-pipeline.test.ts 覆盖。
vi.stubGlobal(
  'Worker',
  class {
    onmessage: unknown = null
    onerror: unknown = null
    postMessage() {}
    terminate() {}
  },
)

let container: HTMLDivElement
let root: Root

beforeEach(() => {
  container = document.createElement('div')
  document.body.appendChild(container)
  root = createRoot(container)
})

afterEach(() => {
  act(() => root.unmount())
  container.remove()
})

async function mountApp() {
  const { default: App } = await import('../src/App')
  await act(async () => {
    root.render(<App />)
  })
}

describe('App 挂载', () => {
  it('渲染不抛异常，并出现主区域', async () => {
    await mountApp()
    expect(container.querySelector('.app')).not.toBeNull()
    expect(container.querySelector('main')).not.toBeNull()
  })

  it('启动时自带一个 Workspace（§3.1 要求自动创建空 Workspace）', async () => {
    await mountApp()
    const s = useWorkbench.getState()
    expect(s.doc.workspaceOrder.length).toBeGreaterThanOrEqual(1)
    expect(s.doc.activeWorkspaceId).not.toBeNull()
  })

  it('新建列并加入 Tile 后，Tile 主体真的被渲染出来（不是"开发中"占位）', async () => {
    await mountApp()
    await act(async () => {
      const s = useWorkbench.getState()
      const col = s.addColumn({ title: '测试列' })
      s.addTile(col, 'kpi')
    })
    // TileBody 挂上了就会有 .tile-body 容器与 data-tile-type
    const body = container.querySelector('[data-tile-type="kpi"]')
    expect(body).not.toBeNull()
    expect(container.textContent).not.toContain('开发中')
  })
})

describe('布局操作与撤销（§12、§17）', () => {
  beforeEach(async () => {
    // 每个用例都从干净状态开始，避免相互污染。
    await act(async () => {
      const s = useWorkbench.getState()
      const w = s.activeWorkspace()
      if (w) for (const c of [...w.columnIds]) s.removeColumn(c)
    })
  })

  it('Consume 把 Tile 收进相邻列，Undo 能还原', async () => {
    await mountApp()
    let colA = ''
    let colB = ''
    let tile = ''
    await act(async () => {
      const s = useWorkbench.getState()
      colA = s.addColumn({ title: 'A' })
      tile = s.addTile(colA, 'kpi')
      s.addTile(colA, 'logs')
      colB = s.addColumn({ title: 'B' })
      s.addTile(colB, 'diagnostics')
    })

    await act(async () => {
      useWorkbench.getState().consume(tile, 'right')
    })
    let d = useWorkbench.getState().doc
    expect(d.columns[colB]?.tileIds).toContain(tile)
    expect(d.columns[colA]?.tileIds).not.toContain(tile)

    await act(async () => {
      useWorkbench.getState().undo()
    })
    d = useWorkbench.getState().doc
    expect(d.columns[colA]?.tileIds).toContain(tile)
  })

  it('Expel 把 Tile 拆到相邻新列，并保留 Tile 的推荐宽度', async () => {
    await mountApp()
    let col = ''
    let tile = ''
    await act(async () => {
      const s = useWorkbench.getState()
      col = s.addColumn({ title: 'A' })
      s.addTile(col, 'kpi')
      tile = s.addTile(col, 'event_table')
    })
    await act(async () => {
      useWorkbench.getState().expel(tile)
    })
    const s = useWorkbench.getState()
    const w = s.activeWorkspace()!
    const idx = w.columnIds.indexOf(col)
    const newCol = s.doc.columns[w.columnIds[idx + 1]!]
    expect(newCol?.tileIds).toEqual([tile])
    // event_table 的 preferredWidth 是 2/3
    expect(newCol?.width).toBe('2/3')
  })

  it('Drill Down 在右侧建新列、继承来源关系，且不动 Pin 列（§4.1.5、§4.1.7）', async () => {
    await mountApp()
    let colA = ''
    let tile = ''
    let pinned = ''
    await act(async () => {
      const s = useWorkbench.getState()
      colA = s.addColumn({ title: 'A' })
      tile = s.addTile(colA, 'pass_ranking')
      pinned = s.addColumn({ title: '固定列' })
      s.addTile(pinned, 'kpi')
      s.togglePin(pinned)
      // 一个未固定的列，应当被下钻替换掉
      const doomed = s.addColumn({ title: '会被替换' })
      s.addTile(doomed, 'logs')
    })

    await act(async () => {
      useWorkbench.getState().drillDown(tile, { kind: 'pass', value: 'fold_constant' })
    })

    const s = useWorkbench.getState()
    const w = s.activeWorkspace()!
    expect(w.columnIds).toContain(pinned) // Pin 列还在
    expect(w.columnIds.some((c) => s.doc.columns[c]?.title === '会被替换')).toBe(false)
    const newCol = s.doc.columns[w.focusedColumnId!]!
    expect(newCol.title).toContain('fold_constant')
    expect(newCol.sourceColumnId).toBe(colA)
    expect(newCol.sourceTileId).toBe(tile)
    expect(newCol.context.pass).toBe('fold_constant')
  })

  it('Hover 不进入 Undo 历史（§17 明确要求）', async () => {
    await mountApp()
    await act(async () => {
      useWorkbench.getState().addColumn({ title: 'A' })
    })
    const depth = useWorkbench.getState().undoStack.length
    await act(async () => {
      useWorkbench.getState().setHover({ kind: 'pass', value: 'x' })
    })
    expect(useWorkbench.getState().undoStack.length).toBe(depth)
  })
})

describe('恢复布局后的 id 分配', () => {
  /**
   * id 计数器是模块级变量，页面一刷新就归零，而从 localStorage 恢复出来的
   * 文档里已经带着 tile-3 这种 id。不推进计数器就会分配出重复 id——
   * 两个对象共用一个 key，删一个连带删另一个，而且要等操作到那一步才暴露。
   */
  it('导入快照后新建的对象不会与已有 id 冲突', async () => {
    await mountApp()

    // 先造出若干对象，拿到一份带较大 id 的快照
    await act(async () => {
      const s = useWorkbench.getState()
      const col = s.addColumn({ title: 'A' })
      for (let i = 0; i < 6; i += 1) s.addTile(col, 'kpi')
    })
    const snap = useWorkbench.getState().exportSnapshot()
    const existing = new Set([
      ...Object.keys(snap.doc.tiles),
      ...Object.keys(snap.doc.columns),
    ])
    expect(existing.size).toBeGreaterThan(6)

    // 模拟"刷新后恢复"：直接导入这份快照
    await act(async () => {
      useWorkbench.getState().importSnapshot(snap)
    })

    // 再新建一批，全部 id 必须是新的
    const created: string[] = []
    await act(async () => {
      const s = useWorkbench.getState()
      const col = s.addColumn({ title: 'B' })
      created.push(col)
      for (let i = 0; i < 6; i += 1) created.push(s.addTile(col, 'logs'))
    })

    for (const id of created) {
      expect(existing.has(id)).toBe(false)
      // id 必须仍是 prefix-<base36> 的正常形状。非生成 id（如默认联动组 'global'）
      // 一旦被误当成计数器来源，就会把序号顶飞，产出 tile-globam 这种乱码。
      expect(id).toMatch(/^(?:ws|col|tile|lg|gather)-[0-9a-z]{1,6}$/)
    }
    expect(new Set(created).size).toBe(created.length)
  })
})

describe('Gather 语义（§4.2）', () => {
  it('从 Gather 移除引用不会删掉原始 Tile', async () => {
    await mountApp()
    let tile = ''
    await act(async () => {
      const s = useWorkbench.getState()
      const col = s.addColumn({ title: 'A' })
      tile = s.addTile(col, 'kpi')
      s.addTile(col, 'logs')
      s.toggleGatherSelection(tile)
      s.enterGather('测试聚合')
    })
    await act(async () => {
      useWorkbench.getState().removeTileFromGather(tile)
    })
    const s = useWorkbench.getState()
    // 引用没了，但原始 Tile 还在
    const g = s.activeWorkspace()!.gatherLayouts.at(-1)!
    expect(g.slots.some((sl) => sl.tileId === tile)).toBe(false)
    expect(s.doc.tiles[tile]).toBeDefined()
  })
})

describe('渲染不得失控', () => {
  /**
   * 这一组专门盯"无限重渲染"。
   *
   * 它出现过两次，都是同一个模式：组件订阅整个 store（useWorkbench() 不带 selector），
   * 或 selector 每次返回新对象，再配合 effect 里 setState —— set() 换掉引用，
   * effect 依赖变化，再 setState，循环不止。前面的用例抓不到它，
   * 因为 React 只在超过嵌套更新上限时才抛错，而那需要真的把 App 完整挂起来跑。
   */
  it('localStorage 里有已保存布局时，启动恢复不会把自己反复触发', async () => {
    // 先造一份自动保存数据，逼 usePersistence 走恢复分支——这正是线上炸掉的路径。
    const seed = useWorkbench.getState().exportSnapshot()
    localStorage.setItem('kxc-workbench-doc-v1', JSON.stringify(seed))
    try {
      await mountApp()
      expect(container.querySelector('.app')).not.toBeNull()
    } finally {
      localStorage.removeItem('kxc-workbench-doc-v1')
    }
  })

  it('命令面板开着时连续改状态，不触发 Maximum update depth', async () => {
    await mountApp()
    await act(async () => {
      useWorkbench.getState().setCommandPaletteOpen(true)
    })
    // 连续写入：每次 commit 都会 structuredClone 整个 doc，把所有引用换新，
    // 足以让"订阅整个 store + effect setState"的写法立刻爆炸。
    await act(async () => {
      const s = useWorkbench.getState()
      for (let i = 0; i < 12; i += 1) {
        const c = s.addColumn({ title: `列 ${i}` })
        s.addTile(c, 'kpi')
        s.setHover({ kind: 'pass', value: `p${i}` })
      }
    })
    expect(useWorkbench.getState().commandPaletteOpen).toBe(true)
    expect(container.querySelector('.command-palette')).not.toBeNull()
  })
})

describe('模式切换都能渲染（§4.2/§4.3/§4.4、§15）', () => {
  it('Gather / Overview / Focus 三种模式挂载均不抛异常', async () => {
    await mountApp()
    let tile = ''
    await act(async () => {
      const s = useWorkbench.getState()
      const col = s.addColumn({ title: 'A' })
      tile = s.addTile(col, 'kpi')
      s.addTile(col, 'pass_ranking')
    })

    for (const mode of ['overview', 'gather'] as const) {
      await act(async () => {
        if (mode === 'gather') {
          useWorkbench.getState().toggleGatherSelection(tile)
          useWorkbench.getState().enterGather('冒烟')
        } else {
          useWorkbench.getState().setMode(mode)
        }
      })
      expect(useWorkbench.getState().mode).toBe(mode)
      expect(container.querySelector('main')?.children.length).toBeGreaterThan(0)
    }

    await act(async () => {
      useWorkbench.getState().enterFocus('tile', tile)
    })
    expect(useWorkbench.getState().mode).toBe('focus')
    expect(container.querySelector('main')?.children.length).toBeGreaterThan(0)

    // Esc 语义：退出 Focus 回到进入前的模式（§4.4 快速返回原位置）
    await act(async () => {
      useWorkbench.getState().exitFocus()
    })
    expect(useWorkbench.getState().mode).toBe('gather')
  })

  it('命令面板能打开并渲染出可执行命令', async () => {
    await mountApp()
    await act(async () => {
      useWorkbench.getState().setCommandPaletteOpen(true)
    })
    const input = container.querySelector('input')
    expect(input).not.toBeNull()
    // 至少要能列出"打开图表"这类命令，空面板说明 commands 没接上
    expect(container.textContent?.length ?? 0).toBeGreaterThan(20)
  })
})

describe('Gather 自动布局规则（§4.2.1）', () => {
  const mk = (n: number, type: TileType = 'kpi') =>
    Array.from({ length: n }, (_, i) => ({ tileId: `t${i}`, type }))

  it('2 个 → 1/2 + 1/2', () => {
    const s = autoGatherLayout(mk(2))
    expect(s.map((x) => x.width)).toEqual(['1/2', '1/2'])
    expect(s.map((x) => x.col)).toEqual([0, 1])
  })

  it('3 个 → 2/3 主视图 + 1/3 两个纵向', () => {
    const s = autoGatherLayout(mk(3))
    expect(s[0]!.width).toBe('2/3')
    expect(s[1]!.width).toBe('1/3')
    expect(s[1]!.col).toBe(1)
    expect(s[2]!.col).toBe(1)
    expect(s[2]!.row).toBe(1)
  })

  it('4 个 → 2×2', () => {
    const s = autoGatherLayout(mk(4))
    expect(groupSlots(s).size).toBe(4)
    expect(new Set(s.map((x) => x.col))).toEqual(new Set([0, 1]))
    expect(new Set(s.map((x) => x.row))).toEqual(new Set([0, 1]))
  })

  it('6 个 → 3 列每列 2 个', () => {
    const s = autoGatherLayout(mk(6))
    expect(new Set(s.map((x) => x.col))).toEqual(new Set([0, 1, 2]))
    expect(s.every((x) => x.width === '1/3')).toBe(true)
  })

  it('超过 6 个不再无限缩小，改为建立 Tab，且列数不超过 3', () => {
    const s = autoGatherLayout(mk(10))
    expect(Math.max(...s.map((x) => x.col))).toBeLessThanOrEqual(2)
    expect(s.some((x) => x.tabbed)).toBe(true)
  })

  it('宽视图优先拿到更大宽度', () => {
    const s = autoGatherLayout([
      { tileId: 'a', type: 'kpi' },
      { tileId: 'b', type: 'perfetto_timeline' },
      { tileId: 'c', type: 'kpi' },
    ])
    const wide = s.find((x) => x.tileId === 'b')!
    expect(wide.width).toBe('2/3')
    expect(wide.col).toBe(0)
  })
})
