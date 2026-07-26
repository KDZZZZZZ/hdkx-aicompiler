/**
 * Gather 自动布局（需求 §4.2.1）。
 *
 * 规则原文：
 *   2 个 → 1/2 + 1/2
 *   3 个 → 2/3 主视图 + 1/3 两个纵向
 *   4 个 → 2 × 2
 *   5~6 个 → 3 列，每列最多 2 个
 *   >6 个 → 不再无限缩小，按类型与优先级自动建立 Tab；
 *           Timeline / IR Diff / Execution DAG 等宽视图优先拿 2/3 或 Full；
 *           KPI / 诊断 / 排行榜等紧凑视图填充剩余区域。
 */

import type { GatherSlot, TileType, WidthTier } from './types'

/** 宽视图：需要横向空间才有意义，超过 6 个时优先保宽。 */
const WIDE_TYPES: ReadonlySet<TileType> = new Set<TileType>([
  'perfetto_timeline',
  'ir_diff',
  'pass_waterfall',
  'event_table',
  'shape_cache_heatmap',
])

/** 紧凑视图：小面积仍可读，用来填缝。 */
const COMPACT_TYPES: ReadonlySet<TileType> = new Set<TileType>([
  'kpi',
  'diagnostics',
  'pass_ranking',
  'hotspot_topn',
  'kernel_topn',
  'regression_delta',
])

/** 数值越小越优先获得大面积。 */
function priority(type: TileType): number {
  if (WIDE_TYPES.has(type)) return 0
  if (COMPACT_TYPES.has(type)) return 2
  return 1
}

export interface GatherInput {
  tileId: string
  type: TileType
  /** 已固定的 tile 不参与重排，保持原位（§4.2.2）。 */
  pinned?: boolean
}

/** 一个 tile 在自动布局下最少需要多宽，避免把宽视图压进 1/3。 */
export function minWidthFor(type: TileType): WidthTier {
  if (WIDE_TYPES.has(type)) return '1/2'
  return '1/3'
}

export function autoGatherLayout(tiles: GatherInput[]): GatherSlot[] {
  const n = tiles.length
  if (n === 0) return []

  // 宽视图排前面，保证它们先拿到大格子。
  const ordered = [...tiles].sort((a, b) => priority(a.type) - priority(b.type))
  const slot = (t: GatherInput, col: number, row: number, width: WidthTier, heightFr = 1): GatherSlot => ({
    tileId: t.tileId,
    col,
    row,
    width,
    tabbed: false,
    pinned: t.pinned ?? false,
    heightFr,
  })

  if (n === 1) {
    const t = ordered[0]!
    return [slot(t, 0, 0, 'full')]
  }

  if (n === 2) {
    return [slot(ordered[0]!, 0, 0, '1/2'), slot(ordered[1]!, 1, 0, '1/2')]
  }

  if (n === 3) {
    // 2/3 主视图 + 右侧一列两个纵向 tile
    return [
      slot(ordered[0]!, 0, 0, '2/3'),
      slot(ordered[1]!, 1, 0, '1/3'),
      slot(ordered[2]!, 1, 1, '1/3'),
    ]
  }

  if (n === 4) {
    return [
      slot(ordered[0]!, 0, 0, '1/2'),
      slot(ordered[1]!, 1, 0, '1/2'),
      slot(ordered[2]!, 0, 1, '1/2'),
      slot(ordered[3]!, 1, 1, '1/2'),
    ]
  }

  if (n <= 6) {
    // 3 列，每列最多 2 个：按列优先填，保证第一列先满。
    return ordered.map((t, i) => slot(t, Math.floor(i / 2), i % 2, '1/3'))
  }

  // >6：固定成 3 列，宽视图独占整列高度，其余按列堆叠；
  // 每格超过 2 个时转为 Tab，而不是继续压扁。
  return layoutManyWithTabs(ordered, slot)
}

function layoutManyWithTabs(
  ordered: GatherInput[],
  slot: (t: GatherInput, col: number, row: number, width: WidthTier, heightFr?: number) => GatherSlot,
): GatherSlot[] {
  const out: GatherSlot[] = []
  const wide = ordered.filter((t) => WIDE_TYPES.has(t.type))
  const rest = ordered.filter((t) => !WIDE_TYPES.has(t.type))

  // 第 0 列留给宽视图：第一个独占，其余进同格 Tab。
  let col = 0
  if (wide.length > 0) {
    const width: WidthTier = wide.length === 1 && rest.length <= 2 ? '2/3' : '1/2'
    wide.forEach((t, i) => {
      const s = slot(t, 0, 0, width)
      s.tabbed = i > 0 // 同一 (col,row) 上的后续 tile 转 Tab
      out.push(s)
    })
    col = 1
  }

  // 其余按列填，每列最多 2 行；列数上限 3（含宽视图列）。
  const maxCols = 3
  const cells: GatherInput[][] = []
  const cellCount = Math.max(1, (maxCols - col) * 2)
  for (const t of rest) {
    const idx = cells.length < cellCount ? cells.length : cells.findIndex((c) => c.length === Math.min(...cells.map((x) => x.length)))
    if (cells.length < cellCount) cells.push([t])
    else cells[idx]!.push(t)
  }
  cells.forEach((group, i) => {
    const c = col + Math.floor(i / 2)
    const r = i % 2
    group.forEach((t, k) => {
      const s = slot(t, c, r, '1/3')
      s.tabbed = k > 0
      out.push(s)
    })
  })

  return out
}

/** 同一格里的 tile 集合（用于渲染 Tab 组）。 */
export function groupSlots(slots: GatherSlot[]): Map<string, GatherSlot[]> {
  const m = new Map<string, GatherSlot[]>()
  for (const s of slots) {
    const key = `${s.col}:${s.row}`
    const arr = m.get(key)
    if (arr) arr.push(s)
    else m.set(key, [s])
  }
  return m
}

/** Gather 中的列数。 */
export function gatherColumnCount(slots: GatherSlot[]): number {
  return slots.reduce((max, s) => Math.max(max, s.col + 1), 0)
}
